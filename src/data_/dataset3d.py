# src/data_/dataset3d.py
from __future__ import annotations

from pathlib import Path
import random
from typing import Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MRIVolumeDataset(Dataset):
    """
    Працює з індексом (CSV → pandas.DataFrame), де є колонки:
      - path         : повний шлях до NIfTI (.nii/.nii.gz/.img/.hdr набори)
      - label        : 'dementia' або 'non_demented' (інші → non_demented)
      - subject_id   : ідентифікатор пацієнта/обстеження
      - split        : train/val/test (для інформації; спліт робиться поза датасетом)

    Повертає:
      x : Tensor (3, D, H, W) — нормований об'єм (z-score), ресемпл до (depth, size, size)
      y : Tensor scalar float (0. або 1.)
      sid : subject_id (str)
    """

    def __init__(
        self,
        df,
        size: int = 96,
        depth: int = 64,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.depth = int(depth)
        self.augment = bool(augment)

        # Перевіримо мінімальний набір колонок
        needed = {"path", "label", "subject_id"}
        if not needed.issubset(set(self.df.columns)):
            raise ValueError(f"DataFrame must contain columns: {needed}, got: {self.df.columns.tolist()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        p = Path(str(row["path"]))
        x = self._load_nifti_tensor(p)  # (3, D, H, W)
        y = 1.0 if str(row["label"]).strip().lower() == "dementia" else 0.0
        sid = str(row["subject_id"])
        return x.float(), torch.tensor(y, dtype=torch.float32), sid

    # ---------------------
    # ВНУТРІШНІ МЕТОДИ
    # ---------------------

    @torch.no_grad()
    def _load_nifti_tensor(self, p: Path) -> torch.Tensor:
        """
        Робустне читання NIfTI в (3, D, H, W).
        Підтримує 2D/3D/4D вхід (4D усереднюємо по каналу, якщо C > 1).
        Ресемпл до (depth, size, size). z-score нормалізація.
        Легкі аугментації для train.
        """
        if not p.exists():
            raise FileNotFoundError(f"NIfTI not found: {p}")

        img = nib.load(str(p))
        vol = img.get_fdata(dtype=np.float32)
        vol = np.nan_to_num(vol)      # без NaN/Inf
        vol = np.squeeze(vol)         # прибрати сингл-тони

        # Привести до (D, H, W)
        if vol.ndim == 2:
            # (H, W) → (1, H, W)
            vol = vol[None, ...]
        elif vol.ndim == 3:
            # Зазвичай NIfTI: (X, Y, Z) → (D, H, W) = (Z, X, Y)
            vol = np.transpose(vol, (2, 0, 1))
        elif vol.ndim == 4:
            # (X, Y, Z, C) → якщо C==1, стискаємо; інакше усереднюємо по C
            if vol.shape[-1] == 1:
                vol = vol[..., 0]
            else:
                vol = vol.mean(axis=-1)
            vol = np.transpose(vol, (2, 0, 1))  # (D, H, W)
        else:
            raise ValueError(f"Unexpected ndim={vol.ndim} for {p}")

        t = torch.from_numpy(vol)  # (D, H, W) float32

        # Ресемпл (trilinear) до цілі (depth, size, size)
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        t = F.interpolate(t, size=(self.depth, self.size, self.size),
                          mode="trilinear", align_corners=False)
        t = t[0, 0]  # (D, H, W)

        # z-score
        mean = t.mean()
        std = t.std().clamp_min(1e-6)
        t = (t - mean) / std

        # Аугментації (very light), тільки якщо augment=True
        if self.augment:
            t = self._augment_3d(t)

        # Канальний стек C=3 для сумісності з r3d-18
        x = t.unsqueeze(0).repeat(3, 1, 1, 1)  # (3, D, H, W)
        return x

    def _augment_3d(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (D, H, W) float32
        Дуже легкі аугментації: фліпи і легка інтенсивність.
        """
        # Фліп по ширині
        if random.random() < 0.5:
            t = torch.flip(t, dims=[2])  # W

        # Фліп по висоті
        if random.random() < 0.3:
            t = torch.flip(t, dims=[1])  # H

        # Фліп по глибині
        if random.random() < 0.3:
            t = torch.flip(t, dims=[0])  # D

        # Легка гамма/контраст
        if random.random() < 0.3:
            gamma = float(np.clip(np.random.normal(1.0, 0.1), 0.7, 1.3))
            t = torch.sign(t) * (t.abs() ** gamma)

        # Невеликий шум
        if random.random() < 0.2:
            noise = torch.randn_like(t) * 0.02
            t = t + noise

        return t