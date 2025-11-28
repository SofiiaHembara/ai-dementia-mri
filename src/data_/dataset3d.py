# src/data_/dataset3d.py
from pathlib import Path
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Alzheimer3DDataset(Dataset):
    """
    Повертає тензор форми (C, D, H, W) для 3D-CNN.
    За замовчуванням C=3 (для r3d_18). Якщо хочеш 1 канал — задай channels=1.
    """
    def __init__(self, df, image_shape=(64, 128, 128), augment=False, channels=3, label_map=None):
        self.df = df.reset_index(drop=True)
        self.shape = tuple(image_shape)  # (D, H, W)
        self.augment = augment
        self.channels = int(channels)
        self.label_map = label_map or {"non_demented": 0, "dementia": 1}

    def __len__(self):
        return len(self.df)

    def _load_nifti_tensor(self, path: Path) -> torch.Tensor:
        vol = nib.load(str(path)).get_fdata(dtype=np.float32)  # (H,W,D) або (H,W,D,1)
        # Стиснути зайвий розмір 1, якщо він є
        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = vol[..., 0]
        if vol.ndim != 3:
            raise ValueError(f"Unexpected NIfTI shape {vol.shape} for {path}")

        # Робастна нормалізація + Z-score
        p1, p99 = np.percentile(vol, [1, 99])
        vol = np.clip(vol, p1, p99)
        m, s = vol.mean(), vol.std() + 1e-6
        vol = (vol - m) / s

        vol = np.ascontiguousarray(vol)      # (H,W,D)
        t = torch.from_numpy(vol).float()    # (H,W,D)

        # -> (1,1,D,H,W) для інтерполяції
        t = t.unsqueeze(0).unsqueeze(0)      # (1,1,H,W,D)
        t = t.permute(0, 1, 4, 2, 3)         # (1,1,D,H,W)

        # Ресемплінг у цільовий розмір
        t = F.interpolate(t, size=self.shape, mode="trilinear", align_corners=False)  # (1,1,D,H,W)

        # Канали: 3 для r3d_18 або 1 (якщо мінятимеш перший Conv на in_ch=1)
        if self.channels == 1:
            t = t.squeeze(0)                 # (1,D,H,W)
        else:
            t = t.repeat(1, self.channels, 1, 1, 1).squeeze(0)  # (C,D,H,W)

        return t

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self._load_nifti_tensor(Path(row["path"]))  # (C,D,H,W)

        # Прості аугментації
        if self.augment:
            if random.random() < 0.5:
                x = x.flip(-1)  # W
            if random.random() < 0.5:
                x = x.flip(-2)  # H
            if random.random() < 0.5:
                x = x.flip(-3)  # D

        y = self.label_map[str(row["label"])]
        return x, torch.tensor(y).long()