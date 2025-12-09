# src/data_/dataset3d.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import random

def _to_3ch(x1: torch.Tensor) -> torch.Tensor:
    # x1: (1, D, H, W) -> (3, D, H, W)
    return x1.repeat(3, 1, 1, 1)

def _percentile_norm(vol: np.ndarray, p_lo=10.0, p_hi=99.0, eps=1e-6) -> np.ndarray:
    lo = np.percentile(vol, p_lo)
    hi = np.percentile(vol, p_hi)
    v = (vol - lo) / max(hi - lo, eps)
    v = np.clip(v, 0.0, 1.0)
    return v

def _maybe_reorder_to_DHW(vol: np.ndarray) -> np.ndarray:
    # приводимо до (D, H, W)
    if vol.ndim == 4:
        # інколи форма (H, W, D, C) — беремо перший канал
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Unexpected volume ndim: {vol.ndim}")
    # найчастіше OASIS зберігається як (H, W, D) → (D, H, W)
    H, W, D = vol.shape
    # якщо остання вісь найбільша — імовірно це D
    if D >= H and D >= W:
        vol = np.transpose(vol, (2, 0, 1))
    # тепер vol: (D, H, W)
    return vol

def _resize_trilinear(x: torch.Tensor, D: int, S: int) -> torch.Tensor:
    # x: (1, D0, H0, W0) float32
    x = x.unsqueeze(0)  # (1, 1, D0, H0, W0)
    x = F.interpolate(x, size=(D, S, S), mode="trilinear", align_corners=False)
    return x.squeeze(0)  # (1, D, S, S)

def _random_intensity(x: torch.Tensor) -> torch.Tensor:
    # x: (1 or 3, D, H, W) in [0,1]
    if random.random() < 0.5:
        scale = 0.9 + 0.2 * random.random()  # [0.9, 1.1]
        bias  = (random.random() - 0.5) * 0.1 # [-0.05, 0.05]
        x = x * scale + bias
        x = torch.clamp(x, 0.0, 1.0)
    return x

def _random_flip(x: torch.Tensor) -> torch.Tensor:
    # фліпи по H/W
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])  # flip W
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-2])  # flip H
    return x

class VolumeDataset(Dataset):
    """
    df: pandas.DataFrame з колонками: subject_id, nifti_path, label (dementia|non_demented), split
    Повертає: (x, y, sid), де x: (3, D, S, S)
    """
    def __init__(self, df, size=128, depth=64, augment=False, cache=False):
        import pandas as pd
        assert {"subject_id","nifti_path","label"}.issubset(df.columns)
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.depth = int(depth)
        self.augment = bool(augment)
        self.cache = cache
        self._mem = {} if cache else None

    def __len__(self):
        return len(self.df)

    @torch.no_grad()
    def _load_nifti_tensor(self, path: str) -> torch.Tensor:
        if self.cache and path in self._mem:
            return self._mem[path].clone()

        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = _maybe_reorder_to_DHW(vol)  # (D,H,W)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        vol = _percentile_norm(vol)       # [0,1]

        # -> torch (1, D, H, W), потім трілінійний ресайз
        x = torch.from_numpy(vol).unsqueeze(0)  # (1,D,H,W)
        x = _resize_trilinear(x, D=self.depth, S=self.size)  # (1,D,S,S)

        if self.cache:
            self._mem[path] = x.clone()
        return x

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        sid = str(row["subject_id"])
        y = 1.0 if str(row["label"]).lower() == "dementia" else 0.0

        x = self._load_nifti_tensor(str(row["nifti_path"]))  # (1,D,S,S)
        if self.augment:
            x = _random_flip(x)
            x = _random_intensity(x)

        # очікуємо 3 канали для відеомоделей — дублюємо
        x = _to_3ch(x)  # (3,D,S,S)

        return x.float(), torch.tensor(y, dtype=torch.float32), sid
