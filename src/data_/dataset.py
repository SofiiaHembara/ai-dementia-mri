from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def _zscore(img: np.ndarray) -> np.ndarray:
    m, s = img.mean(), img.std() + 1e-6
    return (img - m) / s

def _random_affine(img: np.ndarray, max_deg=10, max_shift=0.05, scale_range=(0.95, 1.05)):
    h, w = img.shape
    deg = np.random.uniform(-max_deg, max_deg)
    tx = np.random.uniform(-max_shift, max_shift) * w
    ty = np.random.uniform(-max_shift, max_shift) * h
    sc = np.random.uniform(scale_range[0], scale_range[1])
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, sc)
    M[:, 2] += [tx, ty]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def _random_intensity(img: np.ndarray, gamma_range=(0.9, 1.1), contrast_range=(0.9, 1.1), bias_range=(-0.1, 0.1)):
    g = np.random.uniform(*gamma_range)
    c = np.random.uniform(*contrast_range)
    b = np.random.uniform(*bias_range)
    x = img.copy()
    x = np.sign(x) * (np.abs(x) ** g)
    x = c * x + b
    return np.clip(x, -3.5, 3.5)

def _random_crop(img: np.ndarray, out_size=224, crop_scale=(0.90, 1.0)):
    h, w = img.shape
    scale = np.random.uniform(*crop_scale)
    th, tw = int(h * scale), int(w * scale)
    if th < out_size or tw < out_size:
        return img
    y0 = np.random.randint(0, h - th + 1)
    x0 = np.random.randint(0, w - tw + 1)
    crop = img[y0:y0+th, x0:x0+tw]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

def _maybe(p: float) -> bool:
    return np.random.rand() < p

class Alzheimer2DDataset(Dataset):
    def __init__(
        self,
        df,
        label_map=None,
        augment: bool = False,
        image_size: int = 224,
        use_clahe: bool = False,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.label_map = label_map or {"non_demented": 0, "dementia": 1}
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

        # підготуємо CLAHE, якщо треба
        self._clahe = None
        if self.use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))

    def __len__(self):
        return len(self.df)

    def _read_gray(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if self._clahe is not None:
            img = self._clahe.apply(img)
        return img.astype(np.float32)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if _maybe(0.5):
            img = np.ascontiguousarray(img[:, ::-1])
        if _maybe(0.30):
            img = _random_affine(img, max_deg=10, max_shift=0.05, scale_range=(0.95, 1.05))
        if _maybe(0.25):
            if _maybe(0.5):
                img = cv2.GaussianBlur(img, (3, 3), 0.6)
            else:
                noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
                img = img + noise
        if _maybe(0.35):
            img = _random_intensity(img, gamma_range=(0.95, 1.05), contrast_range=(0.9, 1.1), bias_range=(-0.05, 0.05))
        if _maybe(0.30):
            img = _random_crop(img, out_size=self.image_size, crop_scale=(0.92, 1.0))
        return img

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = Path(row["path"])
        y = self.label_map[row["label"]]

        img = self._read_gray(p)

        img = _zscore(img)

        if self.augment:
            img = self._augment(img)
        x = np.stack([img, img, img], axis=0).astype(np.float32)  # (3, H, W)

        x = torch.from_numpy(x).float()
        y = torch.tensor(int(y)).long()
        return x, y
