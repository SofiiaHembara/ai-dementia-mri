# src/data_/meddino_dataset.py
# -*- coding: utf-8 -*-
"""
MEDdino-specific dataset for medical MRI preprocessing.
Implements preprocessing tailored for medical imaging foundation models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2


class MEDdinoSliceDataset(Dataset):
    """
    Dataset for MEDdino model with medical-specific preprocessing.
    
    MEDdino expects:
    - Grayscale medical images converted to 3-channel
    - Intensity normalization specific to MRI
    - Medical-aware augmentations
    
    CSV columns: slice_path, subject_id, split, label
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        size: int = 224, 
        train: bool = True,
        use_clahe: bool = False,
        normalize_mode: str = "zscore"  # "zscore", "minmax", or "imagenet"
    ):
        """
        Args:
            df: DataFrame with columns [slice_path, subject_id, split, label]
            size: Image size (default 224 for ViT-B/16)
            train: Enable training augmentations
            use_clahe: Apply CLAHE histogram equalization (helps with MRI contrast)
            normalize_mode: Normalization strategy
                - "zscore": z-score normalization per image
                - "minmax": min-max to [0, 1]
                - "imagenet": ImageNet mean/std (for comparison)
        """
        self.df = df.reset_index(drop=True).copy()
        self.size = size
        self.train = train
        self.use_clahe = use_clahe
        self.normalize_mode = normalize_mode
        
        # Ensure paths are strings
        self.df["slice_path"] = self.df["slice_path"].astype(str)
        
        # CLAHE for contrast enhancement (common in medical imaging)
        self.clahe = None
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Build transforms
        self.transform = self._build_transforms()
    
    def _build_transforms(self):
        """Build torchvision transforms pipeline."""
        transforms = []
        
        if self.train:
            # Medical-aware augmentations (conservative to preserve anatomy)
            transforms.extend([
                T.Resize(int(self.size * 1.1), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomCrop(self.size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomAffine(
                    degrees=0, 
                    translate=(0.05, 0.05), 
                    scale=(0.95, 1.05),
                    interpolation=T.InterpolationMode.BICUBIC
                ),
            ])
        else:
            # Validation/test: simple resize + center crop
            transforms.extend([
                T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.size),
            ])
        
        # Convert grayscale to 3-channel (required for ViT)
        transforms.append(T.Grayscale(num_output_channels=3))
        
        # To tensor
        transforms.append(T.ToTensor())
        
        # Normalization based on mode
        if self.normalize_mode == "imagenet":
            # Standard ImageNet normalization
            transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        elif self.normalize_mode == "minmax":
            # Min-max to [0, 1] (ToTensor already does this)
            pass
        elif self.normalize_mode == "zscore":
            # Per-channel z-score (will be applied per-image in __getitem__)
            pass
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        
        return T.Compose(transforms)
    
    def _preprocess_medical_image(self, img: np.ndarray) -> np.ndarray:
        """
        Medical-specific preprocessing before torchvision transforms.
        
        Args:
            img: Grayscale numpy array (uint8)
            
        Returns:
            Preprocessed image (uint8)
        """
        # Apply CLAHE for better contrast (optional)
        if self.clahe is not None:
            img = self.clahe.apply(img)
        
        return img
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row["slice_path"]
        try:
            img = Image.open(img_path).convert("L")  # Load as grayscale
        except Exception as e:
            raise FileNotFoundError(f"Cannot load {img_path}: {e}")
        
        # Convert PIL to numpy for medical preprocessing
        img_np = np.array(img, dtype=np.uint8)
        
        # Medical preprocessing
        img_np = self._preprocess_medical_image(img_np)
        
        # Convert back to PIL for torchvision transforms
        img = Image.fromarray(img_np, mode="L")
        
        # Apply torchvision transforms
        x = self.transform(img)  # (3, H, W)
        
        # Per-image z-score normalization (after transform, if enabled)
        if self.normalize_mode == "zscore":
            # x is already a tensor in [0, 1] from ToTensor
            # Apply z-score per channel (all channels are identical for grayscale)
            mean = x.mean(dim=[1, 2], keepdim=True)
            std = x.std(dim=[1, 2], keepdim=True) + 1e-6
            x = (x - mean) / std
        
        # Label: dementia (1) vs non_demented (0)
        label_str = str(row["label"]).lower().strip()
        y = 1.0 if label_str == "dementia" else 0.0
        y = torch.tensor(y, dtype=torch.float32)
        
        # Subject ID for patient-level aggregation
        subject_id = str(row["subject_id"])
        
        return x, y, subject_id


class MEDdino3DDataset(Dataset):
    """
    3D volumetric dataset for MEDdino 3D models (future extension).
    Currently placeholder for potential 3D MEDdino variants.
    """
    
    def __init__(self, df: pd.DataFrame, size: int = 224, depth: int = 64, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.depth = depth
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        raise NotImplementedError("3D MEDdino dataset not yet implemented")


def create_meddino_dataloaders(
    index_csv: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    use_clahe: bool = False,
    normalize_mode: str = "zscore"
):
    """
    Convenience function to create train/val/test dataloaders.
    
    Args:
        index_csv: Path to CSV with columns [slice_path, subject_id, split, label]
        batch_size: Batch size
        img_size: Image size (224 for ViT-B/16, 384 for ViT-L/16)
        num_workers: Number of data loading workers
        use_clahe: Apply CLAHE preprocessing
        normalize_mode: Normalization mode ("zscore", "minmax", "imagenet")
        
    Returns:
        train_loader, val_loader, test_loader
    """
    import pandas as pd
    from torch.utils.data import DataLoader
    
    # Load and split data
    df = pd.read_csv(index_csv)
    
    # Check required columns
    required_cols = {"slice_path", "subject_id", "split", "label"}
    assert required_cols.issubset(df.columns), f"CSV must have columns: {required_cols}"
    
    # Filter by split
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} slices, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_ds = MEDdinoSliceDataset(
        train_df, size=img_size, train=True, 
        use_clahe=use_clahe, normalize_mode=normalize_mode
    )
    val_ds = MEDdinoSliceDataset(
        val_df, size=img_size, train=False,
        use_clahe=use_clahe, normalize_mode=normalize_mode
    )
    test_ds = MEDdinoSliceDataset(
        test_df, size=img_size, train=False,
        use_clahe=use_clahe, normalize_mode=normalize_mode
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

