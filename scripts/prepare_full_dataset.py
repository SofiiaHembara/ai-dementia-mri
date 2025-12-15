#!/usr/bin/env python3
"""
Prepare Full Dataset from OASIS Discs
- Extracts all discs (disc1-disc12) if needed
- Processes all subjects and creates 2D slices
- Splits into train/test/validation sets (patient-level)
- Creates index CSV for cross-validation
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.oasis1_prepare import (
    autodiscover_clinical,
    find_best_t1,
    find_subject_dirs,
    load_clinical,
)


def extract_disc(disc_num: int, data_dir: Path) -> Path:
    """Extract a disc tar.gz file if not already extracted"""
    tar_file = data_dir / f"oasis_cross-sectional_disc{disc_num}.tar.gz"
    disc_dir = data_dir / "raw" / f"disc{disc_num}"

    if disc_dir.exists() and any(disc_dir.iterdir()):
        print(f"✅ Disc {disc_num} already extracted")
        return disc_dir

    if not tar_file.exists():
        print(f"⚠️  Disc {disc_num} tar.gz not found: {tar_file}")
        return None

    print(f"📦 Extracting disc {disc_num}...")
    disc_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["tar", "-xzf", str(tar_file), "-C", str(disc_dir)],
            check=True,
            capture_output=True,
        )
        print(f"✅ Extracted disc {disc_num}")
        return disc_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting disc {disc_num}: {e}")
        return None


def process_all_discs(data_dir: Path, extract_all: bool = False) -> list:
    """Process all available discs and return list of subject directories"""
    # Check both data_dir and data_dir/raw for discs
    possible_dirs = [data_dir, data_dir / "raw"]
    all_subjects = []

    # Find all disc directories
    disc_dirs = []
    for base_dir in possible_dirs:
        if base_dir.exists():
            discs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("disc")])
            disc_dirs.extend(discs)
    
    disc_dirs = sorted(set(disc_dirs))  # Remove duplicates
    print(f"📁 Found {len(disc_dirs)} disc directories")

    # Extract missing discs if requested
    if extract_all:
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for disc_num in range(1, 13):  # disc1 to disc12
            disc_dir = raw_dir / f"disc{disc_num}"
            if not disc_dir.exists() or not any(disc_dir.iterdir()):
                extracted = extract_disc(disc_num, data_dir)
                if extracted and extracted not in disc_dirs:
                    disc_dirs.append(extracted)
        # Re-check after extraction
        if raw_dir.exists():
            new_discs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("disc")])
            disc_dirs.extend(new_discs)
            disc_dirs = sorted(set(disc_dirs))  # Remove duplicates

    # Find all subject directories from all disc directories
    for disc_dir in disc_dirs:
        if disc_dir.is_dir():
            subjects = find_subject_dirs(disc_dir)
            all_subjects.extend(subjects)
            print(f"📊 Found {len(subjects)} subjects in {disc_dir.name}")

    print(f"📊 Total subjects found: {len(all_subjects)}")
    return all_subjects


def create_2d_slices_from_nifti(
    nifti_path: Path,
    out_dir: Path,
    subject_id: str,
    label: str,
    split: str,
    k: int = 10,
    size: int = 224,
) -> list:
    """Create 2D slices from NIfTI file (reusing logic from make_2d_slices.py)"""
    import cv2

    def load_nifti(path: Path):
        img = nib.load(str(path))
        return img.get_fdata()

    def zscore(img: np.ndarray):
        m = img.mean()
        s = img.std() + 1e-6
        return (img - m) / s

    def minmax01(x: np.ndarray):
        x = x - x.min()
        d = x.max() + 1e-8
        return x / d

    def resize_2d(img: np.ndarray, size: int):
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    def pick_k_slices(depth: int, k: int):
        if k >= depth:
            idx = np.arange(depth)
        else:
            idx = np.linspace(0, depth - 1, k)
            idx = np.unique(np.round(idx).astype(int))
        return idx.tolist()

    try:
        data = load_nifti(nifti_path)

        # Normalize orientation to (H,W,Z)
        if data.ndim == 4:
            data = data[..., 0]
        if data.shape[0] < 16 and data.shape[-1] > 32:
            data = np.moveaxis(data, -1, 0)
        if data.shape[2] < 8 and data.shape[0] > 32:
            data = np.moveaxis(data, 0, 2)

        H, W, Z = data.shape
        idxs = pick_k_slices(Z, k)

        rows = []
        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)

        for z in idxs:
            sl = data[:, :, z].astype(np.float32)

            # Contrast enhancement
            lo, hi = np.percentile(sl, [1, 99])
            sl = np.clip(sl, lo, hi)
            sl = (sl - lo) / (hi - lo + 1e-6)

            # Normalize
            sl = zscore(sl)
            sl = minmax01(sl)
            sl = resize_2d(sl, size)

            # Save as PNG
            png = (sl * 255).astype(np.uint8)
            out_path = class_dir / f"{subject_id}_z{z:03d}.png"
            cv2.imwrite(str(out_path), png)

            rows.append(
                {
                    "subject_id": subject_id,
                    "path": str(out_path),
                    "label": label,
                    "split": split,
                    "slice_z": int(z),
                }
            )
        return rows
    except Exception as e:
        print(f"⚠️  Error processing {nifti_path}: {e}")
        return []


def main(args):
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OASIS Dataset Preparation")
    print("=" * 60)

    # Step 1: Process all discs
    print("\n📦 Step 1: Processing discs...")
    all_subjects = process_all_discs(data_dir, extract_all=args.extract)

    if not all_subjects:
        print("❌ No subjects found!")
        return

    # Step 2: Find clinical data
    print("\n📄 Step 2: Loading clinical data...")
    if args.clinical:
        clinical_path = Path(args.clinical).resolve()
    else:
        clinical_path = autodiscover_clinical(data_dir / "raw")
        if not clinical_path:
            print("⚠️  Clinical file not found. Using dummy labels (all non_demented)")
            clinical_path = None

    if clinical_path:
        clin_df = load_clinical(clinical_path)
        print(f"✅ Loaded clinical data for {len(clin_df)} subjects")
    else:
        # Create dummy clinical data
        clin_df = pd.DataFrame(
            {
                "subject_id": [s.name for s in all_subjects],
                "cdr": [0.0] * len(all_subjects),
                "label": ["non_demented"] * len(all_subjects),
            }
        )
        print("⚠️  Using dummy labels (all non_demented)")

    # Step 3: Find NIfTI files and match with clinical data
    print("\n🔍 Step 3: Finding NIfTI files...")
    nifti_rows = []
    for subj_dir in all_subjects:
        sid = subj_dir.name
        t1 = find_best_t1(subj_dir)
        if t1 is None:
            continue

        # Copy to processed directory
        nifti_out = out_dir / "nifti" / f"{sid}_T1w.nii.gz"
        nifti_out.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = nib.load(str(t1))
            nib.save(img, str(nifti_out))
            nifti_rows.append({"subject_id": sid, "nifti_path": str(nifti_out)})
        except Exception as e:
            print(f"⚠️  Error copying NIfTI for {sid}: {e}")

    df_nifti = pd.DataFrame(nifti_rows)
    print(f"✅ Found {len(df_nifti)} NIfTI files")

    # Step 4: Merge with clinical data
    print("\n🔗 Step 4: Merging with clinical data...")
    df = df_nifti.merge(clin_df, on="subject_id", how="left")
    df = df.dropna(subset=["label"]).copy()
    print(f"✅ Merged dataset: {len(df)} subjects")
    print(f"   Label distribution:\n{df['label'].value_counts()}")

    # Step 5: Patient-level split
    print("\n✂️  Step 5: Creating patient-level splits...")
    subj_unique = df[["subject_id", "label"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        subj_unique["subject_id"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=subj_unique["label"],
    )
    labels_train = subj_unique.set_index("subject_id").loc[train_ids]["label"]
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=args.val_size / (1 - args.test_size),  # Adjust for remaining after test split
        random_state=args.seed,
        stratify=labels_train,
    )

    split_map = {sid: "train" for sid in train_ids}
    split_map.update({sid: "val" for sid in val_ids})
    split_map.update({sid: "test" for sid in test_ids})
    df["split"] = df["subject_id"].map(split_map)

    print(f"✅ Splits created:")
    print(df["split"].value_counts().sort_index())
    print(f"\n   Per split label distribution:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    # Step 6: Create 2D slices
    print(f"\n🖼️  Step 6: Creating 2D slices (k={args.k_slices}, size={args.size})...")
    all_slice_rows = []
    slices_dir = out_dir / "2d_slices"

    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"   Processing {idx+1}/{len(df)}...")

        slice_rows = create_2d_slices_from_nifti(
            Path(row["nifti_path"]),
            slices_dir,
            row["subject_id"],
            row["label"],
            row["split"],
            k=args.k_slices,
            size=args.size,
        )
        all_slice_rows.extend(slice_rows)

    # Step 7: Create index CSV
    print("\n💾 Step 7: Creating index CSV...")
    df_slices = pd.DataFrame(all_slice_rows)
    index_path = out_dir / "index.csv"
    df_slices[["path", "label", "split"]].to_csv(index_path, index=False)

    print(f"✅ Saved index: {index_path}")
    print(f"   Total slices: {len(df_slices)}")
    print(f"\n   Final distribution:")
    print(df_slices.groupby(["split", "label"]).size().unstack(fill_value=0))

    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"\n📁 Output directory: {out_dir}")
    print(f"📄 Index CSV: {index_path}")
    print(f"🖼️  2D slices: {slices_dir}")
    print(f"\n💡 Next step: Run cross-validation with:")
    print(f"   python -m src.eval.cross_validation --index_csv {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare full OASIS dataset with train/test/val splits")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory containing raw discs and tar.gz files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/full_dataset",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--clinical",
        type=str,
        default=None,
        help="Path to clinical CSV/XLSX (auto-discovered if not provided)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract all tar.gz files if not already extracted",
    )
    parser.add_argument(
        "--k_slices",
        type=int,
        default=10,
        help="Number of slices per subject",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output image size",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test set size (fraction)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Validation set size (fraction of remaining after test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    main(args)

