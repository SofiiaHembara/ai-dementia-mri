# scripts/make_2d_slices.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import cv2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_nifti(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata()  # float64
    return data  # (X,Y,Z) або (Z,Y,X) залежно від набору

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
    # рівномірно розкидаємо індекси вздовж осі Z
    if k >= depth:
        idx = np.arange(depth)
    else:
        idx = np.linspace(0, depth - 1, k)
        idx = np.unique(np.round(idx).astype(int))
    return idx.tolist()

def make_slices(nifti_path: Path, out_dir: Path, k: int, size: int, subject_id: str, label: str, split: str):
    data = load_nifti(nifti_path)  # очікуємо (X,Y,Z) або (Z,Y,X)
    # Нормалізуємо орієнтацію до (H,W,Z): в більшості OASIS1 це (X,Y,Z)
    if data.ndim == 4:
        # якщо раптом 4D — беремо перший том
        data = data[..., 0]
    if data.shape[0] < 16 and data.shape[-1] > 32:
        # дуже грубе припущення на випадок незвичної орієнтації
        data = np.moveaxis(data, -1, 0)  # (Z,*,*) → далі приведемо до (H,W,Z)
    # тепер приведемо до (H,W,Z)
    if data.shape[2] < 8 and data.shape[0] > 32:
        # якщо Z випадково на першій осі, обернемо у (H,W,Z)
        data = np.moveaxis(data, 0, 2)

    H, W, Z = data.shape
    idxs = pick_k_slices(Z, k)

    rows = []
    class_dir = out_dir / label
    ensure_dir(class_dir)

    for z in idxs:
        sl = data[:, :, z].astype(np.float32)

        # легке стабільне підсилення контрасту: обріжемо по перцентилях
        lo, hi = np.percentile(sl, [1, 99])
        sl = np.clip(sl, lo, hi)
        sl = (sl - lo) / (hi - lo + 1e-6)

        # z-score (локальний до зрізу) + minmax для 0..1
        sl = zscore(sl)
        sl = minmax01(sl)
        sl = resize_2d(sl, size)

        # перетворимо у uint8 і збережемо PNG (1-канальний)
        png = (sl * 255).astype(np.uint8)
        out_path = class_dir / f"{subject_id}_z{z:03d}.png"
        cv2.imwrite(str(out_path), png)

        rows.append({
            "subject_id": subject_id,
            "path": str(out_path),
            "label": label,
            "split": split,
            "slice_z": int(z),
        })
    return rows

def main(args):
    index_csv = Path(args.index_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(index_csv)
    needed = {"subject_id", "nifti_path", "label", "split"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"❌ {index_csv} must contain columns: {needed}")

    all_rows = []
    for _, row in df.iterrows():
        subject_id = str(row["subject_id"])
        nifti_path = Path(row["nifti_path"])
        label = str(row["label"])
        split = str(row["split"])

        if not nifti_path.exists():
            print(f"⚠️  Missing NIfTI for {subject_id}: {nifti_path}")
            continue

        rows = make_slices(
            nifti_path=nifti_path,
            out_dir=out_dir,
            k=args.k,
            size=args.size,
            subject_id=subject_id,
            label=label,
            split=split,
        )
        all_rows.extend(rows)

    out_index = out_dir / "index_2d.csv"
    pd.DataFrame(all_rows).to_csv(out_index, index=False)
    print(f"✅ Saved: {out_index} (N={len(all_rows)})")
    df2 = pd.DataFrame(all_rows)
    if not df2.empty:
        print(df2.groupby(["split", "label"]).size())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, help="CSV з колонками subject_id,nifti_path,label,split")
    ap.add_argument("--out_dir",   required=True, help="Куди зберегти PNG зрізи та index_2d.csv")
    ap.add_argument("--k", type=int, default=10, help="Кількість зрізів на пацієнта")
    ap.add_argument("--size", type=int, default=224, help="Розмір вихідного PNG (size x size)")
    args = ap.parse_args()
    main(args)
