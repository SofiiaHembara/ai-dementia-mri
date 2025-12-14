#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_index_oasis_full_2d.py

Будує повний 2D-індекс для OASIS:
- обходить data/raw/disc*/OAS1_XXXX_MR1
- для кожного суб'єкта бере оброблений .img (або RAW fallback)
- ріже об'єм на центральні осмислені слайси
- зберігає PNG у data/processed/oasis_full_2d/<subject_id>/
- створює data/index_oasis_full_2d.csv з колонками:
    subject_id, slice_path, label, split

label:
- non_demented  : CDR == 0
- dementia      : CDR > 0
"""

from __future__ import annotations
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image


def find_subject_dirs(nifti_root: Path) -> list[Path]:
    """
    Шукає всі папки OAS1_XXXX_MR1 в disc* всередині nifti_root.
    """
    subj_dirs: list[Path] = []
    for disc in sorted(nifti_root.glob("disc*")):
        if not disc.is_dir():
            continue
        for subj in sorted(disc.glob("OAS1_*_MR1")):
            if subj.is_dir():
                subj_dirs.append(subj)
    return subj_dirs


def choose_img_path(subj_dir: Path) -> Path | None:
    """
    Обирає один .img для даного суб'єкта:
    1) PROCESSED/MPRAGE/T88_111/*masked_gfc.img (найкращий варіант)
    2) PROCESSED/MPRAGE/T88_111/*.img
    3) RAW/*mpr-1_anon.img

    Якщо нічого не знайдено — повертає None.
    """
    # 1) masked_gfc
    cand1 = sorted(
        subj_dir.glob(
            "PROCESSED/MPRAGE/T88_111/*masked_gfc.img"
        )
    )
    if cand1:
        return cand1[0]

    # 2) будь-який .img з T88_111
    cand2 = sorted(subj_dir.glob("PROCESSED/MPRAGE/T88_111/*.img"))
    if cand2:
        return cand2[0]

    # 3) RAW mpr-1_anon
    cand3 = sorted(subj_dir.glob("RAW/*mpr-1_anon.img"))
    if cand3:
        return cand3[0]

    return None


def load_volume(img_path: Path) -> np.ndarray:
    """
    Читає .img/.hdr як об'єм (X, Y, Z) у float32.
    """
    img = nib.load(str(img_path))
    data = img.get_fdata().astype(np.float32)
    # іноді dim порядок може бути не (X,Y,Z), але для нашого демо цього достатньо
    return data


def slice_volume_to_pngs(
    vol: np.ndarray,
    subject_id: str,
    out_root: Path,
    axis: int = 2,
    keep_mid_fraction: float = 0.6,
    min_dynamic_range: float = 1e-3,
) -> list[Path]:
    """
    Ріже 3D-об'єм на 2D-слайси й зберігає їх як PNG.

    - axis: вздовж якої осі різати (2 — типово аксіальні слайси).
    - keep_mid_fraction: яку частину центральних слайсів лишити (0.6 = 60% по центру).
    - min_dynamic_range: якщо max-min < threshold, вважаємо слайс "порожнім" і пропускаємо.
    """
    vol = np.nan_to_num(vol)

    # Перекладаємо так, щоб вісь слайсів була останньою
    if axis != vol.ndim - 1:
        vol = np.moveaxis(vol, axis, -1)

    nz = vol.shape[-1]
    if nz < 4:
        return []

    start = int(nz * (1.0 - keep_mid_fraction) / 2.0)
    end = int(nz * (1.0 + keep_mid_fraction) / 2.0)
    start = max(start, 0)
    end = min(end, nz)

    out_dir = out_root / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for idx in range(start, end):
        sl = vol[..., idx]
        sl = np.nan_to_num(sl)

        # 🔹 КЛЮЧ: стискаємо всі singleton-виміри
        sl = np.squeeze(sl)

        # Нам потрібен 2D-слайс (H, W). Все інше — скіпаємо.
        if sl.ndim != 2:
            print(f"[WARN] Slice {subject_id} z={idx} має форму {sl.shape}, пропускаю.")
            continue

        # Фільтр по динамічному діапазону (щоб викинути майже чорні/константні слайси)
        dr = float(sl.max() - sl.min())
        if dr < min_dynamic_range:
            continue

        # Нормалізація до [0, 255]
        sl_norm = sl - sl.min()
        if sl_norm.max() > 0:
            sl_norm = sl_norm / sl_norm.max()
        sl_uint8 = (sl_norm * 255.0).clip(0, 255).astype(np.uint8)

        out_path = out_dir / f"{subject_id}_z{idx:03d}.png"
        try:
            Image.fromarray(sl_uint8).save(out_path)
            paths.append(out_path)
        except Exception as e:
            print(f"[ERROR] Не вдалося зберегти слайс {out_path}: {e}")
            continue

    return paths

def load_clinical_labels(clinical_csv: Path) -> dict[str, str]:
    """
    Читає clinical CSV (oasis_cross-sectional.csv) і створює mapping:
        subject_id -> label ('dementia' / 'non_demented')

    Вважаємо:
        - ID має формат 'OAS1_0001_MR1'
        - CDR == 0 -> non_demented
        - CDR >  0 -> dementia
    """
    df = pd.read_csv(clinical_csv)
    # Намагаємось знайти колонку з ID
    id_col = None
    for cand in ["ID", "id", "Subject ID", "Subject", "MR ID"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(
            f"Не знайшла колонку з ID в clinical_csv. Є колонки: {list(df.columns)}"
        )

    if "CDR" not in df.columns:
        raise ValueError("clinical_csv не містить колонки 'CDR'.")

    labels: dict[str, str] = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        if not sid:
            continue
        cdr = row["CDR"]
        try:
            cdr_val = float(cdr)
        except Exception:
            continue
        if np.isnan(cdr_val):
            continue

        if cdr_val == 0.0:
            label = "non_demented"
        else:
            label = "dementia"
        labels[sid] = label

    return labels


def stratified_split_subjects(
    subj_df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict[str, str]:
    """
    Робить patient-level split з максимальною стратіфікацією.
    Повертає dict: subject_id -> split ('train'/'val'/'test').

    Робимо:
        - train / temp  (temp ~ val+test, 40%)
        - temp -> val / test (50/50 від temp)
    Якщо стратіфікація неможлива (мало зразків) — fallback на non-stratified split.
    """
    assert abs(train_frac + val_frac - 0.8) < 1e-6, "train_frac + val_frac має бути 0.8 (test=0.2)"
    ids = subj_df["subject_id"].values
    labels = subj_df["label"].values

    def safe_split(X, y, test_size, stratify_labels, stage_name):
        # Якщо замало семплів певного класу — робимо без stratify
        vc = pd.Series(stratify_labels).value_counts()
        if len(vc) < 2 or (vc < 2).any():
            # попередження можна вивести, але для простоти просто fallback
            return train_test_split(
                X, test_size=test_size, random_state=seed
            )
        else:
            return train_test_split(
                X,
                test_size=test_size,
                random_state=seed,
                stratify=stratify_labels,
            )

    # 1) train / temp
    train_ids, temp_ids = safe_split(
        ids, labels, test_size=(1.0 - train_frac), stratify_labels=labels, stage_name="train/temp"
    )

    # 2) val / test з temp
    temp_df = subj_df[subj_df["subject_id"].isin(temp_ids)].copy()
    temp_labels = temp_df["label"].values
    val_size = val_frac / (val_frac + (1 - train_frac - val_frac))  # 0.2 / 0.4 = 0.5
    val_ids, test_ids = safe_split(
        temp_df["subject_id"].values,
        temp_labels,
        test_size=0.5,
        stratify_labels=temp_labels,
        stage_name="val/test",
    )

    split_map: dict[str, str] = {}
    for sid in train_ids:
        split_map[str(sid)] = "train"
    for sid in val_ids:
        split_map[str(sid)] = "val"
    for sid in test_ids:
        split_map[str(sid)] = "test"

    return split_map


def build_full_2d_index(
    nifti_root: Path,
    clinical_csv: Path,
    out_index_csv: Path,
    out_slices_root: Path,
) -> None:
    """
    Основна функція: будує full 2D index по всіх суб'єктах.
    """
    nifti_root = nifti_root.resolve()
    out_slices_root = out_slices_root.resolve()
    out_slices_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] NIFTI root:      {nifti_root}")
    print(f"[INFO] Clinical CSV:    {clinical_csv}")
    print(f"[INFO] Slices out root: {out_slices_root}")
    print(f"[INFO] Out index CSV:   {out_index_csv}")

    labels_map = load_clinical_labels(clinical_csv)
    print(f"[INFO] Clinical subjects with labels: {len(labels_map)}")

    subj_dirs = find_subject_dirs(nifti_root)
    print(f"[INFO] Found subject dirs: {len(subj_dirs)}")

    rows = []
    subj_label_records = []

    for subj_dir in tqdm(subj_dirs, desc="Processing subjects"):
        subject_id = subj_dir.name  # типу OAS1_0001_MR1

        if subject_id not in labels_map:
            # немає клінічного лейблу → пропускаємо
            continue
        label = labels_map[subject_id]

        img_path = choose_img_path(subj_dir)
        if img_path is None:
            print(f"[WARN] Не знайшла жодного .img для {subj_dir}, пропускаю.")
            continue

        try:
            vol = load_volume(img_path)
        except Exception as e:
            print(f"[WARN] Не вдалося прочитати {img_path}: {e}")
            continue

        slice_paths = slice_volume_to_pngs(
            vol,
            subject_id=subject_id,
            out_root=out_slices_root,
            axis=2,
            keep_mid_fraction=0.6,
            min_dynamic_range=1e-3,
        )

        if not slice_paths:
            print(f"[WARN] Немає валідних слайсів для {subject_id}, пропускаю.")
            continue

        # записуємо індекс для кожного слайсу (split заповнимо пізніше)
        for sp in slice_paths:
            rows.append(
                {
                    "subject_id": subject_id,
                    "slice_path": str(sp.relative_to(out_slices_root.parent)),  # щоб шлях був від data/
                    "label": label,
                }
            )
        subj_label_records.append({"subject_id": subject_id, "label": label})

    if not rows:
        raise RuntimeError("Не вдалося згенерувати жодного слайсу. Перевір шляхи та структуру даних.")

    df = pd.DataFrame(rows)
    subj_df = pd.DataFrame(subj_label_records).drop_duplicates()

    print("[INFO] Unique subjects with volume & label:", len(subj_df))
    print("[INFO] Label counts (subjects):")
    print(subj_df["label"].value_counts())

    # робимо patient-level split
    split_map = stratified_split_subjects(subj_df, seed=42, train_frac=0.6, val_frac=0.2)

    df["split"] = df["subject_id"].map(split_map)

    # можуть бути суб'єкти без split (якщо щось пішло не так)
    df = df.dropna(subset=["split"]).reset_index(drop=True)

    print("[INFO] Final slice counts by split:")
    print(df["split"].value_counts())

    out_index_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_index_csv, index=False)
    print(f"[OK] Saved index to: {out_index_csv}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nifti_root",
        type=str,
        required=True,
        help="Корінь з disc*/OAS1_XXXX_MR1 (у тебе: data/raw)",
    )
    ap.add_argument(
        "--clinical_csv",
        type=str,
        required=True,
        help="Шлях до oasis_cross-sectional.csv",
    )
    ap.add_argument(
        "--out_index_csv",
        type=str,
        required=True,
        help="Куди зберегти CSV-індекс (наприклад data/index_oasis_full_2d.csv)",
    )
    ap.add_argument(
        "--out_slices_root",
        type=str,
        required=True,
        help="Куди зберігати PNG-слайси (наприклад data/processed/oasis_full_2d)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_full_2d_index(
        nifti_root=Path(args.nifti_root),
        clinical_csv=Path(args.clinical_csv),
        out_index_csv=Path(args.out_index_csv),
        out_slices_root=Path(args.out_slices_root),
    )
