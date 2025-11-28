# scripts/oasis1_prepare.py
import argparse
import re
from pathlib import Path
import pandas as pd
import nibabel as nib

# ---------- auto-discovery of clinical file ----------
def autodiscover_clinical(root: Path) -> Path | None:
    # пріоритезуємо cross-sectional, потім longitudinal
    cands = list(root.rglob("*cross*section*.csv")) + list(root.rglob("*longitud*.csv")) \
          + list(root.rglob("*cross*section*.xlsx")) + list(root.rglob("*longitud*.xlsx"))
    return sorted(cands)[0] if cands else None

# ---------- subject/CDR parsing ----------
SUBJECT_HINTS = ["subject", "id", "oasis id", "subject id", "oasisid"]
CDR_HINTS     = ["cdr", "clinical dementia rating"]

def _pick_col(cols, hints):
    cols_l = [c.lower() for c in cols]
    for h in hints:
        for i, c in enumerate(cols_l):
            if h in c:
                return list(cols)[i]
    raise KeyError(f"Не знайшов колонку серед {list(cols)} для ключів {hints}")

def _canon_subject(x: str) -> str:
    # приймає 'OAS1_0001', 'OAS1_0001_MR1', 'oas1_0001  ' — нормалізує до 'OAS1_0001_MR1'
    s = str(x).strip()
    m = re.search(r"(OAS1_\d{4})", s, flags=re.I)
    if not m:
        return s
    base = m.group(1).upper()
    return f"{base}_MR1"

def load_clinical(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    subj_col = _pick_col(df.columns, SUBJECT_HINTS)
    cdr_col  = _pick_col(df.columns, CDR_HINTS)

    df = df[[subj_col, cdr_col]].copy()
    df.rename(columns={subj_col: "subject_raw", cdr_col: "cdr"}, inplace=True)
    df["subject_id"] = df["subject_raw"].map(_canon_subject)

    # інколи CDR приходить як текст; приведемо до float, потім до категорій {0,0.5,1,2}
    def _to_cdr(v):
        try:
            fv = float(str(v).strip())
        except Exception:
            return None
        # обмежимося стандартним набором
        if fv in (0.0, 0.5, 1.0, 2.0, 3.0):
            return fv
        # іноді бувають 4/5 — відкинемо
        return None

    df["cdr"] = df["cdr"].map(_to_cdr)
    df = df.dropna(subset=["cdr"]).copy()
    # для нашого бінарного скринінгу: 0 → non_demented, >0 → dementia
    df["label"] = df["cdr"].apply(lambda x: "non_demented" if x == 0.0 else "dementia")
    return df[["subject_id", "cdr", "label"]]

# ---------- MRI discovery ----------
def find_subject_dirs(root: Path):
    # рекурсивно збирає всі папки на кшталт OAS1_####_MR1
    return sorted([p for p in root.rglob("OAS1_*_MR1") if p.is_dir()])

def find_best_t1(subj_dir: Path) -> Path | None:
    # стандартний усереднений MPRAGE: PROCESSED/MPRAGE/SUBJ_111
    cand = subj_dir / "PROCESSED" / "MPRAGE" / "SUBJ_111"
    if not cand.exists():
        return None
    nii = list(cand.glob("*.nii")) + list(cand.glob("*.nii.gz"))
    if nii:
        return nii[0]
    img = list(cand.glob("*.img"))
    if img:
        return img[0]
    return None

def main(args):
    root = Path(args.root).resolve()
    if args.clinical:
        clinical_path = Path(args.clinical).resolve()
    else:
        clinical_path = autodiscover_clinical(root)
        if not clinical_path:
            raise SystemExit("❌ Не вдалося знайти clinical CSV/XLSX (шукав *cross*section* або *longitud*). "
                             "Передай явний шлях через --clinical")

    out_nifti = Path(args.out_nifti).resolve()
    out_index = Path(args.out_index).resolve()
    out_nifti.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    print(f"📄 Clinical file: {clinical_path}")
    clin = load_clinical(clinical_path)
    if clin.empty:
        raise SystemExit("❌ Клінічна таблиця порожня або без валідних CDR.")

    subjects = find_subject_dirs(root)
    if not subjects:
        raise SystemExit(f"❌ Не знайшов жодного 'OAS1_*_MR1' під {root}")

    rows = []
    for subj in subjects:
        sid = subj.name  # OAS1_####_MR1
        t1 = find_best_t1(subj)
        if t1 is None:
            continue
        dst = out_nifti / f"{sid}_T1w.nii.gz"
        img = nib.load(str(t1))
        nib.save(img, str(dst))
        rows.append({"subject_id": sid, "nifti_path": str(dst)})

    df_img = pd.DataFrame(rows)
    if df_img.empty:
        raise SystemExit("❌ Не вдалося зібрати жодного T1 (перевір, чи є PROCESSED/MPRAGE/SUBJ_111).")

    # merge images + clinical
    df = df_img.merge(clin, on="subject_id", how="left")
    df = df.dropna(subset=["label"]).copy()

    # --- PATIENT-LEVEL SPLIT ---
    from sklearn.model_selection import train_test_split
    subj = df[["subject_id", "label"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        subj["subject_id"], test_size=0.15, random_state=args.seed,
        stratify=subj["label"]
    )
    # валід. ~15% від train → загалом ~12.75% датасету
    labels_train = subj.set_index("subject_id").loc[train_ids]["label"]
    train_ids, val_ids = train_test_split(
        train_ids, test_size=0.1765, random_state=args.seed,
        stratify=labels_train
    )

    split_map = {sid: "train" for sid in train_ids}
    split_map.update({sid: "val" for sid in val_ids})
    split_map.update({sid: "test" for sid in test_ids})
    df["split"] = df["subject_id"].map(split_map)

    df = df.sort_values(["split", "subject_id"]).reset_index(drop=True)
    df.to_csv(out_index, index=False)

    print(f"✅ Saved index: {out_index} (N={len(df)})")
    print(df["split"].value_counts())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",      required=True, help="Корінь з дисками (наприклад, data/raw)")
    ap.add_argument("--clinical",  default=None,  help="(необов’язково) шлях до clinical CSV/XLSX; інакше автопошук")
    ap.add_argument("--out_nifti", required=True, help="Куди класти NIfTI (наприклад, data/processed/oasis1_nifti)")
    ap.add_argument("--out_index", required=True, help="CSV-індекс з split (наприклад, data/index_oasis1.csv)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
