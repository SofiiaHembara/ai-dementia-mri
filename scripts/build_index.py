from pathlib import Path
import pandas as pd
import random

ROOT = Path("data/raw/2d_mri")

files = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    files += list(ROOT.rglob(ext))
if not files:
    raise SystemExit("У 2D наборі не знайдено jpg/png. Перевірити шлях.")

def map_label_from_path(p: Path):
    low = p.as_posix().lower()
    if "nondemented" in low:        return "non_demented"
    if "verymilddemented" in low:   return "very_mild"
    if "milddemented" in low:       return "mild"
    if "moderatedemented" in low:   return "moderate"
    if "dement" in low or "alzheimer" in low: return "dementia"
    return "non_demented"

rows = [{"path": str(p), "label": map_label_from_path(p)} for p in files]
df = pd.DataFrame(rows).drop_duplicates()

BINARIZE = True
if BINARIZE:
    df["label"] = df["label"].map(lambda x: "dementia" if x in {"very_mild","mild","moderate","dementia"} else "non_demented")

rng = random.Random(42)
idx = list(range(len(df))); rng.shuffle(idx)
n = len(idx); test_n = max(1, int(0.15*n)); val_n = max(1, int(0.15*n))
test_idx = set(idx[:test_n]); val_idx = set(idx[test_n:test_n+val_n])

splits = []
for i in range(len(df)):
    if i in test_idx: splits.append("test")
    elif i in val_idx: splits.append("val")
    else: splits.append("train")
df["split"] = splits

Path("data").mkdir(parents=True, exist_ok=True)
df.to_csv("data/index.csv", index=False)
print("Збережено data/index.csv")
print(df["label"].value_counts()); print(df["split"].value_counts())
