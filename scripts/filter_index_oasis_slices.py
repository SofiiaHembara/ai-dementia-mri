# scripts/filter_index_oasis_slices.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True,
                    help="Вхідний index CSV (наприклад data/index_oasis_full_2d.csv)")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Куди зберегти відфільтрований CSV")
    ap.add_argument(
        "--min_fg_frac",
        type=float,
        default=0.03,
        help="Мін. частка 'не-чорних' пікселів (0–1). Нижче — дропаємо слайс.",
    )
    ap.add_argument(
        "--min_std",
        type=float,
        default=0.015,
        help="Мін. std інтенсивностей (0–1). Якщо зображення надто 'плоске', дропаємо.",
    )
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="Корінь проєкту — звідси будуються шляхи до PNG (за замовчуванням поточна директорія).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    print(f"[INFO] Читаю CSV: {in_csv}")
    df = pd.read_csv(in_csv)
    print(f"[INFO] Рядків до фільтру: {len(df)}")

    required_cols = {"subject_id", "slice_path", "label", "split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV повинен мати колонки {required_cols}, а маємо: {df.columns.tolist()}"
        )

    kept_rows = []
    dropped_content = 0
    missing = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Фільтрую слайси"):
        rel_path = row["slice_path"]
        p = root / rel_path

        if not p.exists():
            print(f"[WARN] Файл не знайдено: {p}, дропаю цей рядок.")
            missing += 1
            continue

        try:
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"[ERROR] Не можу прочитати {p}: {e}, дропаю.")
            missing += 1
            continue

        # частка "не-чорних" пікселів — беремо поріг трохи вищий
        fg_frac = (arr > 0.10).mean()
        std = float(arr.std())

        if fg_frac < args.min_fg_frac or std < args.min_std:
            dropped_content += 1
            continue

        kept_rows.append(row)

    new_df = pd.DataFrame(kept_rows)
    print(
        f"[INFO] Залишилось рядків: {len(new_df)} "
        f"(дропнуто за контентом: {dropped_content}, проблеми з файлом: {missing})"
    )

    print("\n[INFO] Split value_counts після фільтру:")
    print(new_df["split"].value_counts())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(out_csv, index=False)
    print(f"[OK] Зберегла відфільтрований index до: {out_csv}")


if __name__ == "__main__":
    main()