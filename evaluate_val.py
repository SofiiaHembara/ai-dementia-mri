#!/usr/bin/env python3
"""
Швидка оцінка моделі на валідаційному наборі
"""

import subprocess
import sys
from pathlib import Path

# Параметри
CHECKPOINT = "artifacts/checkpoints_meddino/meddino/best_meddino.pt"
INDEX_CSV = "data/index_oasis1_2d.csv"
BACKBONE = "vit_base_patch16_224.dino"
OUTPUT_DIR = "artifacts/meddino_val"

def main():
    # Перевірка checkpoint
    if not Path(CHECKPOINT).exists():
        print(f"❌ Checkpoint не знайдено: {CHECKPOINT}")
        print("Спочатку натренуйте модель:")
        print("  python -m src.models.train_meddino ...")
        sys.exit(1)
    
    print("🔍 Оцінка моделі на валідаційному наборі...")
    print(f"   Checkpoint: {CHECKPOINT}")
    print(f"   Output: {OUTPUT_DIR}")
    print()
    
    # Команда для eval
    cmd = [
        "python", "-m", "src.models.eval_meddino",
        "--index_csv", INDEX_CSV,
        "--checkpoint", CHECKPOINT,
        "--backbone", BACKBONE,
        "--img_size", "224",
        "--split", "val",
        "--output_dir", OUTPUT_DIR,
        "--use_clahe",
        "--normalize_mode", "zscore",
        "--patient_aggregation", "mean",
        "--batch_size", "32",
        "--num_workers", "0"
    ]
    
    # Запуск
    try:
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("✅ Оцінка завершена!")
        print("="*60)
        print(f"\nРезультати збережені в: {OUTPUT_DIR}/")
        print("\nПерегляньте:")
        print(f"  1. Метрики: cat {OUTPUT_DIR}/metrics_val.json")
        print(f"  2. ROC крива: open {OUTPUT_DIR}/roc_patient_val.png")
        print(f"  3. Передбачення: head {OUTPUT_DIR}/predictions_patient_val.csv")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Помилка при оцінці: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

