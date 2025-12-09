# AI Dementia MRI — Patient-level OASIS (2D DINO + 3D R3D)

Working pipeline for dementia screening from MRI with **patient-level** validation on OASIS.  
Primary approach: **Self-supervised ViT-Base/16 (DINO)** on 2D slices with **aggregation to patient level**.  
Secondary baseline: **3D R3D-18 (Kinetics)** on NIfTI volumes.


---

## What’s inside

- **2D (DINO ViT-B/16)**
  - Training: `src/models/train_2d_dino.py`
  - Evaluation (test + ROC plots): `src/models/eval_2d_dino.py`
- **3D (R3D-18)**
  - Training: `src/models/train_3d.py`
- **Data indices (CSV)**
  - `data/index_oasis1_2d.csv` — columns: `subject_id,path,label,split`
  - `data/index_oasis1_for3d.csv` — columns: `subject_id,nifti_path,label,split`
- **Artifacts**
  - `artifacts/roc_slice.png`, `artifacts/roc_patient.png`
  - `artifacts/test_preds_*.csv`, `artifacts/test_patient_agg_*.csv`

---

## Data layout

Place files so that the CSV paths point to actual files:

```
ai-dementia-mri/
├─ data/                          # NOT tracked
│  ├─ processed/oasis1_nifti/*.nii.gz
│  ├─ interim/oasis1_2d/{non_demented,dementia}/*.png
│  ├─ index_oasis1_for3d.csv
│  └─ index_oasis1_2d.csv
├─ src/
│  ├─ data_/dataset3d.py
│  ├─ models/
│  │  ├─ train_2d_dino.py
│  │  ├─ eval_2d_dino.py
│  │  └─ train_3d.py
│  └─ utils/...
├─ artifacts/                     # ROC plots, CSVs (tracked if small)
├─ checkpoints/                   # model weights (ignored)
├─ notebooks/
│  └─ colab_2d_dinp_3d.ipynb
├─ docs/
│  ├─ figures/roc_slice.png
│  ├─ figures/roc_patient.png
│  └─ baselines/kaggle_2d.md
├─ requirements.txt
├─ .gitignore
└─ README.md

**Splits** are patient-level via the `split` column: `train / val / test`.

---

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt   # GPU builds recommended (Colab/torch >= 2.2)
```

> If using Google Colab: mount Drive, `cd` into the repo, and ensure CSV paths are valid in the Colab filesystem.

---

## Train — 2D (ViT-B/16 DINO)

```bash
python -m src.models.train_2d_dino   --index_csv data/index_oasis1_2d.csv   --out_dir .   --epochs 60 --freeze_epochs 30   --unfreeze_top_k 4   --batch 32 --size 224   --lr 2e-4 --ft_lr 1e-5 --weight_decay 1e-2   --patience 15 --accum_steps 2 --num_workers 0   --monitor slice_auc --min_delta 1e-4   --backbone vit_base_patch16_224.dino
```

### Evaluate + export ROC plots

```bash
python -m src.models.eval_2d_dino   --index_csv data/index_oasis1_2d.csv   --weights best_2d_dino.pt   --backbone vit_base_patch16_224.dino   --size 224 --batch 64 --num_workers 0   --thr_json artifacts/thresholds.json   --out_dir artifacts
```

Outputs:
- Slice-level metrics and ROC: `artifacts/roc_slice.png`
- Patient-level aggregation and ROC: `artifacts/roc_patient.png`
- CSVs with per-slice and per-patient predictions in `artifacts/`.

---

## Train — 3D (R3D-18)

**Basic run (BCE):**
```bash
python -m src.models.train_3d   --index_csv data/index_oasis1_for3d.csv   --epochs 80 --batch 2   --size 128 --depth 64   --lr 1e-4 --weight_decay 5e-4   --freeze_epochs 6 --unfreeze_lr_mult 0.25   --patience 12 --num_workers 0   --accum_steps 4 --groupnorm 8 --balanced --cache
```

**With focal loss & extras (use when class imbalance hurts F1):**
```bash
python -m src.models.train_3d   --index_csv data/index_oasis1_for3d.csv   --epochs 80 --batch 2   --size 128 --depth 64   --lr 1e-4 --weight_decay 5e-4   --freeze_epochs 6 --unfreeze_lr_mult 0.25   --patience 15 --min_delta 1e-4   --num_workers 2 --accum_steps 4   --groupnorm 8 --balanced --cache --zscore   --agg mean   --loss focal --alpha 0.6 --gamma 2.0
```

> Tips: `--groupnorm 8` stabilizes small-batch 3D training; `--balanced` uses a weighted sampler; `--zscore` applies intensity normalization; `--agg` controls patient aggregation (`mean|median|topk`).

---

## Results (snapshot from our runs)

- **2D DINO (test)**
  - **Slice level:** ROC-AUC ≈ **0.75**, F1 ≈ **0.63**, Balanced Accuracy ≈ **0.72**
  - **Patient level:** ROC-AUC ≈ **0.75**, F1 ≈ **0.80**, Balanced Accuracy ≈ **0.875**

- **3D R3D-18 (val)**
  - Best episode (BCE): patient ROC-AUC ≈ **0.71**
  - Focal configuration: best patient Balanced Accuracy ≈ **0.58** (on our small split)

Numbers may vary with seeds and preprocessing; 2D DINO has been more stable given dataset size.

---

## Why DINO ViT-B/16?

DINO pretraining provides strong, general visual features. For a small medical dataset, **freezing most of the backbone and training a lightweight head** on 2D MR slices works well. We then **aggregate slice predictions to patient level** (e.g., **mean** or **top-k** of highest-confidence slices), which aligns with clinical use (decisions per patient, not per slice) and stabilizes metrics.

---

## Preprocessing (concise)

- **2D:** intensity centering/normalization, resize to **224×224**, convert to 3-channel for ViT.
- **3D:** (if available) resampling to a common voxel space, **z-score** or Kinetics mean/std normalization, crop/pad to fixed `D×S×S`.

---

## Artifacts & repo policy

- Do **not** commit data (`data/`), checkpoints (`checkpoints/*.pt`), or large binaries — already in `.gitignore`.
- Keep poster-ready figures (e.g., ROC) under `docs/figures/` and reference them from slides/posters.

---

## Troubleshooting

- **“FileNotFoundError” for images/NIfTI:** check that CSV paths exist inside your current runtime (Colab vs local) and match `data/...`.
- **MPS (macOS) errors:** some ops (e.g., bicubic AA) are not implemented on MPS; either set `PYTORCH_ENABLE_MPS_FALLBACK=1` or run on CUDA/CPU.
- **3D training unstable:** try `--groupnorm 8`, lower LR after unfreezing (`--unfreeze_lr_mult 0.25`), or switch to `--loss focal`.

---

## License & acknowledgements

Backbones based on torchvision/timm weights (Kinetics, DINO). Dataset derived from OASIS; please follow original dataset licenses and citation guidelines.
