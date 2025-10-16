AI Dementia MRI — Baseline README
A minimal, reproducible baseline for screening dementia from MRI using a 2D Kaggle dataset (binary: dementia vs non_demented). This repo is meant to get the team to a working midterm demo quickly (metrics + saved model + evaluation). After midterm we’ll switch to patient-level OASIS NIfTI for proper validation.

What you get
- Download script (Kaggle → data/raw/2d_mri/…)
- Index builder (data/index.csv with path,label,split)
- Training script (ResNet-18 transfer learning; logs AUC/F1/BalAcc; saves checkpoints/best.pt)
- Evaluation script (metrics on val and test + confusion matrix + classification report)

# 0) clone repo & cd
git clone <your-repo-url>.git
cd ai-dementia-mri

# 1) create venv (Python 3.12 recommended; 3.10 also fine)
python3 -m venv .venv
source .venv/bin/activate

# 2) install deps (CPU wheels for PyTorch)
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# 3) configure Kaggle CLI (put kaggle.json to ~/.kaggle/ with chmod 600)
bash scripts/download_data.sh

# 4) build index (creates data/index.csv)
python scripts/build_index.py

# 5) train (5 epochs)
python -m src.models.train_baseline --epochs 5 --batch 32 --lr 3e-4

# 6) evaluate saved checkpoint on val & test
python -m src.eval.evaluate_val_test --ckpt checkpoints/best.pt
Do not commit data. data/ is in .gitignore.

Environment setup
We recommend Python 3.12 (works great with the CPU PyTorch wheels). If you use macOS and hit PyTorch/NumPy issues, 3.10 is also OK.

python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

Install dependencies (with pinned, compatible versions):
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

requirements.txt (already in repo) pins:
numpy==1.26.4
opencv-python==4.10.0.84
pandas==2.2.2
scikit-learn==1.4.2
torch
torchvision
torchaudio
grad-cam==1.5.5

Data download (Kaggle)
Create a Kaggle API token: Kaggle Profile → Create New API Token
Move kaggle.json to:
macOS/Linux: ~/.kaggle/kaggle.json and chmod 600 ~/.kaggle/kaggle.json
Windows (PowerShell): C:\Users\<you>\.kaggle\kaggle.json
Download the dataset:
bash scripts/download_data.sh
# expects: data/raw/2d_mri/Alzheimer_MRI_4_classes_dataset/{NonDemented,VeryMildDemented,MildDemented,ModerateDemented}
If you change the dataset slug, update it inside scripts/download_data.sh.

Build index
Creates data/index.csv with columns: path,label,split.
python scripts/build_index.py
For midterm we binarize into non_demented vs dementia (where dementia collapses VeryMild/Mild/Moderate).
We do a simple image-level split (train/val/test).
After midterm we’ll move to patient-level splits on OASIS NIfTI.

Train
Run as a module (so src/ is on the Python path):
python -m src.models.train_baseline --epochs 5 --batch 32 --lr 3e-4
Backbone: ResNet-18 (ImageNet weights), head replaced with 1-logit layer (BCEWithLogits).
Input: grayscale images normalized (z-score), resized to 224×224, replicated to 3 channels.
Outputs logs per epoch: AUC / F1 / Balanced Accuracy on val.
Saves the best model to checkpoints/best.pt (by AUC).

Evaluate (val & test)
python -m src.eval.evaluate_val_test --ckpt checkpoints/best.pt
Prints:
AUC, F1, Balanced Accuracy for val and test
Confusion matrix [ [TN, FP], [FN, TP] ]
classification_report (precision/recall/F1)

Results (baseline)
Example validation results from our run (5 epochs on CPU):
Epoch 05 | loss=0.1545 | AUC=0.971 | F1=0.899 | BA=0.900
Saved: checkpoints/best.pt
Important caveat: this Kaggle dataset is 2D and not patient-level.
These metrics are great for a midterm demo, but they can be optimistic.
In the final project we will:
Switch to OASIS NIfTI with patient-level splits,
Add 2.5D / lightweight 3D models,
Provide calibration + explainability (Grad-CAM, occlusion).

Repo structure
ai-dementia-mri/
├─ data/                     # NOT committed (see .gitignore)
│  ├─ raw/
│  │  └─ 2d_mri/Alzheimer_MRI_4_classes_dataset/...
│  ├─ processed/
│  └─ index.csv             # built by scripts/build_index.py
├─ scripts/
│  ├─ download_data.sh      # Kaggle -> data/raw/2d_mri
│  └─ build_index.py        # creates data/index.csv
├─ src/
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ dataset.py         # Alzheimer2DDataset
│  ├─ eval/
│  │  ├─ __init__.py
│  │  ├─ evaluate_val_test.py
│  │  └─ gradcam.py         # (optional)
│  └─ models/
│     ├─ __init__.py
│     └─ train_baseline.py  # ResNet-18 TL baseline
├─ checkpoints/             # saved weights (ignored by git)
├─ notebooks/               # EDA (optional)
├─ configs/                 # YAML configs (optional)
├─ requirements.txt
├─ .gitignore               # includes `data/`, `*.pt`, `.venv/`, etc.
└─ README.md

