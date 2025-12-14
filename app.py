# app.py
# Streamlit demo: patient-level dementia screening on OASIS 2D slices (DINO ViT)
# - Only patient-centric view (no file names in UI)
# - Uses index_oasis1_2d.csv (slice_path, subject_id, label, split)
# - Aggregates slice predictions -> patient prediction (mean or top-k)
# - Shows all slices for selected patient
# NOTE: Research/education demo only — NOT for clinical use.

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import random

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import streamlit as st
import timm

# ------------------------------
# Config
# ------------------------------
INDEX_CSV = Path("data/index_oasis1_2d.csv")
CHECKPOINT = Path("best_2d_dino.pt")  # твій чекпойнт з train_2d_dino
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

LABEL_MAP = {"non_demented": 0, "dementia": 1}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ------------------------------
# Model: DINO ViT + 1-logit голова
# (максимально проста та близька до того, як ти тренувала)
# ------------------------------
class DinoClassifier(nn.Module):
    def __init__(self, backbone_name: str = "vit_base_patch16_224.dino"):
        super().__init__()
        # Використовуємо in_chans=3 (як у тренуванні),
        # num_classes=0 -> отримаємо фічі, а не класи
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            in_chans=3,
            num_classes=0,     # фічі
            global_pool="avg",  # середнє по патчах
        )
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Очікуємо (B, 1, H, W) або (B, 3, H, W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.backbone(x)           # (B, F)
        logit = self.head(feats).squeeze(1)  # (B,)
        return logit


@st.cache_resource
def load_model() -> nn.Module:
    if not CHECKPOINT.exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}. Make sure best_2d_dino.pt is in the project root.")
        st.stop()

    model = DinoClassifier(backbone_name="vit_base_patch16_224.dino").to(DEVICE)

    state = torch.load(CHECKPOINT, map_location="cpu")

    # Спробуємо завантажити максимально «мʼяко»
    new_state = {}
    for k, v in state.items():
        # Якщо чекпойнт зберігався як plain state_dict з тими ж іменами
        if k.startswith("backbone.") or k.startswith("head."):
            new_state[k] = v

    incompatible = model.load_state_dict(new_state, strict=False)
    # Просто інформативне повідомлення в консолі, UI не чіпаємо
    print("Loaded checkpoint, missing:", incompatible.missing_keys, "unexpected:", incompatible.unexpected_keys)

    model.eval()
    return model


# ------------------------------
# Preprocess (той же самий пайплайн, що й у тренуванні)
# ------------------------------
def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """Grayscale -> resize -> z-score -> stack to 3ch -> (1,3,H,W)."""
    img = np.array(pil_img.convert("L"))  # (H, W)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    m, s = img.mean(), img.std() + 1e-6
    img = (img - m) / s
    x = np.stack([img, img, img], axis=0)  # (3, H, W)
    x = torch.from_numpy(x).unsqueeze(0).float()  # (1,3,H,W)
    return x


@st.cache_data
def load_index() -> pd.DataFrame:
    if not INDEX_CSV.exists():
        st.error(f"Index CSV not found: {INDEX_CSV}")
        st.stop()
    df = pd.read_csv(INDEX_CSV)
    required = {"slice_path", "subject_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Index is missing columns: {missing}. Found: {list(df.columns)}")
        st.stop()

    # Нормалізуємо шляхи до слайсів
    df["slice_path"] = df["slice_path"].astype(str).apply(lambda p: str(Path(p)))
    # Залишаємо лише існуючі файли
    df = df[df["slice_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    return df


@st.cache_data
def build_patient_index(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Створюємо індекс:
    {
      subject_id: {
         "label": 0/1,
         "split": "train"/"val"/"test",
         "slices": [ "path/to/img1.png", ... ]
      },
      ...
    }
    """
    patients: Dict[str, Dict] = {}
    for sid, grp in df.groupby("subject_id"):
        labels = grp["label"].str.lower().tolist()
        # очікуємо один label на пацієнта → беремо перший
        lbl = labels[0]
        label_int = LABEL_MAP[lbl]
        split = grp["split"].iloc[0]
        slices = grp["slice_path"].tolist()
        patients[sid] = {
            "label_str": lbl,
            "label_int": label_int,
            "split": split,
            "slices": slices,
        }
    return patients


# ------------------------------
# Інференс по пацієнту
# ------------------------------
def predict_patient(
    model: nn.Module,
    patient: Dict,
    agg: str = "mean",
    topk: int = 8,
) -> Tuple[float, List[float]]:
    """
    Повертаємо:
      p_patient (float), slice_probs (list of floats, len = n_slices)
    """
    model.eval()
    slices = patient["slices"]
    probs: List[float] = []

    with torch.no_grad():
        batch_tensors: List[torch.Tensor] = []
        for p in slices:
            img = Image.open(p)
            x = preprocess_pil(img)  # (1,3,H,W)
            batch_tensors.append(x)

        x_all = torch.cat(batch_tensors, dim=0).to(DEVICE)  # (N,3,H,W)
        logits = model(x_all)  # (N,)
        p = torch.sigmoid(logits).cpu().numpy().astype(float)
        probs = p.tolist()

    probs_arr = np.asarray(probs)
    if agg == "mean":
        p_patient = float(probs_arr.mean())
    elif agg == "topk":
        k = min(topk, len(probs_arr))
        top_vals = np.sort(probs_arr)[-k:]
        p_patient = float(top_vals.mean())
    else:
        p_patient = float(probs_arr.mean())

    return p_patient, probs


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Dementia MRI — Patient Demo", page_icon="🧠", layout="wide")
st.title("🧠 Dementia screening demo (OASIS, patient-level)")
st.caption("Research/education prototype on OASIS 2D slices — NOT for clinical use.")

df = load_index()
patients_index = build_patient_index(df)
model = load_model()

# ----- Sidebar -----
with st.sidebar:
    st.header("Settings")

    subset = st.selectbox(
        "Subset",
        options=["test", "val", "train", "all"],
        index=0,
        help="Which split of the dataset to browse.",
    )

    if subset == "all":
        available_ids = list(patients_index.keys())
    else:
        available_ids = [sid for sid, info in patients_index.items() if info["split"] == subset]

    available_ids = sorted(available_ids)

    if not available_ids:
        st.error(f"No patients found for subset={subset}")
        st.stop()

    random_btn = st.button("🎲 Random patient")

    if random_btn:
        chosen_id = random.choice(available_ids)
    else:
        chosen_id = st.selectbox("Patient ID", options=available_ids, index=0)

    threshold = st.slider(
        "Decision threshold (p[dementia])",
        0.0, 1.0,
        value=0.5,
        step=0.01,
        help="Lower → more sensitive (more positives); higher → more specific (fewer false positives).",
    )

    agg = st.radio(
        "Aggregation across slices",
        options=["mean", "topk"],
        index=0,
        help="How to aggregate slice probabilities into a single patient-level score.",
    )

    topk = st.slider(
        "k for top-k (if selected)",
        min_value=1,
        max_value=32,
        value=8,
        step=1,
        help="Average over k most suspicious slices (highest p[dementia]).",
    )

# ----- Main content -----
patient = patients_index[chosen_id]
p_patient, slice_probs = predict_patient(model, patient, agg=agg, topk=topk)

true_label_int = patient["label_int"]
true_label_str = patient["label_str"]
pred_label_int = 1 if p_patient >= threshold else 0
pred_label_str = INV_LABEL[pred_label_int]

col_info, col_slices = st.columns([1, 2])

with col_info:
    st.subheader(f"Patient: {chosen_id}")
    st.markdown(f"**True label:** `{true_label_str}`")
    st.markdown(f"**Split:** `{patient['split']}`")

    st.metric(
        "Patient-level prediction",
        f"{pred_label_str}",
        delta=f"p(dementia) = {p_patient:.3f}  |  thr={threshold:.2f}",
    )

    if pred_label_int == true_label_int:
        st.success("Prediction matches the true label (for this patient).")
    else:
        st.error("Prediction does **not** match the true label for this patient.")

    # Невеликий опис
    st.markdown(
        """
**How this works:**

- For this patient we take **all available MRI slices** from the index.
- The DINO ViT model predicts p(dementia) for **each slice**.
- We aggregate them (mean or top-k) to get one **patient-level score**.
- If `p(dementia) ≥ threshold`, we show `"dementia"`, otherwise `"non_demented"`.
"""
    )

    # Табличка з кількома найпідозрілішими слайсами (без назв файлів)
    probs_arr = np.asarray(slice_probs)
    order = np.argsort(-probs_arr)
    top_show = min(10, len(order))
    df_top = pd.DataFrame(
        {
            "slice_index": list(range(len(slice_probs))),
            "p(dementia)": slice_probs,
        }
    ).iloc[order[:top_show]]
    st.markdown("**Most suspicious slices (by model):**")
    st.dataframe(df_top.reset_index(drop=True), use_container_width=True)

with col_slices:
    st.subheader("All slices for this patient")

    imgs: List[Image.Image] = []
    for p in patient["slices"]:
        try:
            img = Image.open(p).convert("L")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            imgs.append(img)
        except Exception:
            continue

    if not imgs:
        st.warning("No images could be loaded for this patient.")
    else:
        # Показуємо всі слайси в гріді (Streamlit сам розібʼє список по рядах)
        st.image(imgs, caption=[f"slice {i}" for i in range(len(imgs))], use_container_width=True)

        # Проста візуалізація розподілу p(dementia) по слайсам
        st.markdown("**Slice-level probabilities (model view):**")
        df_probs = pd.DataFrame(
            {"slice_index": list(range(len(slice_probs))), "p(dementia)": slice_probs}
        )
        st.line_chart(df_probs.set_index("slice_index"))

st.markdown("---")
st.markdown(
    """
**Disclaimer:**  
This is a research/education demo on a 2D slice dataset with a DINO ViT model.  
Predictions are **not** medical advice and must not be used for clinical decisions.
"""
)
