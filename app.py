# app.py
# Streamlit demo: Binary dementia screening on a 2D MRI dataset
# - Browse samples by class subfolder (NonDemented / VeryMildDemented / MildDemented / ModerateDemented)
# - Upload your own JPG/PNG slice
# - Shows prediction p(dementia) and Grad-CAM heatmap
# NOTE: Educational prototype only — NOT for clinical use.

from __future__ import annotations
import os
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import json

# ------------------------------
# Config
# ------------------------------
CHECKPOINT = "checkpoints/best.pt"  # trained with train_baseline.py
# Point this to your Kaggle 4-classes dataset root:
DATASET_ROOT = Path("data/raw/2d_mri/Alzheimer_MRI_4_classes_dataset")

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Binary mapping used during training (VeryMild/Mild/Moderate => dementia)
LABEL_MAP = {"non_demented": 0, "dementia": 1}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Canonical class folder names we expect under the dataset root
CLASS_ALIASES = {
    "nondemented": "NonDemented",
    "non_demented": "NonDemented",
    "verymilddemented": "VeryMildDemented",
    "very_mild_demented": "VeryMildDemented",
    "milddemented": "MildDemented",
    "moderatedemented": "ModerateDemented",
}
CANONICAL_CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


# ------------------------------
# Model (must match train_baseline.py)
# ------------------------------
def build_resnet18(num_classes: int = 1) -> nn.Module:
    import torchvision
    from torchvision.models import resnet18

    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    except Exception:
        # compatibility with older torchvision
        model = resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_thresholds(path: str = "artifacts/thresholds.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_resource
def load_model() -> nn.Module:
    if not Path(CHECKPOINT).exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}. Train the model first.")
        st.stop()
    model = build_resnet18(num_classes=1).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------
# Preprocess
# ------------------------------
def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """Convert to grayscale, resize to 224, z-score normalize, replicate to 3 channels."""
    img = np.array(pil_img.convert("L"))  # (H, W)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    m, s = img.mean(), img.std() + 1e-6
    img = (img - m) / s
    x = np.stack([img, img, img], axis=0)  # (3, H, W)
    x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)  # (1, 3, H, W)
    return x


# ------------------------------
# Grad-CAM (simple)
# ------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.gradients = None
        self.activations = None
        # get target layer
        modules = dict([*model.named_modules()])
        if target_layer_name not in modules:
            raise KeyError(f"Layer {target_layer_name} not found in the model.")
        self.target_layer = modules[target_layer_name]
        # hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor):
        logits = self.model(x).squeeze(1)  # (B,)
        prob = torch.sigmoid(logits)
        self.model.zero_grad()
        prob.backward(gradient=torch.ones_like(prob), retain_graph=True)

        cams = []
        for i in range(x.size(0)):
            grads = self.gradients[i]  # (C, H, W)
            acts = self.activations[i]  # (C, H, W)
            weights = grads.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
            cam = (weights * acts).sum(dim=0).cpu().numpy()  # (H, W)
            cam = np.maximum(cam, 0)
            cam = (cam - cam.min()) / (cam.max() + 1e-8)
            cams.append(cam)
        return prob.detach().cpu().numpy(), cams


def overlay_cam(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(overlay)


# ------------------------------
# Dataset utilities
# ------------------------------
def canonicalize_class_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("-", "_")
    return CLASS_ALIASES.get(key, name)


def discover_class_folders(root: Path) -> List[Tuple[str, Path]]:
    """Return list of (canonical_name, dir_path) for immediate subfolders."""
    if not root.exists():
        return []
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    classes = []
    for d in subdirs:
        canon = canonicalize_class_name(d.name)
        classes.append((canon, d))
    classes_sorted = sorted(
        classes,
        key=lambda x: CANONICAL_CLASSES.index(x[0]) if x[0] in CANONICAL_CLASSES else 999,
    )
    return classes_sorted


def list_images_in_dir(dir_path: Path, limit: int | None = None) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += list(dir_path.rglob(ext))
    files = sorted(files)
    if limit:
        files = files[:limit]
    return files


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Dementia MRI Demo", page_icon="🧠", layout="wide")
st.title("🧠 Dementia Screening (Demo) — 2D MRI")
st.caption("Educational prototype only — not for clinical use.")

# Sidebar controls
mode = st.sidebar.radio("Mode", ["Browse samples", "Upload image"], index=0)
threshold = st.sidebar.slider("Decision threshold (p[dementia])", 0.0, 1.0, 0.5, 0.01)
page_size = st.sidebar.number_input("Items per page (browse)", min_value=24, max_value=512, value=128, step=8)
random_btn = st.sidebar.button("🎲 Pick random image (from current class)")

# Load recommended thresholds (if exists)
thr_cfg = load_thresholds()
if thr_cfg:
    rec_thr_global = thr_cfg.get("global", {}).get("thr", None)
    metric_name = thr_cfg.get("metric", "f1")
    if rec_thr_global is not None:
        st.sidebar.info(f"Recommended global threshold ({metric_name} on val): {rec_thr_global:.2f}")
        if st.sidebar.button("Use suggested global threshold"):
            threshold = rec_thr_global

# Load model + Grad-CAM
model = load_model()
cam = GradCAM(model, target_layer_name="layer4")

col_left, col_right = st.columns([1, 1])

# -------- Browse mode --------
if mode == "Browse samples":
    root = DATASET_ROOT
    if not root.exists():
        st.error(f"Dataset folder not found:\n{root}\nPlease download/unzip it first.")
        st.stop()

    class_dirs = discover_class_folders(root)
    if not class_dirs:
        st.warning("No class subfolders found under the dataset root.")
        st.stop()

    class_names = ["All classes"] + [c for c, _ in class_dirs]
    chosen = st.selectbox("Class folder", class_names, index=0)

    # Per-class suggested threshold (if available)
    if thr_cfg and chosen != "All classes":
        per_class = thr_cfg.get("per_class_folder", {})
        rec_thr_class = per_class.get(chosen, {}).get("thr", None)
        if rec_thr_class is not None:
            st.sidebar.success(f"{chosen}: suggested thr = {rec_thr_class:.2f}")
            if st.sidebar.button(f"Use suggested for {chosen}"):
                threshold = rec_thr_class

    # Aggregate files
    if chosen == "All classes":
        paths: List[Path] = []
        for cname, cdir in class_dirs:
            paths.extend(list_images_in_dir(cdir))
    else:
        cdir = dict(class_dirs)[chosen]
        paths = list_images_in_dir(cdir)

    if not paths:
        st.warning("No JPG/PNG images found for the selected class.")
        st.stop()

    # Pagination
    total = len(paths)
    pages = max(1, (total + page_size - 1) // page_size)
    page = st.sidebar.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_paths = paths[start:end]

    # Random pick (within filtered set)
    if random_btn and page_paths:
        sel_path = random.choice(page_paths)
    else:
        sel = st.selectbox(
            f"Choose an image ({start+1}-{end}/{total})",
            options=[str(p) for p in page_paths],
            index=0,
        )
        sel_path = Path(sel)

    pil = Image.open(sel_path)

    with torch.no_grad():
        x = preprocess_pil(pil)
    prob, cams = cam(x)
    p1 = float(prob[0])
    pred = 1 if p1 >= threshold else 0

    st.metric(
        "Prediction",
        f"{INV_LABEL[pred]}",
        delta=f"p(dementia) = {p1:.3f} (thr={threshold:.2f})",
        delta_color="inverse" if pred == 0 else "normal",
    )

    st.write("Grad-CAM attention:")
    col_left.image(pil, caption="Original", use_container_width=True)
    col_right.image(overlay_cam(pil, cams[0]), caption="Grad-CAM overlay", use_container_width=True)

# -------- Upload mode --------
else:
    f = st.file_uploader("Upload a JPG/PNG axial MRI slice", type=["jpg", "jpeg", "png"])
    if f is not None:
        pil = Image.open(f).convert("L")
        with torch.no_grad():
            x = preprocess_pil(pil)
        prob, cams = cam(x)
        p1 = float(prob[0])
        pred = 1 if p1 >= threshold else 0

        st.metric(
            "Prediction",
            f"{INV_LABEL[pred]}",
            delta=f"p(dementia) = {p1:.3f} (thr={threshold:.2f})",
            delta_color="inverse" if pred == 0 else "normal",
        )

        st.write("Grad-CAM attention:")
        col_left.image(pil, caption="Uploaded (grayscale)", use_container_width=True)
        col_right.image(overlay_cam(pil.convert("RGB"), cams[0]), caption="Grad-CAM overlay", use_container_width=True)

# Footer disclaimer
st.markdown("---")
st.markdown(
    """
**Disclaimer:** This is a research/education demo built on a 2D dataset (not patient-level).  
Predictions are **not** medical advice. For clinical scenarios we will switch to OASIS NIfTI with patient-level validation, 2.5D/3D modeling, calibration and robust preprocessing.
"""
)
