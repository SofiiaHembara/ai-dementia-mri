# app.py
# Streamlit demo: Binary dementia screening on a 2D MRI dataset (simplified, no pagination)
# - Browse samples by class subfolder (NonDemented / VeryMildDemented / MildDemented / ModerateDemented)
# - Upload your own JPG/PNG slice
# - Shows prediction p(dementia) and Grad-CAM heatmap
# NOTE: Educational prototype only — NOT for clinical use.

from __future__ import annotations
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image

# ------------------------------
# Config
# ------------------------------
CHECKPOINT = "checkpoints/best.pt"  # trained with train_baseline.py (ResNet-18)
DATASET_ROOT = Path("data/raw/2d_mri/Alzheimer_MRI_4_classes_dataset")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {"non_demented": 0, "dementia": 1}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

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
        model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # single logit for BCEWithLogits
    return model

@st.cache_resource
def load_model() -> nn.Module:
    ckpt = Path(CHECKPOINT)
    if not ckpt.exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}. Train the model first.")
        st.stop()
    model = build_resnet18(num_classes=1).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# ------------------------------
# Preprocess
# ------------------------------
def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """grayscale -> resize(224) -> z-score -> stack to 3ch -> tensor (1,3,H,W)"""
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
        modules = dict([*model.named_modules()])
        if target_layer_name not in modules:
            raise KeyError(f"Layer {target_layer_name} not found in the model.")
        self.target_layer = modules[target_layer_name]
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor):
        logits = self.model(x).squeeze(1)  # (B,)
        prob = torch.sigmoid(logits)
        # Build graph for Grad-CAM — do NOT wrap in torch.no_grad()
        self.model.zero_grad(set_to_none=True)
        prob.backward(gradient=torch.ones_like(prob), retain_graph=False)

        cams = []
        for i in range(x.size(0)):
            grads = self.gradients[i]      # (C, H, W)
            acts = self.activations[i]     # (C, H, W)
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
    if limit is not None:
        files = files[:limit]
    return files

# ------------------------------
# Streamlit UI (simplified, no pagination, no extra labels)
# ------------------------------
st.set_page_config(page_title="Dementia MRI Demo", page_icon="🧠", layout="wide")
st.title("🧠 Dementia Screening (Demo) — 2D MRI")
st.caption("Educational prototype only — not for clinical use.")

# single source of truth for the threshold via session_state
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5

with st.sidebar:
    mode = st.radio("Mode", ["Browse samples", "Upload image"], index=0)
    st.session_state.threshold = st.slider(
        "Decision threshold (p[dementia])",
        0.0, 1.0,
        value=float(st.session_state.threshold),
        step=0.01,
        help="Lower → more sensitive; higher → more specific.",
    )
    threshold = float(st.session_state.threshold)
    if mode == "Browse samples":
        max_items = st.number_input(
            "Max items to list (for performance)",
            min_value=50, max_value=5000, value=500, step=50
        )
        random_btn = st.button("🎲 Random from current selection")

# Load model + Grad-CAM
model = load_model()
cam = GradCAM(model, target_layer_name="layer4")

col_left, col_right = st.columns([1, 1])

# -------- Browse mode (no pagination) --------
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

    # Gather files (silently limit by max_items for performance)
    if chosen == "All classes":
        paths: List[Path] = []
        for _, cdir in class_dirs:
            paths.extend(list_images_in_dir(cdir))
    else:
        cdir = dict(class_dirs)[chosen]
        paths = list_images_in_dir(cdir)

    if len(paths) == 0:
        st.warning("No JPG/PNG images found for the selected class.")
        st.stop()

    if len(paths) > int(max_items):
        paths = paths[: int(max_items)]

    # Choose image (no extra labels shown)
    if random_btn and paths:
        sel_path = random.choice(paths)
    else:
        sel = st.selectbox("Choose an image", options=[str(p) for p in paths], index=0)
        sel_path = Path(sel)

    pil = Image.open(sel_path)
    x = preprocess_pil(pil)
    prob, cams = cam(x)
    p1 = float(prob[0])
    pred = 1 if p1 >= threshold else 0

    st.metric(
        "Prediction",
        f"{INV_LABEL[pred]}",
        delta=f"p(dementia) = {p1:.3f}  |  thr={threshold:.2f}",
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
        x = preprocess_pil(pil)
        prob, cams = cam(x)
        p1 = float(prob[0])
        pred = 1 if p1 >= st.session_state.threshold else 0

        st.metric(
            "Prediction",
            f"{INV_LABEL[pred]}",
            delta=f"p(dementia) = {p1:.3f}  |  thr={st.session_state.threshold:.2f}",
            delta_color="inverse" if pred == 0 else "normal",
        )

        st.write("Grad-CAM attention:")
        col_left.image(pil, caption="Uploaded (grayscale)", use_container_width=True)
        col_right.image(overlay_cam(pil.convert("RGB"), cams[0]), caption="Grad-CAM overlay", use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
**Disclaimer:** Research/education demo on a 2D slice dataset (not patient-level).  
Predictions are **not** medical advice.
"""
)
