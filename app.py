import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from src.models.train_2d_dino_patient import DinoClassifier


# =====================
# Конфіг
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_CSV = PROJECT_ROOT / "data/index_oasis_full_2d.csv"
CKPT = PROJECT_ROOT / "best_2d_dino_patient.pt"
IMG_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


# =====================
# Утиліти для шляхів та даних
# =====================

def resolve_slice_path(p: str) -> Path:
    """
    Робимо шлях robust:
    - якщо в CSV абсолютний шлях -> беремо як є;
    - якщо відносний:
        * пробуємо PROJECT_ROOT / p
        * якщо не існує, пробуємо PROJECT_ROOT / "data/processed" / p
    """
    p = Path(p)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(PROJECT_ROOT / p)
        candidates.append(PROJECT_ROOT / "data/processed" / p)

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def to_int_label(x):
    s = str(x).strip().lower()
    if s in {"0", "1"}:
        return int(s)
    if s in {"non_demented", "control", "healthy"}:
        return 0
    if s in {"dementia", "alzheimers", "ad"}:
        return 1
    raise ValueError(f"Unknown label value: {x}")


@st.cache_data(show_spinner="Завантажую індекс зрізів...")
def load_index() -> pd.DataFrame:
    df = pd.read_csv(INDEX_CSV)
    df["label"] = df["label"].apply(to_int_label).astype(int)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    return df


@st.cache_resource(show_spinner="Завантажую модель...")
def load_model() -> DinoClassifier:
    model = DinoClassifier(backbone_name="vit_base_patch16_224.dino")
    state = torch.load(CKPT, map_location=DEVICE)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[WARN] Missing:", incompatible.missing_keys)
        print("[WARN] Unexpected:", incompatible.unexpected_keys)
    model.to(DEVICE)
    model.eval()
    return model


def load_gray_resize(path: Path, size: int = IMG_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("L")
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32)
    m, s = arr.mean(), arr.std() + 1e-6
    arr = (arr - m) / s
    arr3 = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
    return torch.from_numpy(arr3)


def predict_patient_with_slices(
    model: DinoClassifier,
    df: pd.DataFrame,
    subject_id: str,
    max_slices: int = 48,
    agg_mode: str = "topk",
    topk_frac: float = 0.3,
):
    """
    Повертає:
    - patient_prob: агрегована ймовірність деменції для пацієнта
    - paths: список Path до зрізів, які використали
    - probs: масив ймовірностей по зрізах
    """
    df_p = df[df["subject_id"] == subject_id].copy()
    if df_p.empty:
        raise ValueError(f"Немає рядків для {subject_id}")

    raw_paths = df_p["slice_path"].tolist()
    paths = [resolve_slice_path(p) for p in raw_paths]
    paths = [p for p in paths if p.exists()]

    if not paths:
        raise FileNotFoundError(f"Немає PNG для {subject_id}")

    # Обмежуємо кількість зрізів, щоб не захлинутися по пам'яті
    if len(paths) > max_slices:
        idxs = np.linspace(0, len(paths) - 1, max_slices).astype(int)
        paths = [paths[i] for i in idxs]

    imgs = [load_gray_resize(p) for p in paths]
    x = torch.stack(imgs, dim=0).to(DEVICE)  # (N,3,H,W)

    with torch.no_grad():
        logits = model(x).squeeze(1)  # (N,)
        probs = torch.sigmoid(logits).cpu().numpy()

    if agg_mode == "mean":
        patient_prob = float(probs.mean())
    else:
        k = max(1, int(len(probs) * topk_frac))
        topk = np.sort(probs)[-k:]
        patient_prob = float(topk.mean())

    return patient_prob, paths, probs


# =====================
# Streamlit UI
# =====================

st.set_page_config(
    page_title="AI Dementia MRI Demo",
    layout="wide",
)

st.title("🧠 AI-демо: оцінка деменції за МРТ (patient-level)")

st.markdown(
    """
Це демо показує **patient-level** передбачення моделі:

- модель дивиться на **кілька зрізів мозку** для одного пацієнта,
- по кожному зрізу оцінює ймовірність деменції,
- агрегує по зрізах (mean або top-k) і видає одну **ймовірність деменції для пацієнта**.
"""
)

df_index = load_index()
model = load_model()

# --- Sidebar: налаштування ---
st.sidebar.header("Налаштування")

split = st.sidebar.selectbox("Спліт даних", ["test", "val", "train", "all"], index=0)

if split == "all":
    df_split = df_index
else:
    df_split = df_index[df_index["split"] == split]

subjects = sorted(df_split["subject_id"].unique())
if not subjects:
    st.error(f"Для спліту '{split}' немає пацієнтів.")
    st.stop()

subject_id = st.sidebar.selectbox("Пацієнт (subject_id)", subjects)

agg_mode_ui = st.sidebar.radio("Агрегація по зрізах", ["top-k", "mean"], index=0)
agg_mode = "topk" if agg_mode_ui == "top-k" else "mean"

if agg_mode == "topk":
    topk_frac = st.sidebar.slider("Частка top-k зрізів", 0.1, 1.0, 0.3, 0.1)
else:
    topk_frac = 1.0

max_slices = st.sidebar.slider("Максимум зрізів на пацієнта", 8, 96, 48, 8)
threshold = st.sidebar.slider("Поріг для діагнозу 'деменція'", 0.1, 0.9, 0.5, 0.05)

# --- Основна частина: інференс для одного пацієнта ---

df_p = df_index[df_index["subject_id"] == subject_id]
true_label = int(df_p["label"].iloc[0])
true_label_name = "Деменція" if true_label == 1 else "non-demented"

st.subheader(f"Пацієнт: `{subject_id}`")

with st.spinner("Рахую прогноз для пацієнта..."):
    try:
        patient_prob, slice_paths, slice_probs = predict_patient_with_slices(
            model,
            df_index,
            subject_id,
            max_slices=max_slices,
            agg_mode=agg_mode,
            topk_frac=topk_frac,
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Помилка при інференсі: {e}")
        st.stop()

pred_label = 1 if patient_prob >= threshold else 0
pred_label_name = "Деменція" if pred_label == 1 else "non-demented"

# 🔥 Тепер показуємо тільки модель великим, а істинний клас — маленьким
col1, col2 = st.columns(2)
col1.metric("Ймовірність деменції", f"{patient_prob:.3f}")
col2.metric("Передбачення моделі", pred_label_name)

# істинний клас — дрібним сірим текстом
st.caption(f"Лейбл з даних (ground truth): **{true_label_name}**")

st.caption(
    f"Агрегація: **{agg_mode_ui}**, top-k fraction = **{topk_frac:.2f}**, "
    f"поріг = **{threshold:.2f}**"
)

# --- Візуалізація зрізів ---

st.markdown("### Візуалізація зрізів пацієнта")

if len(slice_paths) == 0:
    st.warning("Для цього пацієнта не знайдено жодного зрізу.")
else:
    # Відсортуємо зрізи за ймовірністю деменції (від більш інформативних)
    order = np.argsort(slice_probs)[::-1]
    sorted_paths = [slice_paths[i] for i in order]
    sorted_probs = slice_probs[order]

    n_show = min(len(sorted_paths), 12)
    st.write(f"Показано **{n_show}** зрізів (з {len(sorted_paths)}) з найвищою ймовірністю деменції.")

    cols = st.columns(4)
    for i in range(n_show):
        col = cols[i % 4]
        p = sorted_paths[i]
        prob = sorted_probs[i]
        try:
            col.image(str(p), caption=f"p={prob:.3f}\n{p.name}", use_column_width=True)
        except Exception:
            col.write(f"{p.name} (p={prob:.3f}) — не вдалося відобразити зображення.")