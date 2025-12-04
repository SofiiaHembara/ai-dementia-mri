import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def metrics_binary(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return out

def plot_roc(y_true, y_prob, path_png):
    try:
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        if len(np.unique(y_true)) < 2:
            return
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend(loc="lower right")
        Path(path_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_png, bbox_inches="tight", dpi=160); plt.close()
    except Exception:
        pass

class DinoSliceTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size=224):
        self.df = df.reset_index(drop=True).copy()
        if "path" in self.df.columns:
            self.img_col = "path"
        elif "slice_path" in self.df.columns:
            self.img_col = "slice_path"
        else:
            raise ValueError("Не знайдено колонки 'path' або 'slice_path' у CSV.")
        self.tf = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = str(row[self.img_col])
        img = Image.open(p).convert("L").convert("RGB")
        x = self.tf(img)
        lab = str(row["label"]).lower()
        y = 1.0 if lab in ("dementia", "1", "pos", "positive", "true") else 0.0
        sid = row.get("subject_id", os.path.basename(os.path.dirname(p)))
        return x, torch.tensor(y, dtype=torch.float32), str(sid), p

class DinoBinaryHead_LNLinear(nn.Module):
    """LayerNorm(D) -> Linear(D,1)"""
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        D = getattr(self.backbone, "num_features", None)
        if D is None:
            raise ValueError("Не вдалося визначити num_features у backbone.")
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

    def forward(self, x):
        feats = self.backbone(x)              # (B, D)
        logits = self.head(feats).squeeze(1)  # (B,)
        return logits

class LinearOnlyContainer(nn.Module):
    """Контейнер для сумісності зі збереженим 'head.head.weight' / 'head.head.bias'."""
    def __init__(self, D: int):
        super().__init__()
        self.head = nn.Linear(D, 1)

    def forward(self, feats):
        return self.head(feats)

class DinoBinaryHead_LinearOnly(nn.Module):
    """Backbone -> Linear(D,1) всередині контейнера 'head' з підмодулем 'head'."""
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        D = getattr(self.backbone, "num_features", None)
        if D is None:
            raise ValueError("Не вдалося визначити num_features у backbone.")
        self.head = LinearOnlyContainer(D)

    def forward(self, x):
        feats = self.backbone(x)             # (B, D)
        logits = self.head(feats).squeeze(1) # (B,)
        return logits

def load_threshold(thr_json: str | None, default_thr: float = 0.5) -> float:
    if thr_json and Path(thr_json).exists():
        try:
            obj = json.load(open(thr_json, "r"))
            return float(obj.get("thr", default_thr))
        except Exception:
            return default_thr
    return default_thr

def try_load_compatible(model_lnlin, model_linlinear, weights_path, device):
    """
    Повертає (model, used_variant, missing, unexpected)
    used_variant ∈ {'ln_linear', 'linear_only', 'ln_linear_renamed'}
    """
    sd = torch.load(weights_path, map_location=device)
    keys = list(sd.keys())

    # Випадок 1: вага під LN+Linear ('head.0.*', 'head.1.*')
    if any(k.startswith("head.0.") or k.startswith("head.1.") for k in keys):
        missing, unexpected = model_lnlin.load_state_dict(sd, strict=False)
        return model_lnlin, "ln_linear", missing, unexpected

    # Випадок 2: вага під 'head.head.*' (Linear-тільки всередині контейнера)
    if any(k.startswith("head.head.") for k in keys):
        # Спробуємо 2a) модель LinearOnly (повний збіг імен)
        missing, unexpected = model_linlinear.load_state_dict(sd, strict=False)
        # якщо завантажилося OK — використовуємо
        if not missing and not unexpected:
            return model_linlinear, "linear_only", missing, unexpected

        # 2b) або перейменуємо ключі head.head.* -> head.1.* (для LN+Linear),
        # і підженемо LN до identity (втім це все одно не матиме збережених ваг LN)
        sd2 = {}
        for k,v in sd.items():
            if k.startswith("head.head."):
                new_k = k.replace("head.head.", "head.1.")
                sd2[new_k] = v
            else:
                sd2[k] = v
        missing, unexpected = model_lnlin.load_state_dict(sd2, strict=False)
        return model_lnlin, "ln_linear_renamed", missing, unexpected

    # Випадок 3: інші нестандартні варіанти — пробуємо як є у LN+Linear
    missing, unexpected = model_lnlin.load_state_dict(sd, strict=False)
    return model_lnlin, "ln_linear", missing, unexpected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--weights", default="best_2d_dino.pt", type=str)
    ap.add_argument("--backbone", default="vit_base_patch16_224.dino", type=str)
    ap.add_argument("--size", default=224, type=int)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--num_workers", default=0, type=int)
    ap.add_argument("--thr_json", default="artifacts/thresholds.json", type=str)
    ap.add_argument("--out_dir", default="artifacts", type=str)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    df = pd.read_csv(args.index_csv)
    if "split" not in df.columns:  raise ValueError("У індексі немає 'split'.")
    if "label" not in df.columns:  raise ValueError("У індексі немає 'label'.")

    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if len(test_df) == 0: raise ValueError("У CSV немає рядків зі split == 'test'.")
    path_col = "path" if "path" in test_df.columns else ("slice_path" if "slice_path" in test_df.columns else None)
    if path_col is None: raise ValueError("Не знайдено 'path' або 'slice_path' у CSV.")
    before = len(test_df)
    test_df = test_df[test_df[path_col].notna() & (test_df[path_col].astype(str).str.len() > 0)]
    print(f"Test rows: {before} -> {len(test_df)} (після фільтру NaN шляхів)")

    ds = DinoSliceTestDataset(test_df, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # створюємо обидві варіації моделей
    model_lnlin   = DinoBinaryHead_LNLinear(args.backbone).to(device)
    model_linlinear = DinoBinaryHead_LinearOnly(args.backbone).to(device)

    used_variant = "random_init"
    missing = unexpected = []
    if Path(args.weights).exists():
        model, used_variant, missing, unexpected = try_load_compatible(
            model_lnlin, model_linlinear, args.weights, device
        )
        print(f"Loaded weights: {args.weights} (variant={used_variant})")
        if missing:    print("  Missing keys:", list(missing))
        if unexpected: print("  Unexpected keys:", list(unexpected))
    else:
        print(f"⚠️ Увага: файл ваг не знайдено: {args.weights}. Оцінимо з ImageNet-фічами (результати будуть гірші).")
        model = model_lnlin

    model.eval()
    all_prob, all_y, all_sid, all_path = [], [], [], []

    with torch.no_grad():
        for x, y, sid, p in dl:
            x = x.to(device)
            logits = model(x)                  # (B,)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            all_prob.extend(prob.tolist())
            all_y.extend(y.numpy().tolist())
            all_sid.extend(list(sid))
            all_path.extend(list(p))

    thr = load_threshold(args.thr_json, 0.5)
    m_slice = metrics_binary(all_y, all_prob, thr=thr)
    print(f"TEST slice-level @thr={thr:.2f}: {m_slice}")

    df_pred = pd.DataFrame({
        "subject_id": all_sid,
        "slice_path": all_path,
        "label": ["dementia" if int(v)==1 else "non_demented" for v in all_y],
        "prob": all_prob
    })
    # агрегуємо середнім по subject_id
    def label_to_bin(vals):
        arr = np.array([1.0 if str(v)=="dementia" else 0.0 for v in vals])
        return float(1.0 if arr.mean()>0.5 else 0.0)
    agg = df_pred.groupby("subject_id").agg(
        prob=("prob", "mean"),
        y=("label", label_to_bin)
    ).reset_index()
    m_pat = metrics_binary(agg["y"].values, agg["prob"].values, thr=thr)
    print(f"TEST patient-level @thr={thr:.2f}: {m_pat}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "test_preds_2d_dino.csv"
    agg_csv  = out_dir / "test_patient_agg_2d_dino.csv"
    df_pred.to_csv(pred_csv, index=False); agg.to_csv(agg_csv, index=False)
    print(f"Saved: {pred_csv}"); print(f"Saved: {agg_csv}")

    plot_roc(all_y, all_prob, out_dir / "roc_slice.png")
    plot_roc(agg["y"].values, agg["prob"].values, out_dir / "roc_patient.png")
    print(f"ROC curves saved to: {out_dir/'roc_slice.png'} and {out_dir/'roc_patient.png'}")

if __name__ == "__main__":
    main()