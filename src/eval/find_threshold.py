import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from src.models.train_baseline import build_model
from src.data_.dataset import Alzheimer2DDataset

def collect_probs(model, df, device, batch=64):
    ds = Alzheimer2DDataset(df, augment=False, image_size=224)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            prob = torch.sigmoid(model(x).squeeze(1)).cpu().numpy()
            ys.extend(y.numpy().tolist()); ps.extend(prob.tolist())
    ys = np.array(ys); ps = np.array(ps)
    return ys, ps

def best_threshold(ys, ps, metric="f1"):
    best_thr, best_val = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 181):
        preds = (ps >= thr).astype(int)
        if metric == "f1":
            val = f1_score(ys, preds)
        elif metric == "youden":
            # Youden J = TPR - FPR
            tp = ((preds==1) & (ys==1)).sum()
            fn = ((preds==0) & (ys==1)).sum()
            tn = ((preds==0) & (ys==0)).sum()
            fp = ((preds==1) & (ys==0)).sum()
            tpr = tp / max(1, tp+fn)
            fpr = fp / max(1, fp+tn)
            val = tpr - fpr
        else:
            raise ValueError("metric must be 'f1' or 'youden'")
        if val > best_val:
            best_val, best_thr = val, thr
    return float(best_thr), float(best_val)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--metric", type=str, default="f1", choices=["f1","youden"])
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18","efficientnet_b0"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    df = pd.read_csv("data/index.csv")
    val_df = df[df["split"]=="val"].copy()
    if val_df.empty:
        raise SystemExit("No val split in data/index.csv")

    # load model
    model = build_model(args.model, num_classes=1).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    # global
    ys, ps = collect_probs(model, val_df, device)
    thr_global, val_global = best_threshold(ys, ps, metric=args.metric)

    tokens = ["NonDemented","VeryMildDemented","MildDemented","ModerateDemented"]
    per_class = {}
    for tok in tokens:
        sub = val_df[val_df["path"].str.contains(tok, case=False, na=False)]
        if len(sub) >= 10:
            y_sub, p_sub = collect_probs(model, sub, device)
            thr, val = best_threshold(y_sub, p_sub, metric=args.metric)
            per_class[tok] = {"thr": thr, args.metric: val, "n": int(len(sub))}

    out = {
        "metric": args.metric,
        "global": {"thr": thr_global, args.metric: val_global, "n": int(len(val_df))},
        "per_class_folder": per_class,
    }
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open("artifacts/thresholds.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved artifacts/thresholds.json")
    print(out)
