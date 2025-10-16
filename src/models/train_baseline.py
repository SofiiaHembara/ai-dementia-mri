import argparse
from pathlib import Path
import warnings

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from src.data.dataset import Alzheimer2DDataset

def build_resnet18(num_classes=1):
    import torchvision
    from torchvision.models import resnet18

    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    except Exception:
        model = resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def compute_metrics(y_true, y_prob):
    y_true = y_true.astype("int64")
    y_pred = (y_prob >= 0.5).astype("int64")

    out = {}
    if len(set(y_true.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    return out

def main(args):
    index_path = Path("data/index.csv")
    if not index_path.exists():
        raise SystemExit("❌ Немає data/index.csv. Спочатку запустити python scripts/build_index.py")

    df = pd.read_csv(index_path)
    needed_cols = {"path", "label", "split"}
    if not needed_cols.issubset(df.columns):
        raise SystemExit(f"❌ У data/index.csv мають бути колонки: {needed_cols}")

    # train/val
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit("❌ Порожній train або val. Перевірити спліти в data/index.csv")

    train_ds = Alzheimer2DDataset(train_df, augment=True,  image_size=args.size)
    val_ds   = Alzheimer2DDataset(val_df,   augment=False, image_size=args.size)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_resnet18(num_classes=1).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # -------- train --------
        model.train()
        total_loss = 0.0
        n_samples = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.float().to(device)

            opt.zero_grad()
            logits = model(x).squeeze(1)  # (B,)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            n_samples  += x.size(0)

        train_loss = total_loss / max(1, n_samples)

        # -------- eval --------
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                logits = model(x).squeeze(1)
                prob = torch.sigmoid(logits).cpu().numpy()
                ys.extend(y.numpy().tolist())
                ps.extend(prob.tolist())

        import numpy as np
        ys = np.array(ys)
        ps = np.array(ps)
        m = compute_metrics(ys, ps)

        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
              f"AUC={m['roc_auc']:.3f} | F1={m['f1_macro']:.3f} | BA={m['bal_acc']:.3f}")

        score = m["roc_auc"]
        if score != score:  
            score = m["bal_acc"]

        if score > best_auc:
            best_auc = score
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ ✅ saved: {best_path} (score={best_auc:.3f})")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--size",   type=int, default=224)
    args = ap.parse_args()
    main(args)
