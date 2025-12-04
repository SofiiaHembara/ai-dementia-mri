# src/models/train_2d_dino.py
# -*- coding: utf-8 -*-
"""
2D fine-tuning on OASIS slices with DINO/DINOv2 ViT backbones (timm).
- Works on CPU/CUDA/MPS
- Proper val_loss computation
- Flexible early stopping monitor: pat_auc | slice_auc | loss
- Patient-level aggregation by subject_id
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

# -------------------------
# Dataset
# CSV columns required: slice_path, subject_id, split, label
# label in {"dementia","non_demented"} (case-insensitive)
# -------------------------
class SliceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 224, train: bool = True):
        self.df = df.reset_index(drop=True).copy()
        self.size = size
        self.train = train

        # ensure paths are strings
        self.df["slice_path"] = self.df["slice_path"].astype(str)

        aug = []
        if self.train:
            aug += [
                T.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]
        else:
            aug += [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]

        # Grayscale → 3ch + to tensor + normalize
        self.tf = T.Compose(
            aug
            + [
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = row["slice_path"]
        img = Image.open(p).convert("L")  # read gray
        x = self.tf(img)  # (3,H,W)

        y = 1.0 if str(row["label"]).lower() == "dementia" else 0.0
        sid = str(row["subject_id"])
        return x, torch.tensor(y, dtype=torch.float32), sid

# -------------------------
# Model builder
# -------------------------
class BinHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 1)
        )

    def forward(self, x):
        # x: (B, D)
        return self.head(x)  # (B,1)

def build_model(backbone_name: str, img_size: int, device: torch.device):
    import timm
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0,          # фіч-екстрактор
        img_size=img_size,      # фіксуємо 224 → без ресемплінгу поз.ембедінгів
        dynamic_img_size=False,  # критично для MPS
    )
    feature_dim = model.num_features

    class BinHead(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.head = nn.Linear(in_dim, 1)
        def forward(self, x):
            return self.head(x)

    class Wrap(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            feats = self.backbone(x)      # (B, D)
            logits = self.head(feats)     # (B, 1)
            return logits

    net = Wrap(model, BinHead(feature_dim)).to(device)
    return net, model, net.head

def freeze_backbone(model: nn.Module, freeze: bool = True, unfreeze_top_k: int = 0):
    """
    Freeze all backbone params except the last `unfreeze_top_k` blocks (for ViT).
    """
    # Freeze everything
    for p in model.backbone.parameters():
        p.requires_grad = not freeze

    if freeze:
        return

    # try to unfreeze last blocks (ViT style)
    # many timm ViTs have blocks as model.blocks
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is None:
        return

    if unfreeze_top_k > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        for blk in blocks[-unfreeze_top_k:]:
            for p in blk.parameters():
                p.requires_grad = True

# -------------------------
# Metrics
# -------------------------
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

def compute_slice_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_pred))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    return out

def compute_patient_metrics(y_true_slice, y_prob_slice, sids_slice, thr=0.5):
    df = pd.DataFrame({"sid": sids_slice, "y": y_true_slice, "p": y_prob_slice})
    agg = df.groupby("sid").agg(y=("y", "mean"), p=("p", "mean"))
    y_true = (agg["y"].values > 0.5).astype(int)
    y_prob = agg["p"].values
    return compute_slice_metrics(y_true, y_prob, thr=thr)

# -------------------------
# Train / Eval loop
# -------------------------
from tqdm import tqdm

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn=None,
    optimizer: torch.optim.Optimizer = None,
    train: bool = True,
    accum_steps: int = 1,
):
    model.train(train)

    total_loss = 0.0
    total_n = 0

    y_all, p_all, sid_all = [], [], []

    # Autocast: disable on MPS (unsupported), enable on CUDA
    use_autocast = (device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_autocast else torch.cpu.amp.autocast

    if train and optimizer is None:
        raise ValueError("optimizer must be provided in train mode")

    optimizer_zeroed = False
    if train:
        optimizer.zero_grad(set_to_none=True)
        optimizer_zeroed = True

    with torch.set_grad_enabled(train):
        for step, batch in enumerate(tqdm(loader, leave=False)):
            x, y, sid = batch
            x = to_device(x, device)
            y = to_device(y, device)

            if use_autocast:
                ctx = autocast_ctx(dtype=torch.float16)
            else:
                # do not use mps autocast; cpu autocast is no-op
                ctx = torch.cpu.amp.autocast(enabled=False)

            with ctx:
                logits = model(x)                         # (B,1)
                logits = logits.reshape(x.size(0))        # -> (B,)
                prob = torch.sigmoid(logits).detach().cpu().numpy()

                if loss_fn is not None:
                    loss = loss_fn(logits, y)
                else:
                    # if no loss_fn (eval fallback), set 0
                    loss = torch.zeros((), device=device, dtype=logits.dtype)

            if train:
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs

            y_all.extend(y.detach().cpu().numpy().tolist())
            p_all.extend(prob.tolist())
            sid_all.extend(list(sid))

        # flush remaining grads if not divisible by accum_steps
        if train and optimizer_zeroed and (len(loader) % accum_steps != 0):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, total_n)
    slice_m = compute_slice_metrics(y_all, p_all, thr=0.5)

    # pick best patient threshold by F1
    thrs = np.linspace(0.05, 0.95, 19)
    f1s = []
    for t in thrs:
        f1s.append(compute_patient_metrics(y_all, p_all, sid_all, thr=t)["f1"])
    best_thr = float(thrs[int(np.argmax(f1s))])
    pat_m = compute_patient_metrics(y_all, p_all, sid_all, thr=best_thr)

    return avg_loss, np.array(y_all), np.array(p_all), np.array(sid_all), slice_m, pat_m, best_thr

# -------------------------
# Main
# -------------------------
def main(args):
    set_seed(args.seed)
    device = pick_device()
    print(f"Device: {device.type}")

    # ----------------- Load CSV -----------------
    index_csv = Path(args.index_csv)
    df = pd.read_csv(index_csv)
    required = {"slice_path", "subject_id", "split", "label"}
    assert required.issubset(df.columns), f"CSV must contain {required}, got {df.columns.tolist()}"

    # Drop rows with missing files
    df = df[df["slice_path"].astype(str).apply(os.path.exists)].reset_index(drop=True)

    # Train/Val/Test splits by 'split'
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df   = df[df.split == "val"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)

    # Safety: subject_id disjointness
    tr = set(train_df.subject_id)
    va = set(val_df.subject_id)
    te = set(test_df.subject_id)
    assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0, "Leakage across splits!"

    # ----------------- Datasets / Loaders -----------------
    train_ds = SliceDataset(train_df, size=args.size, train=True)
    val_ds   = SliceDataset(val_df,   size=args.size, train=False)
    test_ds  = SliceDataset(test_df,  size=args.size, train=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # ----------------- Pos weight & loss -----------------
    pos = (train_df["label"].str.lower() == "dementia").sum()
    neg = (train_df["label"].str.lower() == "non_demented").sum()
    pos_weight = torch.tensor([neg / max(1, pos)], device=device, dtype=torch.float32)
    print(f"Train counts -> neg={neg}, pos={pos}, pos_weight={float(pos_weight.cpu().numpy()[0]):.9f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ----------------- Model -----------------
    model, backbone, head = build_model(args.backbone, args.size, device)

    # Freeze backbone for probe phase
    freeze_backbone(model, freeze=True)
    for p in head.parameters():
        p.requires_grad = True

    # Optimizers: warm (head only), then full/partial unfreeze
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_score = -float("inf")
    best_thr = 0.5
    stale = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_2d_dino.pt"
    last_path = out_dir / "last_2d_dino.pt"

    # ----------------- Training -----------------
    for epoch in range(1, args.epochs + 1):
        # Unfreeze policy
        if epoch == args.freeze_epochs + 1:
            # unfreeze some top blocks + reduce LR
            freeze_backbone(model, freeze=False, unfreeze_top_k=args.unfreeze_top_k)
            # two-param groups: (backbone small lr), (head ft_lr)
            backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
            head_params     = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
            opt = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": args.ft_lr, "weight_decay": args.weight_decay},
                    {"params": head_params,     "lr": args.lr,    "weight_decay": args.weight_decay},
                ]
            )

        tr_loss, _, _, _, tr_m_slice, tr_m_pat, _ = run_epoch(
            model, train_dl, device, loss_fn=loss_fn, optimizer=opt, train=True, accum_steps=args.accum_steps
        )
        with torch.no_grad():
            va_loss, yv, pv, sidv, va_m_slice, va_m_pat, cur_thr = run_epoch(
                model, val_dl, device, loss_fn=loss_fn, optimizer=None, train=False
            )

        # what to monitor
        if args.monitor == "pat_auc":
            score = va_m_pat["roc_auc"]
        elif args.monitor == "slice_auc":
            score = va_m_slice["roc_auc"]
        elif args.monitor == "loss":
            score = -va_loss
        else:
            raise ValueError("--monitor must be one of pat_auc | slice_auc | loss")

        improved = (score > best_score + args.min_delta)

        # Log
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"slice: AUC={va_m_slice['roc_auc']:.3f} F1={va_m_slice['f1']:.3f} BA={va_m_slice['bal_acc']:.3f} | "
            f"patient@thr={cur_thr:.2f}: AUC={va_m_pat['roc_auc']:.3f} F1={va_m_pat['f1']:.3f} BA={va_m_pat['bal_acc']:.3f}"
        )

        # Save last
        torch.save(model.state_dict(), str(last_path))

        # Save best
        if improved:
            best_score = score
            best_thr = cur_thr
            stale = 0
            torch.save(model.state_dict(), str(best_path))
            with open(out_dir / "thresholds.json", "w") as f:
                json.dump({"metric": args.monitor, "thr": best_thr}, f, indent=2)
            print(f"  ↳ ✅ saved best: {best_path.name} (score={best_score:.3f}, thr={best_thr:.2f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"⏹ Early stopping: no improvement for {args.patience} epoch(s).")
                break

    # ----------------- Final eval on test -----------------
    with torch.no_grad():
        # Load best
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        te_loss, yt, pt, sidt, te_slice, te_pat, thr_te = run_epoch(
            model, test_dl, device, loss_fn=loss_fn, optimizer=None, train=False
        )
        print("TEST slice-level:", te_slice)
        print("TEST patient-level @best_thr:", te_pat)

# -------------------------
# CLI
# -------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--freeze_epochs", type=int, default=25)
    p.add_argument("--unfreeze_top_k", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ft_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m",
                   help="e.g. vit_small_patch14_dinov2.lvd142m or vit_base_patch16_224.dino")
    p.add_argument("--monitor", type=str, default="pat_auc", choices=["pat_auc", "slice_auc", "loss"])
    p.add_argument("--min_delta", type=float, default=1e-4, help="required improvement over best_score")
    return p.parse_args()

if __name__ == "__main__":
    main(get_args())