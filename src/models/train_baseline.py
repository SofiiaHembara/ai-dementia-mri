import argparse
import os
import random
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from src.data_.dataset import Alzheimer2DDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = y_true.astype("int64")
    y_pred = (y_prob >= thr).astype("int64")
    out = {}
    if len(set(y_true.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    return out


def build_model(model_name="resnet18", num_classes=1):
    import torchvision

    if model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m
    else:
        from torchvision.models import resnet18, ResNet18_Weights

        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m


def freeze_backbone(model: nn.Module, freeze: bool = True):
    for name, p in model.named_parameters():
        head = name.startswith("fc") or "classifier.1" in name
        if not head:
            p.requires_grad = not (freeze)

def run_epoch(model, loader, device, loss_fn, optimizer=None, amp=False, grad_clip=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_samples = 0
    ys, ps = [], []

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x).squeeze(1)  # (B,)
            loss = loss_fn(logits, y)
            prob = torch.sigmoid(logits).detach().cpu().numpy()

        if is_train:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(prob.tolist())

    ys = np.array(ys)
    ps = np.array(ps)
    avg_loss = total_loss / max(1, n_samples)
    return avg_loss, ys, ps


def main(args):
    set_seed(args.seed)

    index_path = Path("data/index.csv")
    if not index_path.exists():
        raise SystemExit("❌ Немає data/index.csv. Спочатку запустити: python scripts/build_index.py")

    df = pd.read_csv(index_path)
    needed = {"path", "label", "split"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"❌ У data/index.csv мають бути колонки: {needed}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit("❌ Порожній train або val. Перевірити спліти в data/index.csv")

    # pos_weight від дисбалансу (train)
    pos_count = (train_df["label"] == "dementia").sum()
    neg_count = (train_df["label"] == "non_demented").sum()
    if pos_count == 0 or neg_count == 0:
        print("⚠️  Один із класів відсутній у train. pos_weight=1.0")
        pos_weight = 1.0
    else:
        pos_weight = neg_count / float(pos_count)
    print(f"ℹ️  Train class counts: non_demented={neg_count}, dementia={pos_count}, pos_weight={pos_weight:.3f}")

    # Datasets / Loaders
    train_ds = Alzheimer2DDataset(train_df, augment=True, image_size=args.size)
    val_ds = Alzheimer2DDataset(val_df, augment=False, image_size=args.size)

    if args.balanced:
        weights = train_df["label"].map({"non_demented": 1.0, "dementia": pos_weight}).values
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_dl = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=True
        )
    else:
        train_dl = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )

    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, num_classes=1).to(device)


    if args.freeze_epochs > 0:
        print(f"🧊 Freezing backbone for {args.freeze_epochs} epoch(s)")
        freeze_backbone(model, True)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    best_score = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone for fine-tuning")
            freeze_backbone(model, False)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * args.unfreeze_lr_mult, weight_decay=args.weight_decay)
            remaining = max(1, args.epochs - epoch + 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=args.min_lr)

        train_loss, _, _ = run_epoch(
            model, train_dl, device, loss_fn, optimizer=opt, amp=args.amp, grad_clip=args.grad_clip
        )

        with torch.no_grad():
            val_loss, y_true, y_prob = run_epoch(model, val_dl, device, loss_fn, optimizer=None, amp=False)

        m = compute_metrics(y_true, y_prob, thr=args.val_thr)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"AUC={m['roc_auc']:.3f} | F1={m['f1_macro']:.3f} | BA={m['bal_acc']:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        torch.save(model.state_dict(), last_path)

        score = m["roc_auc"]
        if np.isnan(score):
            score = m["bal_acc"]

        if score > best_score:
            best_score = score
            stale = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ ✅ saved: {best_path} (score={best_score:.3f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"⏹ Early stopping: no improvement for {args.patience} epoch(s).")
                break

        scheduler.step()

    print(f"Best score: {best_score:.3f} (AUC if available, else BA)")
    print(f"Models saved: {best_path.name} (best), {last_path.name} (last)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)

    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (AMP) if CUDA is available")

    # Transfer learning strategy
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Freeze backbone for N initial epochs")
    parser.add_argument("--unfreeze_lr_mult", type=float, default=0.3, help="LR multiplier after unfreezing")

    # Regularization / stopping
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--balanced", action="store_true", help="Use WeightedRandomSampler for class balancing")
    parser.add_argument("--val_thr", type=float, default=0.5, help="Threshold for val metrics (F1/BA)")

    # System
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
