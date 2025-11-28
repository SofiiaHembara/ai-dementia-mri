# src/models/train_3d.py
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from src.data_.dataset3d import Alzheimer3DDataset

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

def build_r3d18(num_classes=1):
    from torchvision.models.video import r3d_18, R3D_18_Weights
    m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # один логіт для BCEWithLogits
    return m

def run_epoch(model, loader, device, loss_fn, optimizer=None, amp=False, grad_clip=None):
    is_train = optimizer is not None
    model.train(is_train)
    ys, ps = [], []
    total_loss, n = 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y in loader:
        # x: (B, C=3, D, H, W) → r3d_18 очікує (B, C, T, H, W); T=D
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
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(prob.tolist())

    return total_loss / max(1, n), np.array(ys), np.array(ps)

def main(args):
    index_csv = Path(args.index_csv)
    if not index_csv.exists():
        raise SystemExit(f"❌ Not found: {index_csv}")

    df = pd.read_csv(index_csv)
    needed = {"path", "label", "split"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"❌ index must contain columns: {needed}")

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise SystemExit("❌ Empty train/val split")

    # класовий дисбаланс
    pos = (train_df["label"] == "dementia").sum()
    neg = (train_df["label"] == "non_demented").sum()
    pos_weight = neg / float(pos) if pos > 0 else 1.0
    print(f"ℹ️ train counts: non_demented={neg}, dementia={pos}, pos_weight={pos_weight:.3f}")

    # датасети/лоадери
    shape = (args.depth, args.size, args.size)
    train_ds = Alzheimer3DDataset(train_df, image_shape=shape, augment=True)
    val_ds   = Alzheimer3DDataset(val_df,   image_shape=shape, augment=False)

    if args.balanced:
        weights = train_df["label"].map({"non_demented": 1.0, "dementia": pos_weight}).values
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_r3d18(num_classes=1).to(device)

    if args.freeze_epochs > 0:
        # заморозимо все крім fc
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
        print(f"🧊 Freezing backbone for {args.freeze_epochs} epoch(s)")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True, parents=True)
    best_path = ckpt_dir / "best_3d.pt"
    last_path = ckpt_dir / "last_3d.pt"

    best_score, stale = -1.0, 0
    for ep in range(1, args.epochs + 1):
        # розморозка після freeze_epochs
        if ep == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone")
            for p in model.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * args.unfreeze_lr_mult,
                                    weight_decay=args.weight_decay)
            remain = max(1, args.epochs - ep + 1)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remain, eta_min=args.min_lr)

        tr_loss, _, _ = run_epoch(model, train_dl, device, loss_fn, optimizer=opt,
                                  amp=args.amp, grad_clip=args.grad_clip)
        with torch.no_grad():
            val_loss, y_true, y_prob = run_epoch(model, val_dl, device, loss_fn, optimizer=None, amp=False)

        m = compute_metrics(y_true, y_prob, thr=args.val_thr)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"AUC={m['roc_auc']:.3f} | F1={m['f1_macro']:.3f} | BA={m['bal_acc']:.3f} | "
              f"lr={sched.get_last_lr()[0]:.2e}")

        torch.save(model.state_dict(), last_path)
        score = m["roc_auc"] if not np.isnan(m["roc_auc"]) else m["bal_acc"]
        if score > best_score:
            best_score, stale = score, 0
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ ✅ saved: {best_path} (score={best_score:.3f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"⏹ Early stopping ({args.patience})")
                break

        sched.step()

    print(f"Best score: {best_score:.3f}")
    print(f"Models: {best_path.name} (best), {last_path.name} (last)")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", type=str, default="data/index_oasis1.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)      # 3D → пам'ять обмежує
    ap.add_argument("--size", type=int, default=128)     # H,W
    ap.add_argument("--depth", type=int, default=64)     # D (T)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--freeze_epochs", type=int, default=2)
    ap.add_argument("--unfreeze_lr_mult", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--val_thr", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    main(args)