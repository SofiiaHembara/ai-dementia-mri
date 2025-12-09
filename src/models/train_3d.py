# src/models/train_3d.py
from __future__ import annotations
import argparse, json, os, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision.models.video import r3d_18, R3D_18_Weights

from src.data_.dataset3d import VolumeDataset

warnings.filterwarnings("ignore", category=UserWarning)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def convert_bn_to_gn(module: nn.Module, num_groups: int = 8):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            gn = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels, affine=True)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)
    return module

class VideoHead(nn.Module):
    def __init__(self, in_features: int, p: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(x)

def build_model(groupnorm: int | None):
    # 3D backbone (Kinetics pretrain). Очікує (B,3,T,H,W) з [0..1] або нормалізовано.
    weights = R3D_18_Weights.KINETICS400_V1
    backbone = r3d_18(weights=weights)
    in_feats = backbone.fc.in_features
    backbone.fc = nn.Identity()
    head = VideoHead(in_feats)
    model = nn.Sequential(backbone, head)

    if groupnorm and groupnorm > 0:
        convert_bn_to_gn(model, groupnorm)
        print(f"✅ Converted BatchNorm3d → GroupNorm({groupnorm})")

    return model, weights

def patient_metrics(y_true, y_prob, sids, thr=0.5):
    df = pd.DataFrame(dict(sid=sids, y=y_true, p=y_prob))
    agg = df.groupby("sid").agg(y=("y","mean"), p=("p","mean")).reset_index()
    y = (agg["y"].values > 0.5).astype(int)
    p = agg["p"].values
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["roc_auc"] = float("nan")
    yhat = (p >= thr).astype(int)
    out["f1"] = float(f1_score(y, yhat))
    out["bal_acc"] = float(balanced_accuracy_score(y, yhat))
    return out

@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None, weights_norm=None, find_thr=False):
    model.eval()
    ys, ps, sids = [], [], []
    for x, y, sid in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=False)  # (B,3,D,S,S)
        if weights_norm is not None:
            # нормалізація під Kinetics (стандартизовані mean/std)
            mean = torch.as_tensor(weights_norm.mean, device=device)[None, :, None, None, None]
            std  = torch.as_tensor(weights_norm.std,  device=device)[None, :, None, None, None]
            x = (x - mean) / std

        logits = model(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (B,1) → (B,)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        ys.extend(y.numpy().tolist())
        ps.extend(p.tolist())
        sids.extend(list(sid))

    ys = np.asarray(ys).astype(int)
    ps = np.asarray(ps)

    # slice-level (насправді тут «volume-level», але лишимо назву для сумісності)
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
    out_slice = {}
    try:
        out_slice["roc_auc"] = float(roc_auc_score(ys, ps))
    except Exception:
        out_slice["roc_auc"] = float("nan")
    yhat = (ps >= 0.5).astype(int)
    out_slice["f1"] = float(f1_score(ys, yhat))
    out_slice["bal_acc"] = float(balanced_accuracy_score(ys, yhat))

    thr = 0.5
    if find_thr:
        thrs = np.linspace(0.05, 0.95, 19)
        f1s = []
        for t in thrs:
            f1s.append(patient_metrics(ys, ps, sids, thr=t)["f1"])
        thr = float(thrs[int(np.argmax(f1s))])

    out_pat = patient_metrics(ys, ps, sids, thr=thr)
    return out_slice, out_pat, thr

def run_epoch(model, loader, device, loss_fn, optimizer, scaler, weights_norm, train=True, accum_steps=1, use_amp=False):
    if train:
        model.train(True)
    else:
        model.train(False)

    total_loss, total_n = 0.0, 0
    ys, ps, sids = [], [], []

    for step, (x, y, sid) in enumerate(tqdm(loader, leave=False)):
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)

        if weights_norm is not None:
            mean = torch.as_tensor(weights_norm.mean, device=device)[None, :, None, None, None]
            std  = torch.as_tensor(weights_norm.std,  device=device)[None, :, None, None, None]
            x = (x - mean) / std

        if train:
            with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (B,)
                loss = loss_fn(logits, y)
            (loss/accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(x).squeeze(-1).squeeze(-1).squeeze(-1)
            loss = loss_fn(logits, y)

        p = torch.sigmoid(logits).detach().cpu().numpy()
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(p.tolist())
        sids.extend(list(sid))

        total_loss += loss.item() * x.size(0)
        total_n    += x.size(0)

    # «slice-level»
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
    ys = np.asarray(ys).astype(int)
    ps = np.asarray(ps)
    out_slice = {}
    try:
        out_slice["roc_auc"] = float(roc_auc_score(ys, ps))
    except Exception:
        out_slice["roc_auc"] = float("nan")
    yhat = (ps >= 0.5).astype(int)
    out_slice["f1"] = float(f1_score(ys, yhat))
    out_slice["bal_acc"] = float(balanced_accuracy_score(ys, yhat))

    return total_loss/max(1,total_n), out_slice, ys, ps, sids

def main(args):
    device = get_device()
    print("Device:", device.type)

    df = pd.read_csv(args.index_csv)

    # ✓ Узгодження назв колонок
    if "nifti_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "nifti_path"})

    # ✓ Перевірка потрібних колонок
    need = {"subject_id", "nifti_path", "label", "split"}
    missing = need - set(df.columns)
    assert not missing, f"В індексі бракує колонок: {missing}. Є: {list(df.columns)}"

    # ✓ Лишаємо лише записи з валідними шляхами
    df["nifti_path"] = df["nifti_path"].astype(str)
    if "path_exists" in df.columns:
        df = df[df["path_exists"] == True].copy()
    else:
        import os
        df = df[df["nifti_path"].apply(os.path.exists)].copy()

    df = df.dropna(subset=["nifti_path"]).reset_index(drop=True)
    tr_df = df[df.split=="train"].reset_index(drop=True)
    va_df = df[df.split=="val"].reset_index(drop=True)

    print(f"ℹ️ train counts:", dict(tr_df["label"].value_counts()))
    # датасети
    tr_ds = VolumeDataset(tr_df, size=args.size, depth=args.depth, augment=True,  cache=args.cache)
    va_ds = VolumeDataset(va_df, size=args.size, depth=args.depth, augment=False, cache=args.cache)

    # балансування
    if args.balanced:
        labels = (tr_df["label"].str.lower()=="dementia").astype(int).values
        class_sample_count = np.bincount(labels, minlength=2)
        weight = 1. / np.maximum(class_sample_count, 1)
        samples_weight = weight[labels]
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=False, drop_last=False)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.num_workers, pin_memory=False, drop_last=False)

    va_loader = DataLoader(va_ds, batch_size=max(1,args.batch*2), shuffle=False,
                           num_workers=args.num_workers, pin_memory=False)

    # модель
    model, weights = build_model(groupnorm=args.groupnorm)
    model.to(device)

    # нормалізація під Kinetics
    weights_norm = weights.transforms()
    # лосс з pos_weight
    pos = (tr_df["label"].str.lower()=="dementia").sum()
    neg = (tr_df["label"].str.lower()=="non_demented").sum()
    pos_weight = torch.tensor([neg/max(pos,1)], device=device, dtype=torch.float32)
    print(f"pos_weight={float(pos_weight.cpu().numpy()[0]):.3f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

    # заморозка бекбону
    def freeze_backbone(flag: bool):
        for n, p in model[0].named_parameters():
            p.requires_grad = not flag
    if args.freeze_epochs > 0:
        freeze_backbone(True)
        print(f"🧊 Freezing backbone for {args.freeze_epochs} epoch(s)")

    best = -1.0
    best_thr = 0.5
    stale = 0

    for epoch in range(1, args.epochs+1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone")
            freeze_backbone(False)
            # зменшуємо LR при розморозці, щоб не зламати фічі
            for g in opt.param_groups:
                g["lr"] = args.lr * args.unfreeze_lr_mult

        tr_loss, tr_m_slice, _, _, _ = run_epoch(
            model, tr_loader, device, loss_fn, opt, None, weights_norm,
            train=True, accum_steps=args.accum_steps, use_amp=(device.type=="cuda")
        )
        with torch.no_grad():
            va_loss, va_m_slice, ys, ps, sids = run_epoch(
                model, va_loader, device, loss_fn, None, None, weights_norm,
                train=False, accum_steps=1, use_amp=False
            )
            # підбір порога по пацієнту
            thrs = np.linspace(0.05, 0.95, 19)
            f1s = []
            for t in thrs:
                f1s.append(patient_metrics(ys, ps, sids, thr=t)["f1"])
            cur_thr = float(thrs[int(np.argmax(f1s))])
            va_m_pat = patient_metrics(ys, ps, sids, thr=cur_thr)

        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"slice: AUC={tr_m_slice['roc_auc']:.3f}->{va_m_slice['roc_auc']:.3f} F1={va_m_slice['f1']:.3f} BA={va_m_slice['bal_acc']:.3f} | "
            f"patient@thr={cur_thr:.2f}: AUC={va_m_pat['roc_auc']:.3f} F1={va_m_pat['f1']:.3f} BA={va_m_pat['bal_acc']:.3f}"
        )

        # обираємо найкраще за patient AUC (fallback на BA)
        score = va_m_pat["roc_auc"]
        if math.isnan(score):
            score = va_m_pat["bal_acc"]

        # зберігаємо
        Path("checkpoints").mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), "checkpoints/last_3d.pt")
        if score > best:
            best = score
            best_thr = cur_thr
            stale = 0
            torch.save(model.state_dict(), "checkpoints/best_3d.pt")
            Path("artifacts").mkdir(exist_ok=True, parents=True)
            with open("artifacts/thresholds_3d.json","w") as f:
                json.dump({"metric":"f1_patient","thr":best_thr}, f, indent=2)
            print(f"  ↳ ✅ saved: checkpoints/best_3d.pt (score={best:.3f}, thr={best_thr:.2f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"⏹ Early stopping ({args.patience})")
                break

        sched.step()

    print(f"Best score (patient AUC): {best:.3f} @ thr={best_thr:.2f}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--freeze_epochs", type=int, default=4)
    ap.add_argument("--unfreeze_lr_mult", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--groupnorm", type=int, default=8)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--cache", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)
    main(args)
