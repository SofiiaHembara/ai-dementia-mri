# src/models/train_3d.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models.video import r3d_18, R3D_18_Weights

# локальний датасет
from src.data_.dataset3d import MRIVolumeDataset


# -----------------------
# УТИЛІТИ
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)


def label_to01(s: str) -> int:
    s = str(s).lower().strip()
    return 1 if s == "dementia" else 0  # ВАЖЛИВО: тільки рівно 'dementia'


def compute_metrics(y_true, y_prob, thr=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(thr)).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_pred))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    return out


def patient_level_metrics(y_true, y_prob, sids, thr=0.5) -> Dict[str, float]:
    df = pd.DataFrame({"sid": sids, "y": y_true, "p": y_prob})
    agg = df.groupby("sid").agg(y=("y", "mean"), p=("p", "mean"))
    y = (agg["y"].values > 0.5).astype(int)  # справжня мітка пацієнта (якщо волюми з'єднані — середнє)
    p = agg["p"].values
    return compute_metrics(y, p, thr=thr)


def choose_gn_groups(num_channels: int, requested_groups: int) -> int:
    g = min(requested_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


def convert_bn3d_to_groupnorm(module: nn.Module, groups: int):
    """
    Рекурсивно замінює BatchNorm3d → GroupNorm зі збереженням кількості каналів.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            num_channels = child.num_features
            g = max(1, choose_gn_groups(num_channels, groups))
            gn = nn.GroupNorm(g, num_channels)
            setattr(module, name, gn)
        else:
            convert_bn3d_to_groupnorm(child, groups)


def freeze_backbone(model: nn.Module, freeze: bool = True):
    """
    Заморожуємо всі параметри, крім класифікатора (model.fc).
    """
    for n, p in model.named_parameters():
        if n.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = not freeze


# -----------------------
# ЕПОХА
# -----------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    loss_fn,
    optimizer=None,
    scaler=None,
    train: bool = True,
    accum_steps: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray, List[str], Dict[str, float]]:
    model.train(train)

    total_loss, total_n = 0.0, 0
    all_y, all_p, all_sid = [], [], []

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, (x, y, sid) in enumerate(tqdm(loader, leave=False)):
        x = x.to(device, non_blocking=False)  # (B,3,D,H,W)
        y = y.to(device, non_blocking=False)  # (B,)

        with torch.cuda.amp.autocast(enabled=(device == "cuda" and scaler is not None)):
            logits = model(x).squeeze(1)           # (B,)
            loss = loss_fn(logits, y)

        if train:
            (scaler.scale(loss) if scaler else loss).backward()

            if (step + 1) % accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)

        all_y.extend(y.detach().cpu().numpy().tolist())
        all_p.extend(prob.tolist())
        all_sid.extend(list(sid))

    # якщо останній step не співпав з кроком акумуляції — зробимо крок
    if train and ((step + 1) % accum_steps != 0):
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    metrics = compute_metrics(all_y, all_p, thr=0.5)
    return total_loss / max(1, total_n), np.array(all_y), np.array(all_p), all_sid, metrics


# -----------------------
# MAIN
# -----------------------
def main(args):
    set_seed(args.seed)
    ensure_dirs()

    # Завантаження індексу
    index_csv = Path(args.index_csv)
    df = pd.read_csv(index_csv)

    # Уніфікуємо назви колонок
    if "path" not in df.columns and "nifti_path" in df.columns:
        df["path"] = df["nifti_path"]

    need = {"path", "label", "split", "subject_id"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {need}, got: {df.columns.tolist()}")

    # прибираємо відсутні шляхи
    df = df.dropna(subset=["path"]).copy()
    df["path"] = df["path"].astype(str)
    df = df[df["path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    # Бінарні мітки (обережно з 'non_demented')
    df["_y"] = df["label"].map(label_to01).astype(int)

    # Спліт
    if not {"train", "val", "test"}.issubset(set(df["split"].unique())):
        # fallback: створити спліт по subject_id (70/15/15)
        sids = df["subject_id"].drop_duplicates().sample(frac=1.0, random_state=args.seed).tolist()
        n = len(sids)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        train_s, val_s, test_s = set(sids[:n_train]), set(sids[n_train:n_train+n_val]), set(sids[n_train+n_val:])
        df["split"] = df["subject_id"].apply(
            lambda s: "train" if s in train_s else ("val" if s in val_s else "test")
        )

    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df   = df[df.split == "val"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)

    # Підрахунок класів і pos_weight
    pos_count = int(train_df["_y"].sum())
    neg_count = int(len(train_df) - pos_count)
    if pos_count == 0:
        pos_weight = torch.tensor([1.0], dtype=torch.float32)
    else:
        pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight = pos_weight.to(device)
    print(f"ℹ️ train counts: non_demented={neg_count}, dementia={pos_count}, pos_weight={float(pos_weight.cpu().numpy()[0]):.3f}")

    # Датасети/Даталоадери
    train_ds = MRIVolumeDataset(train_df, size=args.size, depth=args.depth, augment=True)
    val_ds   = MRIVolumeDataset(val_df,   size=args.size, depth=args.depth, augment=False)
    test_ds  = MRIVolumeDataset(test_df,  size=args.size, depth=args.depth, augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=(device=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=max(1,args.batch//2), shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))
    test_dl  = DataLoader(test_ds,  batch_size=max(1,args.batch//2), shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))

    # Модель (r3d_18) + класифікатор (1 логіт)
    try:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    except Exception:
        model = r3d_18(weights="DEFAULT")  # для старіших версій torchvision

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    # GroupNorm замість BatchNorm3d (опційно)
    if args.groupnorm and args.groupnorm > 0:
        convert_bn3d_to_groupnorm(model, args.groupnorm)
        print(f"✅ Converted BatchNorm3d → GroupNorm({args.groupnorm})")

    model = model.to(device)

    # Фриз беку
    if args.freeze_epochs > 0:
        freeze_backbone(model, True)

    # Оптимізація/шедулер/AMP
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_score = -1.0
    best_thr = 0.5
    stale = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone")
            freeze_backbone(model, False)
            for g in optimizer.param_groups:
                g["lr"] = args.lr  # підняти LR для fine-tune

        tr_loss, yt, pt, sidt, tr_m = run_epoch(
            model, train_dl, device, loss_fn, optimizer, scaler,
            train=True, accum_steps=max(1, args.accum_steps)
        )

        with torch.no_grad():
            va_loss, yv, pv, sidv, va_m = run_epoch(
                model, val_dl, device, loss_fn, optimizer=None, scaler=None,
                train=False, accum_steps=1
            )

        # підбір порогу по F1 на patient-level
        thrs = np.linspace(0.05, 0.95, 19)
        f1s = []
        for t in thrs:
            f1s.append(patient_level_metrics(yv, pv, sidv, thr=t)["f1"])
        t_idx = int(np.argmax(f1s))
        cur_thr = float(thrs[t_idx])
        va_pat = patient_level_metrics(yv, pv, sidv, thr=cur_thr)

        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"AUC={va_m['roc_auc']:.3f} F1={va_m['f1']:.3f} BA={va_m['bal_acc']:.3f} | "
            f"patient@thr={cur_thr:.2f}: AUC={va_pat['roc_auc']:.3f} F1={va_pat['f1']:.3f} BA={va_pat['bal_acc']:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # зберігаємо last
        torch.save(model.state_dict(), "checkpoints/last_3d.pt")

        # критерій вибору best — patient-level AUC (fallback на BA)
        score = va_pat["roc_auc"]
        if math.isnan(score):
            score = va_pat["bal_acc"]

        if score > best_score:
            best_score = score
            best_thr = cur_thr
            stale = 0
            torch.save(model.state_dict(), "checkpoints/best_3d.pt")
            with open("artifacts/thresholds_3d.json", "w") as f:
                json.dump({"metric": "f1_patient", "thr": best_thr}, f, indent=2)
            print(f"  ↳ ✅ saved: checkpoints/best_3d.pt (score={best_score:.3f}, thr={best_thr:.2f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"⏹ Early stopping ({args.patience})")
                break

        scheduler.step()

    print(f"Best score: {best_score:.3f} @thr={best_thr:.2f}")

    # Оцінка на test
    with torch.no_grad():
        te_loss, yt, pt, sidt, te_m = run_epoch(
            model, test_dl, device, loss_fn, optimizer=None, scaler=None,
            train=False, accum_steps=1
        )
        te_pat = patient_level_metrics(yt, pt, sidt, thr=best_thr)

    print("TEST slice-level:", te_m)
    print("TEST patient-level @best_thr:", te_pat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--unfreeze_lr_mult", type=float, default=0.3)  # залишено для сумісності
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--groupnorm", type=int, default=0)  # 0 = не вмикати
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true")  # для сумісності з твоїми скриптами
    args = parser.parse_args()
    main(args)