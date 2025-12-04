# src/models/train_3d_monai.py
import argparse, os, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# MONAI
try:
    from monai.data import CacheDataset
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing, Resize,
        ScaleIntensityRange, RandFlip, RandRotate, RandAffine, EnsureType
    )
    from monai.networks.nets import DenseNet121
except Exception as e:
    raise RuntimeError(
        "Не знайдено MONAI. Встанови: pip install 'monai[nibabel,torch]' \n"
        f"Помилка імпорту: {e}"
    )

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def lab2int(v):
    s = str(v).strip().lower()
    if s in ["1","true","yes","dementia"]:
        return 1
    if s in ["0","false","no","non_demented","non-demented","control"]:
        return 0
    try:
        f = float(s)
        return 1 if f >= 0.5 else 0
    except:
        return 0

def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    out["f1"]      = float(f1_score(y_true, y_pred, zero_division=0))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    return out

def build_transforms(depth, size, augment=False):
    # MONAI pipeline: Load NIfTI → channel-first → орієнтація RAS → scale/intensity → resize → (augment) → tensor
    t = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),     # (1, D, H, W)
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size=(depth, size, size), mode="trilinear"),
        EnsureType()
    ]
    if augment:
        t.extend([
            RandFlip(spatial_axis=0, prob=0.2),
            RandFlip(spatial_axis=1, prob=0.2),
            RandFlip(spatial_axis=2, prob=0.2),
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.2),
            RandAffine(prob=0.15, translate_range=(2,2,2), scale_range=(0.1,0.1,0.1))
        ])
    return Compose(t)

class VolDataset(CacheDataset):
    """
    CacheDataset очікує список dictів: {"img": <path>, "y": 0/1, "sid": str}
    Transforms повертають тензор 3D (1,D,H,W)
    """
    def __init__(self, items, transforms):
        super().__init__(data=items, transform=transforms, cache_rate=0.0, num_workers=0)

def freeze_backbone(model, freeze=True):
    for p in model.parameters():
        p.requires_grad = not freeze

def replace_norm_with_group(model, num_groups=8):
    # Для DenseNet121 (monai) параметр 'norm' керує типом нормалізації.
    # Якщо створено з norm='GROUP', то нічого не робимо.
    # Якщо інший тип — тут можна пройтись і замінити BatchNorm на GroupNorm вручну (не потрібно, якщо ми одразу задаємо norm='GROUP').
    return model

def main(args):
    device = get_device()
    print("Device:", device)

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # ---- Load index (subject-level for 3D) ----
    df = pd.read_csv(args.index_csv)
    # авто-визначення колонки з шляхом до NIfTI
    path_col = "path" if "path" in df.columns else ("nifti_path" if "nifti_path" in df.columns else None)
    if path_col is None:
        raise ValueError(f"У CSV немає 'path' або 'nifti_path'. Є: {df.columns.tolist()}")

    need_cols = set([path_col, "label", "split"])
    assert need_cols.issubset(df.columns), f"Потрібні колонки {need_cols}, але отримано {df.columns.tolist()}"

    df = df.copy()
    df["y"] = df["label"].apply(lab2int)
    if "subject_id" not in df.columns:
        df["subject_id"] = df[path_col].apply(lambda p: Path(str(p)).stem)

    tr_df = df[df["split"]=="train"].reset_index(drop=True)
    va_df = df[df["split"]=="val"].reset_index(drop=True)
    te_df = df[df["split"]=="test"].reset_index(drop=True)

    print(f"train counts: non_demented={(tr_df['y']==0).sum()}, dementia={(tr_df['y']==1).sum()}")

    # MONAI transforms
    train_tf = build_transforms(args.depth, args.size, augment=True)
    val_tf   = build_transforms(args.depth, args.size, augment=False)
    test_tf  = build_transforms(args.depth, args.size, augment=False)

    def to_items(df_):
        return [{"img": row[path_col], "y": int(row["y"]), "sid": str(row["subject_id"])} for _, row in df_.iterrows()]

    train_ds = VolDataset(to_items(tr_df), train_tf)
    val_ds   = VolDataset(to_items(va_df),   val_tf)
    test_ds  = VolDataset(to_items(te_df),  test_tf)

    # Балансований семплер
    y_train = np.array([it["y"] for it in train_ds.data])
    pos = int((y_train==1).sum()); neg = len(y_train)-pos
    pos_weight = torch.tensor([neg/max(1,pos)], device=device, dtype=torch.float32)
    print(f"pos_weight: {float(pos_weight):.3f}")

    w = np.where(y_train==1, neg/(pos+1e-9), pos/(neg+1e-9))
    sampler = WeightedRandomSampler(weights=torch.tensor(w, dtype=torch.double), num_samples=len(y_train), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=False)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # ---- Model (MONAI DenseNet121 3D) ----
    norm_kind = "GROUP" if args.groupnorm > 0 else "BATCH"
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.0, norm=norm_kind, norm_name=None)
    model = model.to(device)
    if args.groupnorm > 0:
        print(f"✅ Using GroupNorm({args.groupnorm}) via norm='GROUP'")

    # заморозити на кілька епох (як «feature extractor»)
    freeze_backbone(model, True)
    # але останній класифікаційний шар зробимо trainable
    if hasattr(model, "class_layers"):  # DenseNet у MONAI має class_layers
        for p in model.class_layers.parameters():
            p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = -1.0
    best_thr = 0.5
    stale = 0

    def run_epoch(loader, train=True, accum_steps=args.accum_steps):
        if train:
            model.train()
        else:
            model.eval()

        total_loss, total_n = 0.0, 0
        all_y, all_p, all_sid = [], [], []

        for step, batch in enumerate(tqdm(loader, leave=False)):
            x = batch["img"].to(device, non_blocking=False)
            y = torch.tensor(batch["y"], dtype=torch.float32, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type != "cpu" else torch.bfloat16, enabled=(device.type != "cpu")):
                logits = model(x).squeeze(1)  # (B,)
                loss   = loss_fn(logits, y)
                prob   = torch.sigmoid(logits).detach().cpu().numpy()

            if train:
                (loss/accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            total_loss += loss.item()*x.size(0)
            total_n    += x.size(0)
            all_y.extend(y.detach().cpu().numpy().tolist())
            all_p.extend(prob.tolist())
            all_sid.extend(list(batch["sid"]))

        mets = compute_metrics(all_y, all_p, thr=0.5)
        return total_loss/max(1,total_n), np.array(all_y), np.array(all_p), np.array(all_sid), mets

    for epoch in range(1, args.epochs+1):
        # анфріз після freeze_epochs
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone")
            freeze_backbone(model, False)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * args.unfreeze_lr_mult, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

        tr_loss, _, _, _, tr_m = run_epoch(train_dl, train=True)
        with torch.no_grad():
            va_loss, yv, pv, sidv, va_m = run_epoch(val_dl, train=False)

        # підбір порога
        thrs = np.linspace(0.05, 0.95, 19)
        f1s = [compute_metrics(yv, pv, thr=float(t))["f1"] for t in thrs]
        cur_thr = float(thrs[int(np.argmax(f1s))])
        va_pat = compute_metrics(yv, pv, thr=cur_thr)  # тут volume==patient

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
              f"AUC={va_m['roc_auc']:.3f} F1={va_m['f1']:.3f} BA={va_m['bal_acc']:.3f} | "
              f"patient@thr={cur_thr:.2f}: AUC={va_pat['roc_auc']:.3f} F1={va_pat['f1']:.3f} BA={va_pat['bal_acc']:.3f} | "
              f"lr={sched.get_last_lr()[0]:.2e}")

        # save last
        torch.save(model.state_dict(), str(out_dir / "checkpoints" / "last_3d_monai.pt"))

        score = va_pat["roc_auc"]
        if math.isnan(score):
            score = va_pat["bal_acc"]

        if score > best_score:
            best_score = score
            best_thr = cur_thr
            stale = 0
            torch.save(model.state_dict(), str(out_dir / "checkpoints" / "best_3d_monai.pt"))
            with open(out_dir / "artifacts" / "thresholds_3d_monai.json","w") as f:
                json.dump({"metric":"f1_patient", "thr":best_thr}, f, indent=2)
            print(f"  ↳ ✅ saved best: best_3d_monai.pt (score={best_score:.3f}, thr={best_thr:.2f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"⏹ Early stopping: {args.patience} epochs w/o improvement.")
                break

        sched.step()

    print(f"Best patient-level score: {best_score:.3f} at thr={best_thr:.2f}")

    # Тест
    with torch.no_grad():
        _, yt, pt, sidt, sl_m = run_epoch(test_dl, train=False)
        te_pat = compute_metrics(yt, pt, thr=best_thr)
    print("TEST volume-level:", sl_m)
    print("TEST patient-level @best_thr:", te_pat)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", type=str, required=True, help="subject-level CSV (path/nifti_path, label, split, subject_id)")
    ap.add_argument("--out_dir", type=str, default=".", help="де зберігати checkpoints/artifacts")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--depth", type=int, default=64)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--freeze_epochs", type=int, default=6)
    ap.add_argument("--unfreeze_lr_mult", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--accum_steps", type=int, default=2)
    ap.add_argument("--groupnorm", type=int, default=0, help=">0 => DenseNet з GROUP norm")

    args = ap.parse_args()
    main(args)