# src/eval/cross_validation.py
"""
Cross-Validation Script for Dementia MRI Classification
Uses EfficientNet-B2 (pretrained on ImageNet) for better performance
Performs k-fold cross-validation and reports comprehensive metrics
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_.dataset import Alzheimer2DDataset

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_efficientnet_b2(num_classes: int = 1):
    """Build EfficientNet-B2 with ImageNet pretrained weights"""
    try:
        import torchvision
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    except Exception as e:
        print(f"Warning: Could not load EfficientNet-B2, falling back to EfficientNet-B0: {e}")
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model


def freeze_backbone(model: nn.Module, freeze: bool = True):
    """Freeze/unfreeze backbone parameters"""
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = not freeze


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer=None,
    amp: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_samples = 0
    ys, ps = [], []

    scaler = torch.cuda.amp.GradScaler() if amp else None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        if is_train:
            optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x).squeeze(1)  # (B,)
            loss = loss_fn(logits, y.float())

        if is_train:
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        n_samples += batch_size

        with torch.no_grad():
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.cpu().numpy())
            ps.append(prob)

    avg_loss = total_loss / max(1, n_samples)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    return avg_loss, y_true, y_prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive metrics"""
    y_pred = (y_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true_int, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_int, y_pred))
    metrics["f1"] = float(f1_score(y_true_int, y_pred, average="binary"))
    metrics["precision"] = float(precision_score(y_true_int, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true_int, y_pred, zero_division=0))

    # AUC-ROC
    if len(np.unique(y_true_int)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_int, y_prob))
    else:
        metrics["roc_auc"] = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_true_int, y_pred)
    if cm.shape == (2, 2):
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])
        metrics["sensitivity"] = float(metrics["tp"] / (metrics["tp"] + metrics["fn"] + 1e-8))
        metrics["specificity"] = float(metrics["tn"] / (metrics["tn"] + metrics["fp"] + 1e-8))
    else:
        metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = 0
        metrics["sensitivity"] = metrics["specificity"] = 0.0

    return metrics


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Find optimal threshold for given metric"""
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_thr = 0.5
    best_val = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn + 1e-8)
                spec = tn / (tn + fp + 1e-8)
                val = sens + spec - 1
            else:
                val = 0.0
        else:
            val = balanced_accuracy_score(y_true, y_pred)

        if val > best_val:
            best_val = val
            best_thr = thr

    return best_thr


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold: int,
    args,
    device: torch.device,
) -> Dict:
    """Train and evaluate one fold"""
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/{args.n_folds}")
    print(f"{'='*60}")

    # Datasets
    train_ds = Alzheimer2DDataset(train_df, augment=True, image_size=args.size)
    val_ds = Alzheimer2DDataset(val_df, augment=False, image_size=args.size)

    # DataLoaders
    if args.balanced:
        labels = train_df["label"].map({"non_demented": 0, "dementia": 1}).values
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_dl = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=True
        )
    else:
        train_dl = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )

    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_efficientnet_b2(num_classes=1).to(device)
    freeze_backbone(model, freeze=True)

    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # Training loop
    best_score = -1.0
    best_metrics = None
    best_y_true = None
    best_y_prob = None

    for epoch in range(1, args.epochs + 1):
        # Unfreeze after freeze_epochs
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone")
            freeze_backbone(model, freeze=False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * args.unfreeze_lr_mult, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1, eta_min=args.min_lr
            )

        train_loss, _, _ = run_epoch(model, train_dl, device, loss_fn, optimizer=optimizer, amp=args.amp)

        with torch.no_grad():
            val_loss, y_true, y_prob = run_epoch(model, val_dl, device, loss_fn, optimizer=None, amp=False)

        metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
        score = metrics["roc_auc"] if not np.isnan(metrics["roc_auc"]) else metrics["balanced_accuracy"]

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"AUC={metrics['roc_auc']:.3f} | F1={metrics['f1']:.3f} | "
                f"BA={metrics['balanced_accuracy']:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if score > best_score:
            best_score = score
            best_metrics = metrics.copy()
            best_y_true = y_true.copy()
            best_y_prob = y_prob.copy()

        scheduler.step()

    # Find optimal threshold
    best_thr = find_best_threshold(best_y_true, best_y_prob, metric="f1")
    best_metrics_thr = compute_metrics(best_y_true, best_y_prob, threshold=best_thr)
    best_metrics_thr["optimal_threshold"] = best_thr

    print(f"\n✅ Fold {fold + 1} Best Results:")
    print(f"   AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   F1: {best_metrics['f1']:.4f}")
    print(f"   Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"   Optimal Threshold: {best_thr:.3f} (F1: {best_metrics_thr['f1']:.4f})")

    return {
        "fold": fold + 1,
        **best_metrics,
        **{f"{k}_opt": v for k, v in best_metrics_thr.items()},
    }


def main(args):
    set_seed(args.seed)

    # Load data
    index_path = Path(args.index_csv)
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    df = pd.read_csv(index_path)
    print(f"📊 Loaded {len(df)} samples from {index_path}")

    # Check required columns
    required_cols = ["path", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter valid labels
    valid_labels = {"non_demented", "dementia"}
    df = df[df["label"].isin(valid_labels)].copy()
    print(f"📊 After filtering: {len(df)} samples")
    print(f"   Label distribution:\n{df['label'].value_counts()}")

    if len(df) < args.n_folds:
        raise ValueError(f"Not enough samples ({len(df)}) for {args.n_folds}-fold CV")

    # Prepare labels for stratification
    labels = df["label"].map({"non_demented": 0, "dementia": 1}).values

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"🖥️  Using device: {device}")

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        result = train_fold(train_df, val_df, fold, args, device)
        fold_results.append(result)

    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    
    # Summary statistics
    summary = {
        "n_folds": args.n_folds,
        "n_samples": len(df),
        "model": "EfficientNet-B2",
        "metrics": {}
    }

    metric_cols = ["roc_auc", "f1", "balanced_accuracy", "accuracy", "precision", "recall", 
                   "sensitivity", "specificity", "f1_opt", "balanced_accuracy_opt"]
    
    for col in metric_cols:
        if col in results_df.columns:
            values = results_df[col].dropna()
            if len(values) > 0:
                summary["metrics"][col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }

    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: EfficientNet-B2 (pretrained on ImageNet)")
    print(f"Folds: {args.n_folds}")
    print(f"Total samples: {len(df)}")
    print(f"\nMetrics (mean ± std across folds):")
    
    for metric in ["roc_auc", "f1", "balanced_accuracy", "accuracy", "precision", "recall"]:
        if metric in summary["metrics"]:
            m = summary["metrics"][metric]
            print(f"  {metric.upper():20s}: {m['mean']:.4f} ± {m['std']:.4f} (range: {m['min']:.4f}-{m['max']:.4f})")

    print(f"\n{'='*60}")
    print("Per-fold results:")
    print(results_df.to_string(index=False))
    print(f"{'='*60}")

    # Save results
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(output_dir / "cv_results.csv", index=False)
    print(f"\n💾 Saved per-fold results to: {output_dir / 'cv_results.csv'}")

    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved summary to: {output_dir / 'cv_summary.json'}")

    return summary, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Validation for Dementia MRI Classification")
    
    # Data
    parser.add_argument("--index_csv", type=str, default="data/index.csv",
                       help="Path to index CSV with columns: path, label")
    
    # CV
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    
    # Model
    parser.add_argument("--size", type=int, default=224, help="Input image size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per fold")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--freeze_epochs", type=int, default=2, help="Epochs to freeze backbone")
    parser.add_argument("--unfreeze_lr_mult", type=float, default=0.3, help="LR multiplier after unfreezing")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--balanced", action="store_true", help="Use weighted sampling")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    
    # System
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (even if CUDA available)")
    
    args = parser.parse_args()
    main(args)

