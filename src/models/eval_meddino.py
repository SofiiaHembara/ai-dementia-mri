# src/models/eval_meddino.py
# -*- coding: utf-8 -*-
"""
Evaluation script for trained MEDdino models.

Features:
- Load trained checkpoint
- Evaluate on test set
- Generate ROC curves (slice-level and patient-level)
- Export predictions to CSV
- Compute comprehensive metrics
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, balanced_accuracy_score, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_.meddino_dataset import MEDdinoSliceDataset
from src.models.train_meddino import MEDdinoClassifier, get_device


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def predict(model, loader, device):
    """
    Run inference on a dataloader.
    
    Returns:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        subject_ids: Subject IDs
    """
    model.eval()
    
    all_y_true = []
    all_y_prob = []
    all_subject_ids = []
    
    for x, y, subject_ids in tqdm(loader, desc="Inference"):
        x = x.to(device)
        
        logits = model(x)
        # Convert to float32 for MPS compatibility
        probs = torch.sigmoid(logits).cpu().float().numpy()
        
        all_y_true.extend(y.float().numpy())
        all_y_prob.extend(probs)
        all_subject_ids.extend(subject_ids)
    
    return np.array(all_y_true), np.array(all_y_prob), all_subject_ids


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics."""
    # Ensure float32 for MPS compatibility
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold)
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    })
    
    return metrics


def aggregate_to_patient_level(y_true_slice, y_prob_slice, subject_ids, aggregation="mean"):
    """
    Aggregate slice predictions to patient level.
    
    Args:
        y_true_slice: Slice-level labels
        y_prob_slice: Slice-level predictions
        subject_ids: Subject IDs
        aggregation: "mean", "median", or "max"
        
    Returns:
        patient_df: DataFrame with patient-level predictions
    """
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "y_true": y_true_slice,
        "y_prob": y_prob_slice
    })
    
    if aggregation == "mean":
        patient_df = df.groupby("subject_id").agg(
            y_true=("y_true", "max"),  # Patient has dementia if any slice is positive
            y_prob=("y_prob", "mean")
        ).reset_index()
    elif aggregation == "median":
        patient_df = df.groupby("subject_id").agg(
            y_true=("y_true", "max"),
            y_prob=("y_prob", "median")
        ).reset_index()
    elif aggregation == "max":
        patient_df = df.groupby("subject_id").agg(
            y_true=("y_true", "max"),
            y_prob=("y_prob", "max")
        ).reset_index()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return patient_df


def plot_roc_curve(y_true, y_prob, title, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved ROC curve: {save_path}")


def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """Find optimal threshold by maximizing a metric."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = []
    
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return float(thresholds[best_idx]), float(scores[best_idx])


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_meddino(args):
    """Main evaluation pipeline."""
    
    device = get_device()
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.index_csv)
    
    if args.split == "test":
        eval_df = df[df["split"] == "test"].reset_index(drop=True)
    elif args.split == "val":
        eval_df = df[df["split"] == "val"].reset_index(drop=True)
    elif args.split == "all":
        eval_df = df.reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split: {args.split}")
    
    print(f"Evaluating on {args.split} set: {len(eval_df)} slices")
    print(f"Unique subjects: {eval_df['subject_id'].nunique()}")
    
    # Create dataset and loader
    eval_ds = MEDdinoSliceDataset(
        eval_df, size=args.img_size, train=False,
        use_clahe=args.use_clahe, normalize_mode=args.normalize_mode
    )
    
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    
    # Load model
    print("\nLoading model...")
    model = MEDdinoClassifier(
        backbone_name=args.backbone,
        img_size=args.img_size,
        dropout=0.0,  # No dropout during inference
        head_type=args.head_type
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✓ Loaded checkpoint: {args.checkpoint}")
    
    # Run inference
    print("\nRunning inference...")
    y_true, y_prob, subject_ids = predict(model, eval_loader, device)
    
    # Load or compute threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold:.3f}")
    elif args.threshold_json is not None:
        with open(args.threshold_json, "r") as f:
            threshold = json.load(f)["threshold"]
        print(f"Loaded threshold from {args.threshold_json}: {threshold:.3f}")
    else:
        threshold, _ = find_optimal_threshold(y_true, y_prob, metric=args.optimize_metric)
        print(f"Computed optimal threshold: {threshold:.3f}")
    
    # Slice-level evaluation
    print("\n" + "="*60)
    print("SLICE-LEVEL EVALUATION")
    print("="*60)
    
    slice_metrics = compute_metrics(y_true, y_prob, threshold)
    
    for k, v in slice_metrics.items():
        print(f"  {k}: {v}")
    
    # Patient-level evaluation
    print("\n" + "="*60)
    print(f"PATIENT-LEVEL EVALUATION (aggregation: {args.patient_aggregation})")
    print("="*60)
    
    patient_df = aggregate_to_patient_level(
        y_true, y_prob, subject_ids, aggregation=args.patient_aggregation
    )
    
    patient_metrics = compute_metrics(
        patient_df["y_true"].values,
        patient_df["y_prob"].values,
        threshold
    )
    
    for k, v in patient_metrics.items():
        print(f"  {k}: {v}")
    
    # Plot ROC curves
    print("\nGenerating ROC curves...")
    
    plot_roc_curve(
        y_true, y_prob,
        title=f"ROC Curve - Slice Level ({args.split} set)",
        save_path=output_dir / f"roc_slice_{args.split}.png"
    )
    
    plot_roc_curve(
        patient_df["y_true"].values,
        patient_df["y_prob"].values,
        title=f"ROC Curve - Patient Level ({args.split} set)",
        save_path=output_dir / f"roc_patient_{args.split}.png"
    )
    
    # Save predictions
    print("\nSaving predictions...")
    
    # Slice-level predictions
    slice_preds_df = pd.DataFrame({
        "subject_id": subject_ids,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (y_prob >= threshold).astype(int)
    })
    slice_preds_df.to_csv(output_dir / f"predictions_slice_{args.split}.csv", index=False)
    print(f"  ✓ Saved: predictions_slice_{args.split}.csv")
    
    # Patient-level predictions
    patient_df["y_pred"] = (patient_df["y_prob"] >= threshold).astype(int)
    patient_df.to_csv(output_dir / f"predictions_patient_{args.split}.csv", index=False)
    print(f"  ✓ Saved: predictions_patient_{args.split}.csv")
    
    # Save metrics
    results = {
        "split": args.split,
        "threshold": threshold,
        "slice_metrics": slice_metrics,
        "patient_metrics": patient_metrics,
        "n_slices": int(len(y_true)),
        "n_patients": int(len(patient_df)),
        "aggregation": args.patient_aggregation
    }
    
    with open(output_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MEDdino model")
    
    # Data
    parser.add_argument("--index_csv", type=str, required=True, help="Path to index CSV")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "all"])
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224.dino")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    
    # Threshold
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (if None, computed)")
    parser.add_argument("--threshold_json", type=str, default=None, help="Path to threshold JSON")
    parser.add_argument("--optimize_metric", type=str, default="f1", choices=["f1", "balanced_accuracy"])
    
    # Preprocessing
    parser.add_argument("--use_clahe", action="store_true")
    parser.add_argument("--normalize_mode", type=str, default="zscore", choices=["zscore", "minmax", "imagenet"])
    
    # Patient aggregation
    parser.add_argument("--patient_aggregation", type=str, default="mean", choices=["mean", "median", "max"])
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./artifacts/meddino")
    
    # DataLoader
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_meddino(args)

