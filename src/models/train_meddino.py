# src/models/train_meddino.py
# -*- coding: utf-8 -*-
"""
MEDdino fine-tuning pipeline for dementia classification on OASIS MRI slices.

MEDdino is a medical imaging foundation model (ViT-based) pre-trained on diverse medical data.
This script:
- Loads MEDdino pre-trained weights (or falls back to DINO/DINOv2)
- Fine-tunes with progressive unfreezing
- Tracks both slice-level and patient-level metrics
- Supports early stopping and checkpointing
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_.meddino_dataset import MEDdinoSliceDataset
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Auto-select device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Model Building
# ============================================================================

class MEDdinoClassifier(nn.Module):
    """
    MEDdino-based binary classifier.
    
    Architecture:
        - Backbone: ViT encoder (MEDdino/DINO/DINOv2)
        - Head: Simple linear layer or small MLP
    """
    
    def __init__(self, backbone_name: str, img_size: int = 224, dropout: float = 0.1, head_type: str = "linear"):
        """
        Args:
            backbone_name: timm model name, e.g., 
                - "vit_base_patch16_224.dino" (DINO ViT-B/16)
                - "vit_small_patch14_dinov2.lvd142m" (DINOv2 ViT-S/14)
                - "vit_large_patch14_dinov2.lvd142m" (DINOv2 ViT-L/14)
            img_size: Input image size
            dropout: Dropout rate for head
            head_type: "linear" or "mlp"
        """
        super().__init__()
        
        import timm
        
        # Load backbone (feature extractor)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=img_size,
            dynamic_img_size=False  # Critical for MPS compatibility
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Build classification head
        if head_type == "linear":
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim, 1)
            )
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(256, 1)
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
            
        Returns:
            logits: (B,) - raw logits (before sigmoid)
        """
        features = self.backbone(x)  # (B, D)
        logits = self.head(features).squeeze(-1)  # (B,)
        return logits
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_last_n_blocks(self, n: int):
        """
        Unfreeze the last n transformer blocks (for ViT).
        
        Args:
            n: Number of blocks to unfreeze (from the end)
        """
        # First freeze everything
        self.freeze_backbone(freeze=True)
        
        # Try to access ViT blocks
        if hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {n} blocks of backbone")
        else:
            print("Warning: Could not find 'blocks' attribute in backbone. Full backbone remains frozen.")


def build_meddino_model(
    backbone_name: str = "vit_base_patch16_224.dino",
    img_size: int = 224,
    dropout: float = 0.1,
    head_type: str = "linear",
    device: torch.device = None
):
    """
    Build MEDdino model.
    
    Args:
        backbone_name: timm model identifier
        img_size: Image size
        dropout: Dropout for head
        head_type: "linear" or "mlp"
        device: Device to move model to
        
    Returns:
        model: MEDdinoClassifier
    """
    if device is None:
        device = get_device()
    
    model = MEDdinoClassifier(
        backbone_name=backbone_name,
        img_size=img_size,
        dropout=dropout,
        head_type=head_type
    )
    
    model = model.to(device)
    
    print(f"Built MEDdino model: {backbone_name}")
    print(f"  Feature dim: {model.feature_dim}")
    print(f"  Head type: {head_type}")
    print(f"  Device: {device.type}")
    
    return model


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy and ensure float32 (MPS compatibility)
    y_true = np.asarray(y_true, dtype=np.float32).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    
    # ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")
    
    # Classification metrics
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    return metrics


def compute_patient_metrics(y_true_slice, y_prob_slice, subject_ids, threshold=0.5, aggregation="mean"):
    """
    Aggregate slice predictions to patient level.
    
    Args:
        y_true_slice: Slice-level ground truth
        y_prob_slice: Slice-level predictions
        subject_ids: Subject IDs for each slice
        threshold: Decision threshold
        aggregation: "mean", "median", or "max"
        
    Returns:
        Patient-level metrics
    """
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "y_true": y_true_slice,
        "y_prob": y_prob_slice
    })
    
    # Aggregate to patient level
    if aggregation == "mean":
        patient_df = df.groupby("subject_id").agg(
            y_true=("y_true", "max"),  # Any slice with label 1 → patient has dementia
            y_prob=("y_prob", "mean")  # Mean probability across slices
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
    
    # Compute metrics on patient level
    return compute_metrics(patient_df["y_true"].values, patient_df["y_prob"].values, threshold)


def find_best_threshold(y_true, y_prob, subject_ids=None, metric="f1", aggregation="mean"):
    """
    Find optimal threshold by grid search.
    
    Args:
        y_true: Ground truth
        y_prob: Predictions
        subject_ids: If provided, optimize at patient level
        metric: Metric to optimize ("f1", "balanced_accuracy")
        aggregation: Patient-level aggregation method
        
    Returns:
        best_threshold, best_score
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = []
    
    for thr in thresholds:
        if subject_ids is not None:
            m = compute_patient_metrics(y_true, y_prob, subject_ids, threshold=thr, aggregation=aggregation)
        else:
            m = compute_metrics(y_true, y_prob, threshold=thr)
        scores.append(m[metric])
    
    best_idx = np.argmax(scores)
    return float(thresholds[best_idx]), float(scores[best_idx])


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accum_steps: int = 1,
    use_amp: bool = False
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_y_true = []
    all_y_prob = []
    all_subject_ids = []
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for step, (x, y, subject_ids) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        
        # Forward
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
        
        # Backward (with gradient accumulation)
        loss = loss / accum_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item() * accum_steps * x.size(0)
        
        # Convert to float32 for MPS compatibility
        probs = torch.sigmoid(logits).detach().cpu().float().numpy()
        all_y_true.extend(y.cpu().float().numpy())
        all_y_prob.extend(probs)
        all_subject_ids.extend(subject_ids)
        
        pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})
    
    # Final step if needed
    if len(loader) % accum_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, np.array(all_y_true), np.array(all_y_prob), all_subject_ids


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    all_y_true = []
    all_y_prob = []
    all_subject_ids = []
    
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for x, y, subject_ids in pbar:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * x.size(0)
        
        # Convert to float32 for MPS compatibility
        probs = torch.sigmoid(logits).cpu().float().numpy()
        all_y_true.extend(y.cpu().float().numpy())
        all_y_prob.extend(probs)
        all_subject_ids.extend(subject_ids)
    
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, np.array(all_y_true), np.array(all_y_prob), all_subject_ids


# ============================================================================
# Main Training Function
# ============================================================================

def train_meddino(args):
    """Main training pipeline."""
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.index_csv)
    
    # Filter existing files
    df = df[df["slice_path"].apply(os.path.exists)].reset_index(drop=True)
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} slices")
    print(f"Val: {len(val_df)} slices")
    print(f"Test: {len(test_df)} slices")
    
    # Check for patient-level splits
    train_subjects = set(train_df["subject_id"])
    val_subjects = set(val_df["subject_id"])
    test_subjects = set(test_df["subject_id"])
    
    assert len(train_subjects & val_subjects) == 0, "Patient leakage: train-val overlap!"
    assert len(train_subjects & test_subjects) == 0, "Patient leakage: train-test overlap!"
    assert len(val_subjects & test_subjects) == 0, "Patient leakage: val-test overlap!"
    print("✓ No patient leakage across splits")
    
    # Create datasets
    train_ds = MEDdinoSliceDataset(
        train_df, size=args.img_size, train=True,
        use_clahe=args.use_clahe, normalize_mode=args.normalize_mode
    )
    val_ds = MEDdinoSliceDataset(
        val_df, size=args.img_size, train=False,
        use_clahe=args.use_clahe, normalize_mode=args.normalize_mode
    )
    test_ds = MEDdinoSliceDataset(
        test_df, size=args.img_size, train=False,
        use_clahe=args.use_clahe, normalize_mode=args.normalize_mode
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_meddino_model(
        backbone_name=args.backbone,
        img_size=args.img_size,
        dropout=args.dropout,
        head_type=args.head_type,
        device=device
    )
    
    # Loss function (with class balancing)
    pos_count = (train_df["label"].str.lower() == "dementia").sum()
    neg_count = (train_df["label"].str.lower() == "non_demented").sum()
    # Use float32 for MPS compatibility
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    
    print(f"\nClass balance: neg={neg_count}, pos={pos_count}, pos_weight={pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Phase 1: Freeze backbone, train head only
    print("\n" + "="*60)
    print("PHASE 1: Training head with frozen backbone")
    print("="*60)
    
    model.freeze_backbone(freeze=True)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    
    best_val_score = -float("inf")
    best_threshold = 0.5
    patience_counter = 0
    
    for epoch in range(1, args.freeze_epochs + 1):
        print(f"\nEpoch {epoch}/{args.freeze_epochs}")
        
        # Train
        train_loss, train_y, train_p, train_sids = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            accum_steps=args.accum_steps, use_amp=(device.type == "cuda")
        )
        
        # Validate
        val_loss, val_y, val_p, val_sids = evaluate(model, val_loader, criterion, device)
        
        # Find best threshold on validation set
        thr, _ = find_best_threshold(
            val_y, val_p, val_sids,
            metric=args.optimize_metric,
            aggregation=args.patient_aggregation
        )
        
        # Compute metrics
        train_metrics_slice = compute_metrics(train_y, train_p, threshold=thr)
        val_metrics_slice = compute_metrics(val_y, val_p, threshold=thr)
        
        train_metrics_patient = compute_patient_metrics(
            train_y, train_p, train_sids, threshold=thr, aggregation=args.patient_aggregation
        )
        val_metrics_patient = compute_patient_metrics(
            val_y, val_p, val_sids, threshold=thr, aggregation=args.patient_aggregation
        )
        
        # Monitor score
        if args.monitor == "patient_auc":
            val_score = val_metrics_patient["roc_auc"]
        elif args.monitor == "patient_f1":
            val_score = val_metrics_patient["f1"]
        elif args.monitor == "slice_auc":
            val_score = val_metrics_slice["roc_auc"]
        else:
            val_score = -val_loss
        
        # Logging
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Slice  → Val AUC: {val_metrics_slice['roc_auc']:.3f}, F1: {val_metrics_slice['f1']:.3f}, Bal Acc: {val_metrics_slice['balanced_accuracy']:.3f}")
        print(f"  Patient→ Val AUC: {val_metrics_patient['roc_auc']:.3f}, F1: {val_metrics_patient['f1']:.3f}, Bal Acc: {val_metrics_patient['balanced_accuracy']:.3f}")
        print(f"  Threshold: {thr:.3f}")
        
        # Save best model
        if val_score > best_val_score + args.min_delta:
            best_val_score = val_score
            best_threshold = thr
            patience_counter = 0
            
            torch.save(model.state_dict(), output_dir / "best_meddino.pt")
            
            with open(output_dir / "best_threshold.json", "w") as f:
                json.dump({"threshold": best_threshold, "metric": args.monitor}, f, indent=2)
            
            print(f"  ✓ Saved best model (score: {best_val_score:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹ Early stopping: no improvement for {args.patience} epochs")
            break
    
    # Phase 2: Unfreeze top blocks, fine-tune
    if args.unfreeze_blocks > 0:
        print("\n" + "="*60)
        print(f"PHASE 2: Fine-tuning with last {args.unfreeze_blocks} blocks unfrozen")
        print("="*60)
        
        # Load best from phase 1
        model.load_state_dict(torch.load(output_dir / "best_meddino.pt", map_location=device))
        
        # Unfreeze last n blocks
        model.unfreeze_last_n_blocks(args.unfreeze_blocks)
        
        # New optimizer with lower LR for backbone
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" in n]
        
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_finetune},
            {"params": head_params, "lr": args.lr_head}
        ], weight_decay=args.weight_decay)
        
        best_val_score = -float("inf")
        patience_counter = 0
        
        for epoch in range(1, args.finetune_epochs + 1):
            print(f"\nEpoch {epoch}/{args.finetune_epochs}")
            
            train_loss, train_y, train_p, train_sids = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                accum_steps=args.accum_steps, use_amp=(device.type == "cuda")
            )
            
            val_loss, val_y, val_p, val_sids = evaluate(model, val_loader, criterion, device)
            
            thr, _ = find_best_threshold(
                val_y, val_p, val_sids,
                metric=args.optimize_metric,
                aggregation=args.patient_aggregation
            )
            
            val_metrics_slice = compute_metrics(val_y, val_p, threshold=thr)
            val_metrics_patient = compute_patient_metrics(
                val_y, val_p, val_sids, threshold=thr, aggregation=args.patient_aggregation
            )
            
            if args.monitor == "patient_auc":
                val_score = val_metrics_patient["roc_auc"]
            elif args.monitor == "patient_f1":
                val_score = val_metrics_patient["f1"]
            elif args.monitor == "slice_auc":
                val_score = val_metrics_slice["roc_auc"]
            else:
                val_score = -val_loss
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Slice  → Val AUC: {val_metrics_slice['roc_auc']:.3f}, F1: {val_metrics_slice['f1']:.3f}")
            print(f"  Patient→ Val AUC: {val_metrics_patient['roc_auc']:.3f}, F1: {val_metrics_patient['f1']:.3f}")
            
            if val_score > best_val_score + args.min_delta:
                best_val_score = val_score
                best_threshold = thr
                patience_counter = 0
                
                torch.save(model.state_dict(), output_dir / "best_meddino.pt")
                
                with open(output_dir / "best_threshold.json", "w") as f:
                    json.dump({"threshold": best_threshold, "metric": args.monitor}, f, indent=2)
                
                print(f"  ✓ Saved best model (score: {best_val_score:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping: no improvement for {args.patience} epochs")
                break
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    model.load_state_dict(torch.load(output_dir / "best_meddino.pt", map_location=device))
    
    test_loss, test_y, test_p, test_sids = evaluate(model, test_loader, criterion, device)
    
    test_metrics_slice = compute_metrics(test_y, test_p, threshold=best_threshold)
    test_metrics_patient = compute_patient_metrics(
        test_y, test_p, test_sids, threshold=best_threshold, aggregation=args.patient_aggregation
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Threshold: {best_threshold:.3f}")
    print("\nSlice-level metrics:")
    for k, v in test_metrics_slice.items():
        print(f"  {k}: {v:.4f}")
    print("\nPatient-level metrics:")
    for k, v in test_metrics_patient.items():
        print(f"  {k}: {v:.4f}")
    
    # Save test results
    test_results = {
        "test_loss": test_loss,
        "threshold": best_threshold,
        "slice_metrics": test_metrics_slice,
        "patient_metrics": test_metrics_patient
    }
    
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to {output_dir}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train MEDdino for dementia classification")
    
    # Data
    parser.add_argument("--index_csv", type=str, required=True, help="Path to index CSV")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/meddino", help="Output directory")
    
    # Model
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224.dino",
                        help="timm model name (e.g., vit_base_patch16_224.dino)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    
    # Training - Phase 1 (frozen backbone)
    parser.add_argument("--freeze_epochs", type=int, default=20, help="Epochs with frozen backbone")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for head")
    
    # Training - Phase 2 (partial unfreeze)
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Epochs for fine-tuning")
    parser.add_argument("--unfreeze_blocks", type=int, default=4, help="Number of top blocks to unfreeze (0 = skip phase 2)")
    parser.add_argument("--lr_finetune", type=float, default=1e-5, help="Learning rate for unfrozen backbone")
    
    # Optimization
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement")
    parser.add_argument("--monitor", type=str, default="patient_auc",
                        choices=["patient_auc", "patient_f1", "slice_auc", "loss"])
    parser.add_argument("--optimize_metric", type=str, default="f1", choices=["f1", "balanced_accuracy"])
    
    # Preprocessing
    parser.add_argument("--use_clahe", action="store_true", help="Use CLAHE preprocessing")
    parser.add_argument("--normalize_mode", type=str, default="zscore", choices=["zscore", "minmax", "imagenet"])
    
    # Patient-level aggregation
    parser.add_argument("--patient_aggregation", type=str, default="mean", choices=["mean", "median", "max"])
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_meddino(args)

