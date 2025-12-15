#!/usr/bin/env python3
# examples/meddino_quickstart.py
"""
Quick start example for MEDdino pipeline.

This script demonstrates:
1. Loading a trained MEDdino model
2. Making predictions on single images
3. Making patient-level predictions
4. Saving results

Usage:
    python examples/meddino_quickstart.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.models.infer_meddino import MEDdinoPredictor


def example_single_image():
    """Example 1: Predict on a single image."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = MEDdinoPredictor(
        checkpoint_path="checkpoints/meddino/best_meddino.pt",
        backbone_name="vit_base_patch16_224.dino",
        img_size=224,
        threshold=0.5,
        use_clahe=True,
        normalize_mode="zscore"
    )
    
    # Predict on a single image
    image_path = "data/interim/oasis1_2d/dementia/OAS1_0001_MR1_slice_080.png"
    
    if not Path(image_path).exists():
        print(f"⚠ Image not found: {image_path}")
        print("Please update the path to a valid MRI slice image.")
        return
    
    prob = predictor.predict_image(image_path)
    label = predictor.classify(prob)
    
    print(f"\nImage: {image_path}")
    print(f"Probability of dementia: {prob:.4f}")
    print(f"Predicted label: {label}")
    print(f"Confidence: {max(prob, 1-prob):.4f}")


def example_patient_prediction():
    """Example 2: Predict on a patient (multiple slices)."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Patient-Level Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = MEDdinoPredictor(
        checkpoint_path="checkpoints/meddino/best_meddino.pt",
        backbone_name="vit_base_patch16_224.dino",
        threshold=0.5,
        use_clahe=True
    )
    
    # Example: Get all slices for a patient
    patient_dir = Path("data/interim/oasis1_2d/dementia")
    
    if not patient_dir.exists():
        print(f"⚠ Directory not found: {patient_dir}")
        print("Please update the path to a valid directory with MRI slices.")
        return
    
    # Find all PNG images (slices) for this patient
    slice_paths = sorted(list(patient_dir.glob("OAS1_0001_MR1_slice_*.png")))[:10]  # First 10 slices
    
    if not slice_paths:
        print(f"⚠ No slices found in {patient_dir}")
        return
    
    print(f"\nPatient: OAS1_0001_MR1")
    print(f"Number of slices: {len(slice_paths)}")
    
    # Predict with different aggregation methods
    for agg in ["mean", "median", "max"]:
        patient_prob, slice_probs = predictor.predict_patient(
            slice_paths,
            aggregation=agg,
            return_slice_probs=True
        )
        
        label = predictor.classify(patient_prob)
        
        print(f"\nAggregation: {agg}")
        print(f"  Patient probability: {patient_prob:.4f}")
        print(f"  Predicted label: {label}")
        print(f"  Slice prob range: [{slice_probs.min():.3f}, {slice_probs.max():.3f}]")
        print(f"  Slice prob std: {slice_probs.std():.3f}")


def example_batch_processing():
    """Example 3: Batch processing multiple images."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    # Initialize predictor
    predictor = MEDdinoPredictor(
        checkpoint_path="checkpoints/meddino/best_meddino.pt",
        backbone_name="vit_base_patch16_224.dino",
        threshold=0.5
    )
    
    # Get a batch of images
    data_dir = Path("data/interim/oasis1_2d/dementia")
    
    if not data_dir.exists():
        print(f"⚠ Directory not found: {data_dir}")
        return
    
    image_paths = list(data_dir.glob("*.png"))[:20]  # First 20 images
    
    if not image_paths:
        print(f"⚠ No images found in {data_dir}")
        return
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    # Batch prediction (much faster than loop)
    probs = predictor.predict_batch(image_paths, batch_size=8)
    labels = [predictor.classify(p) for p in probs]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "image": [p.name for p in image_paths],
        "probability": probs,
        "prediction": labels
    })
    
    print("\nResults:")
    print(results_df.head(10))
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Mean probability: {probs.mean():.3f}")
    print(f"  Std probability: {probs.std():.3f}")
    print(f"  Predicted dementia: {sum(1 for l in labels if l == 'dementia')}/{len(labels)}")


def example_csv_processing():
    """Example 4: Process images from CSV index."""
    print("\n" + "="*60)
    print("EXAMPLE 4: CSV-Based Processing")
    print("="*60)
    
    # Initialize predictor
    predictor = MEDdinoPredictor(
        checkpoint_path="checkpoints/meddino/best_meddino.pt",
        backbone_name="vit_base_patch16_224.dino",
        threshold=0.5
    )
    
    # Load index CSV
    index_csv = "data/index_oasis1_2d.csv"
    
    if not Path(index_csv).exists():
        print(f"⚠ Index CSV not found: {index_csv}")
        return
    
    df = pd.read_csv(index_csv)
    
    # Filter to test set only
    test_df = df[df["split"] == "test"].copy()
    
    # Sample 50 slices for demo
    sample_df = test_df.sample(n=min(50, len(test_df)), random_state=42)
    
    print(f"\nProcessing {len(sample_df)} test images...")
    
    # Predict
    probs = predictor.predict_batch(sample_df["slice_path"].tolist(), batch_size=16)
    
    # Add predictions to dataframe
    sample_df["predicted_prob"] = probs
    sample_df["predicted_label"] = [predictor.classify(p) for p in probs]
    sample_df["correct"] = sample_df["label"].str.lower() == sample_df["predicted_label"]
    
    # Compute accuracy
    accuracy = sample_df["correct"].mean()
    
    print(f"\nSlice-level accuracy: {accuracy:.3f}")
    
    # Patient-level accuracy
    patient_results = []
    
    for subject_id, group in sample_df.groupby("subject_id"):
        true_label = group["label"].iloc[0].lower()
        slice_probs = group["predicted_prob"].values
        patient_prob = slice_probs.mean()
        pred_label = predictor.classify(patient_prob)
        
        patient_results.append({
            "subject_id": subject_id,
            "true_label": true_label,
            "patient_prob": patient_prob,
            "predicted_label": pred_label,
            "correct": true_label == pred_label
        })
    
    patient_df = pd.DataFrame(patient_results)
    patient_accuracy = patient_df["correct"].mean()
    
    print(f"Patient-level accuracy: {patient_accuracy:.3f}")
    print(f"\nSample results:")
    print(patient_df.head())


def example_comparison():
    """Example 5: Compare preprocessing options."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Preprocessing Comparison")
    print("="*60)
    
    image_path = "data/interim/oasis1_2d/dementia/OAS1_0001_MR1_slice_080.png"
    
    if not Path(image_path).exists():
        print(f"⚠ Image not found: {image_path}")
        return
    
    print(f"\nTesting different preprocessing options on: {image_path}\n")
    
    configs = [
        {"use_clahe": False, "normalize_mode": "zscore"},
        {"use_clahe": True, "normalize_mode": "zscore"},
        {"use_clahe": False, "normalize_mode": "minmax"},
        {"use_clahe": False, "normalize_mode": "imagenet"},
    ]
    
    for config in configs:
        predictor = MEDdinoPredictor(
            checkpoint_path="checkpoints/meddino/best_meddino.pt",
            backbone_name="vit_base_patch16_224.dino",
            **config
        )
        
        prob = predictor.predict_image(image_path)
        
        print(f"Config: {config}")
        print(f"  → Probability: {prob:.4f}, Label: {predictor.classify(prob)}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*20 + "MEDdino Quick Start Examples")
    print("="*70)
    
    print("\n⚠ NOTE: These examples assume you have:")
    print("  1. Trained a MEDdino model (checkpoints/meddino/best_meddino.pt)")
    print("  2. Prepared MRI slice data in data/interim/oasis1_2d/")
    print("  3. Created an index CSV at data/index_oasis1_2d.csv")
    print("\nIf any of these are missing, some examples will be skipped.\n")
    
    # Check if checkpoint exists
    checkpoint_path = Path("checkpoints/meddino/best_meddino.pt")
    
    if not checkpoint_path.exists():
        print("\n" + "="*70)
        print("⚠ WARNING: Checkpoint not found!")
        print("="*70)
        print(f"\nExpected checkpoint at: {checkpoint_path}")
        print("\nTo train a model, run:")
        print("  python -m src.models.train_meddino \\")
        print("      --index_csv data/index_oasis1_2d.csv \\")
        print("      --output_dir checkpoints/meddino \\")
        print("      --freeze_epochs 5 \\")
        print("      --finetune_epochs 5")
        print("\nExiting...")
        return
    
    try:
        # Run examples
        example_single_image()
        example_patient_prediction()
        example_batch_processing()
        example_csv_processing()
        example_comparison()
        
        print("\n" + "="*70)
        print("✓ All examples completed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nPlease check that:")
        print("  1. Your checkpoint path is correct")
        print("  2. Your data paths exist")
        print("  3. All dependencies are installed")


if __name__ == "__main__":
    main()

