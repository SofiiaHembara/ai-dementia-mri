# src/models/infer_meddino.py
# -*- coding: utf-8 -*-
"""
MEDdino inference utilities for production deployment.

Provides:
- Single image inference
- Batch inference
- Patient-level aggregation
- Integration helpers for Flask/Streamlit/FastAPI
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.train_meddino import MEDdinoClassifier, get_device


# ============================================================================
# Inference Wrapper
# ============================================================================

class MEDdinoPredictor:
    """
    Easy-to-use wrapper for MEDdino inference.
    
    Usage:
        predictor = MEDdinoPredictor(
            checkpoint_path="checkpoints/best_meddino.pt",
            backbone_name="vit_base_patch16_224.dino"
        )
        
        # Single image
        prob = predictor.predict_image("path/to/slice.png")
        
        # Patient (multiple slices)
        patient_prob = predictor.predict_patient([
            "path/to/slice1.png",
            "path/to/slice2.png",
            ...
        ])
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        backbone_name: str = "vit_base_patch16_224.dino",
        img_size: int = 224,
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
        use_clahe: bool = False,
        normalize_mode: str = "zscore"
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint (.pt file)
            backbone_name: timm model identifier
            img_size: Input image size
            threshold: Decision threshold (if None, use 0.5)
            device: Device (if None, auto-select)
            use_clahe: Apply CLAHE preprocessing
            normalize_mode: "zscore", "minmax", or "imagenet"
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.backbone_name = backbone_name
        self.img_size = img_size
        self.threshold = threshold if threshold is not None else 0.5
        self.use_clahe = use_clahe
        self.normalize_mode = normalize_mode
        
        # Device
        self.device = device if device is not None else get_device()
        
        # CLAHE
        self.clahe = None
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Load model
        self.model = self._load_model()
        
        print(f"MEDdinoPredictor initialized:")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Backbone: {self.backbone_name}")
        print(f"  Device: {self.device}")
        print(f"  Threshold: {self.threshold:.3f}")
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Build model
        model = MEDdinoClassifier(
            backbone_name=self.backbone_name,
            img_size=self.img_size,
            dropout=0.0,  # No dropout during inference
            head_type="linear"
        )
        
        # Load weights
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _preprocess_image(self, img: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            img: Can be path (str/Path), PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Load image
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("L")
        elif isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                # Convert RGB to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(img.astype(np.uint8), mode="L")
        elif isinstance(img, Image.Image):
            img = img.convert("L")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Convert to numpy
        img_np = np.array(img, dtype=np.uint8)
        
        # Resize
        img_np = cv2.resize(img_np, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # CLAHE (optional)
        if self.clahe is not None:
            img_np = self.clahe.apply(img_np)
        
        # Convert to float
        img_np = img_np.astype(np.float32)
        
        # Normalize
        if self.normalize_mode == "zscore":
            mean = img_np.mean()
            std = img_np.std() + 1e-6
            img_np = (img_np - mean) / std
        elif self.normalize_mode == "minmax":
            img_np = img_np / 255.0
        elif self.normalize_mode == "imagenet":
            img_np = img_np / 255.0
            # ImageNet normalization (will be applied as 3-channel below)
        
        # Stack to 3 channels (ViT expects RGB)
        img_3ch = np.stack([img_np, img_np, img_np], axis=0)  # (3, H, W)
        
        # ImageNet normalization (if enabled)
        if self.normalize_mode == "imagenet":
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_3ch = (img_3ch - mean) / std
        
        # To tensor
        tensor = torch.from_numpy(img_3ch).float().unsqueeze(0)  # (1, 3, H, W)
        
        return tensor
    
    @torch.no_grad()
    def predict_image(
        self,
        img: Union[str, Path, Image.Image, np.ndarray],
        return_logit: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Predict for a single image.
        
        Args:
            img: Image (path, PIL, or numpy array)
            return_logit: If True, return (probability, logit)
            
        Returns:
            Probability of dementia (float)
            Or (probability, logit) if return_logit=True
        """
        self.model.eval()
        
        # Preprocess
        x = self._preprocess_image(img).to(self.device)
        
        # Forward pass
        logit = self.model(x).cpu().float().item()
        prob = float(torch.sigmoid(torch.tensor(logit, dtype=torch.float32)).item())
        
        if return_logit:
            return prob, logit
        return prob
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict for a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for inference
            
        Returns:
            Array of probabilities (N,)
        """
        self.model.eval()
        
        all_probs = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = [self._preprocess_image(img) for img in batch_imgs]
            x_batch = torch.cat(batch_tensors, dim=0).to(self.device)  # (B, 3, H, W)
            
            # Forward pass
            logits = self.model(x_batch)  # (B,)
            # Convert to float32 for MPS compatibility
            probs = torch.sigmoid(logits).cpu().float().numpy()
            
            all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def predict_patient(
        self,
        slice_paths: List[Union[str, Path]],
        aggregation: str = "mean",
        return_slice_probs: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Predict for a patient (multiple slices).
        
        Args:
            slice_paths: List of paths to MRI slices
            aggregation: "mean", "median", or "max"
            return_slice_probs: If True, return (patient_prob, slice_probs)
            
        Returns:
            Patient-level probability
            Or (patient_prob, slice_probs) if return_slice_probs=True
        """
        # Predict all slices
        slice_probs = self.predict_batch(slice_paths)
        
        # Aggregate
        if aggregation == "mean":
            patient_prob = float(slice_probs.mean())
        elif aggregation == "median":
            patient_prob = float(np.median(slice_probs))
        elif aggregation == "max":
            patient_prob = float(slice_probs.max())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        if return_slice_probs:
            return patient_prob, slice_probs
        return patient_prob
    
    def classify(self, prob: float) -> str:
        """
        Convert probability to class label using threshold.
        
        Args:
            prob: Probability of dementia
            
        Returns:
            "dementia" or "non_demented"
        """
        return "dementia" if prob >= self.threshold else "non_demented"


# ============================================================================
# Standalone Inference Script
# ============================================================================

def main():
    """Command-line inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MEDdino inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224.dino")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_clahe", action="store_true")
    parser.add_argument("--normalize_mode", type=str, default="zscore", choices=["zscore", "minmax", "imagenet"])
    
    # Input mode
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--image_list", type=str, default=None, help="Text file with image paths (one per line)")
    parser.add_argument("--patient_dir", type=str, default=None, help="Directory with patient slices")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "max"])
    
    # Output
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MEDdinoPredictor(
        checkpoint_path=args.checkpoint,
        backbone_name=args.backbone,
        img_size=args.img_size,
        threshold=args.threshold,
        use_clahe=args.use_clahe,
        normalize_mode=args.normalize_mode
    )
    
    results = {}
    
    # Single image
    if args.image is not None:
        print(f"\nPredicting single image: {args.image}")
        prob = predictor.predict_image(args.image)
        label = predictor.classify(prob)
        print(f"  Probability: {prob:.4f}")
        print(f"  Prediction: {label}")
        results = {"image": args.image, "probability": prob, "prediction": label}
    
    # Image list
    elif args.image_list is not None:
        print(f"\nPredicting from list: {args.image_list}")
        with open(args.image_list, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        probs = predictor.predict_batch(image_paths)
        
        results["images"] = []
        for path, prob in zip(image_paths, probs):
            label = predictor.classify(prob)
            print(f"  {path}: {prob:.4f} → {label}")
            results["images"].append({
                "path": path,
                "probability": float(prob),
                "prediction": label
            })
    
    # Patient directory
    elif args.patient_dir is not None:
        print(f"\nPredicting patient from directory: {args.patient_dir}")
        patient_dir = Path(args.patient_dir)
        
        # Find all images in directory
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        slice_paths = [
            str(p) for p in patient_dir.iterdir()
            if p.suffix.lower() in image_exts
        ]
        
        if not slice_paths:
            print(f"No images found in {patient_dir}")
            return
        
        print(f"  Found {len(slice_paths)} slices")
        
        patient_prob, slice_probs = predictor.predict_patient(
            slice_paths, aggregation=args.aggregation, return_slice_probs=True
        )
        patient_label = predictor.classify(patient_prob)
        
        print(f"\n  Patient-level probability: {patient_prob:.4f}")
        print(f"  Patient-level prediction: {patient_label}")
        print(f"  Aggregation: {args.aggregation}")
        
        results = {
            "patient_dir": str(patient_dir),
            "n_slices": len(slice_paths),
            "patient_probability": float(patient_prob),
            "patient_prediction": patient_label,
            "aggregation": args.aggregation,
            "slice_probabilities": slice_probs.tolist()
        }
    
    else:
        print("Error: Must specify --image, --image_list, or --patient_dir")
        return
    
    # Save results
    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output_json}")


# ============================================================================
# Flask/FastAPI Integration Example
# ============================================================================

class MEDdinoAPI:
    """
    Wrapper for web API integration (Flask, FastAPI, etc.).
    
    Example with Flask:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        api = MEDdinoAPI(checkpoint_path="best_meddino.pt")
        
        @app.route("/predict", methods=["POST"])
        def predict():
            file = request.files["image"]
            img = Image.open(file.stream)
            result = api.predict_single(img)
            return jsonify(result)
    
    Example with FastAPI:
        from fastapi import FastAPI, File, UploadFile
        
        app = FastAPI()
        api = MEDdinoAPI(checkpoint_path="best_meddino.pt")
        
        @app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            img = Image.open(file.file)
            return api.predict_single(img)
    """
    
    def __init__(self, checkpoint_path: str, threshold: float = 0.5, **kwargs):
        self.predictor = MEDdinoPredictor(
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            **kwargs
        )
    
    def predict_single(self, img: Union[Image.Image, np.ndarray]) -> Dict:
        """Predict for a single image (API format)."""
        prob = self.predictor.predict_image(img)
        label = self.predictor.classify(prob)
        
        return {
            "probability": float(prob),
            "prediction": label,
            "confidence": float(max(prob, 1 - prob))
        }
    
    def predict_patient(self, images: List[Union[Image.Image, np.ndarray]], aggregation: str = "mean") -> Dict:
        """Predict for a patient (multiple images, API format)."""
        patient_prob, slice_probs = self.predictor.predict_patient(
            images, aggregation=aggregation, return_slice_probs=True
        )
        patient_label = self.predictor.classify(patient_prob)
        
        return {
            "patient_probability": float(patient_prob),
            "patient_prediction": patient_label,
            "n_slices": len(images),
            "aggregation": aggregation,
            "slice_probabilities": slice_probs.tolist()
        }


if __name__ == "__main__":
    main()

