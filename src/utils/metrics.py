"""
Evaluation metrics for semantic segmentation.

Provides computation of:
- Per-class IoU (Intersection over Union)
- Per-class F1 Score
- Overall Accuracy (OA)
- Mean IoU (mIoU)
- Mean F1 (mF1)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from typing import Dict, List, Optional


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int = 255
) -> np.ndarray:
    """
    Compute confusion matrix from predictions and targets.
    
    Args:
        predictions: Predicted labels (N,)
        targets: Ground truth labels (N,)
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    valid = targets != ignore_index
    return confusion_matrix(
        targets[valid],
        predictions[valid],
        labels=list(range(num_classes))
    )


def compute_metrics_from_cm(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute segmentation metrics from confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: Optional list of class names
        
    Returns:
        Dictionary with IoU, F1, OA, mIoU, mF1
    """
    # Per-class metrics
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1)
    
    precision = intersection / np.maximum(cm.sum(axis=0), 1)
    recall = intersection / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    
    # Overall accuracy
    oa = intersection.sum() / np.maximum(cm.sum(), 1)
    
    # Mean metrics
    miou = iou.mean()
    mf1 = f1.mean()
    
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "oa": oa,
        "miou": miou,
        "mf1": mf1,
        "confusion_matrix": cm
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate segmentation model on a dataset.
    
    Args:
        model: Segmentation model
        dataloader: Evaluation dataloader
        num_classes: Number of classes
        device: Device for computation
        class_names: Optional class names for printing
        verbose: Whether to print results
        
    Returns:
        Dictionary with all computed metrics
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
    
    for imgs, masks in iterator:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        preds = model(imgs).argmax(dim=1)
        
        valid = masks != 255
        cm += confusion_matrix(
            masks[valid].cpu().numpy(),
            preds[valid].cpu().numpy(),
            labels=list(range(num_classes))
        )
        
    metrics = compute_metrics_from_cm(cm, class_names)
    
    if verbose:
        print_metrics(metrics, class_names, num_classes)
        
    return metrics


def print_metrics(
    metrics: Dict,
    class_names: Optional[List[str]] = None,
    num_classes: int = 6
):
    """Print formatted metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Class':<25s}  {'IoU':>10s}  {'F1':>10s}")
    print("-" * 50)
    
    for i in range(num_classes):
        name = class_names[i] if class_names else f"Class {i}"
        print(f"{name:<25s}  {metrics['iou'][i]:>10.4f}  {metrics['f1'][i]:>10.4f}")
        
    print("-" * 50)
    print(f"{'Overall Accuracy':<25s}  {metrics['oa']:>10.4f}")
    print(f"{'Mean IoU':<25s}  {metrics['miou']:>10.4f}")
    print(f"{'Mean F1':<25s}  {metrics['mf1']:>10.4f}")
    print("=" * 60)


def compute_class_weights(
    dataloader: DataLoader,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency.
    
    Args:
        dataloader: Training dataloader
        num_classes: Number of classes
        class_names: Optional class names
        verbose: Whether to print distribution
        
    Returns:
        Class weight tensor
    """
    if verbose:
        print("\nComputing class weights from training data...")
        
    class_counts = torch.zeros(num_classes)
    
    iterator = tqdm(dataloader, desc="Analyzing") if verbose else dataloader
    
    for _, masks in iterator:
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum()
            
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    
    if verbose:
        print("-" * 60)
        for i in range(num_classes):
            name = class_names[i] if class_names else f"Class {i}"
            pct = 100 * class_counts[i] / total
            print(f"{i}: {name:<20s} | Pixels: {int(class_counts[i]):>10,} "
                  f"({pct:>5.2f}%) | Weight: {weights[i]:.4f}")
        print("-" * 60)
        
    return weights
