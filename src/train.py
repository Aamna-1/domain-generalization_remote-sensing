"""
Main training script for domain generalization experiments.

Supports:
- Multiple PEFT methods (SoRA, LoRAReins)
- Multiple datasets (ISPRS, LoveDA)
- Augmentation grid search
- Multi-seed evaluation

Usage:
    python src/train.py --config configs/experiments/isprs_sora.yaml
    python src/train.py --method sora --config L3 --scenario V2P --seeds 42 123 456
"""

import os
import sys
import random
import argparse
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backbone import DINOv3MultiScaleExtractor
from models.decoder import SegformerDecoder, SegModel
from models.sora import SoraConfig, SoraModel, SORA_CONFIGS
from models.reins import DINOv3MultiScaleExtractorLoRAReins, REINS_CONFIGS
from data.isprs import ISPRSDataset, get_isprs_dataloaders
from data.loveda import LoveDADataset, get_loveda_dataloaders
from utils.metrics import evaluate_model, compute_class_weights


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_model_sora(
    config_name: str,
    num_classes: int,
    img_size: int,
    device: str
) -> SegModel:
    """Create model with SoRA adaptation."""
    import timm
    
    cfg = SORA_CONFIGS[config_name]
    sora_config = SoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        target_modules=cfg["targets"],
        start_lora_idx=cfg["start_idx"],
        lora_dropout=0.1,
        lora_weight_init="sora_niter_4",
        first_eigen=False,
        merge_weights=False,
    )
    
    # Create backbone with SoRA
    backbone = timm.create_model(
        'vit_small_plus_patch16_dinov3.lvd1689m',
        pretrained=True,
        features_only=True,
        out_indices=[2, 5, 8, 11],
    )
    
    class SoRAExtractor(nn.Module):
        def __init__(self, sora_config, backbone):
            super().__init__()
            self.backbone = SoraModel(sora_config, backbone).model
            
        def forward(self, x):
            return self.backbone(x)
    
    feature_extractor = SoRAExtractor(sora_config, backbone).to(device)
    decoder = SegformerDecoder(
        in_channels=384,
        num_scales=4,
        embed_dim=256,
        num_classes=num_classes,
        target_size=img_size
    ).to(device)
    
    model = SegModel(feature_extractor, decoder).to(device)
    
    # Print parameter info
    backbone_total = sum(p.numel() for p in feature_extractor.backbone.parameters())
    sora_trainable = sum(p.numel() for p in feature_extractor.backbone.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in decoder.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Backbone: {backbone_total:,} | SoRA: {sora_trainable:,} | "
          f"Decoder: {decoder_trainable:,} | Total trainable: {total_trainable:,}")
    
    return model


def create_model_reins(
    config_name: str,
    num_classes: int,
    img_size: int,
    device: str
) -> Tuple[SegModel, nn.Module]:
    """Create model with LoRAReins adaptation."""
    cfg = REINS_CONFIGS[config_name]
    
    feature_extractor = DINOv3MultiScaleExtractorLoRAReins(
        reins_cfg=cfg,
        out_indices=(2, 5, 8, 11),
        embed_dim=384,
        patch_size=16,
        num_layers=12,
    ).to(device)
    
    decoder = SegformerDecoder(
        in_channels=384,
        num_scales=4,
        embed_dim=256,
        num_classes=num_classes,
        target_size=img_size
    ).to(device)
    
    model = SegModel(feature_extractor, decoder).to(device)
    
    # Print parameter info
    reins_trainable = sum(p.numel() for p in feature_extractor.lorareins.parameters())
    decoder_trainable = sum(p.numel() for p in decoder.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  LoRAReins: {reins_trainable:,} | Decoder: {decoder_trainable:,} | "
          f"Total trainable: {total_trainable:,}")
    
    return model, feature_extractor


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_classes: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for imgs, masks in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks[masks >= num_classes] = 255
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int
) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks[masks >= num_classes] = 255
        
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    num_classes: int,
    device: str,
    epochs: int = 10,
    lr: float = 2e-4,
    patience: int = 10,
    seed: int = 42
) -> Dict:
    """
    Full training loop with early stopping.
    
    Returns:
        Dictionary with training history and best model state
    """
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        val_loss = validate_epoch(
            model, val_loader, criterion, device, num_classes
        )
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"[Seed {seed}] Epoch {epoch+1}/{epochs}: "
              f"Train={train_loss:.4f} Val={val_loss:.4f}", end="")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(" ← Best!")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model (val loss: {best_val_loss:.4f})")
        
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "final_epoch": epoch + 1
    }


def run_experiment(
    method: str,
    config: str,
    dataset: str,
    scenario: str,
    seeds: List[int],
    alpha_p: float = 0.0,
    alpha_g: float = 0.0,
    epochs: int = 10,
    data_root: str = "./data"
) -> Dict:
    """
    Run a complete experiment with multiple seeds.
    
    Args:
        method: 'sora' or 'reins'
        config: Configuration name (e.g., 'L3', 'R5')
        dataset: 'isprs' or 'loveda'
        scenario: Domain generalization scenario (e.g., 'V2P', 'R2U')
        seeds: List of random seeds
        alpha_p: Photometric augmentation intensity
        alpha_g: Geometric augmentation (max rotation)
        epochs: Number of training epochs
        data_root: Root directory for datasets
        
    Returns:
        Dictionary with aggregated results
    """
    device = get_device()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {method.upper()} {config} | {dataset.upper()} {scenario}")
    print(f"Augmentation: αp={alpha_p}, αg={alpha_g}")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Dataset configuration
    if dataset == "isprs":
        num_classes = 6
        img_size = 512
        batch_size = 8
        class_names = ISPRSDataset.CLASS_NAMES
        
        # Set paths based on scenario
        if scenario == "V2P":
            train_img = f"{data_root}/vaihingen_irrg/img_dir/train"
            train_mask = f"{data_root}/vaihingen_irrg/ann_dir/train"
            val_img = f"{data_root}/vaihingen_irrg/img_dir/val"
            val_mask = f"{data_root}/vaihingen_irrg/ann_dir/val"
            test_img = f"{data_root}/potsdam_rgb/img_dir/val"
            test_mask = f"{data_root}/potsdam_rgb/ann_dir/val"
        else:  # P2V
            train_img = f"{data_root}/potsdam_rgb/img_dir/train"
            train_mask = f"{data_root}/potsdam_rgb/ann_dir/train"
            val_img = f"{data_root}/potsdam_rgb/img_dir/val"
            val_mask = f"{data_root}/potsdam_rgb/ann_dir/val"
            test_img = f"{data_root}/vaihingen_irrg/img_dir/val"
            test_mask = f"{data_root}/vaihingen_irrg/ann_dir/val"
            
    else:  # loveda
        num_classes = 7
        img_size = 1024
        batch_size = 2
        class_names = LoveDADataset.CLASS_NAMES
        
        if scenario == "R2U":
            train_img = f"{data_root}/loveda/Train/Rural/images_png"
            train_mask = f"{data_root}/loveda/Train/Rural/masks_png"
            val_img = f"{data_root}/loveda/Val/Rural/images_png"
            val_mask = f"{data_root}/loveda/Val/Rural/masks_png"
            test_img = f"{data_root}/loveda/Val/Urban/images_png"
            test_mask = f"{data_root}/loveda/Val/Urban/masks_png"
        else:  # U2R
            train_img = f"{data_root}/loveda/Train/Urban/images_png"
            train_mask = f"{data_root}/loveda/Train/Urban/masks_png"
            val_img = f"{data_root}/loveda/Val/Urban/images_png"
            val_mask = f"{data_root}/loveda/Val/Urban/masks_png"
            test_img = f"{data_root}/loveda/Val/Rural/images_png"
            test_mask = f"{data_root}/loveda/Val/Rural/masks_png"
    
    # Create dataloaders
    DatasetClass = ISPRSDataset if dataset == "isprs" else LoveDADataset
    
    train_dataset = DatasetClass(
        train_img, train_mask,
        img_size=img_size,
        augment=True,
        alpha_p=alpha_p,
        alpha_g=alpha_g
    )
    val_dataset = DatasetClass(val_img, val_mask, img_size=img_size, augment=False)
    test_dataset = DatasetClass(test_img, test_mask, img_size=img_size, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=True
    )
    
    # Compute class weights
    class_weights = compute_class_weights(
        train_loader, num_classes, class_names, verbose=True
    )
    
    # Run experiments for each seed
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        
        # Create model
        feature_extractor = None
        if method == "sora":
            model = create_model_sora(config, num_classes, img_size, device)
        else:  # reins
            model, feature_extractor = create_model_reins(
                config, num_classes, img_size, device
            )
        
        # Train
        train_result = train_model(
            model, train_loader, val_loader,
            class_weights, num_classes, device,
            epochs=epochs, seed=seed
        )
        
        # Evaluate
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_model(
            model, val_loader, num_classes, device, class_names
        )
        
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(
            model, test_loader, num_classes, device, class_names
        )
        
        # Cleanup hooks if using Reins
        if feature_extractor is not None:
            feature_extractor.remove_hooks()
        
        all_results.append({
            "seed": seed,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "train_result": train_result
        })
    
    # Aggregate results
    val_miou = np.array([r["val_metrics"]["miou"] for r in all_results])
    test_miou = np.array([r["test_metrics"]["miou"] for r in all_results])
    test_oa = np.array([r["test_metrics"]["oa"] for r in all_results])
    
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"Val mIoU:  {val_miou.mean():.4f} ± {val_miou.std():.4f}")
    print(f"Test mIoU: {test_miou.mean():.4f} ± {test_miou.std():.4f}")
    print(f"Test OA:   {test_oa.mean():.4f} ± {test_oa.std():.4f}")
    
    return {
        "method": method,
        "config": config,
        "dataset": dataset,
        "scenario": scenario,
        "alpha_p": alpha_p,
        "alpha_g": alpha_g,
        "seeds": seeds,
        "all_results": all_results,
        "val_miou_mean": val_miou.mean(),
        "val_miou_std": val_miou.std(),
        "test_miou_mean": test_miou.mean(),
        "test_miou_std": test_miou.std(),
        "test_oa_mean": test_oa.mean(),
        "test_oa_std": test_oa.std(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Domain Generalization Training Script"
    )
    parser.add_argument(
        "--method", type=str, default="sora",
        choices=["sora", "reins"],
        help="PEFT method to use"
    )
    parser.add_argument(
        "--config", type=str, default="L3",
        help="Configuration name (L1-L5 for SoRA, R1-R5 for Reins)"
    )
    parser.add_argument(
        "--dataset", type=str, default="isprs",
        choices=["isprs", "loveda"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--scenario", type=str, default="V2P",
        help="DG scenario (V2P, P2V for ISPRS; R2U, U2R for LoveDA)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds for experiments"
    )
    parser.add_argument(
        "--alpha_p", type=float, default=0.0,
        help="Photometric augmentation intensity"
    )
    parser.add_argument(
        "--alpha_g", type=float, default=0.0,
        help="Geometric augmentation (max rotation degrees)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        method=args.method,
        config=args.config,
        dataset=args.dataset,
        scenario=args.scenario,
        seeds=args.seeds,
        alpha_p=args.alpha_p,
        alpha_g=args.alpha_g,
        epochs=args.epochs,
        data_root=args.data_root
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.method}_{args.config}_{args.dataset}_{args.scenario}_"
        f"ap{args.alpha_p}_ag{args.alpha_g}.pt"
    )
    torch.save(results, output_file)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
