"""
LoveDA dataset loader for Urban/Rural domain generalization.

Classes (7):
    0: Background
    1: Building
    2: Road
    3: Water
    4: Barren
    5: Forest
    6: Agricultural
"""

import os
import random
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode


class LoveDADataset(Dataset):
    """
    LoveDA semantic segmentation dataset.
    
    Supports Urban and Rural domains with two-axis augmentation
    control for domain generalization experiments.
    
    Args:
        img_dir: Path to images_png directory
        mask_dir: Path to masks_png directory
        img_size: Target image size. Default 1024.
        augment: Whether to apply augmentation. Default False.
        alpha_p: Photometric augmentation intensity [0, 1]. Default 0.2.
        alpha_g: Maximum rotation angle in degrees. Default 15.0.
    """
    
    CLASS_NAMES = [
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agricultural"
    ]
    NUM_CLASSES = 7
    
    def __init__(
        self,
        img_dir: str,
        mask_dir: Optional[str] = None,
        img_size: int = 1024,
        augment: bool = False,
        alpha_p: float = 0.2,
        alpha_g: float = 15.0
    ):
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
            
        self.imgs = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith('.png')
        ])
        
        if not self.imgs:
            raise ValueError(f"No PNG images found in: {img_dir}")
            
        self.masks = None
        if mask_dir and os.path.exists(mask_dir):
            self.masks = sorted([
                os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                if f.lower().endswith('.png')
            ])
            if len(self.imgs) != len(self.masks):
                print(f"Warning: {len(self.imgs)} images != {len(self.masks)} masks")
                
        self.img_size = img_size
        self.augment = augment
        self.alpha_p = alpha_p
        self.alpha_g = alpha_g
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # Photometric augmentation
        if alpha_p > 0:
            self.color_aug = transforms.ColorJitter(
                brightness=alpha_p,
                contrast=alpha_p,
                saturation=alpha_p,
                hue=min(alpha_p * 0.2, 0.5),
            )
        else:
            self.color_aug = None
            
    def __len__(self) -> int:
        return len(self.imgs)
        
    def _apply_geometric_aug(
        self,
        img: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply geometric augmentations."""
        if self.alpha_g > 0:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
                
        if self.alpha_g > 0:
            angle = random.uniform(-self.alpha_g, self.alpha_g)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(
                mask.unsqueeze(0), angle,
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0)
            
        return img, mask
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load and resize image
        img = read_image(self.imgs[idx]).float() / 255.0
        img = TF.resize(img, (self.img_size, self.img_size))
        
        # Initialize mask
        mask = torch.full(
            (self.img_size, self.img_size),
            255,
            dtype=torch.long
        )
        
        # Load and process mask
        if self.masks:
            mask = read_image(self.masks[idx])[0].to(torch.long)
            mask = TF.resize(
                mask.unsqueeze(0),
                (self.img_size, self.img_size),
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0)
            
            # LoveDA label mapping: 0 is ignore/unlabeled, 1-7 are classes
            ignore = (mask == 0)
            mask = mask.clone()
            mask[~ignore] = mask[~ignore] - 1  # Shift to 0-6
            mask[ignore] = 255  # Set ignore regions
            
        # Apply augmentations
        if self.augment:
            img, mask = self._apply_geometric_aug(img, mask)
            if self.color_aug is not None:
                img = self.color_aug(img)
                
        # Normalize
        img = self.normalize(img)
        
        return img, mask
        
    @staticmethod
    def get_class_names() -> List[str]:
        """Get list of class names."""
        return LoveDADataset.CLASS_NAMES.copy()


def get_loveda_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    test_img_dir: str,
    test_mask_dir: str,
    batch_size: int = 2,
    img_size: int = 1024,
    alpha_p: float = 0.2,
    alpha_g: float = 15.0,
    num_workers: int = 4
):
    """
    Create train/val/test dataloaders for LoveDA dataset.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_dataset = LoveDADataset(
        train_img_dir, train_mask_dir,
        img_size=img_size,
        augment=True,
        alpha_p=alpha_p,
        alpha_g=alpha_g
    )
    
    val_dataset = LoveDADataset(
        val_img_dir, val_mask_dir,
        img_size=img_size,
        augment=False
    )
    
    test_dataset = LoveDADataset(
        test_img_dir, test_mask_dir,
        img_size=img_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
