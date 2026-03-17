"""
ISPRS Potsdam/Vaihingen dataset loader.

Handles both Potsdam (RGB) and Vaihingen (IRRG) datasets
with configurable augmentation for domain generalization experiments.

Classes (6):
    0: Impervious surfaces
    1: Buildings  
    2: Low vegetation
    3: Trees
    4: Cars
    5: Clutter/background
"""

import os
import random
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode


class ISPRSDataset(Dataset):
    """
    ISPRS semantic segmentation dataset.
    
    Supports both Potsdam (RGB) and Vaihingen (IRRG) with
    two-axis augmentation control (photometric αp and geometric αg).
    
    Args:
        img_dir: Path to image directory
        mask_dir: Path to annotation directory
        img_size: Target image size. Default 512.
        augment: Whether to apply augmentation. Default False.
        alpha_p: Photometric augmentation intensity [0, 1]. Default 0.2.
        alpha_g: Maximum rotation angle in degrees. Default 15.0.
    """
    
    # Class names for ISPRS benchmark
    CLASS_NAMES = [
        "Impervious surfaces",
        "Buildings",
        "Low vegetation", 
        "Trees",
        "Cars",
        "Clutter/background"
    ]
    NUM_CLASSES = 6
    
    def __init__(
        self,
        img_dir: str,
        mask_dir: Optional[str] = None,
        img_size: int = 512,
        augment: bool = False,
        alpha_p: float = 0.2,
        alpha_g: float = 15.0
    ):
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
            
        self.imgs = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.tif', '.jpg'))
        ])
        
        if not self.imgs:
            raise ValueError(f"No images found in: {img_dir}")
            
        self.masks = None
        if mask_dir and os.path.exists(mask_dir):
            self.masks = sorted([
                os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                if f.lower().endswith(('.png', '.tif'))
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
        """Apply geometric augmentations to image and mask."""
        # Random flips (controlled by alpha_g > 0)
        if self.alpha_g > 0:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
                
        # Random rotation
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
        
        # Load and process mask if available
        if self.masks:
            mask = read_image(self.masks[idx])[0].to(torch.long)
            mask = TF.resize(
                mask.unsqueeze(0),
                (self.img_size, self.img_size),
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0)
            
            # ISPRS label mapping: 0 is ignore, 1-6 are classes
            mask[mask == 0] = 255  # Ignore class
            valid = (mask >= 1) & (mask <= 6)
            mask[valid] = mask[valid] - 1  # Shift to 0-5
            mask[~valid & (mask != 255)] = 255
            
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
        return ISPRSDataset.CLASS_NAMES.copy()


def get_isprs_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    test_img_dir: str,
    test_mask_dir: str,
    batch_size: int = 8,
    img_size: int = 512,
    alpha_p: float = 0.2,
    alpha_g: float = 15.0,
    num_workers: int = 4
):
    """
    Create train/val/test dataloaders for ISPRS dataset.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_dataset = ISPRSDataset(
        train_img_dir, train_mask_dir,
        img_size=img_size,
        augment=True,
        alpha_p=alpha_p,
        alpha_g=alpha_g
    )
    
    val_dataset = ISPRSDataset(
        val_img_dir, val_mask_dir,
        img_size=img_size,
        augment=False
    )
    
    test_dataset = ISPRSDataset(
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
