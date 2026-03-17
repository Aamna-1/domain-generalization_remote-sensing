"""
SegFormer-style decoder for semantic segmentation.

This module implements a lightweight MLP-based decoder that fuses
multi-scale features from the backbone for dense prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLP(nn.Module):
    """
    Simple MLP for feature projection.
    
    Reshapes spatial features, applies linear projection,
    and reshapes back to spatial format.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Projected tensor of shape (B, out_features, H, W)
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.linear(x)                 # (B, H*W, out_features)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


class SegformerDecoder(nn.Module):
    """
    SegFormer-style decoder for multi-scale feature fusion.
    
    Takes multi-scale features from the backbone, projects them
    to a common dimension, fuses them, and predicts per-pixel classes.
    
    Args:
        in_channels: Number of channels in backbone features. Default 384.
        num_scales: Number of feature scales. Default 4.
        embed_dim: Hidden dimension for feature fusion. Default 256.
        num_classes: Number of output classes. Default 6.
        target_size: Output spatial size. Default 512.
    """
    
    def __init__(
        self,
        in_channels: int = 384,
        num_scales: int = 4,
        embed_dim: int = 256,
        num_classes: int = 6,
        target_size: int = 512,
    ):
        super().__init__()
        
        self.target_size = target_size
        self.num_scales = num_scales
        
        # Per-scale MLP projections
        self.mlp_stage = nn.ModuleList([
            MLP(in_channels, embed_dim) for _ in range(num_scales)
        ])
        
        # Fusion layers
        self.fuse_stage = nn.Sequential(
            nn.Conv2d(embed_dim * num_scales, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features and predict segmentation.
        
        Args:
            features: List of feature tensors from backbone
            
        Returns:
            Logits tensor of shape (B, num_classes, target_size, target_size)
        """
        # Project each scale
        projected = [mlp(f) for mlp, f in zip(self.mlp_stage, features)]
        
        # Upsample all to first scale's resolution
        target_h, target_w = projected[0].shape[2], projected[0].shape[3]
        resized = [
            F.interpolate(p, size=(target_h, target_w), mode='bilinear', align_corners=False)
            for p in projected
        ]
        
        # Concatenate and fuse
        x = torch.cat(resized, dim=1)
        x = self.fuse_stage(x)
        
        # Predict
        x = self.head(x)
        
        # Upsample to target size
        x = F.interpolate(
            x, size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=False
        )
        
        return x


class SegModel(nn.Module):
    """
    Complete segmentation model combining backbone and decoder.
    
    Args:
        feature_extractor: Backbone feature extractor
        decoder: Segmentation decoder
    """
    
    def __init__(self, feature_extractor: nn.Module, decoder: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete model.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        features = self.feature_extractor(x)
        return self.decoder(features)
    
    def get_trainable_params(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick test
    from backbone import DINOv3MultiScaleExtractor
    
    backbone = DINOv3MultiScaleExtractor()
    decoder = SegformerDecoder(
        in_channels=384,
        num_scales=4,
        embed_dim=256,
        num_classes=6,
        target_size=512
    )
    model = SegModel(backbone, decoder)
    
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Total params: {model.get_total_params():,}")
    print(f"Trainable params: {model.get_trainable_params():,}")
