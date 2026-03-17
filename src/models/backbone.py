"""
DINOv3 Vision Transformer backbone with multi-scale feature extraction.

This module provides the feature extraction backbone using DINOv3 ViT-S+
pretrained model from timm, with support for multi-scale feature outputs.
"""

import torch
import torch.nn as nn
import timm
from typing import List, Tuple


def get_transformer_blocks(backbone: nn.Module) -> nn.Sequential:
    """
    Robustly locate the transformer blocks within a timm backbone.
    
    Args:
        backbone: The timm model backbone
        
    Returns:
        The nn.Sequential containing transformer blocks
        
    Raises:
        AttributeError: If blocks cannot be found
    """
    search_paths = [
        "blocks",
        "model.blocks", 
        "trunk.blocks",
        "patch_model.blocks"
    ]
    
    for attr_path in search_paths:
        obj = backbone
        found = True
        for attr in attr_path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found:
            return obj
            
    raise AttributeError(
        f"Cannot find transformer blocks in backbone. "
        f"Available attributes: {[n for n, _ in backbone.named_children()]}"
    )


class DINOv3MultiScaleExtractor(nn.Module):
    """
    Multi-scale feature extractor using DINOv3 ViT-S+ backbone.
    
    Extracts features at multiple stages of the transformer for
    dense prediction tasks like semantic segmentation.
    
    Args:
        out_indices: Tuple of block indices to extract features from.
                    Default (2, 5, 8, 11) provides 4 scales.
        pretrained: Whether to load pretrained weights. Default True.
        freeze_backbone: Whether to freeze backbone parameters. Default True.
    """
    
    def __init__(
        self,
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            'vit_small_plus_patch16_dinov3.lvd1689m',
            pretrained=pretrained,
            features_only=True,
            out_indices=list(out_indices),
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Model info
        self.embed_dim = 384
        self.patch_size = 16
        self.num_layers = 12
        self.out_indices = out_indices
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of feature tensors at each output index
        """
        return self.backbone(x)
    
    def get_num_params(self) -> Tuple[int, int]:
        """
        Get parameter counts.
        
        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Quick test
    model = DINOv3MultiScaleExtractor()
    x = torch.randn(2, 3, 512, 512)
    features = model(x)
    
    print("DINOv3 Multi-Scale Extractor")
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")
    
    total, trainable = model.get_num_params()
    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,}")
