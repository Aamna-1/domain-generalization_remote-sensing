"""
LoRAReins implementation for feature-space adaptation.

Reins (Representation Engineering via In-context Learning) provides
an alternative to weight-space adaptation by introducing learnable
tokens that modulate feature representations through attention.

LoRAReins extends this by parameterizing the tokens using low-rank
decomposition for parameter efficiency.

Reference: Stronger, Fewer, & Superior: Harnessing Vision Foundation
Models for Domain Generalized Semantic Segmentation
"""

import math
from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import get_transformer_blocks


class Reins(nn.Module):
    """
    Reins module for feature-space adaptation.
    
    Introduces learnable tokens that interact with image features
    through attention, enabling dynamic and input-dependent adaptation.
    
    Args:
        num_layers: Number of transformer layers to adapt
        embed_dims: Feature embedding dimension
        patch_size: Patch size of the ViT
        token_length: Number of learnable tokens
        use_softmax: Whether to use softmax in attention
        scale_init: Initial scale for the adaptation
    """
    
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        token_length: int = 100,
        use_softmax: bool = True,
        scale_init: float = 0.001
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.token_length = token_length
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.create_model()
        
    def create_model(self):
        """Initialize learnable parameters."""
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        # Xavier initialization
        val = math.sqrt(6.0 / float(
            3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
        ))
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        
    def get_tokens(self, layer: int) -> Tensor:
        """Get tokens for a specific layer or all layers."""
        if layer == -1:
            return self.learnable_tokens
        return self.learnable_tokens[layer]
        
    def forward(
        self,
        feats: Tensor,
        layer: int,
        batch_first: bool = False,
        has_cls_token: bool = True
    ) -> Tensor:
        """
        Apply feature adaptation.
        
        Args:
            feats: Input features (N, B, C) or (B, N, C) if batch_first
            layer: Layer index
            batch_first: Whether batch dimension is first
            has_cls_token: Whether features include CLS token
            
        Returns:
            Adapted features with same shape as input
        """
        if batch_first:
            feats = feats.permute(1, 0, 2)
            
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
            
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(feats, tokens, layer) * self.scale
        feats = feats + delta_feat
        
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
            
        if batch_first:
            feats = feats.permute(1, 0, 2)
            
        return feats
        
    def forward_delta_feat(
        self,
        feats: Tensor,
        tokens: Tensor,
        layers: int
    ) -> Tensor:
        """Compute feature delta using token attention."""
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        
        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)
            
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :])
        )
        
        return self.mlp_delta_f(delta_f + feats)


class LoRAReins(Reins):
    """
    LoRA-parameterized Reins for efficient adaptation.
    
    Instead of full learnable tokens, uses low-rank decomposition:
    tokens = learnable_tokens_a @ learnable_tokens_b
    
    This reduces parameters from (L * T * D) to (L * T * r + L * r * D)
    where L=layers, T=token_length, D=embed_dims, r=lora_dim.
    
    Args:
        lora_dim: Rank of the low-rank decomposition
        **kwargs: Arguments passed to Reins
    """
    
    def __init__(self, lora_dim: int = 16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)
        
    def create_model(self):
        """Initialize with low-rank token parameterization."""
        super().create_model()
        
        # Replace full tokens with low-rank factors
        del self.learnable_tokens
        
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        
        # Initialization
        val = math.sqrt(6.0 / float(
            3 * reduce(mul, (self.patch_size, self.patch_size), 1)
            + (self.embed_dims * self.lora_dim) ** 0.5
        ))
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)
        
    def get_tokens(self, layer: int) -> Tensor:
        """Get tokens by computing A @ B."""
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]


class DINOv3MultiScaleExtractorLoRAReins(nn.Module):
    """
    DINOv3 backbone with LoRAReins adaptation.
    
    Injects LoRAReins hooks into transformer blocks to adapt
    features during forward pass.
    
    Args:
        reins_cfg: Configuration dict with keys:
            - token_length: Number of learnable tokens
            - lora_dim: Low-rank dimension
            - scale_init: Initial adaptation scale
            - start_idx: First block to adapt
        out_indices: Block indices for feature extraction
        embed_dim: Feature dimension
        patch_size: ViT patch size
        num_layers: Total number of transformer layers
    """
    
    def __init__(
        self,
        reins_cfg: dict,
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        embed_dim: int = 384,
        patch_size: int = 16,
        num_layers: int = 12
    ):
        super().__init__()
        
        import timm
        
        self.backbone = timm.create_model(
            'vit_small_plus_patch16_dinov3.lvd1689m',
            pretrained=True,
            features_only=True,
            out_indices=list(out_indices),
        )
        
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        # Create LoRAReins
        start_idx = reins_cfg["start_idx"]
        adapted_layers = num_layers - start_idx
        
        self.lorareins = LoRAReins(
            lora_dim=reins_cfg["lora_dim"],
            num_layers=adapted_layers,
            embed_dims=embed_dim,
            patch_size=patch_size,
            token_length=reins_cfg["token_length"],
            use_softmax=True,
            scale_init=reins_cfg["scale_init"],
        )
        
        # Register hooks
        blocks = get_transformer_blocks(self.backbone)
        self._hooks = []
        
        for abs_idx in range(start_idx, num_layers):
            rel_idx = abs_idx - start_idx
            handle = blocks[abs_idx].register_forward_hook(
                self._make_hook(rel_idx)
            )
            self._hooks.append(handle)
            
        # Log info
        frozen = sum(p.numel() for p in self.backbone.parameters())
        reins = sum(p.numel() for p in self.lorareins.parameters())
        print(f"Backbone frozen: {frozen:,} | LoRAReins trainable: {reins:,} | "
              f"Adapting blocks {start_idx}–{num_layers-1} ({adapted_layers} layers)")
              
    def _make_hook(self, rel_idx: int):
        """Create a forward hook for a specific layer."""
        reins = self.lorareins
        
        def hook(module, input, output):
            x = output.permute(1, 0, 2)
            x = reins(x, layer=rel_idx, batch_first=False, has_cls_token=True)
            return x.permute(1, 0, 2)
            
        return hook
        
    def forward(self, x: Tensor):
        """Extract features with LoRAReins adaptation."""
        return self.backbone(x)
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


# Predefined configurations
REINS_CONFIGS = {
    "R1": {"token_length": 50,  "lora_dim": 8,  "scale_init": 0.001, "start_idx": 0},
    "R2": {"token_length": 100, "lora_dim": 16, "scale_init": 0.001, "start_idx": 0},
    "R3": {"token_length": 100, "lora_dim": 16, "scale_init": 0.001, "start_idx": 6},
    "R4": {"token_length": 150, "lora_dim": 16, "scale_init": 0.001, "start_idx": 0},
    "R5": {"token_length": 100, "lora_dim": 32, "scale_init": 0.001, "start_idx": 0},
}


def get_reins_config(config_name: str) -> dict:
    """Get a predefined Reins configuration."""
    return REINS_CONFIGS[config_name].copy()
