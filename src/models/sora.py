"""
SoRA (SVD-initialized LoRA) implementation.

SoRA improves upon standard LoRA by initializing the low-rank
adaptation matrices using Singular Value Decomposition of the
pretrained weights, aiming to preserve domain-agnostic transformations.

Reference: SoMA: Singular Value Decomposed Minor Components Adaptation
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank


@dataclass
class SoraConfig:
    """Configuration for SoRA adaptation."""
    r: int = field(default=8)
    target_modules: Optional[List[str]] = field(default=None)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    lora_weight_init: str = field(default="sora_niter_4")
    first_eigen: bool = field(default=False)
    merge_weights: bool = field(default=False)
    fan_in_fan_out: bool = field(default=False)
    start_lora_idx: int = field(default=0)
    bias: str = field(default="none")


class LoraLayer:
    """Base class for LoRA layer functionality."""
    
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        first_eigen: bool,
        merge_weights: bool
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else lambda x: x
        self.first_eigen = first_eigen
        self.merged = False
        self.merge_weights = merge_weights


class SoraLinear(nn.Linear, LoraLayer):
    """
    Linear layer with SoRA (SVD-initialized LoRA) adaptation.
    
    The key difference from standard LoRA is that the low-rank
    matrices A and B are initialized using SVD of the pretrained
    weights, rather than random/zero initialization.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        first_eigen: bool = False,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self, r=r, first_eigen=first_eigen,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=merge_weights
        )
        
        self.fan_in_fan_out = fan_in_fan_out
        
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            
    def sora_init(self, init: str):
        """
        Initialize LoRA matrices using SVD of pretrained weights.
        
        Args:
            init: Initialization method - "sora" for full SVD or
                  "sora_niter_N" for randomized SVD with N iterations
        """
        weight = self.weight
        dtype = weight.dtype
        
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError("SoRA init requires float32, float16, or bfloat16.")
            
        weight = weight.to(torch.float32)
        
        if init == "sora":
            # Full SVD
            U, S, Vh = torch.linalg.svd(weight.data, full_matrices=False)
            if self.first_eigen:
                Ur, Sr, Vhr = U[:, :self.r], S[:self.r], Vh[:self.r, :]
            else:
                Ur, Sr, Vhr = U[:, -self.r:], S[-self.r:], Vh[-self.r:, :]
            Sr = Sr / self.scaling
            
        elif len(init.split("_niter_")) == 2:
            # Randomized SVD
            niter = int(init.split("_niter_")[-1])
            Ur, Sr, Vr = svd_lowrank(weight.data, self.r, niter=niter)
            Sr = Sr / self.scaling
            Vhr = Vr.t()
            
        else:
            raise ValueError(f"init should be 'sora' or 'sora_niter_[N]', got {init}")
            
        # Initialize A and B from SVD components
        self.lora_A.weight.data = torch.diag(torch.sqrt(Sr)) @ Vhr
        self.lora_B.weight.data = Ur @ torch.diag(torch.sqrt(Sr))
        
        # Subtract the low-rank approximation from original weights
        self.weight.data = (
            weight.data - self.scaling * self.lora_B.weight.data @ self.lora_A.weight.data
        ).to(dtype)
        
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        
        if not mode and self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.lora_B.weight @ self.lora_A.weight * self.scaling
            self.merged = True
        elif self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= self.lora_B.weight @ self.lora_A.weight * self.scaling
            self.merged = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            return (
                F.linear(x, self.weight, bias=self.bias) +
                ((x @ self.lora_A.weight.T) @ self.lora_B.weight.T) * self.scaling
            )
        return F.linear(x, self.weight, bias=self.bias)


class SoraModel(nn.Module):
    """
    Wrapper that applies SoRA adaptation to target modules in a model.
    
    Finds and replaces specified linear layers with SoraLinear layers,
    then initializes them using SVD.
    """
    
    def __init__(self, config: SoraConfig, model: nn.Module):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        self._mark_only_lora_as_trainable()
        self.forward = self.model.forward
        
    def _extract_block_index(self, module_name: str) -> Optional[int]:
        """Extract transformer block index from module name."""
        match = re.search(r'blocks\.(\d+)\.', module_name)
        return int(match.group(1)) if match else None
        
    def _get_parent_module(self, module_name: str):
        """Get parent module and child name from full module path."""
        names = module_name.split('.')
        parent = self.model
        for name in names[:-1]:
            parent = getattr(parent, name)
        return parent, names[-1]
        
    def _find_and_replace(self):
        """Find target modules and replace with SoraLinear."""
        kwargs = {
            "first_eigen": self.peft_config.first_eigen,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": self.peft_config.merge_weights,
        }
        
        for name, module in self.model.named_modules():
            block_idx = self._extract_block_index(name)
            
            # Skip if before start index
            if block_idx is None or block_idx < self.peft_config.start_lora_idx:
                continue
                
            # Check if this is a target module
            if any(name.endswith(t) for t in self.peft_config.target_modules):
                if isinstance(module, nn.Linear):
                    parent, target_name = self._get_parent_module(name)
                    
                    # Create SoRA layer
                    new_module = SoraLinear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        r=self.peft_config.r,
                        **kwargs
                    )
                    
                    # Copy weights
                    new_module.weight = module.weight
                    if module.bias is not None:
                        new_module.bias = module.bias
                        
                    # Replace module
                    setattr(parent, target_name, new_module)
                    new_module.to(module.weight.device)
                    
                    # Initialize with SVD
                    new_module.sora_init(self.peft_config.lora_weight_init)
                    
                    # Ensure LoRA params are on correct device
                    for n, m in new_module.named_modules():
                        if "lora_" in n:
                            m.to(module.weight.device)
                            
    def _mark_only_lora_as_trainable(self):
        """Freeze all params except LoRA parameters."""
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name
            
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# Predefined configurations
SORA_CONFIGS = {
    "L1": {"r": 8,  "alpha": 16, "targets": ["attn.qkv"], "start_idx": 0},
    "L2": {"r": 16, "alpha": 32, "targets": ["attn.qkv"], "start_idx": 0},
    "L3": {"r": 16, "alpha": 32, "targets": ["attn.qkv", "attn.proj"], "start_idx": 0},
    "L4": {"r": 16, "alpha": 32, "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"], "start_idx": 6},
    "L5": {"r": 32, "alpha": 64, "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"], "start_idx": 0},
}


def get_sora_config(config_name: str) -> SoraConfig:
    """Get a predefined SoRA configuration."""
    cfg = SORA_CONFIGS[config_name]
    return SoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        target_modules=cfg["targets"],
        start_lora_idx=cfg["start_idx"],
        lora_dropout=0.1,
        lora_weight_init="sora_niter_4",
        first_eigen=False,
        merge_weights=False,
    )
