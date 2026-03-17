"""
Standard LoRA (Low-Rank Adaptation) implementation.

This provides a baseline LoRA implementation without the SVD
initialization used in SoRA.

Reference: LoRA: Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    r: int = field(default=8)
    target_modules: Optional[List[str]] = field(default=None)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    merge_weights: bool = field(default=False)
    fan_in_fan_out: bool = field(default=False)
    start_lora_idx: int = field(default=0)
    bias: str = field(default="none")


class LoRALinear(nn.Linear):
    """
    Linear layer with LoRA adaptation.
    
    Adds low-rank matrices A and B such that:
    output = Wx + (BA)x * scaling
    
    where A ∈ R^{r×d}, B ∈ R^{d×r}, and r << d.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: Rank of low-rank matrices
        lora_alpha: Scaling factor
        lora_dropout: Dropout probability
        merge_weights: Whether to merge weights at eval
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False
        
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            
            # Freeze the pretrained weight
            self.weight.requires_grad = False
            
        self.reset_lora_parameters()
        
    def reset_lora_parameters(self):
        """Initialize LoRA parameters."""
        if hasattr(self, 'lora_A'):
            # Initialize A with Kaiming uniform
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            # Initialize B with zeros (start with identity transformation)
            nn.init.zeros_(self.lora_B.weight)
            
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if hasattr(self, 'lora_A'):
            self.lora_A.train(mode)
            self.lora_B.train(mode)
            
        if not mode and self.merge_weights and not self.merged:
            # Merge weights for inference
            if self.r > 0:
                self.weight.data += (
                    self.lora_B.weight @ self.lora_A.weight
                ) * self.scaling
            self.merged = True
        elif self.merge_weights and self.merged:
            # Unmerge weights for training
            if self.r > 0:
                self.weight.data -= (
                    self.lora_B.weight @ self.lora_A.weight
                ) * self.scaling
            self.merged = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # Compute original output + LoRA output
            result = F.linear(x, self.weight, bias=self.bias)
            lora_output = self.lora_B(self.lora_A(self.lora_dropout(x)))
            return result + lora_output * self.scaling
        else:
            return F.linear(x, self.weight, bias=self.bias)


class LoRAModel(nn.Module):
    """
    Wrapper that applies LoRA to target modules in a model.
    """
    
    def __init__(self, config: LoRAConfig, model: nn.Module):
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
        """Get parent module and child name."""
        names = module_name.split('.')
        parent = self.model
        for name in names[:-1]:
            parent = getattr(parent, name)
        return parent, names[-1]
        
    def _find_and_replace(self):
        """Find target modules and replace with LoRALinear."""
        for name, module in self.model.named_modules():
            block_idx = self._extract_block_index(name)
            
            if block_idx is None or block_idx < self.peft_config.start_lora_idx:
                continue
                
            if any(name.endswith(t) for t in self.peft_config.target_modules):
                if isinstance(module, nn.Linear):
                    parent, target_name = self._get_parent_module(name)
                    
                    new_module = LoRALinear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        r=self.peft_config.r,
                        lora_alpha=self.peft_config.lora_alpha,
                        lora_dropout=self.peft_config.lora_dropout,
                        merge_weights=self.peft_config.merge_weights,
                    )
                    
                    new_module.weight = module.weight
                    if module.bias is not None:
                        new_module.bias = module.bias
                        
                    setattr(parent, target_name, new_module)
                    new_module.to(module.weight.device)
                    
    def _mark_only_lora_as_trainable(self):
        """Freeze all params except LoRA."""
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name
            
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# Predefined configurations
LORA_CONFIGS = {
    "L1": {"r": 8,  "alpha": 16, "targets": ["attn.qkv"], "start_idx": 0},
    "L2": {"r": 16, "alpha": 32, "targets": ["attn.qkv"], "start_idx": 0},
    "L3": {"r": 16, "alpha": 32, "targets": ["attn.qkv", "attn.proj"], "start_idx": 0},
    "L4": {"r": 16, "alpha": 32, "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"], "start_idx": 6},
    "L5": {"r": 32, "alpha": 64, "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"], "start_idx": 0},
}


def get_lora_config(config_name: str) -> LoRAConfig:
    """Get a predefined LoRA configuration."""
    cfg = LORA_CONFIGS[config_name]
    return LoRAConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        target_modules=cfg["targets"],
        start_lora_idx=cfg["start_idx"],
        lora_dropout=0.1,
        merge_weights=False,
    )
