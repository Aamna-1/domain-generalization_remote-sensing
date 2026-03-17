"""
Dataset loaders for remote sensing semantic segmentation.

Supports:
- ISPRS Potsdam (RGB) and Vaihingen (IRRG)
- LoveDA Urban and Rural
"""

from .isprs import ISPRSDataset
from .loveda import LoveDADataset

__all__ = ['ISPRSDataset', 'LoveDADataset']
