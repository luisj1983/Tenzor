"""
Tenzor: High-performance neural network and tensor library
"""

from .tenzor_core import *

__version__ = "1.0.0"

__all__ = [
    "Tensor",
    "Variable",
    "Device",
    "dtype",
    "zeros",
    "ones",
    "randn",
    "matmul",
    "nn",
    "optim",
]
