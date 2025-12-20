"""
Benchmark Configuration
=======================
Central configuration for all benchmark parameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Iterations
    warmup_iterations: int = 5
    benchmark_iterations: int = 100

    # Devices to test
    devices: List[str] = field(default_factory=lambda: ["cpu", "cuda"])

    # Matrix sizes for matmul benchmarks
    matmul_sizes: List[tuple] = field(default_factory=lambda: [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ])

    # Batch sizes for various tests
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64, 128])

    # Conv2d configurations (in_ch, out_ch, kernel, stride, padding)
    conv2d_configs: List[Dict] = field(default_factory=lambda: [
        # ResNet-style convolutions
        {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3},
        {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1},
        {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1},
        {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1},
        # 1x1 convolutions
        {"in_channels": 256, "out_channels": 64, "kernel_size": 1, "stride": 1, "padding": 0},
        {"in_channels": 512, "out_channels": 128, "kernel_size": 1, "stride": 1, "padding": 0},
    ])

    # Image sizes for conv benchmarks
    image_sizes: List[tuple] = field(default_factory=lambda: [
        (32, 32),    # CIFAR
        (56, 56),    # ResNet intermediate
        (112, 112),  # ResNet intermediate
        (224, 224),  # ImageNet
    ])

    # Model configurations for training benchmarks
    mlp_configs: List[Dict] = field(default_factory=lambda: [
        {"layers": [784, 256, 10], "name": "small_mlp"},
        {"layers": [784, 512, 256, 10], "name": "medium_mlp"},
        {"layers": [784, 1024, 512, 256, 10], "name": "large_mlp"},
    ])

    # Output settings
    output_dir: str = "results"
    save_raw_data: bool = True
    generate_plots: bool = True

    # Comparison settings
    compare_with_pytorch: bool = True
    compare_with_numpy: bool = False  # For CPU-only ops

    # Precision settings
    dtypes: List[str] = field(default_factory=lambda: ["float32"])

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# Default configuration
DEFAULT_CONFIG = BenchmarkConfig()

# Quick benchmark (for testing)
QUICK_CONFIG = BenchmarkConfig(
    warmup_iterations=2,
    benchmark_iterations=10,
    matmul_sizes=[(512, 512, 512), (1024, 1024, 1024)],
    batch_sizes=[1, 32],
    image_sizes=[(56, 56), (224, 224)],
)

# Full benchmark (for release)
FULL_CONFIG = BenchmarkConfig(
    warmup_iterations=10,
    benchmark_iterations=200,
    devices=["cpu", "cuda"],
)
