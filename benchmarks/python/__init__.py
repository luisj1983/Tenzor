"""
Tenzor Python Benchmark Suite
=============================
Comprehensive benchmarks comparing Tenzor with PyTorch and other frameworks.

Usage:
    python -m benchmarks.python.run_benchmarks --quick
    python -m benchmarks.python.run_benchmarks --category matmul
    python -m benchmarks.python.generate_report results/benchmark.json
"""

from .benchmark_config import BenchmarkConfig, DEFAULT_CONFIG, QUICK_CONFIG, FULL_CONFIG
from .benchmark_utils import (
    BenchmarkResult,
    Timer,
    run_benchmark,
    compute_statistics,
    save_results,
    load_results,
    get_system_info,
    print_result,
)

__all__ = [
    "BenchmarkConfig",
    "DEFAULT_CONFIG",
    "QUICK_CONFIG",
    "FULL_CONFIG",
    "BenchmarkResult",
    "Timer",
    "run_benchmark",
    "compute_statistics",
    "save_results",
    "load_results",
    "get_system_info",
    "print_result",
]
