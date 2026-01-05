"""
Benchmark Utilities
===================
Common utilities for running benchmarks and collecting statistics.
"""

# IMPORTANT: Import PyTorch BEFORE Tenzor to avoid MKL library conflicts
# When both libraries use MKL, the first one to load wins and sets up the runtime
try:
    import torch
    import torch.nn
except ImportError:
    pass

import time
import gc
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime
import platform
import subprocess


# =============================================================================
# CUDA Synchronization Helpers
# =============================================================================

def get_tenzor_sync_fn(device: str) -> Optional[Callable]:
    """Get the appropriate synchronization function for Tenzor CUDA operations.

    Returns None for CPU, or a sync function for CUDA.
    For CUDA, falls back to PyTorch's sync if Tenzor doesn't expose one,
    since they typically share the same CUDA context.
    """
    if device != "cuda":
        return None

    try:
        import tenzor as tz
        # Try different possible sync methods Tenzor might have
        if hasattr(tz, 'cuda') and hasattr(tz.cuda, 'synchronize'):
            return tz.cuda.synchronize
        elif hasattr(tz, 'synchronize'):
            return tz.synchronize
        elif hasattr(tz, 'sync'):
            return tz.sync
    except ImportError:
        pass

    # Fallback: use PyTorch's sync if available (they share CUDA context)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass

    # Last resort: warn that timing may be inaccurate
    print("  [WARNING] No CUDA sync function found for Tenzor. GPU timings may be inaccurate.")
    return None


def get_pytorch_sync_fn(device: str) -> Optional[Callable]:
    """Get the appropriate synchronization function for PyTorch CUDA operations."""
    if device != "cuda":
        return None

    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return None


def check_tenzor_cuda_available() -> bool:
    """Check if Tenzor has CUDA support available."""
    try:
        import tenzor as tz
        # Try different possible ways Tenzor might expose CUDA availability
        if hasattr(tz, 'cuda') and hasattr(tz.cuda, 'is_available'):
            return tz.cuda.is_available()
        elif hasattr(tz, 'is_cuda_available'):
            return tz.is_cuda_available()
        elif hasattr(tz, 'cuda_available'):
            return tz.cuda_available()
        else:
            # Try to create a CUDA tensor as a test
            try:
                test = tz.randn([2, 2]).cuda()
                del test
                return True
            except Exception:
                return False
    except ImportError:
        return False


def check_pytorch_cuda_available() -> bool:
    """Check if PyTorch has CUDA support available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def clear_gpu_memory():
    """Clear GPU memory caches for both frameworks."""
    gc.collect()

    # Clear PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # Clear Tenzor cache if available
    try:
        import tenzor as tz
        if hasattr(tz, 'cuda') and hasattr(tz.cuda, 'empty_cache'):
            tz.cuda.empty_cache()
        if hasattr(tz, 'cuda') and hasattr(tz.cuda, 'synchronize'):
            tz.cuda.synchronize()
    except ImportError:
        pass


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""
    name: str
    category: str
    device: str
    framework: str

    # Timing statistics (in milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float

    # Performance metrics
    gflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None

    # Metadata
    iterations: int = 0
    warmup_iterations: int = 0
    parameters: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def speedup_vs(self, other: 'BenchmarkResult') -> float:
        """Calculate speedup compared to another result."""
        return other.mean_ms / self.mean_ms if self.mean_ms > 0 else 0


class Timer:
    """High-precision timer for benchmarking."""

    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


def run_benchmark(
    fn: Callable,
    warmup_iterations: int = 5,
    benchmark_iterations: int = 100,
    sync_fn: Optional[Callable] = None,
) -> List[float]:
    """
    Run a benchmark function multiple times and collect timing data.

    Args:
        fn: Function to benchmark (should take no arguments)
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of timed runs
        sync_fn: Optional synchronization function (e.g., for CUDA)

    Returns:
        List of elapsed times in milliseconds
    """
    # Force garbage collection
    gc.collect()

    # Warmup
    for _ in range(warmup_iterations):
        fn()
        if sync_fn:
            sync_fn()

    # Benchmark
    times = []
    for _ in range(benchmark_iterations):
        with Timer() as t:
            fn()
            if sync_fn:
                sync_fn()
        times.append(t.elapsed_ms)

    return times


def compute_statistics(
    times: List[float],
    name: str,
    category: str,
    device: str,
    framework: str,
    flops: Optional[int] = None,
    bytes_accessed: Optional[int] = None,
    warmup_iterations: int = 5,
    parameters: Dict[str, Any] = None,
) -> BenchmarkResult:
    """
    Compute statistics from benchmark timing data.

    Args:
        times: List of elapsed times in milliseconds
        name: Benchmark name
        category: Benchmark category
        device: Device used (cpu/cuda)
        framework: Framework name (tenzor/pytorch)
        flops: Total floating point operations (for GFLOPS calculation)
        bytes_accessed: Total bytes accessed (for bandwidth calculation)
        warmup_iterations: Number of warmup iterations used
        parameters: Additional parameters to record

    Returns:
        BenchmarkResult with computed statistics
    """
    sorted_times = sorted(times)
    n = len(times)

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if n > 1 else 0

    # Compute percentiles
    p95_idx = int(0.95 * n)
    p99_idx = int(0.99 * n)

    gflops = None
    if flops is not None and mean_ms > 0:
        gflops = (flops / 1e9) / (mean_ms / 1000)

    bandwidth = None
    if bytes_accessed is not None and mean_ms > 0:
        bandwidth = (bytes_accessed / 1e9) / (mean_ms / 1000)

    return BenchmarkResult(
        name=name,
        category=category,
        device=device,
        framework=framework,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min(times),
        max_ms=max(times),
        median_ms=statistics.median(times),
        p95_ms=sorted_times[p95_idx] if p95_idx < n else sorted_times[-1],
        p99_ms=sorted_times[p99_idx] if p99_idx < n else sorted_times[-1],
        gflops=gflops,
        bandwidth_gbps=bandwidth,
        iterations=n,
        warmup_iterations=warmup_iterations,
        parameters=parameters or {},
    )


def get_system_info() -> Dict[str, Any]:
    """Collect system information for benchmark reports."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": platform.os.cpu_count(),
    }

    # Try to get GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(", ")
            info["gpu_name"] = gpu_info[0] if gpu_info else "Unknown"
            info["gpu_memory"] = gpu_info[1] if len(gpu_info) > 1 else "Unknown"
            info["gpu_driver"] = gpu_info[2] if len(gpu_info) > 2 else "Unknown"
    except Exception:
        info["gpu_name"] = "Not available"

    return info


def print_result(result: BenchmarkResult, baseline: Optional[BenchmarkResult] = None):
    """Pretty print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"  {result.name}")
    print(f"  Framework: {result.framework} | Device: {result.device}")
    print(f"{'='*60}")
    print(f"  Mean:      {result.mean_ms:>10.3f} ms")
    print(f"  Std Dev:   {result.std_ms:>10.3f} ms")
    print(f"  Min:       {result.min_ms:>10.3f} ms")
    print(f"  Max:       {result.max_ms:>10.3f} ms")
    print(f"  Median:    {result.median_ms:>10.3f} ms")
    print(f"  P95:       {result.p95_ms:>10.3f} ms")
    print(f"  P99:       {result.p99_ms:>10.3f} ms")

    if result.gflops:
        print(f"  GFLOPS:    {result.gflops:>10.2f}")
    if result.bandwidth_gbps:
        print(f"  Bandwidth: {result.bandwidth_gbps:>10.2f} GB/s")

    if baseline:
        speedup = result.speedup_vs(baseline)
        status = "FASTER" if speedup > 1 else "SLOWER"
        print(f"\n  vs {baseline.framework}: {speedup:.2f}x ({status})")


def save_results(results: List[BenchmarkResult], filepath: str):
    """Save benchmark results to JSON file."""
    data = {
        "system_info": get_system_info(),
        "results": [r.to_dict() for r in results],
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to: {filepath}")


def load_results(filepath: str) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return [BenchmarkResult(**r) for r in data["results"]]
