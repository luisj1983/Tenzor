"""
Matrix Multiplication Benchmarks
================================
Compare matrix multiplication performance between Tenzor and PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))

from typing import List, Tuple
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_device_available,
    check_pytorch_cuda_available, clear_gpu_memory
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


def calculate_matmul_flops(M: int, N: int, K: int) -> int:
    """Calculate FLOPs for matrix multiplication C = A @ B.
    A is M x K, B is K x N, C is M x N.
    Each output element requires K multiplications and K-1 additions.
    """
    return 2 * M * N * K


def benchmark_tenzor_matmul(
    sizes: List[Tuple[int, int, int]],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor matrix multiplication."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    for M, K, N in sizes:
        # Create tensors on whichever device was requested (was hardcoded to
        # only move to GPU for device == "cuda", silently leaving
        # rocm/vulkan/oneapi tensors on CPU).
        if device == "cpu":
            A = tz.randn([M, K])
            B = tz.randn([K, N])
        else:
            # Tensor.to() / tz.randn(device=...) accept a device name string
            # directly (pybind11 implicit Device conversion) — no need to
            # construct a Device object here.
            A = tz.randn([M, K]).to(device)
            B = tz.randn([K, N]).to(device)

        # Benchmark function - use default args to capture current values
        def matmul_fn(a=A, b=B):
            return tz.matmul(a, b)

        # Run benchmark
        times = run_benchmark(
            matmul_fn,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
            sync_fn=sync_fn,
        )

        # Compute statistics
        flops = calculate_matmul_flops(M, N, K)
        result = compute_statistics(
            times=times,
            name=f"MatMul {M}x{K} @ {K}x{N}",
            category="matmul",
            device=device,
            framework="tenzor",
            flops=flops,
            warmup_iterations=config.warmup_iterations,
            parameters={"M": M, "K": K, "N": N},
        )
        results.append(result)
        print_result(result)

    return results


def benchmark_pytorch_matmul(
    sizes: List[Tuple[int, int, int]],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch matrix multiplication."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed, skipping PyTorch benchmarks")
        return []

    results = []
    sync_fn = get_pytorch_sync_fn(device)

    for M, K, N in sizes:
        # Create tensors
        torch_device = torch.device(device)
        A = torch.randn(M, K, device=torch_device)
        B = torch.randn(K, N, device=torch_device)

        # Benchmark function - use default args to capture current values
        def matmul_fn(a=A, b=B):
            return torch.matmul(a, b)

        # Run benchmark
        times = run_benchmark(
            matmul_fn,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
            sync_fn=sync_fn,
        )

        # Compute statistics
        flops = calculate_matmul_flops(M, N, K)
        result = compute_statistics(
            times=times,
            name=f"MatMul {M}x{K} @ {K}x{N}",
            category="matmul",
            device=device,
            framework="pytorch",
            flops=flops,
            warmup_iterations=config.warmup_iterations,
            parameters={"M": M, "K": K, "N": N},
        )
        results.append(result)
        print_result(result)

    return results


def benchmark_batched_matmul_tenzor(
    batch_sizes: List[int],
    matrix_size: Tuple[int, int, int],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark batched matrix multiplication with Tenzor."""
    import tenzor as tz

    # Check if bmm is available
    if not hasattr(tz, 'bmm'):
        print("  [SKIP] Tenzor batched matmul not yet implemented (requires bmm function)")
        return []

    M, K, N = matrix_size
    results = []
    sync_fn = get_tenzor_sync_fn(device)

    for batch in batch_sizes:
        flops = batch * calculate_matmul_flops(M, N, K)

        A = tz.randn([batch, M, K], device=device)
        B = tz.randn([batch, K, N], device=device)

        def bmm_fn(a=A, b=B):
            return tz.bmm(a, b)

        times = run_benchmark(
            bmm_fn,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
            sync_fn=sync_fn,
        )

        result = compute_statistics(
            times=times,
            name=f"BatchedMatMul B={batch} {M}x{K} @ {K}x{N}",
            category="batched_matmul",
            device=device,
            framework="tenzor",
            flops=flops,
            warmup_iterations=config.warmup_iterations,
            parameters={"batch": batch, "M": M, "K": K, "N": N},
        )
        results.append(result)
        print_result(result)

    return results


def benchmark_batched_matmul_pytorch(
    batch_sizes: List[int],
    matrix_size: Tuple[int, int, int],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark batched matrix multiplication with PyTorch."""
    try:
        import torch
    except ImportError:
        return []

    M, K, N = matrix_size
    results = []

    for batch in batch_sizes:
        torch_device = torch.device(device)
        A = torch.randn(batch, M, K, device=torch_device)
        B = torch.randn(batch, K, N, device=torch_device)

        sync_fn = torch.cuda.synchronize if device == "cuda" else None

        def bmm_fn():
            return torch.bmm(A, B)

        times = run_benchmark(
            bmm_fn,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
            sync_fn=sync_fn,
        )

        flops = batch * calculate_matmul_flops(M, N, K)
        result = compute_statistics(
            times=times,
            name=f"BatchedMatMul B={batch} {M}x{K} @ {K}x{N}",
            category="batched_matmul",
            device=device,
            framework="pytorch",
            flops=flops,
            warmup_iterations=config.warmup_iterations,
            parameters={"batch": batch, "M": M, "K": K, "N": N},
        )
        results.append(result)
        print_result(result)

    return results


def run_matmul_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all matrix multiplication benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  MATRIX MULTIPLICATION BENCHMARKS")
    print("=" * 70)

    for device in config.devices:
        print(f"\n{'='*70}")
        print(f"  Device: {device.upper()}")
        print(f"{'='*70}")

        # Skip this device if Tenzor doesn't have it available. PyTorch has
        # no rocm/vulkan/oneapi device type of its own (a ROCm PyTorch build
        # still uses "cuda" as its device string), so the PyTorch comparison
        # only ever runs for device in ("cpu", "cuda").
        tenzor_avail = check_tenzor_device_available(device)
        pytorch_comparable = device in ("cpu", "cuda")
        pytorch_avail = pytorch_comparable and check_pytorch_cuda_available() if device == "cuda" else pytorch_comparable

        if not tenzor_avail and not (config.compare_with_pytorch and pytorch_avail):
            print(f"{device} not available for either framework, skipping...")
            continue

        if not tenzor_avail:
            print(f"  [WARNING] Tenzor {device} not available, will skip Tenzor {device} benchmarks")
        if config.compare_with_pytorch and not pytorch_avail:
            reason = "not a PyTorch device" if not pytorch_comparable else f"PyTorch {device} not available"
            print(f"  [WARNING] {reason}, will skip PyTorch {device} benchmarks")

        # Clear GPU memory before starting
        if device != "cpu":
            clear_gpu_memory()

        # Standard matmul
        print("\n--- Tenzor MatMul ---")
        tenzor_results = []
        if tenzor_avail:
            tenzor_results = benchmark_tenzor_matmul(config.matmul_sizes, device, config)
            all_results.extend(tenzor_results)
        else:
            print(f"  [SKIP] Tenzor {device} not available")

        if config.compare_with_pytorch:
            print("\n--- PyTorch MatMul ---")
            pytorch_results = []
            if pytorch_avail:
                pytorch_results = benchmark_pytorch_matmul(config.matmul_sizes, device, config)
                all_results.extend(pytorch_results)
            else:
                print(f"  [SKIP] PyTorch {device} not available")

            # Print comparison (only if both have results)
            if tenzor_results and pytorch_results:
                print("\n--- Comparison Summary ---")
                for tz_r, pt_r in zip(tenzor_results, pytorch_results):
                    speedup = tz_r.speedup_vs(pt_r)
                    status = "FASTER" if speedup > 1 else "SLOWER"
                    print(f"  {tz_r.name}: Tenzor is {speedup:.2f}x {status} than PyTorch")

        # Batched matmul
        print("\n--- Tenzor Batched MatMul ---")
        if tenzor_avail:
            batch_tz = benchmark_batched_matmul_tenzor(
                config.batch_sizes, (256, 256, 256), device, config
            )
            all_results.extend(batch_tz)
        else:
            print(f"  [SKIP] Tenzor {device} not available")

        if config.compare_with_pytorch:
            print("\n--- PyTorch Batched MatMul ---")
            if pytorch_avail:
                batch_pt = benchmark_batched_matmul_pytorch(
                    config.batch_sizes, (256, 256, 256), device, config
                )
                all_results.extend(batch_pt)
            else:
                print(f"  [SKIP] PyTorch {device} not available")

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_matmul_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    save_results(results, "results/matmul_benchmarks.json")
