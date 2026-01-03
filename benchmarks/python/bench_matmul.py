"""
Matrix Multiplication Benchmarks
================================
Compare matrix multiplication performance between Tenzor and PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from typing import List, Tuple
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_cuda_available,
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
        # Create tensors
        if device == "cuda":
            A = tz.randn([M, K]).cuda()
            B = tz.randn([K, N]).cuda()
        else:
            A = tz.randn([M, K])
            B = tz.randn([K, N])

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

        tz_device = tz.Device.cuda(0) if device == "cuda" else tz.Device.cpu()
        A = tz.randn([batch, M, K], device=tz_device)
        B = tz.randn([batch, K, N], device=tz_device)

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

        # Skip CUDA if not available for either framework
        if device == "cuda":
            tenzor_cuda = check_tenzor_cuda_available()
            pytorch_cuda = check_pytorch_cuda_available()

            if not tenzor_cuda and not pytorch_cuda:
                print("CUDA not available for either framework, skipping...")
                continue

            if not tenzor_cuda:
                print("  [WARNING] Tenzor CUDA not available, will skip Tenzor CUDA benchmarks")
            if not pytorch_cuda and config.compare_with_pytorch:
                print("  [WARNING] PyTorch CUDA not available, will skip PyTorch CUDA benchmarks")

        # Clear GPU memory before starting
        if device == "cuda":
            clear_gpu_memory()

        # Standard matmul
        print("\n--- Tenzor MatMul ---")
        if device != "cuda" or check_tenzor_cuda_available():
            tenzor_results = benchmark_tenzor_matmul(config.matmul_sizes, device, config)
            all_results.extend(tenzor_results)
        else:
            tenzor_results = []
            print("  [SKIP] Tenzor CUDA not available")

        if config.compare_with_pytorch:
            print("\n--- PyTorch MatMul ---")
            if device != "cuda" or check_pytorch_cuda_available():
                pytorch_results = benchmark_pytorch_matmul(config.matmul_sizes, device, config)
                all_results.extend(pytorch_results)
            else:
                pytorch_results = []
                print("  [SKIP] PyTorch CUDA not available")

            # Print comparison (only if both have results)
            if tenzor_results and pytorch_results:
                print("\n--- Comparison Summary ---")
                for tz_r, pt_r in zip(tenzor_results, pytorch_results):
                    speedup = tz_r.speedup_vs(pt_r)
                    status = "FASTER" if speedup > 1 else "SLOWER"
                    print(f"  {tz_r.name}: Tenzor is {speedup:.2f}x {status} than PyTorch")

        # Batched matmul
        print("\n--- Tenzor Batched MatMul ---")
        batch_tz = benchmark_batched_matmul_tenzor(
            config.batch_sizes, (256, 256, 256), device, config
        )
        all_results.extend(batch_tz)

        if config.compare_with_pytorch:
            print("\n--- PyTorch Batched MatMul ---")
            batch_pt = benchmark_batched_matmul_pytorch(
                config.batch_sizes, (256, 256, 256), device, config
            )
            all_results.extend(batch_pt)

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_matmul_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    save_results(results, "results/matmul_benchmarks.json")
