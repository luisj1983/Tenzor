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
    run_benchmark, compute_statistics, BenchmarkResult, print_result
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

    for M, K, N in sizes:
        # Create tensors
        if device == "cuda":
            A = tz.randn([M, K]).cuda()
            B = tz.randn([K, N]).cuda()
            sync_fn = lambda: None  # Tenzor should auto-sync
        else:
            A = tz.randn([M, K])
            B = tz.randn([K, N])
            sync_fn = None

        # Benchmark function
        def matmul_fn():
            return tz.matmul(A, B)

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

    for M, K, N in sizes:
        # Create tensors
        torch_device = torch.device(device)
        A = torch.randn(M, K, device=torch_device)
        B = torch.randn(K, N, device=torch_device)

        if device == "cuda":
            sync_fn = torch.cuda.synchronize
        else:
            sync_fn = None

        # Benchmark function
        def matmul_fn():
            return torch.matmul(A, B)

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
    """Benchmark batched matrix multiplication with Tenzor.

    Note: Tenzor currently only supports 2D matmul. Batched matmul is not yet implemented.
    This function returns empty results until bmm is added.
    """
    print("  [SKIP] Tenzor batched matmul not yet implemented (requires bmm function)")
    return []


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

        # Skip CUDA if not available
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("CUDA not available, skipping...")
                    continue
            except ImportError:
                pass

        # Standard matmul
        print("\n--- Tenzor MatMul ---")
        tenzor_results = benchmark_tenzor_matmul(config.matmul_sizes, device, config)
        all_results.extend(tenzor_results)

        if config.compare_with_pytorch:
            print("\n--- PyTorch MatMul ---")
            pytorch_results = benchmark_pytorch_matmul(config.matmul_sizes, device, config)
            all_results.extend(pytorch_results)

            # Print comparison
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
