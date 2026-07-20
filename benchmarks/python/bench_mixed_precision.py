"""
Mixed Precision Benchmarks
==========================
Compare mixed precision (FP16/BF16/AMP) performance between Tenzor and PyTorch.
Critical for modern GPU training with Tensor Cores.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))

from typing import List, Tuple, Dict, Any
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_device_available,
    check_pytorch_cuda_available, clear_gpu_memory
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


# Matrix multiplication configurations for precision comparison
MATMUL_CONFIGS = [
    (1024, 1024, 1024, "1K x 1K"),
    (2048, 2048, 2048, "2K x 2K"),
    (4096, 4096, 4096, "4K x 4K"),
    # (8192, 8192, 8192, "8K x 8K"),  # Too slow on CPU
]

# Linear layer configurations (batch, in_features, out_features)
LINEAR_CONFIGS = [
    (32, 4096, 4096, "Transformer FFN"),
    (64, 1024, 4096, "Expansion"),
    (64, 4096, 1024, "Projection"),
    (128, 768, 3072, "BERT FFN"),
]

# Conv2d configurations for precision testing
CONV_CONFIGS = [
    (32, 64, 64, 56, 56, 3, "ResNet stage1"),
    (32, 128, 128, 28, 28, 3, "ResNet stage2"),
    (32, 256, 256, 14, 14, 3, "ResNet stage3"),
]


def benchmark_tenzor_matmul_precision(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor matmul at different precisions."""
    import tenzor as tz
    tz.initialize()

    results = []
    dtypes = ["float32", "float16"]
    if device != "cpu":
        dtypes.append("bfloat16")

    # Skip float16 for large CPU matrices (software emulation is extremely slow)
    skip_cpu_fp16_threshold = 1024  # Skip float16 on CPU for matrices > 1K

    for m, k, n, name in configs:
        for dtype in dtypes:
            # Skip float16 on CPU for large matrices (software emulation is too slow)
            if device == "cpu" and dtype == "float16" and m > skip_cpu_fp16_threshold:
                print(f"  [SKIP] {name} {dtype}: CPU float16 too slow for large matrices")
                continue

            try:
                # tz.randn requires a tenzor.dtype enum, not a string — this
                # was broken identically on every backend including CPU
                # (pre-existing bug found while fixing device handling here).
                tz_dtype = getattr(tz.dtype, dtype)
                if device == "cpu":
                    a = tz.randn([m, k], dtype=tz_dtype)
                    b = tz.randn([k, n], dtype=tz_dtype)
                else:
                    a = tz.randn([m, k], dtype=tz_dtype).to(device)
                    b = tz.randn([k, n], dtype=tz_dtype).to(device)

                def matmul_fn():
                    return tz.matmul(a, b)

                times = run_benchmark(
                    matmul_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                )

                flops = 2 * m * k * n
                result = compute_statistics(
                    times=times,
                    name=f"{name} {dtype}",
                    category="matmul_precision",
                    device=device,
                    framework="tenzor",
                    flops=flops,
                    warmup_iterations=config.warmup_iterations,
                    parameters={"m": m, "k": k, "n": n, "dtype": dtype},
                )
                results.append(result)
                print_result(result)

            except Exception as e:
                print(f"  [SKIP] {name} {dtype}: {e}")

    return results


def benchmark_pytorch_matmul_precision(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch matmul at different precisions."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed, skipping")
        return []

    results = []
    torch_device = torch.device(device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    dtypes = ["float32", "float16"]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtypes.append("bfloat16")

    # Skip float16 for large CPU matrices (software emulation is extremely slow)
    skip_cpu_fp16_threshold = 1024  # Skip float16 on CPU for matrices > 1K

    for m, k, n, name in configs:
        for dtype_name in dtypes:
            # Skip float16 on CPU for large matrices (software emulation is too slow)
            if device == "cpu" and dtype_name == "float16" and m > skip_cpu_fp16_threshold:
                print(f"  [SKIP] {name} {dtype_name}: CPU float16 too slow for large matrices")
                continue

            try:
                torch_dtype = dtype_map[dtype_name]
                a = torch.randn(m, k, device=torch_device, dtype=torch_dtype)
                b = torch.randn(k, n, device=torch_device, dtype=torch_dtype)

                sync_fn = torch.cuda.synchronize if device == "cuda" else None

                with torch.no_grad():
                    def matmul_fn():
                        return torch.matmul(a, b)

                    times = run_benchmark(
                        matmul_fn,
                        warmup_iterations=config.warmup_iterations,
                        benchmark_iterations=config.benchmark_iterations,
                        sync_fn=sync_fn,
                    )

                flops = 2 * m * k * n
                result = compute_statistics(
                    times=times,
                    name=f"{name} {dtype_name}",
                    category="matmul_precision",
                    device=device,
                    framework="pytorch",
                    flops=flops,
                    warmup_iterations=config.warmup_iterations,
                    parameters={"m": m, "k": k, "n": n, "dtype": dtype_name},
                )
                results.append(result)
                print_result(result)

            except Exception as e:
                print(f"  [SKIP] {name} {dtype_name}: {e}")

    return results


def benchmark_tenzor_linear_precision(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor linear layers at different precisions."""
    import tenzor as tz
    tz.initialize()

    results = []
    dtypes = ["float32", "float16"]

    # Skip float16 for large CPU layers (software emulation is extremely slow)
    skip_cpu_fp16_threshold = 1024  # Skip float16 on CPU for layers > 1K features

    for batch, in_feat, out_feat, name in configs:
        for dtype in dtypes:
            # Skip float16 on CPU for large layers
            if device == "cpu" and dtype == "float16" and max(in_feat, out_feat) > skip_cpu_fp16_threshold:
                print(f"  [SKIP] {name} {dtype}: CPU float16 too slow for large layers")
                continue

            try:
                # tz.randn requires a tenzor.dtype enum, not a string, and
                # nn.Linear's constructor has no dtype kwarg at all (dtype is
                # set via .to(dtype) like device) — both broken identically
                # on every backend including CPU (pre-existing bugs found
                # while fixing device handling here).
                tz_dtype = getattr(tz.dtype, dtype)
                linear = tz.nn.Linear(in_feat, out_feat)
                linear.to(tz_dtype)

                if device == "cpu":
                    x = tz.randn([batch, in_feat], dtype=tz_dtype)
                else:
                    linear.to(device)
                    x = tz.randn([batch, in_feat], dtype=tz_dtype).to(device)

                x_var = tz.Variable(x, False)

                def linear_fn():
                    return linear.forward(x_var)

                times = run_benchmark(
                    linear_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                )

                flops = 2 * batch * in_feat * out_feat
                result = compute_statistics(
                    times=times,
                    name=f"{name} {dtype}",
                    category="linear_precision",
                    device=device,
                    framework="tenzor",
                    flops=flops,
                    warmup_iterations=config.warmup_iterations,
                )
                results.append(result)
                print_result(result)

            except Exception as e:
                print(f"  [SKIP] {name} {dtype}: {e}")

    return results


def benchmark_pytorch_linear_precision(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch linear layers at different precisions."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }

    # Skip float16 for large CPU layers (software emulation is extremely slow)
    skip_cpu_fp16_threshold = 1024  # Skip float16 on CPU for layers > 1K features

    for batch, in_feat, out_feat, name in configs:
        for dtype_name in ["float32", "float16"]:
            # Skip float16 on CPU for large layers
            if device == "cpu" and dtype_name == "float16" and max(in_feat, out_feat) > skip_cpu_fp16_threshold:
                print(f"  [SKIP] {name} {dtype_name}: CPU float16 too slow for large layers")
                continue

            try:
                torch_dtype = dtype_map[dtype_name]
                linear = nn.Linear(in_feat, out_feat).to(torch_device).to(torch_dtype)
                linear.eval()

                x = torch.randn(batch, in_feat, device=torch_device, dtype=torch_dtype)
                sync_fn = torch.cuda.synchronize if device == "cuda" else None

                with torch.no_grad():
                    def linear_fn():
                        return linear(x)

                    times = run_benchmark(
                        linear_fn,
                        warmup_iterations=config.warmup_iterations,
                        benchmark_iterations=config.benchmark_iterations,
                        sync_fn=sync_fn,
                    )

                flops = 2 * batch * in_feat * out_feat
                result = compute_statistics(
                    times=times,
                    name=f"{name} {dtype_name}",
                    category="linear_precision",
                    device=device,
                    framework="pytorch",
                    flops=flops,
                    warmup_iterations=config.warmup_iterations,
                )
                results.append(result)
                print_result(result)

            except Exception as e:
                print(f"  [SKIP] {name} {dtype_name}: {e}")

    return results


def benchmark_pytorch_amp(
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch Automatic Mixed Precision training."""
    try:
        import torch
        import torch.nn as nn
        from torch.cuda.amp import autocast, GradScaler
    except ImportError:
        return []

    if device != "cuda":
        print("AMP benchmarks require CUDA, skipping...")
        return []

    results = []
    torch_device = torch.device(device)

    # Simple model for AMP testing
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=1024):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        def forward(self, x):
            return self.layers(x)

    configs = [
        (32, 1024, "Batch=32 Hidden=1024"),
        (64, 2048, "Batch=64 Hidden=2048"),
        (128, 4096, "Batch=128 Hidden=4096"),
    ]

    for batch, hidden, name in configs:
        # FP32 baseline
        try:
            model = SimpleModel(hidden).to(torch_device)
            optimizer = torch.optim.Adam(model.parameters())
            x = torch.randn(batch, hidden, device=torch_device)
            target = torch.randn(batch, hidden, device=torch_device)

            def train_fp32():
                optimizer.zero_grad()
                output = model(x)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                optimizer.step()

            sync_fn = torch.cuda.synchronize

            times = run_benchmark(
                train_fp32,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            result = compute_statistics(
                times=times,
                name=f"{name} FP32",
                category="amp_training",
                device=device,
                framework="pytorch",
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name} FP32: {e}")

        # AMP training
        try:
            model = SimpleModel(hidden).to(torch_device)
            optimizer = torch.optim.Adam(model.parameters())
            scaler = GradScaler()
            x = torch.randn(batch, hidden, device=torch_device)
            target = torch.randn(batch, hidden, device=torch_device)

            def train_amp():
                optimizer.zero_grad()
                with autocast():
                    output = model(x)
                    loss = nn.functional.mse_loss(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            sync_fn = torch.cuda.synchronize

            times = run_benchmark(
                train_amp,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            result = compute_statistics(
                times=times,
                name=f"{name} AMP",
                category="amp_training",
                device=device,
                framework="pytorch_amp",
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name} AMP: {e}")

    return results


def print_precision_comparison(results: List[BenchmarkResult], title: str):
    """Print precision comparison table."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

    # Group by configuration name (without dtype)
    grouped = {}
    for r in results:
        base_name = r.name.rsplit(" ", 1)[0]
        dtype = r.name.rsplit(" ", 1)[1]
        key = (base_name, r.framework)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][dtype] = r

    print(f"{'Configuration':<25} {'Framework':<10} {'FP32 (ms)':<12} {'FP16 (ms)':<12} {'Speedup':<10}")
    print("-" * 80)

    for (name, framework), dtypes in grouped.items():
        fp32 = dtypes.get("float32")
        fp16 = dtypes.get("float16")

        if fp32 and fp16:
            speedup = fp32.mean_ms / fp16.mean_ms if fp16.mean_ms > 0 else 0
            print(f"{name:<25} {framework:<10} {fp32.mean_ms:<12.3f} {fp16.mean_ms:<12.3f} {speedup:.2f}x")


def run_mixed_precision_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all mixed precision benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  MIXED PRECISION BENCHMARKS")
    print("=" * 70)

    for device in config.devices:
        print(f"\n{'='*70}")
        print(f"  Device: {device.upper()}")
        print(f"{'='*70}")

        # PyTorch has no rocm/vulkan/oneapi device type of its own, so the
        # comparison only ever runs for device in ("cpu", "cuda").
        tenzor_avail = check_tenzor_device_available(device)
        pytorch_comparable = device in ("cpu", "cuda")
        pytorch_avail = (check_pytorch_cuda_available() if device == "cuda" else pytorch_comparable) if pytorch_comparable else False

        if not tenzor_avail and not (config.compare_with_pytorch and pytorch_avail):
            print(f"{device} not available for either framework, skipping...")
            continue
        if not tenzor_avail:
            print(f"  [WARNING] Tenzor {device} not available")
        run_pytorch = config.compare_with_pytorch and pytorch_avail
        if config.compare_with_pytorch and not pytorch_avail:
            reason = "not a PyTorch device" if not pytorch_comparable else f"PyTorch {device} not available"
            print(f"  [WARNING] {reason}")

        # Matmul precision benchmarks
        print("\n--- Tenzor Matmul (Multi-Precision) ---")
        tenzor_matmul = benchmark_tenzor_matmul_precision(MATMUL_CONFIGS, device, config) if tenzor_avail else []
        all_results.extend(tenzor_matmul)

        if run_pytorch:
            print("\n--- PyTorch Matmul (Multi-Precision) ---")
            pytorch_matmul = benchmark_pytorch_matmul_precision(MATMUL_CONFIGS, device, config)
            all_results.extend(pytorch_matmul)

            print_precision_comparison(tenzor_matmul + pytorch_matmul, "Matmul Precision Comparison")

        # Linear layer precision
        print("\n--- Tenzor Linear (Multi-Precision) ---")
        tenzor_linear = benchmark_tenzor_linear_precision(LINEAR_CONFIGS, device, config) if tenzor_avail else []
        all_results.extend(tenzor_linear)

        if run_pytorch:
            print("\n--- PyTorch Linear (Multi-Precision) ---")
            pytorch_linear = benchmark_pytorch_linear_precision(LINEAR_CONFIGS, device, config)
            all_results.extend(pytorch_linear)

            print_precision_comparison(tenzor_linear + pytorch_linear, "Linear Layer Precision Comparison")

        # AMP training (PyTorch only for comparison baseline)
        if device == "cuda" and config.compare_with_pytorch:
            print("\n--- PyTorch AMP Training ---")
            amp_results = benchmark_pytorch_amp(device, config)
            all_results.extend(amp_results)

            # Print AMP comparison
            print("\n" + "=" * 70)
            print("  AMP vs FP32 Training Comparison")
            print("=" * 70)
            print(f"{'Configuration':<35} {'FP32 (ms)':<12} {'AMP (ms)':<12} {'Speedup':<10}")
            print("-" * 70)

            fp32_results = [r for r in amp_results if "FP32" in r.name]
            amp_only = [r for r in amp_results if "AMP" in r.name]

            for fp32_r, amp_r in zip(fp32_results, amp_only):
                speedup = fp32_r.mean_ms / amp_r.mean_ms if amp_r.mean_ms > 0 else 0
                name = fp32_r.name.replace(" FP32", "")
                print(f"{name:<35} {fp32_r.mean_ms:<12.3f} {amp_r.mean_ms:<12.3f} {speedup:.2f}x")

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_mixed_precision_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    import os
    os.makedirs("results", exist_ok=True)
    save_results(results, "results/mixed_precision_benchmarks.json")
