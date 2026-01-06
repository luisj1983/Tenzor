"""
Normalization Layer Benchmarks
==============================
Compare normalization layer performance between Tenzor and PyTorch.
Critical for transformers (LayerNorm) and CNNs (BatchNorm).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))

from typing import List, Tuple, Dict, Any
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_cuda_available,
    check_pytorch_cuda_available, clear_gpu_memory
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


# LayerNorm configurations (batch, seq_len, hidden_size)
LAYERNORM_CONFIGS = [
    (8, 512, 768, "BERT-base"),
    (16, 128, 768, "BERT-base short"),
    (4, 1024, 768, "GPT-2 Small"),
    # (4, 2048, 768, "GPT-2 long"),  # Too slow on CPU (seq=2048)
    (2, 512, 1024, "Large model"),
    # (4, 512, 4096, "Llama 7B"),  # Too slow on CPU (dim=4096)
    # (2, 2048, 4096, "Llama 7B long"),  # Too slow on CPU (seq=2048, dim=4096)
]

# BatchNorm2d configurations (batch, channels, height, width)
BATCHNORM_CONFIGS = [
    (32, 64, 56, 56, "ResNet stage1"),
    (32, 128, 28, 28, "ResNet stage2"),
    (32, 256, 14, 14, "ResNet stage3"),
    (32, 512, 7, 7, "ResNet stage4"),
    (128, 64, 56, 56, "Large batch"),
]


def benchmark_tenzor_layernorm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor LayerNorm."""
    import tenzor as tz
    tz.initialize()

    results = []

    for batch, seq_len, hidden, name in configs:
        try:
            ln = tz.nn.LayerNorm([hidden])

            if device == "cuda":
                x = tz.randn([batch, seq_len, hidden]).cuda()
            else:
                x = tz.randn([batch, seq_len, hidden])

            x_var = tz.Variable(x, False)

            # Disable gradient computation for inference (like PyTorch's torch.no_grad())
            tz.set_grad_enabled(False)

            def ln_fn():
                return ln.forward(x_var)

            times = run_benchmark(
                ln_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )

            num_elements = batch * seq_len * hidden
            bytes_accessed = num_elements * 4 * 3  # read + write + params

            result = compute_statistics(
                times=times,
                name=name,
                category="layernorm",
                device=device,
                framework="tenzor",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
                parameters={"batch": batch, "seq_len": seq_len, "hidden": hidden},
            )
            results.append(result)
            print_result(result)

            # Re-enable gradients after benchmark
            tz.set_grad_enabled(True)

        except Exception as e:
            tz.set_grad_enabled(True)  # Ensure re-enabled on error
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_pytorch_layernorm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch LayerNorm."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed, skipping")
        return []

    results = []
    torch_device = torch.device(device)

    for batch, seq_len, hidden, name in configs:
        try:
            ln = nn.LayerNorm(hidden).to(torch_device)
            ln.eval()

            x = torch.randn(batch, seq_len, hidden, device=torch_device)
            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            with torch.no_grad():
                def ln_fn():
                    return ln(x)

                times = run_benchmark(
                    ln_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            num_elements = batch * seq_len * hidden
            bytes_accessed = num_elements * 4 * 3

            result = compute_statistics(
                times=times,
                name=name,
                category="layernorm",
                device=device,
                framework="pytorch",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
                parameters={"batch": batch, "seq_len": seq_len, "hidden": hidden},
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_tenzor_batchnorm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
    training: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor BatchNorm2d."""
    import tenzor as tz
    tz.initialize()

    results = []
    mode = "train" if training else "eval"

    for batch, channels, h, w, name in configs:
        try:
            bn = tz.nn.BatchNorm2d(channels)
            if training:
                bn.train()
            else:
                bn.eval()

            if device == "cuda":
                x = tz.randn([batch, channels, h, w]).cuda()
            else:
                x = tz.randn([batch, channels, h, w])

            x_var = tz.Variable(x, training)

            def bn_fn():
                return bn.forward(x_var)

            times = run_benchmark(
                bn_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )

            num_elements = batch * channels * h * w
            bytes_accessed = num_elements * 4 * 3

            result = compute_statistics(
                times=times,
                name=f"{name} ({mode})",
                category="batchnorm",
                device=device,
                framework="tenzor",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
                parameters={"batch": batch, "channels": channels, "h": h, "w": w},
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_pytorch_batchnorm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
    training: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch BatchNorm2d."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)
    mode = "train" if training else "eval"

    for batch, channels, h, w, name in configs:
        try:
            bn = nn.BatchNorm2d(channels).to(torch_device)
            if training:
                bn.train()
            else:
                bn.eval()

            x = torch.randn(batch, channels, h, w, device=torch_device)
            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            if training:
                def bn_fn():
                    return bn(x)
            else:
                with torch.no_grad():
                    def bn_fn():
                        return bn(x)

            times = run_benchmark(
                bn_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            num_elements = batch * channels * h * w
            bytes_accessed = num_elements * 4 * 3

            result = compute_statistics(
                times=times,
                name=f"{name} ({mode})",
                category="batchnorm",
                device=device,
                framework="pytorch",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
                parameters={"batch": batch, "channels": channels, "h": h, "w": w},
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_tenzor_rmsnorm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor RMSNorm (Llama-style normalization)."""
    import tenzor as tz
    tz.initialize()

    results = []

    for batch, seq_len, hidden, name in configs:
        try:
            rms = tz.nn.RMSNorm(hidden)

            if device == "cuda":
                x = tz.randn([batch, seq_len, hidden]).cuda()
            else:
                x = tz.randn([batch, seq_len, hidden])

            x_var = tz.Variable(x, False)

            def rms_fn():
                return rms.forward(x_var)

            times = run_benchmark(
                rms_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )

            num_elements = batch * seq_len * hidden
            bytes_accessed = num_elements * 4 * 2  # RMSNorm is simpler

            result = compute_statistics(
                times=times,
                name=f"RMSNorm {name}",
                category="rmsnorm",
                device=device,
                framework="tenzor",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] RMSNorm {name}: {e}")

    return results


def benchmark_pytorch_rmsnorm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch RMSNorm (manual implementation or from transformers)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)

    # Manual RMSNorm implementation
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x

    for batch, seq_len, hidden, name in configs:
        try:
            rms = RMSNorm(hidden).to(torch_device)
            rms.eval()

            x = torch.randn(batch, seq_len, hidden, device=torch_device)
            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            with torch.no_grad():
                def rms_fn():
                    return rms(x)

                times = run_benchmark(
                    rms_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            num_elements = batch * seq_len * hidden
            bytes_accessed = num_elements * 4 * 2

            result = compute_statistics(
                times=times,
                name=f"RMSNorm {name}",
                category="rmsnorm",
                device=device,
                framework="pytorch",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] RMSNorm {name}: {e}")

    return results


def print_comparison(tenzor_results: List[BenchmarkResult], pytorch_results: List[BenchmarkResult], title: str):
    """Print comparison table."""
    print("\n" + "=" * 75)
    print(f"  COMPARISON: {title}")
    print("=" * 75)
    print(f"{'Configuration':<30} {'Tenzor (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<15}")
    print("-" * 75)

    for tz_r, pt_r in zip(tenzor_results, pytorch_results):
        speedup = tz_r.speedup_vs(pt_r)
        status = "FASTER" if speedup > 1 else "SLOWER"
        print(f"{tz_r.name:<30} {tz_r.mean_ms:<15.3f} {pt_r.mean_ms:<15.3f} {speedup:.2f}x {status}")


def run_normalization_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all normalization benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  NORMALIZATION LAYER BENCHMARKS")
    print("=" * 70)

    for device in config.devices:
        print(f"\n{'='*70}")
        print(f"  Device: {device.upper()}")
        print(f"{'='*70}")

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("CUDA not available, skipping...")
                    continue
            except ImportError:
                pass

        # LayerNorm benchmarks
        print("\n--- Tenzor LayerNorm ---")
        tenzor_ln = benchmark_tenzor_layernorm(LAYERNORM_CONFIGS, device, config)
        all_results.extend(tenzor_ln)

        if config.compare_with_pytorch:
            print("\n--- PyTorch LayerNorm ---")
            pytorch_ln = benchmark_pytorch_layernorm(LAYERNORM_CONFIGS, device, config)
            all_results.extend(pytorch_ln)
            print_comparison(tenzor_ln, pytorch_ln, "LayerNorm")

        # BatchNorm inference benchmarks
        print("\n--- Tenzor BatchNorm2d (Inference) ---")
        tenzor_bn_eval = benchmark_tenzor_batchnorm(BATCHNORM_CONFIGS, device, config, training=False)
        all_results.extend(tenzor_bn_eval)

        if config.compare_with_pytorch:
            print("\n--- PyTorch BatchNorm2d (Inference) ---")
            pytorch_bn_eval = benchmark_pytorch_batchnorm(BATCHNORM_CONFIGS, device, config, training=False)
            all_results.extend(pytorch_bn_eval)
            print_comparison(tenzor_bn_eval, pytorch_bn_eval, "BatchNorm2d (Inference)")

        # BatchNorm training benchmarks
        print("\n--- Tenzor BatchNorm2d (Training) ---")
        tenzor_bn_train = benchmark_tenzor_batchnorm(BATCHNORM_CONFIGS, device, config, training=True)
        all_results.extend(tenzor_bn_train)

        if config.compare_with_pytorch:
            print("\n--- PyTorch BatchNorm2d (Training) ---")
            pytorch_bn_train = benchmark_pytorch_batchnorm(BATCHNORM_CONFIGS, device, config, training=True)
            all_results.extend(pytorch_bn_train)
            print_comparison(tenzor_bn_train, pytorch_bn_train, "BatchNorm2d (Training)")

        # RMSNorm benchmarks
        print("\n--- Tenzor RMSNorm ---")
        tenzor_rms = benchmark_tenzor_rmsnorm(LAYERNORM_CONFIGS, device, config)
        all_results.extend(tenzor_rms)

        if config.compare_with_pytorch:
            print("\n--- PyTorch RMSNorm ---")
            pytorch_rms = benchmark_pytorch_rmsnorm(LAYERNORM_CONFIGS, device, config)
            all_results.extend(pytorch_rms)
            print_comparison(tenzor_rms, pytorch_rms, "RMSNorm")

        # LayerNorm vs RMSNorm comparison (within Tenzor)
        if tenzor_ln and tenzor_rms:
            print("\n" + "=" * 75)
            print("  ANALYSIS: LayerNorm vs RMSNorm (Tenzor)")
            print("=" * 75)
            print(f"{'Configuration':<30} {'LayerNorm (ms)':<15} {'RMSNorm (ms)':<15} {'RMS Speedup':<15}")
            print("-" * 75)

            for ln_r, rms_r in zip(tenzor_ln, tenzor_rms):
                speedup = ln_r.mean_ms / rms_r.mean_ms if rms_r.mean_ms > 0 else 0
                print(f"{ln_r.name:<30} {ln_r.mean_ms:<15.3f} {rms_r.mean_ms:<15.3f} {speedup:.2f}x")

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_normalization_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    import os
    os.makedirs("results", exist_ok=True)
    save_results(results, "results/normalization_benchmarks.json")
