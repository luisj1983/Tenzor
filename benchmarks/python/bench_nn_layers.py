"""
Neural Network Layer Benchmarks
===============================
Compare various neural network layer performance between Tenzor and PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))

# IMPORTANT: Import PyTorch BEFORE Tenzor to avoid MKL library conflicts
# When both libraries use MKL, the first one to load wins and sets up the runtime
try:
    import torch
    import torch.nn
except ImportError:
    pass

from typing import List, Dict
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_cuda_available,
    check_pytorch_cuda_available, clear_gpu_memory
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


def benchmark_linear_layers(device: str, config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark Linear (fully connected) layers."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)
    layer_configs = [
        {"in_features": 784, "out_features": 256, "batch": 32},
        {"in_features": 256, "out_features": 128, "batch": 32},
        {"in_features": 1024, "out_features": 1024, "batch": 64},
        {"in_features": 4096, "out_features": 4096, "batch": 16},
        {"in_features": 768, "out_features": 3072, "batch": 32},   # BERT-style
        {"in_features": 3072, "out_features": 768, "batch": 32},   # BERT-style
    ]

    print("\n--- Linear Layer Benchmarks ---")

    for cfg in layer_configs:
        in_f, out_f, batch = cfg["in_features"], cfg["out_features"], cfg["batch"]

        # Tenzor
        if device == "cuda":
            x = tz.randn([batch, in_f]).cuda()
            layer = tz.nn.Linear(in_f, out_f)
            layer.cuda()
        else:
            x = tz.randn([batch, in_f])
            layer = tz.nn.Linear(in_f, out_f)

        # Tenzor modules expect Variable input
        x_var = tz.Variable(x, requires_grad=False)

        # Disable gradient computation for inference (like PyTorch's torch.no_grad())
        tz.set_grad_enabled(False)

        # Use default args to capture current loop values (fixes closure bug)
        def forward_fn(l=layer, xv=x_var):
            return l(xv)

        times = run_benchmark(forward_fn, config.warmup_iterations, config.benchmark_iterations, sync_fn)

        # Re-enable gradients
        tz.set_grad_enabled(True)
        flops = 2 * batch * in_f * out_f

        result = compute_statistics(
            times=times,
            name=f"Linear {in_f}->{out_f} B={batch}",
            category="linear",
            device=device,
            framework="tenzor",
            flops=flops,
            warmup_iterations=config.warmup_iterations,
            parameters=cfg,
        )
        results.append(result)
        print_result(result)

        # PyTorch comparison
        if config.compare_with_pytorch:
            try:
                import torch
                import torch.nn as nn

                pt_sync_fn = get_pytorch_sync_fn(device)
                torch_device = torch.device(device)
                x_pt = torch.randn(batch, in_f, device=torch_device)
                layer_pt = nn.Linear(in_f, out_f).to(torch_device)
                layer_pt.eval()

                # Use torch.no_grad() for inference benchmarking
                with torch.no_grad():
                    # Use default args to capture current loop values (fixes closure bug)
                    def forward_pt(l=layer_pt, x=x_pt):
                        return l(x)

                    times_pt = run_benchmark(forward_pt, config.warmup_iterations, config.benchmark_iterations, pt_sync_fn)

                result_pt = compute_statistics(
                    times=times_pt,
                    name=f"Linear {in_f}->{out_f} B={batch}",
                    category="linear",
                    device=device,
                    framework="pytorch",
                    flops=flops,
                    warmup_iterations=config.warmup_iterations,
                    parameters=cfg,
                )
                results.append(result_pt)
                print_result(result_pt, baseline=result)
            except ImportError:
                pass

    return results


def benchmark_activation_functions(device: str, config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark activation functions."""
    import tenzor as tz
    tz.initialize()

    results = []
    sizes = [(32, 1024), (64, 4096), (128, 16384)]
    sync_fn = get_tenzor_sync_fn(device)

    activations_tz = {
        "ReLU": tz.nn.ReLU(),
        "GELU": tz.nn.GELU(),
        "Sigmoid": tz.nn.Sigmoid(),
        "Tanh": tz.nn.Tanh(),
    }

    print("\n--- Activation Function Benchmarks ---")

    for name, activation in activations_tz.items():
        for batch, features in sizes:
            if device == "cuda":
                x = tz.randn([batch, features]).cuda()
            else:
                x = tz.randn([batch, features])

            # Tenzor modules expect Variable input
            x_var = tz.Variable(x, requires_grad=False)

            # Use default args to capture current loop values (fixes closure bug)
            def act_fn(act=activation, xv=x_var):
                return act(xv)

            times = run_benchmark(act_fn, config.warmup_iterations, config.benchmark_iterations, sync_fn)

            result = compute_statistics(
                times=times,
                name=f"{name} {batch}x{features}",
                category="activation",
                device=device,
                framework="tenzor",
                bytes_accessed=batch * features * 4 * 2,  # read + write
                warmup_iterations=config.warmup_iterations,
                parameters={"activation": name, "batch": batch, "features": features},
            )
            results.append(result)
            print_result(result)

    # PyTorch comparison
    if config.compare_with_pytorch:
        try:
            import torch
            import torch.nn as nn

            pt_sync_fn = get_pytorch_sync_fn(device)
            torch_device = torch.device(device)

            activations_pt = {
                "ReLU": nn.ReLU(),
                "GELU": nn.GELU(),
                "Sigmoid": nn.Sigmoid(),
                "Tanh": nn.Tanh(),
            }

            for name, activation in activations_pt.items():
                activation = activation.to(torch_device)
                for batch, features in sizes:
                    x_pt = torch.randn(batch, features, device=torch_device)

                    # Use default args to capture current loop values (fixes closure bug)
                    def act_pt(act=activation, x=x_pt):
                        return act(x)

                    times_pt = run_benchmark(act_pt, config.warmup_iterations, config.benchmark_iterations, pt_sync_fn)

                    result_pt = compute_statistics(
                        times=times_pt,
                        name=f"{name} {batch}x{features}",
                        category="activation",
                        device=device,
                        framework="pytorch",
                        bytes_accessed=batch * features * 4 * 2,
                        warmup_iterations=config.warmup_iterations,
                        parameters={"activation": name, "batch": batch, "features": features},
                    )
                    results.append(result_pt)
        except ImportError:
            pass

    return results


def benchmark_normalization_layers(device: str, config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark normalization layers."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    # BatchNorm2d configs
    bn_configs = [
        {"channels": 64, "h": 56, "w": 56, "batch": 32},
        {"channels": 128, "h": 28, "w": 28, "batch": 32},
        {"channels": 256, "h": 14, "w": 14, "batch": 32},
        {"channels": 512, "h": 7, "w": 7, "batch": 32},
    ]

    print("\n--- BatchNorm2d Benchmarks ---")

    for cfg in bn_configs:
        ch, h, w, batch = cfg["channels"], cfg["h"], cfg["w"], cfg["batch"]

        if device == "cuda":
            x = tz.randn([batch, ch, h, w]).cuda()
            bn = tz.nn.BatchNorm2d(ch)
            bn.eval()  # Set eval mode before CUDA transfer
            bn.cuda()
        else:
            x = tz.randn([batch, ch, h, w])
            bn = tz.nn.BatchNorm2d(ch)
            bn.eval()

        # Tenzor modules expect Variable input
        x_var = tz.Variable(x, requires_grad=False)

        # Disable gradient computation for inference
        tz.set_grad_enabled(False)

        # Use default args to capture current loop values (fixes closure bug)
        def bn_fn(layer=bn, xv=x_var):
            return layer(xv)

        times = run_benchmark(bn_fn, config.warmup_iterations, config.benchmark_iterations, sync_fn)

        # Re-enable gradients
        tz.set_grad_enabled(True)

        result = compute_statistics(
            times=times,
            name=f"BatchNorm2d C={ch} {h}x{w} B={batch}",
            category="batchnorm",
            device=device,
            framework="tenzor",
            warmup_iterations=config.warmup_iterations,
            parameters=cfg,
        )
        results.append(result)
        print_result(result)

        # PyTorch
        if config.compare_with_pytorch:
            try:
                import torch
                import torch.nn as nn

                pt_sync_fn = get_pytorch_sync_fn(device)
                torch_device = torch.device(device)
                x_pt = torch.randn(batch, ch, h, w, device=torch_device)
                bn_pt = nn.BatchNorm2d(ch).to(torch_device)

                # Use default args to capture current loop values (fixes closure bug)
                def bn_pt_fn(layer=bn_pt, x=x_pt):
                    return layer(x)

                times_pt = run_benchmark(bn_pt_fn, config.warmup_iterations, config.benchmark_iterations, pt_sync_fn)

                result_pt = compute_statistics(
                    times=times_pt,
                    name=f"BatchNorm2d C={ch} {h}x{w} B={batch}",
                    category="batchnorm",
                    device=device,
                    framework="pytorch",
                    warmup_iterations=config.warmup_iterations,
                    parameters=cfg,
                )
                results.append(result_pt)
                print_result(result_pt, baseline=result)
            except ImportError:
                pass

    # LayerNorm configs
    ln_configs = [
        {"normalized_shape": [768], "batch": 32, "seq_len": 512},   # BERT
        {"normalized_shape": [1024], "batch": 32, "seq_len": 512},  # Large
    ]

    print("\n--- LayerNorm Benchmarks ---")

    # IMPORTANT: Disable gradient tracking for inference benchmarking
    # This allows the fast SIMD path to be used
    prev_grad_state = tz.is_grad_enabled()
    tz.set_grad_enabled(False)

    for cfg in ln_configs:
        norm_shape = cfg["normalized_shape"]
        batch = cfg["batch"]
        seq_len = cfg["seq_len"]

        if device == "cuda":
            x = tz.randn([batch, seq_len, norm_shape[0]]).cuda()
            ln = tz.nn.LayerNorm(norm_shape)
            ln.cuda()
        else:
            x = tz.randn([batch, seq_len, norm_shape[0]])
            ln = tz.nn.LayerNorm(norm_shape)

        # Tenzor modules expect Variable input
        x_var = tz.Variable(x, requires_grad=False)

        # Use default args to capture current loop values (fixes closure bug)
        def ln_fn(layer=ln, xv=x_var):
            return layer(xv)

        times = run_benchmark(ln_fn, config.warmup_iterations, config.benchmark_iterations, sync_fn)

        result = compute_statistics(
            times=times,
            name=f"LayerNorm {norm_shape} B={batch} S={seq_len}",
            category="layernorm",
            device=device,
            framework="tenzor",
            warmup_iterations=config.warmup_iterations,
            parameters=cfg,
        )
        results.append(result)
        print_result(result)

    # Restore gradient state
    tz.set_grad_enabled(prev_grad_state)

    return results


def benchmark_pooling_layers(device: str, config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark pooling layers."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    pool_configs = [
        {"channels": 64, "h": 112, "w": 112, "kernel": 3, "stride": 2, "batch": 32},
        {"channels": 128, "h": 56, "w": 56, "kernel": 2, "stride": 2, "batch": 32},
        {"channels": 256, "h": 28, "w": 28, "kernel": 2, "stride": 2, "batch": 32},
    ]

    print("\n--- MaxPool2d Benchmarks ---")

    for cfg in pool_configs:
        ch, h, w = cfg["channels"], cfg["h"], cfg["w"]
        kernel, stride, batch = cfg["kernel"], cfg["stride"], cfg["batch"]

        if device == "cuda":
            x = tz.randn([batch, ch, h, w]).cuda()
            pool = tz.nn.MaxPool2d(kernel, stride=stride)
        else:
            x = tz.randn([batch, ch, h, w])
            pool = tz.nn.MaxPool2d(kernel, stride=stride)

        # Tenzor modules expect Variable input
        x_var = tz.Variable(x, requires_grad=False)

        # Use default args to capture current loop values (fixes closure bug)
        def pool_fn(p=pool, xv=x_var):
            return p(xv)

        times = run_benchmark(pool_fn, config.warmup_iterations, config.benchmark_iterations, sync_fn)

        result = compute_statistics(
            times=times,
            name=f"MaxPool2d C={ch} {h}x{w} k={kernel}",
            category="maxpool",
            device=device,
            framework="tenzor",
            warmup_iterations=config.warmup_iterations,
            parameters=cfg,
        )
        results.append(result)
        print_result(result)

    return results


def run_nn_layer_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all neural network layer benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  NEURAL NETWORK LAYER BENCHMARKS")
    print("=" * 70)

    for device in config.devices:
        print(f"\n{'='*70}")
        print(f"  Device: {device.upper()}")
        print(f"{'='*70}")

        # Check CUDA availability for both frameworks
        if device == "cuda":
            tenzor_cuda = check_tenzor_cuda_available()
            pytorch_cuda = check_pytorch_cuda_available()

            if not tenzor_cuda and not pytorch_cuda:
                print("CUDA not available for either framework, skipping...")
                continue

            if not tenzor_cuda:
                print("  [WARNING] Tenzor CUDA not available")
            if not pytorch_cuda and config.compare_with_pytorch:
                print("  [WARNING] PyTorch CUDA not available")

        # Clear GPU memory before starting
        if device == "cuda":
            clear_gpu_memory()

        all_results.extend(benchmark_linear_layers(device, config))
        all_results.extend(benchmark_activation_functions(device, config))
        all_results.extend(benchmark_normalization_layers(device, config))
        all_results.extend(benchmark_pooling_layers(device, config))

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_nn_layer_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    save_results(results, "results/nn_layer_benchmarks.json")
