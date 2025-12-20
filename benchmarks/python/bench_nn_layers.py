"""
Neural Network Layer Benchmarks
===============================
Compare various neural network layer performance between Tenzor and PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from typing import List, Dict
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


def benchmark_linear_layers(device: str, config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark Linear (fully connected) layers."""
    import tenzor as tz
    tz.initialize()

    results = []
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

        def forward_fn():
            return layer(x_var)

        times = run_benchmark(forward_fn, config.warmup_iterations, config.benchmark_iterations)
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

                torch_device = torch.device(device)
                x_pt = torch.randn(batch, in_f, device=torch_device)
                layer_pt = nn.Linear(in_f, out_f).to(torch_device)
                sync_fn = torch.cuda.synchronize if device == "cuda" else None

                def forward_pt():
                    return layer_pt(x_pt)

                times_pt = run_benchmark(forward_pt, config.warmup_iterations, config.benchmark_iterations, sync_fn)

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

            def act_fn():
                return activation(x_var)

            times = run_benchmark(act_fn, config.warmup_iterations, config.benchmark_iterations)

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

            activations_pt = {
                "ReLU": nn.ReLU(),
                "GELU": nn.GELU(),
                "Sigmoid": nn.Sigmoid(),
                "Tanh": nn.Tanh(),
            }

            for name, activation in activations_pt.items():
                for batch, features in sizes:
                    torch_device = torch.device(device)
                    x_pt = torch.randn(batch, features, device=torch_device)
                    activation = activation.to(torch_device)
                    sync_fn = torch.cuda.synchronize if device == "cuda" else None

                    def act_pt():
                        return activation(x_pt)

                    times_pt = run_benchmark(act_pt, config.warmup_iterations, config.benchmark_iterations, sync_fn)

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
            bn.cuda()
        else:
            x = tz.randn([batch, ch, h, w])
            bn = tz.nn.BatchNorm2d(ch)

        # Tenzor modules expect Variable input
        x_var = tz.Variable(x, requires_grad=False)

        def bn_fn():
            return bn(x_var)

        times = run_benchmark(bn_fn, config.warmup_iterations, config.benchmark_iterations)

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

                torch_device = torch.device(device)
                x_pt = torch.randn(batch, ch, h, w, device=torch_device)
                bn_pt = nn.BatchNorm2d(ch).to(torch_device)
                sync_fn = torch.cuda.synchronize if device == "cuda" else None

                def bn_pt_fn():
                    return bn_pt(x_pt)

                times_pt = run_benchmark(bn_pt_fn, config.warmup_iterations, config.benchmark_iterations, sync_fn)

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

        def ln_fn():
            return ln(x_var)

        times = run_benchmark(ln_fn, config.warmup_iterations, config.benchmark_iterations)

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

    return results


def benchmark_pooling_layers(device: str, config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark pooling layers."""
    import tenzor as tz
    tz.initialize()

    results = []

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

        def pool_fn():
            return pool(x_var)

        times = run_benchmark(pool_fn, config.warmup_iterations, config.benchmark_iterations)

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

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("CUDA not available, skipping...")
                    continue
            except ImportError:
                pass

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
