"""
Convolution Benchmarks
======================
Compare Conv2d performance between Tenzor and PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))

from typing import List, Dict, Tuple
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_device_available,
    check_pytorch_cuda_available, clear_gpu_memory
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


def calculate_conv2d_flops(
    batch: int,
    in_channels: int,
    out_channels: int,
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Calculate FLOPs for Conv2d operation."""
    output_h = (input_h + 2 * padding - kernel_h) // stride + 1
    output_w = (input_w + 2 * padding - kernel_w) // stride + 1

    # Each output element requires kernel_h * kernel_w * in_channels MACs
    flops_per_output = 2 * kernel_h * kernel_w * in_channels
    total_outputs = batch * out_channels * output_h * output_w

    return flops_per_output * total_outputs


def benchmark_tenzor_conv2d(
    conv_configs: List[Dict],
    image_sizes: List[Tuple[int, int]],
    batch_size: int,
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor Conv2d."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    for conv_cfg in conv_configs:
        for H, W in image_sizes:
            in_ch = conv_cfg["in_channels"]
            out_ch = conv_cfg["out_channels"]
            kernel = conv_cfg["kernel_size"]
            stride = conv_cfg["stride"]
            padding = conv_cfg["padding"]

            # Create input and conv layer
            conv = tz.nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
            if device == "cpu":
                x = tz.randn([batch_size, in_ch, H, W])
            else:
                x = tz.randn([batch_size, in_ch, H, W]).to(device)
                conv.to(device)

            # Tenzor modules expect Variable input
            x_var = tz.Variable(x, requires_grad=False)

            # Use default args to capture current loop values (fixes closure bug)
            def conv_fn(layer=conv, xv=x_var):
                return layer(xv)

            times = run_benchmark(
                conv_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            flops = calculate_conv2d_flops(
                batch_size, in_ch, out_ch, H, W, kernel, kernel, stride, padding
            )

            name = f"Conv2d {in_ch}->{out_ch} k={kernel} s={stride} @ {H}x{W}"
            result = compute_statistics(
                times=times,
                name=name,
                category="conv2d",
                device=device,
                framework="tenzor",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "in_channels": in_ch,
                    "out_channels": out_ch,
                    "kernel_size": kernel,
                    "stride": stride,
                    "padding": padding,
                    "input_size": (H, W),
                    "batch_size": batch_size,
                },
            )
            results.append(result)
            print_result(result)

    return results


def benchmark_pytorch_conv2d(
    conv_configs: List[Dict],
    image_sizes: List[Tuple[int, int]],
    batch_size: int,
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch Conv2d."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed, skipping...")
        return []

    results = []
    sync_fn = get_pytorch_sync_fn(device)

    for conv_cfg in conv_configs:
        for H, W in image_sizes:
            in_ch = conv_cfg["in_channels"]
            out_ch = conv_cfg["out_channels"]
            kernel = conv_cfg["kernel_size"]
            stride = conv_cfg["stride"]
            padding = conv_cfg["padding"]

            torch_device = torch.device(device)
            x = torch.randn(batch_size, in_ch, H, W, device=torch_device)
            conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding).to(torch_device)

            # Use default args to capture current loop values (fixes closure bug)
            def conv_fn(layer=conv, inp=x):
                return layer(inp)

            times = run_benchmark(
                conv_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            flops = calculate_conv2d_flops(
                batch_size, in_ch, out_ch, H, W, kernel, kernel, stride, padding
            )

            name = f"Conv2d {in_ch}->{out_ch} k={kernel} s={stride} @ {H}x{W}"
            result = compute_statistics(
                times=times,
                name=name,
                category="conv2d",
                device=device,
                framework="pytorch",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "in_channels": in_ch,
                    "out_channels": out_ch,
                    "kernel_size": kernel,
                    "stride": stride,
                    "padding": padding,
                    "input_size": (H, W),
                    "batch_size": batch_size,
                },
            )
            results.append(result)
            print_result(result)

    return results


def benchmark_resnet_layers(
    device: str, config: BenchmarkConfig,
    tenzor_avail: bool = True, pytorch_avail: bool = None,
) -> List[BenchmarkResult]:
    """Benchmark convolutions matching ResNet-50 architecture."""
    if pytorch_avail is None:
        pytorch_avail = device in ("cpu", "cuda")

    resnet_layers = [
        # Stage 1
        {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3, "input_size": (224, 224)},
        # Stage 2
        {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "input_size": (56, 56)},
        {"in_channels": 64, "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "input_size": (56, 56)},
        # Stage 3
        {"in_channels": 256, "out_channels": 128, "kernel_size": 1, "stride": 1, "padding": 0, "input_size": (56, 56)},
        {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1, "input_size": (56, 56)},
        {"in_channels": 128, "out_channels": 512, "kernel_size": 1, "stride": 1, "padding": 0, "input_size": (28, 28)},
        # Stage 4
        {"in_channels": 512, "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "input_size": (28, 28)},
        {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1, "input_size": (28, 28)},
        {"in_channels": 256, "out_channels": 1024, "kernel_size": 1, "stride": 1, "padding": 0, "input_size": (14, 14)},
        # Stage 5
        {"in_channels": 1024, "out_channels": 512, "kernel_size": 1, "stride": 1, "padding": 0, "input_size": (14, 14)},
        {"in_channels": 512, "out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1, "input_size": (14, 14)},
        {"in_channels": 512, "out_channels": 2048, "kernel_size": 1, "stride": 1, "padding": 0, "input_size": (7, 7)},
    ]

    print("\n--- ResNet-50 Layer Benchmarks ---")
    results = []

    for i, layer in enumerate(resnet_layers):
        H, W = layer["input_size"]
        configs = [{k: v for k, v in layer.items() if k != "input_size"}]
        sizes = [(H, W)]

        print(f"\n  Layer {i+1}/{len(resnet_layers)}")

        if tenzor_avail:
            tz_results = benchmark_tenzor_conv2d(configs, sizes, 32, device, config)
            results.extend(tz_results)

        if config.compare_with_pytorch and pytorch_avail:
            pt_results = benchmark_pytorch_conv2d(configs, sizes, 32, device, config)
            results.extend(pt_results)

    return results


def run_conv2d_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all Conv2d benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  CONVOLUTION BENCHMARKS")
    print("=" * 70)

    for device in config.devices:
        print(f"\n{'='*70}")
        print(f"  Device: {device.upper()}")
        print(f"{'='*70}")

        # Check availability for both frameworks. PyTorch has no rocm/vulkan/
        # oneapi device type of its own, so the comparison only ever runs for
        # device in ("cpu", "cuda").
        tenzor_avail = check_tenzor_device_available(device)
        pytorch_comparable = device in ("cpu", "cuda")
        pytorch_avail = (check_pytorch_cuda_available() if device == "cuda" else pytorch_comparable) if pytorch_comparable else False

        if not tenzor_avail and not (config.compare_with_pytorch and pytorch_avail):
            print(f"{device} not available for either framework, skipping...")
            continue

        if not tenzor_avail:
            print(f"  [WARNING] Tenzor {device} not available")
        if config.compare_with_pytorch and not pytorch_avail:
            reason = "not a PyTorch device" if not pytorch_comparable else f"PyTorch {device} not available"
            print(f"  [WARNING] {reason}")

        # Clear GPU memory before starting
        if device != "cpu":
            clear_gpu_memory()

        # Standard conv2d benchmarks
        print("\n--- Tenzor Conv2d ---")
        if tenzor_avail:
            tz_results = benchmark_tenzor_conv2d(
                config.conv2d_configs[:3],  # First 3 configs
                config.image_sizes[:2],      # Smaller sizes
                batch_size=32,
                device=device,
                config=config,
            )
            all_results.extend(tz_results)
        else:
            tz_results = []
            print(f"  [SKIP] Tenzor {device} not available")

        if config.compare_with_pytorch:
            print("\n--- PyTorch Conv2d ---")
            if pytorch_avail:
                pt_results = benchmark_pytorch_conv2d(
                    config.conv2d_configs[:3],
                    config.image_sizes[:2],
                    batch_size=32,
                    device=device,
                    config=config,
                )
                all_results.extend(pt_results)
            else:
                print(f"  [SKIP] PyTorch {device} not available")

        # ResNet layer benchmarks
        resnet_results = benchmark_resnet_layers(device, config, tenzor_avail, pytorch_avail)
        all_results.extend(resnet_results)

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_conv2d_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    save_results(results, "results/conv2d_benchmarks.json")
