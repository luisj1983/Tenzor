"""
End-to-End Model Benchmarks
===========================
Compare full model inference and training performance between Tenzor and PyTorch.
Critical for understanding real-world performance differences.
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


# MLP configurations
MLP_CONFIGS = [
    ([784, 256, 128, 10], 64, "Small MLP"),
    ([784, 512, 256, 128, 10], 64, "Medium MLP"),
    ([784, 1024, 512, 256, 10], 64, "Large MLP"),
    ([784, 2048, 1024, 512, 10], 32, "XLarge MLP"),
]

# CNN configurations (simplified ResNet-like)
CNN_CONFIGS = [
    ("ResNet-18", 32, 224),
    ("ResNet-34", 16, 224),
    ("ResNet-50", 8, 224),
]

# Transformer configurations
TRANSFORMER_CONFIGS = [
    (8, 512, 768, 12, 6, "BERT-base 6L"),
    (4, 512, 768, 12, 12, "BERT-base 12L"),
    # (2, 1024, 768, 12, 6, "Long seq 6L"),  # Too slow on CPU
]


def benchmark_tenzor_mlp(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
    training: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor MLP models."""
    import tenzor as tz
    tz.initialize()

    results = []
    mode = "train" if training else "eval"
    sync_fn = get_tenzor_sync_fn(device)

    # Disable gradients for inference (like PyTorch's torch.no_grad())
    if not training:
        tz.set_grad_enabled(False)

    for layer_sizes, batch_size, name in configs:
        try:
            # Build MLP
            layers = []
            for i in range(len(layer_sizes) - 1):
                layers.append(tz.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    layers.append(tz.nn.ReLU())

            model = tz.nn.Sequential(*layers)

            if device == "cpu":
                x = tz.randn([batch_size, layer_sizes[0]])
            else:
                model.to(device)  # Move model to device!
                x = tz.randn([batch_size, layer_sizes[0]]).to(device)

            x_var = tz.Variable(x, training)

            if training:
                # Use optimizer like PyTorch for fair comparison
                optimizer = tz.optim.SGD(model.parameters(), lr=0.01)

                def mlp_fn():
                    optimizer.zero_grad()
                    output = model.forward(x_var)
                    loss = tz.sum(output)
                    loss.backward()
                    optimizer.step()
                    return output
            else:
                def mlp_fn():
                    return model.forward(x_var)

            times = run_benchmark(
                mlp_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            # Calculate FLOPs
            flops = 0
            for i in range(len(layer_sizes) - 1):
                flops += 2 * batch_size * layer_sizes[i] * layer_sizes[i + 1]
            if training:
                flops *= 3  # Forward + backward

            result = compute_statistics(
                times=times,
                name=f"{name} ({mode})",
                category="mlp",
                device=device,
                framework="tenzor",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={"layers": layer_sizes, "batch": batch_size},
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # Restore gradient state
    if not training:
        tz.set_grad_enabled(True)
    return results


def benchmark_pytorch_mlp(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
    training: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch MLP models."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed, skipping")
        return []

    results = []
    torch_device = torch.device(device)
    mode = "train" if training else "eval"

    for layer_sizes, batch_size, name in configs:
        try:
            # Build MLP
            layers = []
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU())

            model = nn.Sequential(*layers).to(torch_device)

            if training:
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            else:
                model.eval()

            x = torch.randn(batch_size, layer_sizes[0], device=torch_device)
            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            if training:
                def mlp_fn():
                    optimizer.zero_grad()
                    output = model(x)
                    loss = output.sum()
                    loss.backward()
                    optimizer.step()
                    return output

                times = run_benchmark(
                    mlp_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )
            else:
                with torch.no_grad():
                    def mlp_fn():
                        return model(x)

                    times = run_benchmark(
                        mlp_fn,
                        warmup_iterations=config.warmup_iterations,
                        benchmark_iterations=config.benchmark_iterations,
                        sync_fn=sync_fn,
                    )

            flops = 0
            for i in range(len(layer_sizes) - 1):
                flops += 2 * batch_size * layer_sizes[i] * layer_sizes[i + 1]
            if training:
                flops *= 3

            result = compute_statistics(
                times=times,
                name=f"{name} ({mode})",
                category="mlp",
                device=device,
                framework="pytorch",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={"layers": layer_sizes, "batch": batch_size},
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_tenzor_transformer(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor transformer encoder."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    # Disable gradients for inference (like PyTorch's torch.no_grad())
    tz.set_grad_enabled(False)

    for batch, seq_len, embed_dim, num_heads, num_layers, name in configs:
        try:
            encoder_layer = tz.nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
            )
            encoder = tz.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            encoder.eval()  # Set to eval mode

            if device == "cpu":
                x = tz.randn([batch, seq_len, embed_dim])
            else:
                encoder.to(device)  # Move model to device!
                x = tz.randn([batch, seq_len, embed_dim]).to(device)

            x_var = tz.Variable(x, False)

            def transformer_fn():
                return encoder.forward(x_var)

            times = run_benchmark(
                transformer_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            # Approximate FLOPs for transformer
            head_dim = embed_dim // num_heads
            attn_flops = batch * num_heads * seq_len * seq_len * head_dim * 2 * 2
            qkv_flops = 3 * batch * seq_len * embed_dim * embed_dim * 2
            out_flops = batch * seq_len * embed_dim * embed_dim * 2
            ffn_flops = 2 * batch * seq_len * embed_dim * embed_dim * 4 * 2
            layer_flops = attn_flops + qkv_flops + out_flops + ffn_flops
            total_flops = layer_flops * num_layers

            result = compute_statistics(
                times=times,
                name=name,
                category="transformer",
                device=device,
                framework="tenzor",
                flops=total_flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "embed_dim": embed_dim, "num_heads": num_heads,
                    "num_layers": num_layers
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # Restore gradient state
    tz.set_grad_enabled(True)
    return results


def benchmark_pytorch_transformer(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch transformer encoder."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)

    for batch, seq_len, embed_dim, num_heads, num_layers, name in configs:
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
            )
            encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            encoder = encoder.to(torch_device)
            encoder.eval()

            x = torch.randn(batch, seq_len, embed_dim, device=torch_device)
            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            with torch.no_grad():
                def transformer_fn():
                    return encoder(x)

                times = run_benchmark(
                    transformer_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            head_dim = embed_dim // num_heads
            attn_flops = batch * num_heads * seq_len * seq_len * head_dim * 2 * 2
            qkv_flops = 3 * batch * seq_len * embed_dim * embed_dim * 2
            out_flops = batch * seq_len * embed_dim * embed_dim * 2
            ffn_flops = 2 * batch * seq_len * embed_dim * embed_dim * 4 * 2
            layer_flops = attn_flops + qkv_flops + out_flops + ffn_flops
            total_flops = layer_flops * num_layers

            result = compute_statistics(
                times=times,
                name=name,
                category="transformer",
                device=device,
                framework="pytorch",
                flops=total_flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "embed_dim": embed_dim, "num_heads": num_heads,
                    "num_layers": num_layers
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_pytorch_resnet(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch ResNet models (reference implementation)."""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)

    model_map = {
        "ResNet-18": models.resnet18,
        "ResNet-34": models.resnet34,
        "ResNet-50": models.resnet50,
    }

    for model_name, batch_size, image_size in configs:
        try:
            model = model_map[model_name](pretrained=False).to(torch_device)
            model.eval()

            x = torch.randn(batch_size, 3, image_size, image_size, device=torch_device)
            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            with torch.no_grad():
                def resnet_fn():
                    return model(x)

                times = run_benchmark(
                    resnet_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            result = compute_statistics(
                times=times,
                name=f"{model_name} B={batch_size}",
                category="resnet",
                device=device,
                framework="pytorch",
                warmup_iterations=config.warmup_iterations,
                parameters={"model": model_name, "batch": batch_size, "image_size": image_size},
            )
            results.append(result)
            print_result(result)

            # Calculate throughput
            throughput = batch_size / (result.mean_ms / 1000)
            print(f"  Throughput: {throughput:.1f} images/sec")

        except Exception as e:
            print(f"  [SKIP] {model_name}: {e}")

    return results


def print_comparison(tenzor_results: List[BenchmarkResult], pytorch_results: List[BenchmarkResult], title: str):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print(f"  COMPARISON: {title}")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Tenzor (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<15}")
    print("-" * 80)

    for tz_r, pt_r in zip(tenzor_results, pytorch_results):
        speedup = tz_r.speedup_vs(pt_r)
        status = "FASTER" if speedup > 1 else "SLOWER"
        print(f"{tz_r.name:<30} {tz_r.mean_ms:<15.3f} {pt_r.mean_ms:<15.3f} {speedup:.2f}x {status}")


def run_model_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all model benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  END-TO-END MODEL BENCHMARKS")
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

        # MLP Inference
        print("\n--- Tenzor MLP (Inference) ---")
        tenzor_mlp_eval = benchmark_tenzor_mlp(MLP_CONFIGS, device, config, training=False) if tenzor_avail else []
        all_results.extend(tenzor_mlp_eval)

        if run_pytorch:
            print("\n--- PyTorch MLP (Inference) ---")
            pytorch_mlp_eval = benchmark_pytorch_mlp(MLP_CONFIGS, device, config, training=False)
            all_results.extend(pytorch_mlp_eval)
            print_comparison(tenzor_mlp_eval, pytorch_mlp_eval, "MLP Inference")

        # MLP Training
        print("\n--- Tenzor MLP (Training) ---")
        tenzor_mlp_train = benchmark_tenzor_mlp(MLP_CONFIGS, device, config, training=True) if tenzor_avail else []
        all_results.extend(tenzor_mlp_train)

        if run_pytorch:
            print("\n--- PyTorch MLP (Training) ---")
            pytorch_mlp_train = benchmark_pytorch_mlp(MLP_CONFIGS, device, config, training=True)
            all_results.extend(pytorch_mlp_train)
            print_comparison(tenzor_mlp_train, pytorch_mlp_train, "MLP Training")

        # Transformer Encoder
        print("\n--- Tenzor Transformer Encoder ---")
        tenzor_transformer = benchmark_tenzor_transformer(TRANSFORMER_CONFIGS, device, config) if tenzor_avail else []
        all_results.extend(tenzor_transformer)

        if run_pytorch:
            print("\n--- PyTorch Transformer Encoder ---")
            pytorch_transformer = benchmark_pytorch_transformer(TRANSFORMER_CONFIGS, device, config)
            all_results.extend(pytorch_transformer)
            print_comparison(tenzor_transformer, pytorch_transformer, "Transformer Encoder")

        # ResNet (PyTorch reference only)
        if run_pytorch:
            print("\n--- PyTorch ResNet (Reference) ---")
            resnet_results = benchmark_pytorch_resnet(CNN_CONFIGS, device, config)
            all_results.extend(resnet_results)

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_model_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    import os
    os.makedirs("results", exist_ok=True)
    save_results(results, "results/model_benchmarks.json")
