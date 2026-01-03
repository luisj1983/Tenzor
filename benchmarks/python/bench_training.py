"""
Training Benchmarks
===================
Compare end-to-end training performance between Tenzor and PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from typing import List, Dict
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_cuda_available,
    check_pytorch_cuda_available, clear_gpu_memory
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


def benchmark_mlp_training_tenzor(
    mlp_config: Dict,
    batch_size: int,
    device: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark MLP training iteration with Tenzor."""
    import tenzor as tz
    tz.initialize()

    layers = mlp_config["layers"]
    name = mlp_config["name"]

    # Build model
    modules = []
    for i in range(len(layers) - 1):
        modules.append(tz.nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            modules.append(tz.nn.ReLU())

    model = tz.nn.Sequential(*modules)
    optimizer = tz.optim.Adam(model.parameters(), lr=1e-3)

    # Create data
    if device == "cuda":
        x = tz.randn([batch_size, layers[0]]).cuda()
        y = tz.randn([batch_size, layers[-1]]).cuda()
        model.cuda()
    else:
        x = tz.randn([batch_size, layers[0]])
        y = tz.randn([batch_size, layers[-1]])

    # Both input and target need to be Variables for mse_loss
    y_var = tz.Variable(y, requires_grad=False)

    def train_step():
        optimizer.zero_grad()
        pred = model(tz.Variable(x, requires_grad=True))
        loss = tz.nn.mse_loss(pred, y_var)
        loss.backward()
        optimizer.step()
        return loss

    times = run_benchmark(
        train_step,
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
    )

    # Calculate FLOPs (forward + backward ≈ 3x forward)
    flops = 0
    for i in range(len(layers) - 1):
        flops += 2 * batch_size * layers[i] * layers[i + 1]
    flops *= 3  # forward + backward

    result = compute_statistics(
        times=times,
        name=f"Train {name} B={batch_size}",
        category="training",
        device=device,
        framework="tenzor",
        flops=flops,
        warmup_iterations=config.warmup_iterations,
        parameters={
            "layers": layers,
            "batch_size": batch_size,
            "optimizer": "Adam",
        },
    )

    return result


def benchmark_mlp_training_pytorch(
    mlp_config: Dict,
    batch_size: int,
    device: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark MLP training iteration with PyTorch."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return None

    layers = mlp_config["layers"]
    name = mlp_config["name"]

    # Build model
    modules = []
    for i in range(len(layers) - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            modules.append(nn.ReLU())

    model = nn.Sequential(*modules)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.randn(batch_size, layers[0], device=torch_device)
    y = torch.randn(batch_size, layers[-1], device=torch_device)

    sync_fn = torch.cuda.synchronize if device == "cuda" else None

    def train_step():
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        return loss

    times = run_benchmark(
        train_step,
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
        sync_fn=sync_fn,
    )

    flops = 0
    for i in range(len(layers) - 1):
        flops += 2 * batch_size * layers[i] * layers[i + 1]
    flops *= 3

    result = compute_statistics(
        times=times,
        name=f"Train {name} B={batch_size}",
        category="training",
        device=device,
        framework="pytorch",
        flops=flops,
        warmup_iterations=config.warmup_iterations,
        parameters={
            "layers": layers,
            "batch_size": batch_size,
            "optimizer": "Adam",
        },
    )

    return result


def benchmark_cnn_training_tenzor(
    batch_size: int,
    device: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark CNN training (simplified LeNet-style)."""
    import tenzor as tz
    tz.initialize()

    # Simple CNN
    model = tz.nn.Sequential(
        tz.nn.Conv2d(1, 32, 3, padding=1),
        tz.nn.ReLU(),
        tz.nn.MaxPool2d(2, 2),
        tz.nn.Conv2d(32, 64, 3, padding=1),
        tz.nn.ReLU(),
        tz.nn.MaxPool2d(2, 2),
        tz.nn.Flatten(),
        tz.nn.Linear(64 * 7 * 7, 128),
        tz.nn.ReLU(),
        tz.nn.Linear(128, 10),
    )

    optimizer = tz.optim.Adam(model.parameters(), lr=1e-3)

    if device == "cuda":
        x = tz.randn([batch_size, 1, 28, 28]).cuda()
        y = tz.randn([batch_size, 10]).cuda()
        model.cuda()
    else:
        x = tz.randn([batch_size, 1, 28, 28])
        y = tz.randn([batch_size, 10])

    # Both input and target need to be Variables for mse_loss
    y_var = tz.Variable(y, requires_grad=False)

    def train_step():
        optimizer.zero_grad()
        pred = model(tz.Variable(x, requires_grad=True))
        loss = tz.nn.mse_loss(pred, y_var)
        loss.backward()
        optimizer.step()

    times = run_benchmark(
        train_step,
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
    )

    result = compute_statistics(
        times=times,
        name=f"Train CNN (LeNet-style) B={batch_size}",
        category="training",
        device=device,
        framework="tenzor",
        warmup_iterations=config.warmup_iterations,
        parameters={"batch_size": batch_size, "model": "cnn_lenet"},
    )

    return result


def benchmark_cnn_training_pytorch(
    batch_size: int,
    device: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark CNN training with PyTorch."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return None

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    torch_device = torch.device(device)
    model = SimpleCNN().to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.randn(batch_size, 1, 28, 28, device=torch_device)
    y = torch.randn(batch_size, 10, device=torch_device)

    sync_fn = torch.cuda.synchronize if device == "cuda" else None

    def train_step():
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    times = run_benchmark(
        train_step,
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
        sync_fn=sync_fn,
    )

    result = compute_statistics(
        times=times,
        name=f"Train CNN (LeNet-style) B={batch_size}",
        category="training",
        device=device,
        framework="pytorch",
        warmup_iterations=config.warmup_iterations,
        parameters={"batch_size": batch_size, "model": "cnn_lenet"},
    )

    return result


def benchmark_optimizer_comparison(device: str, config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Compare different optimizers."""
    import tenzor as tz
    tz.initialize()

    results = []
    batch_size = 64
    layers = [1024, 512, 256, 10]

    # Build model
    modules = []
    for i in range(len(layers) - 1):
        modules.append(tz.nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            modules.append(tz.nn.ReLU())

    optimizers = {
        "SGD": lambda params: tz.optim.SGD(params, lr=0.01, momentum=0.9),
        "Adam": lambda params: tz.optim.Adam(params, lr=1e-3),
        "AdamW": lambda params: tz.optim.AdamW(params, lr=1e-3, weight_decay=0.01),
    }

    print("\n--- Optimizer Comparison ---")

    for opt_name, opt_fn in optimizers.items():
        model = tz.nn.Sequential(*modules)
        optimizer = opt_fn(model.parameters())

        if device == "cuda":
            x = tz.randn([batch_size, layers[0]]).cuda()
            y = tz.randn([batch_size, layers[-1]]).cuda()
            model.cuda()
        else:
            x = tz.randn([batch_size, layers[0]])
            y = tz.randn([batch_size, layers[-1]])

        # Both input and target need to be Variables for mse_loss
        y_var = tz.Variable(y, requires_grad=False)

        def train_step():
            optimizer.zero_grad()
            pred = model(tz.Variable(x, requires_grad=True))
            loss = tz.nn.mse_loss(pred, y_var)
            loss.backward()
            optimizer.step()

        times = run_benchmark(
            train_step,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
        )

        result = compute_statistics(
            times=times,
            name=f"Optimizer: {opt_name}",
            category="optimizer",
            device=device,
            framework="tenzor",
            warmup_iterations=config.warmup_iterations,
            parameters={"optimizer": opt_name, "batch_size": batch_size},
        )
        results.append(result)
        print_result(result)

    return results


def run_training_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all training benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  TRAINING BENCHMARKS")
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

        # MLP training
        print("\n--- MLP Training Benchmarks ---")
        for mlp_cfg in config.mlp_configs:
            for batch in [32, 64, 128]:
                print(f"\n  Tenzor - {mlp_cfg['name']} batch={batch}")
                tz_result = benchmark_mlp_training_tenzor(mlp_cfg, batch, device, config)
                all_results.append(tz_result)
                print_result(tz_result)

                if config.compare_with_pytorch:
                    print(f"  PyTorch - {mlp_cfg['name']} batch={batch}")
                    pt_result = benchmark_mlp_training_pytorch(mlp_cfg, batch, device, config)
                    if pt_result:
                        all_results.append(pt_result)
                        print_result(pt_result, baseline=tz_result)

        # CNN training
        print("\n--- CNN Training Benchmarks ---")
        for batch in [16, 32, 64]:
            print(f"\n  Tenzor CNN batch={batch}")
            tz_cnn = benchmark_cnn_training_tenzor(batch, device, config)
            all_results.append(tz_cnn)
            print_result(tz_cnn)

            if config.compare_with_pytorch:
                print(f"  PyTorch CNN batch={batch}")
                pt_cnn = benchmark_cnn_training_pytorch(batch, device, config)
                if pt_cnn:
                    all_results.append(pt_cnn)
                    print_result(pt_cnn, baseline=tz_cnn)

        # Optimizer comparison
        all_results.extend(benchmark_optimizer_comparison(device, config))

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_training_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    save_results(results, "results/training_benchmarks.json")
