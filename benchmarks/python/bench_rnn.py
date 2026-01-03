"""
RNN/LSTM/GRU Benchmarks
=======================
Compare recurrent neural network performance between Tenzor and PyTorch.
Critical for sequence modeling, time series, and NLP tasks.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from typing import List, Tuple, Dict, Any
from benchmark_utils import (
    run_benchmark, compute_statistics, BenchmarkResult, print_result,
    get_tenzor_sync_fn, get_pytorch_sync_fn, check_tenzor_cuda_available,
    check_pytorch_cuda_available, clear_gpu_memory
)
from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG


# LSTM configurations (batch, seq_len, input_size, hidden_size, num_layers, name)
LSTM_CONFIGS = [
    (32, 128, 256, 256, 1, "Small 1L"),
    (32, 128, 512, 512, 1, "Medium 1L"),
    (32, 128, 512, 512, 2, "Medium 2L"),
    (16, 256, 768, 768, 2, "Large 2L"),
    # (8, 512, 1024, 1024, 3, "XLarge 3L"),  # Too slow on CPU (seq=512, hidden=1024, 3 layers)
]

# GRU configurations (same format)
GRU_CONFIGS = [
    (32, 128, 256, 256, 1, "Small 1L"),
    (32, 128, 512, 512, 1, "Medium 1L"),
    (32, 128, 512, 512, 2, "Medium 2L"),
    (16, 256, 768, 768, 2, "Large 2L"),
]

# Bidirectional configurations
BIDIRECTIONAL_CONFIGS = [
    (32, 128, 256, 256, 1, "BiLSTM Small"),
    (16, 256, 512, 512, 2, "BiLSTM Medium"),
    # (8, 512, 768, 768, 2, "BiLSTM Large"),  # Too slow on CPU (bidirectional, seq=512)
]


def calculate_lstm_flops(batch: int, seq_len: int, input_size: int, hidden_size: int, num_layers: int) -> int:
    """Calculate approximate FLOPs for LSTM forward pass.

    LSTM has 4 gates (i, f, g, o), each requiring:
    - Input projection: input_size * hidden_size
    - Hidden projection: hidden_size * hidden_size
    Plus elementwise ops for gates and cell state.
    """
    # First layer
    first_layer_flops = seq_len * batch * (
        4 * input_size * hidden_size +  # Input projections
        4 * hidden_size * hidden_size +  # Hidden projections
        4 * hidden_size * 5  # Activations and elementwise ops
    )

    # Subsequent layers
    other_layers_flops = (num_layers - 1) * seq_len * batch * (
        4 * hidden_size * hidden_size +  # Input projections (from prev layer)
        4 * hidden_size * hidden_size +  # Hidden projections
        4 * hidden_size * 5
    )

    return int(first_layer_flops + other_layers_flops) * 2  # *2 for multiply-add


def benchmark_tenzor_lstm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
    bidirectional: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor LSTM."""
    import tenzor as tz
    tz.initialize()

    results = []
    bi_str = "Bi" if bidirectional else ""
    sync_fn = get_tenzor_sync_fn(device)

    for batch, seq_len, input_size, hidden_size, num_layers, name in configs:
        try:
            lstm = tz.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )

            if device == "cuda":
                x = tz.randn([batch, seq_len, input_size]).cuda()
            else:
                x = tz.randn([batch, seq_len, input_size])

            x_var = tz.Variable(x, False)

            # Use default args to capture current loop values (fixes closure bug)
            def lstm_fn(layer=lstm, xv=x_var):
                output, (h_n, c_n) = layer.forward(xv)
                return output

            times = run_benchmark(
                lstm_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            flops = calculate_lstm_flops(batch, seq_len, input_size, hidden_size, num_layers)
            if bidirectional:
                flops *= 2

            result = compute_statistics(
                times=times,
                name=f"{bi_str}LSTM {name}",
                category="lstm",
                device=device,
                framework="tenzor",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "input_size": input_size, "hidden_size": hidden_size,
                    "num_layers": num_layers, "bidirectional": bidirectional
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {bi_str}LSTM {name}: {e}")

    return results


def benchmark_pytorch_lstm(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
    bidirectional: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch LSTM."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed, skipping")
        return []

    results = []
    torch_device = torch.device(device)
    bi_str = "Bi" if bidirectional else ""
    sync_fn = get_pytorch_sync_fn(device)

    for batch, seq_len, input_size, hidden_size, num_layers, name in configs:
        try:
            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            ).to(torch_device)
            lstm.eval()

            x = torch.randn(batch, seq_len, input_size, device=torch_device)

            with torch.no_grad():
                # Use default args to capture current loop values (fixes closure bug)
                def lstm_fn(layer=lstm, inp=x):
                    output, (h_n, c_n) = layer(inp)
                    return output

                times = run_benchmark(
                    lstm_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            flops = calculate_lstm_flops(batch, seq_len, input_size, hidden_size, num_layers)
            if bidirectional:
                flops *= 2

            result = compute_statistics(
                times=times,
                name=f"{bi_str}LSTM {name}",
                category="lstm",
                device=device,
                framework="pytorch",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "input_size": input_size, "hidden_size": hidden_size,
                    "num_layers": num_layers, "bidirectional": bidirectional
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {bi_str}LSTM {name}: {e}")

    return results


def benchmark_tenzor_gru(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor GRU."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    for batch, seq_len, input_size, hidden_size, num_layers, name in configs:
        try:
            gru = tz.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )

            if device == "cuda":
                x = tz.randn([batch, seq_len, input_size]).cuda()
            else:
                x = tz.randn([batch, seq_len, input_size])

            x_var = tz.Variable(x, False)

            # Use default args to capture current loop values (fixes closure bug)
            def gru_fn(layer=gru, xv=x_var):
                output, h_n = layer.forward(xv)
                return output

            times = run_benchmark(
                gru_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            # GRU has 3 gates vs LSTM's 4, so roughly 75% of LSTM FLOPs
            flops = int(calculate_lstm_flops(batch, seq_len, input_size, hidden_size, num_layers) * 0.75)

            result = compute_statistics(
                times=times,
                name=f"GRU {name}",
                category="gru",
                device=device,
                framework="tenzor",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "input_size": input_size, "hidden_size": hidden_size,
                    "num_layers": num_layers
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] GRU {name}: {e}")

    return results


def benchmark_pytorch_gru(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch GRU."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)
    sync_fn = get_pytorch_sync_fn(device)

    for batch, seq_len, input_size, hidden_size, num_layers, name in configs:
        try:
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            ).to(torch_device)
            gru.eval()

            x = torch.randn(batch, seq_len, input_size, device=torch_device)

            with torch.no_grad():
                # Use default args to capture current loop values (fixes closure bug)
                def gru_fn(layer=gru, inp=x):
                    output, h_n = layer(inp)
                    return output

                times = run_benchmark(
                    gru_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            flops = int(calculate_lstm_flops(batch, seq_len, input_size, hidden_size, num_layers) * 0.75)

            result = compute_statistics(
                times=times,
                name=f"GRU {name}",
                category="gru",
                device=device,
                framework="pytorch",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "input_size": input_size, "hidden_size": hidden_size,
                    "num_layers": num_layers
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] GRU {name}: {e}")

    return results


def benchmark_tenzor_lstm_backward(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor LSTM with backward pass."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    for batch, seq_len, input_size, hidden_size, num_layers, name in configs:
        try:
            lstm = tz.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )

            if device == "cuda":
                x = tz.randn([batch, seq_len, input_size]).cuda()
            else:
                x = tz.randn([batch, seq_len, input_size])

            x_var = tz.Variable(x, True)

            # Use default args to capture current loop values (fixes closure bug)
            def lstm_backward_fn(layer=lstm, xv=x_var):
                output, _ = layer.forward(xv)
                loss = tz.sum(output)
                loss.backward()
                return output

            times = run_benchmark(
                lstm_backward_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            flops = calculate_lstm_flops(batch, seq_len, input_size, hidden_size, num_layers) * 3

            result = compute_statistics(
                times=times,
                name=f"LSTM {name} (train)",
                category="lstm_train",
                device=device,
                framework="tenzor",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] LSTM {name} (train): {e}")

    return results


def benchmark_pytorch_lstm_backward(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch LSTM with backward pass."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)
    sync_fn = get_pytorch_sync_fn(device)

    for batch, seq_len, input_size, hidden_size, num_layers, name in configs:
        try:
            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            ).to(torch_device)
            lstm.train()

            x = torch.randn(batch, seq_len, input_size, device=torch_device, requires_grad=True)

            # Use default args to capture current loop values (fixes closure bug)
            def lstm_backward_fn(layer=lstm, inp=x):
                layer.zero_grad()
                output, _ = layer(inp)
                loss = output.sum()
                loss.backward()
                return output

            times = run_benchmark(
                lstm_backward_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            flops = calculate_lstm_flops(batch, seq_len, input_size, hidden_size, num_layers) * 3

            result = compute_statistics(
                times=times,
                name=f"LSTM {name} (train)",
                category="lstm_train",
                device=device,
                framework="pytorch",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] LSTM {name} (train): {e}")

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


def run_rnn_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all RNN benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  RNN / LSTM / GRU BENCHMARKS")
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

        # LSTM Forward
        print("\n--- Tenzor LSTM (Forward) ---")
        tenzor_lstm = benchmark_tenzor_lstm(LSTM_CONFIGS, device, config)
        all_results.extend(tenzor_lstm)

        if config.compare_with_pytorch:
            print("\n--- PyTorch LSTM (Forward) ---")
            pytorch_lstm = benchmark_pytorch_lstm(LSTM_CONFIGS, device, config)
            all_results.extend(pytorch_lstm)
            print_comparison(tenzor_lstm, pytorch_lstm, "LSTM Forward")

        # GRU Forward
        print("\n--- Tenzor GRU (Forward) ---")
        tenzor_gru = benchmark_tenzor_gru(GRU_CONFIGS, device, config)
        all_results.extend(tenzor_gru)

        if config.compare_with_pytorch:
            print("\n--- PyTorch GRU (Forward) ---")
            pytorch_gru = benchmark_pytorch_gru(GRU_CONFIGS, device, config)
            all_results.extend(pytorch_gru)
            print_comparison(tenzor_gru, pytorch_gru, "GRU Forward")

        # Bidirectional LSTM
        print("\n--- Tenzor BiLSTM ---")
        tenzor_bilstm = benchmark_tenzor_lstm(BIDIRECTIONAL_CONFIGS, device, config, bidirectional=True)
        all_results.extend(tenzor_bilstm)

        if config.compare_with_pytorch:
            print("\n--- PyTorch BiLSTM ---")
            pytorch_bilstm = benchmark_pytorch_lstm(BIDIRECTIONAL_CONFIGS, device, config, bidirectional=True)
            all_results.extend(pytorch_bilstm)
            print_comparison(tenzor_bilstm, pytorch_bilstm, "Bidirectional LSTM")

        # LSTM Training (with backward)
        print("\n--- Tenzor LSTM (Training) ---")
        tenzor_lstm_train = benchmark_tenzor_lstm_backward(LSTM_CONFIGS[:3], device, config)
        all_results.extend(tenzor_lstm_train)

        if config.compare_with_pytorch:
            print("\n--- PyTorch LSTM (Training) ---")
            pytorch_lstm_train = benchmark_pytorch_lstm_backward(LSTM_CONFIGS[:3], device, config)
            all_results.extend(pytorch_lstm_train)
            print_comparison(tenzor_lstm_train, pytorch_lstm_train, "LSTM Training")

        # LSTM vs GRU comparison
        if tenzor_lstm and tenzor_gru:
            print("\n" + "=" * 70)
            print("  ANALYSIS: LSTM vs GRU (Tenzor)")
            print("=" * 70)
            print(f"{'Configuration':<25} {'LSTM (ms)':<15} {'GRU (ms)':<15} {'GRU Speedup':<15}")
            print("-" * 70)

            for lstm_r, gru_r in zip(tenzor_lstm[:len(tenzor_gru)], tenzor_gru):
                speedup = lstm_r.mean_ms / gru_r.mean_ms if gru_r.mean_ms > 0 else 0
                name = lstm_r.name.replace("LSTM ", "")
                print(f"{name:<25} {lstm_r.mean_ms:<15.3f} {gru_r.mean_ms:<15.3f} {speedup:.2f}x")

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_rnn_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    import os
    os.makedirs("results", exist_ok=True)
    save_results(results, "results/rnn_benchmarks.json")
