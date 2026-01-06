"""
Embedding Layer Benchmarks
==========================
Compare embedding layer performance between Tenzor and PyTorch.
Critical for NLP models (BERT, GPT, etc.) and recommendation systems.
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


# Embedding configurations (vocab_size, embed_dim, batch, seq_len, name)
EMBEDDING_CONFIGS = [
    (30522, 768, 32, 128, "BERT vocab"),
    (50257, 768, 16, 512, "GPT-2 vocab"),
    # (32000, 4096, 4, 2048, "Llama vocab"),  # Too slow on CPU (seq=2048, dim=4096)
    (100000, 256, 64, 64, "Large vocab small embed"),
    (10000, 1024, 32, 256, "Small vocab large embed"),
]

# Position embedding configurations
POSITION_CONFIGS = [
    (512, 768, 32, "BERT max_pos"),
    (1024, 768, 16, "GPT-2 max_pos"),
    # (2048, 768, 8, "Long context"),  # Too slow on CPU
    # (4096, 4096, 4, "Llama style"),  # Too slow on CPU (4096 dim)
]


def benchmark_tenzor_embedding(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor embedding forward pass."""
    import tenzor as tz
    tz.initialize()

    results = []

    for vocab_size, embed_dim, batch, seq_len, name in configs:
        try:
            embedding = tz.nn.Embedding(vocab_size, embed_dim)

            # Create random token indices
            if device == "cuda":
                indices = tz.randint(0, vocab_size, [batch, seq_len]).cuda()
            else:
                indices = tz.randint(0, vocab_size, [batch, seq_len])

            def embed_fn():
                return embedding.forward(indices)

            times = run_benchmark(
                embed_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )

            # Bytes accessed: indices (int64) + output (float32)
            bytes_accessed = batch * seq_len * 8 + batch * seq_len * embed_dim * 4

            result = compute_statistics(
                times=times,
                name=name,
                category="embedding",
                device=device,
                framework="tenzor",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "vocab_size": vocab_size, "embed_dim": embed_dim,
                    "batch": batch, "seq_len": seq_len
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_pytorch_embedding(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch embedding forward pass."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed, skipping")
        return []

    results = []
    torch_device = torch.device(device)

    for vocab_size, embed_dim, batch, seq_len, name in configs:
        try:
            embedding = nn.Embedding(vocab_size, embed_dim).to(torch_device)
            indices = torch.randint(0, vocab_size, (batch, seq_len), device=torch_device)

            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            with torch.no_grad():
                def embed_fn():
                    return embedding(indices)

                times = run_benchmark(
                    embed_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            bytes_accessed = batch * seq_len * 8 + batch * seq_len * embed_dim * 4

            result = compute_statistics(
                times=times,
                name=name,
                category="embedding",
                device=device,
                framework="pytorch",
                bytes_accessed=bytes_accessed,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "vocab_size": vocab_size, "embed_dim": embed_dim,
                    "batch": batch, "seq_len": seq_len
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_tenzor_embedding_backward(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor embedding with backward pass (sparse gradients)."""
    import tenzor as tz
    tz.initialize()

    results = []

    for vocab_size, embed_dim, batch, seq_len, name in configs:
        try:
            embedding = tz.nn.Embedding(vocab_size, embed_dim)

            if device == "cuda":
                indices = tz.randint(0, vocab_size, [batch, seq_len]).cuda()
            else:
                indices = tz.randint(0, vocab_size, [batch, seq_len])

            def embed_backward_fn():
                output = embedding.forward(indices)
                loss = tz.sum(output)
                loss.backward()
                return output

            times = run_benchmark(
                embed_backward_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )

            result = compute_statistics(
                times=times,
                name=f"{name} (backward)",
                category="embedding_backward",
                device=device,
                framework="tenzor",
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "vocab_size": vocab_size, "embed_dim": embed_dim,
                    "batch": batch, "seq_len": seq_len
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name} (backward): {e}")

    return results


def benchmark_pytorch_embedding_backward(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch embedding with backward pass."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)

    for vocab_size, embed_dim, batch, seq_len, name in configs:
        try:
            embedding = nn.Embedding(vocab_size, embed_dim).to(torch_device)
            indices = torch.randint(0, vocab_size, (batch, seq_len), device=torch_device)
            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            def embed_backward_fn():
                embedding.zero_grad()
                output = embedding(indices)
                loss = output.sum()
                loss.backward()
                return output

            times = run_benchmark(
                embed_backward_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            result = compute_statistics(
                times=times,
                name=f"{name} (backward)",
                category="embedding_backward",
                device=device,
                framework="pytorch",
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "vocab_size": vocab_size, "embed_dim": embed_dim,
                    "batch": batch, "seq_len": seq_len
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name} (backward): {e}")

    return results


def benchmark_tenzor_position_embedding(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor positional embedding lookup."""
    import tenzor as tz
    tz.initialize()

    results = []

    for max_pos, embed_dim, batch, name in configs:
        try:
            pos_embedding = tz.nn.Embedding(max_pos, embed_dim)

            # Create position indices
            seq_len = min(max_pos, 512)
            if device == "cuda":
                positions = tz.arange(0, seq_len).cuda().expand(batch, seq_len)
            else:
                positions = tz.arange(0, seq_len).expand(batch, seq_len)

            def pos_embed_fn():
                return pos_embedding.forward(positions)

            times = run_benchmark(
                pos_embed_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )

            result = compute_statistics(
                times=times,
                name=f"PositionEmbed {name}",
                category="position_embedding",
                device=device,
                framework="tenzor",
                warmup_iterations=config.warmup_iterations,
                parameters={"max_pos": max_pos, "embed_dim": embed_dim, "batch": batch},
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] PositionEmbed {name}: {e}")

    return results


def benchmark_pytorch_position_embedding(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch positional embedding lookup."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)

    for max_pos, embed_dim, batch, name in configs:
        try:
            pos_embedding = nn.Embedding(max_pos, embed_dim).to(torch_device)
            seq_len = min(max_pos, 512)
            positions = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch, -1)

            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            with torch.no_grad():
                def pos_embed_fn():
                    return pos_embedding(positions)

                times = run_benchmark(
                    pos_embed_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            result = compute_statistics(
                times=times,
                name=f"PositionEmbed {name}",
                category="position_embedding",
                device=device,
                framework="pytorch",
                warmup_iterations=config.warmup_iterations,
                parameters={"max_pos": max_pos, "embed_dim": embed_dim, "batch": batch},
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] PositionEmbed {name}: {e}")

    return results


def benchmark_combined_embedding(
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark combined token + position embedding (like BERT/GPT)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)

    configs = [
        (30522, 512, 768, 32, 128, "BERT-base"),
        (50257, 1024, 768, 16, 512, "GPT-2"),
        # (32000, 4096, 4096, 4, 2048, "Llama-7B"),  # Too slow on CPU (seq=2048, dim=4096)
    ]

    for vocab_size, max_pos, embed_dim, batch, seq_len, name in configs:
        try:
            token_embed = nn.Embedding(vocab_size, embed_dim).to(torch_device)
            pos_embed = nn.Embedding(max_pos, embed_dim).to(torch_device)

            tokens = torch.randint(0, vocab_size, (batch, seq_len), device=torch_device)
            positions = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch, -1)

            sync_fn = torch.cuda.synchronize if device == "cuda" else None

            with torch.no_grad():
                def combined_fn():
                    return token_embed(tokens) + pos_embed(positions)

                times = run_benchmark(
                    combined_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            result = compute_statistics(
                times=times,
                name=f"Token+Pos {name}",
                category="combined_embedding",
                device=device,
                framework="pytorch",
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] Token+Pos {name}: {e}")

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


def run_embedding_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all embedding benchmarks."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  EMBEDDING LAYER BENCHMARKS")
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

        # Token embedding forward
        print("\n--- Tenzor Embedding (Forward) ---")
        tenzor_embed = benchmark_tenzor_embedding(EMBEDDING_CONFIGS, device, config)
        all_results.extend(tenzor_embed)

        if config.compare_with_pytorch:
            print("\n--- PyTorch Embedding (Forward) ---")
            pytorch_embed = benchmark_pytorch_embedding(EMBEDDING_CONFIGS, device, config)
            all_results.extend(pytorch_embed)
            print_comparison(tenzor_embed, pytorch_embed, "Embedding Forward")

        # Token embedding backward
        print("\n--- Tenzor Embedding (Backward) ---")
        tenzor_embed_bwd = benchmark_tenzor_embedding_backward(EMBEDDING_CONFIGS, device, config)
        all_results.extend(tenzor_embed_bwd)

        if config.compare_with_pytorch:
            print("\n--- PyTorch Embedding (Backward) ---")
            pytorch_embed_bwd = benchmark_pytorch_embedding_backward(EMBEDDING_CONFIGS, device, config)
            all_results.extend(pytorch_embed_bwd)
            print_comparison(tenzor_embed_bwd, pytorch_embed_bwd, "Embedding Backward")

        # Position embedding
        print("\n--- Tenzor Position Embedding ---")
        tenzor_pos = benchmark_tenzor_position_embedding(POSITION_CONFIGS, device, config)
        all_results.extend(tenzor_pos)

        if config.compare_with_pytorch:
            print("\n--- PyTorch Position Embedding ---")
            pytorch_pos = benchmark_pytorch_position_embedding(POSITION_CONFIGS, device, config)
            all_results.extend(pytorch_pos)
            print_comparison(tenzor_pos, pytorch_pos, "Position Embedding")

        # Combined embedding (PyTorch reference)
        if config.compare_with_pytorch:
            print("\n--- Combined Token+Position Embedding (PyTorch) ---")
            combined_results = benchmark_combined_embedding(device, config)
            all_results.extend(combined_results)

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_embedding_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    import os
    os.makedirs("results", exist_ok=True)
    save_results(results, "results/embedding_benchmarks.json")
