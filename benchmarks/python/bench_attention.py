"""
Attention/Transformer Benchmarks
================================
Compare attention mechanism performance between Tenzor and PyTorch.
Critical for modern transformer-based models (BERT, GPT, ViT, etc.)
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


def calculate_attention_flops(batch: int, heads: int, seq_q: int, seq_kv: int, head_dim: int) -> int:
    """Calculate FLOPs for scaled dot-product attention.

    QK^T: batch * heads * seq_q * seq_kv * head_dim * 2
    Softmax: ~5 ops per element
    AV: batch * heads * seq_q * seq_kv * head_dim * 2
    """
    qk_flops = batch * heads * seq_q * seq_kv * head_dim * 2
    av_flops = batch * heads * seq_q * seq_kv * head_dim * 2
    softmax_flops = batch * heads * seq_q * seq_kv * 5
    return qk_flops + av_flops + softmax_flops


def calculate_mha_flops(batch: int, seq_len: int, embed_dim: int, num_heads: int) -> int:
    """Calculate FLOPs for multi-head attention.

    Includes: QKV projections + attention + output projection
    """
    head_dim = embed_dim // num_heads

    # QKV projections: 3 * batch * seq * embed * embed
    qkv_flops = 3 * batch * seq_len * embed_dim * embed_dim * 2

    # Attention: as calculated above
    attn_flops = calculate_attention_flops(batch, num_heads, seq_len, seq_len, head_dim)

    # Output projection: batch * seq * embed * embed
    out_flops = batch * seq_len * embed_dim * embed_dim * 2

    return qkv_flops + attn_flops + out_flops


# Configuration for attention benchmarks
ATTENTION_CONFIGS = [
    # (batch, seq_len, embed_dim, num_heads, name)
    (8, 128, 768, 12, "BERT-base seq=128"),
    (8, 512, 768, 12, "BERT-base seq=512"),
    (4, 256, 768, 12, "GPT-2 Small seq=256"),
    #(4, 1024, 768, 12, "GPT-2 Small seq=1024"),
    #(2, 512, 1024, 16, "GPT-2 Medium"),
    #(1, 512, 1600, 25, "GPT-2 XL"),
    (32, 197, 768, 12, "ViT-Base"),
    #(1, 2048, 768, 12, "Long sequence"),
]


def benchmark_tenzor_attention(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor multi-head attention."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    for batch, seq_len, embed_dim, num_heads, name in configs:
        try:
            # Create multi-head attention layer
            mha = tz.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

            # Create input
            if device == "cuda":
                x = tz.randn([batch, seq_len, embed_dim]).cuda()
            else:
                x = tz.randn([batch, seq_len, embed_dim])

            x_var = tz.Variable(x, False)

            # Use default args to capture current loop values (fixes closure bug)
            def attn_fn(layer=mha, xv=x_var):
                output, _ = layer.forward(xv, xv, xv, need_weights=False)
                return output

            times = run_benchmark(
                attn_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            flops = calculate_mha_flops(batch, seq_len, embed_dim, num_heads)
            result = compute_statistics(
                times=times,
                name=name,
                category="attention",
                device=device,
                framework="tenzor",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "embed_dim": embed_dim, "num_heads": num_heads
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_pytorch_attention(
    configs: List[Tuple],
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch multi-head attention."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed, skipping PyTorch benchmarks")
        return []

    results = []
    torch_device = torch.device(device)
    sync_fn = get_pytorch_sync_fn(device)

    for batch, seq_len, embed_dim, num_heads, name in configs:
        try:
            # Create multi-head attention layer
            mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(torch_device)
            mha.eval()

            # Create input
            x = torch.randn(batch, seq_len, embed_dim, device=torch_device)

            with torch.no_grad():
                # Use default args to capture current loop values (fixes closure bug)
                def attn_fn(layer=mha, inp=x):
                    output, _ = layer(inp, inp, inp, need_weights=False)
                    return output

                times = run_benchmark(
                    attn_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            flops = calculate_mha_flops(batch, seq_len, embed_dim, num_heads)
            result = compute_statistics(
                times=times,
                name=name,
                category="attention",
                device=device,
                framework="pytorch",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
                parameters={
                    "batch": batch, "seq_len": seq_len,
                    "embed_dim": embed_dim, "num_heads": num_heads
                },
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_scaled_dot_product_tenzor(
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark Tenzor scaled dot-product attention kernel."""
    import tenzor as tz
    tz.initialize()

    results = []
    sync_fn = get_tenzor_sync_fn(device)

    configs = [
        (8, 12, 512, 512, 64, "Self-Attn 512x512"),
        # (4, 16, 1024, 1024, 64, "Self-Attn 1024x1024"),  # Too slow on CPU
        # (2, 12, 2048, 2048, 64, "Self-Attn 2048x2048"),  # Too slow on CPU
        (8, 12, 128, 512, 64, "Cross-Attn 128x512"),
    ]

    for batch, heads, seq_q, seq_kv, head_dim, name in configs:
        try:
            if device == "cuda":
                q = tz.randn([batch, heads, seq_q, head_dim]).cuda()
                k = tz.randn([batch, heads, seq_kv, head_dim]).cuda()
                v = tz.randn([batch, heads, seq_kv, head_dim]).cuda()
            else:
                q = tz.randn([batch, heads, seq_q, head_dim])
                k = tz.randn([batch, heads, seq_kv, head_dim])
                v = tz.randn([batch, heads, seq_kv, head_dim])

            scale = 1.0 / (head_dim ** 0.5)

            # Use default args to capture current loop values (fixes closure bug)
            def sdp_fn(q_=q, k_=k, v_=v, s=scale):
                # Manual scaled dot-product attention
                k_t = k_.transpose(-2, -1)
                scores = tz.matmul(q_, k_t)
                scores = tz.mul(scores, s)
                scores_var = tz.Variable(scores, False)
                attn_weights = tz.nn.softmax(scores_var, -1)
                output = tz.matmul(attn_weights.tensor(), v_)
                return output

            times = run_benchmark(
                sdp_fn,
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
                sync_fn=sync_fn,
            )

            flops = calculate_attention_flops(batch, heads, seq_q, seq_kv, head_dim)
            result = compute_statistics(
                times=times,
                name=f"SDP {name}",
                category="scaled_dot_product",
                device=device,
                framework="tenzor",
                flops=flops,
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def benchmark_scaled_dot_product_pytorch(
    device: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Benchmark PyTorch scaled dot-product attention (including Flash Attention if available)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return []

    results = []
    torch_device = torch.device(device)
    sync_fn = get_pytorch_sync_fn(device)

    configs = [
        (8, 12, 512, 512, 64, "Self-Attn 512x512"),
        # (4, 16, 1024, 1024, 64, "Self-Attn 1024x1024"),  # Too slow on CPU
        # (2, 12, 2048, 2048, 64, "Self-Attn 2048x2048"),  # Too slow on CPU
        (8, 12, 128, 512, 64, "Cross-Attn 128x512"),
    ]

    # Check if scaled_dot_product_attention is available (PyTorch 2.0+)
    has_sdpa = hasattr(F, 'scaled_dot_product_attention')

    for batch, heads, seq_q, seq_kv, head_dim, name in configs:
        try:
            q = torch.randn(batch, heads, seq_q, head_dim, device=torch_device)
            k = torch.randn(batch, heads, seq_kv, head_dim, device=torch_device)
            v = torch.randn(batch, heads, seq_kv, head_dim, device=torch_device)

            with torch.no_grad():
                if has_sdpa:
                    # Use PyTorch 2.0 optimized SDPA (includes Flash Attention)
                    # Use default args to capture current loop values (fixes closure bug)
                    def sdp_fn(q_=q, k_=k, v_=v):
                        return F.scaled_dot_product_attention(q_, k_, v_)
                    framework_name = "pytorch_sdpa"
                else:
                    # Manual implementation
                    scale = 1.0 / (head_dim ** 0.5)
                    # Use default args to capture current loop values (fixes closure bug)
                    def sdp_fn(q_=q, k_=k, v_=v, s=scale):
                        scores = torch.matmul(q_, k_.transpose(-2, -1)) * s
                        attn_weights = F.softmax(scores, dim=-1)
                        return torch.matmul(attn_weights, v_)
                    framework_name = "pytorch_manual"

                times = run_benchmark(
                    sdp_fn,
                    warmup_iterations=config.warmup_iterations,
                    benchmark_iterations=config.benchmark_iterations,
                    sync_fn=sync_fn,
                )

            flops = calculate_attention_flops(batch, heads, seq_q, seq_kv, head_dim)
            result = compute_statistics(
                times=times,
                name=f"SDP {name}",
                category="scaled_dot_product",
                device=device,
                framework=framework_name,
                flops=flops,
                warmup_iterations=config.warmup_iterations,
            )
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return results


def run_attention_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all attention benchmarks with Tenzor vs PyTorch comparison."""
    config = config or DEFAULT_CONFIG
    all_results = []

    print("\n" + "=" * 70)
    print("  ATTENTION / TRANSFORMER BENCHMARKS")
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

        # Multi-head attention benchmarks
        print("\n--- Tenzor Multi-Head Attention ---")
        if device != "cuda" or check_tenzor_cuda_available():
            tenzor_mha = benchmark_tenzor_attention(ATTENTION_CONFIGS, device, config)
            all_results.extend(tenzor_mha)
        else:
            tenzor_mha = []
            print("  [SKIP] Tenzor CUDA not available")

        if config.compare_with_pytorch:
            print("\n--- PyTorch Multi-Head Attention ---")
            if device != "cuda" or check_pytorch_cuda_available():
                pytorch_mha = benchmark_pytorch_attention(ATTENTION_CONFIGS, device, config)
                all_results.extend(pytorch_mha)
            else:
                pytorch_mha = []
                print("  [SKIP] PyTorch CUDA not available")

            # Print comparison (only if both have results)
            if tenzor_mha and pytorch_mha:
                print("\n" + "=" * 70)
                print("  COMPARISON: Tenzor vs PyTorch (Multi-Head Attention)")
                print("=" * 70)
                print(f"{'Configuration':<35} {'Tenzor (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
                print("-" * 75)

                for tz_r, pt_r in zip(tenzor_mha, pytorch_mha):
                    speedup = tz_r.speedup_vs(pt_r)
                    status = "FASTER" if speedup > 1 else "SLOWER"
                    print(f"{tz_r.name:<35} {tz_r.mean_ms:<15.3f} {pt_r.mean_ms:<15.3f} {speedup:.2f}x {status}")

        # Scaled dot-product attention benchmarks
        print("\n--- Tenzor Scaled Dot-Product Attention ---")
        if device != "cuda" or check_tenzor_cuda_available():
            tenzor_sdp = benchmark_scaled_dot_product_tenzor(device, config)
            all_results.extend(tenzor_sdp)
        else:
            tenzor_sdp = []
            print("  [SKIP] Tenzor CUDA not available")

        if config.compare_with_pytorch:
            print("\n--- PyTorch Scaled Dot-Product Attention ---")
            if device != "cuda" or check_pytorch_cuda_available():
                pytorch_sdp = benchmark_scaled_dot_product_pytorch(device, config)
                all_results.extend(pytorch_sdp)
            else:
                pytorch_sdp = []
                print("  [SKIP] PyTorch CUDA not available")

            # Print comparison (only if both have results)
            if tenzor_sdp and pytorch_sdp:
                print("\n" + "=" * 70)
                print("  COMPARISON: Tenzor vs PyTorch (Scaled Dot-Product)")
                print("=" * 70)
                print(f"{'Configuration':<35} {'Tenzor (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
                print("-" * 75)

                for tz_r, pt_r in zip(tenzor_sdp, pytorch_sdp):
                    speedup = tz_r.speedup_vs(pt_r)
                    status = "FASTER" if speedup > 1 else "SLOWER"
                    print(f"{tz_r.name:<35} {tz_r.mean_ms:<15.3f} {pt_r.mean_ms:<15.3f} {speedup:.2f}x {status}")

    return all_results


if __name__ == "__main__":
    from benchmark_config import QUICK_CONFIG
    results = run_attention_benchmarks(QUICK_CONFIG)

    from benchmark_utils import save_results
    import os
    os.makedirs("results", exist_ok=True)
    save_results(results, "results/attention_benchmarks.json")
