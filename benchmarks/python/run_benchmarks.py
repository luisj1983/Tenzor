#!/usr/bin/env python3
"""
Tenzor Benchmark Suite - Main Runner
====================================
Run comprehensive benchmarks comparing Tenzor with PyTorch.

Usage:
    python run_benchmarks.py                    # Full benchmark suite
    python run_benchmarks.py --quick            # Quick test run
    python run_benchmarks.py --category matmul  # Specific category
    python run_benchmarks.py --device cuda      # Specific device
    python run_benchmarks.py --no-pytorch       # Skip PyTorch comparison
"""

import os

# Fix UCX library conflict between PyTorch and Tenzor
# Must be set BEFORE importing either library
os.environ['UCX_TLS'] = 'tcp,cuda_copy,cuda_ipc'  # Disable problematic transports
os.environ['UCX_MEMTYPE_CACHE'] = 'n'  # Disable memory type cache
os.environ['UCX_RNDV_SCHEME'] = 'get_zcopy'  # Use zero-copy get
os.environ['UCX_ERROR_SIGNALS'] = ''  # Disable UCX signal handling
os.environ['UCX_LOG_LEVEL'] = 'error'  # Suppress UCX warnings

# CRITICAL: Import PyTorch FIRST before Tenzor to avoid UCX signal handler conflicts
# PyTorch's UCX initialization must happen before Tenzor loads its backends
try:
    import torch
    _pytorch_available = True
except ImportError:
    _pytorch_available = False

import sys
import argparse
from datetime import datetime
from typing import List

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_config import BenchmarkConfig, DEFAULT_CONFIG, QUICK_CONFIG, FULL_CONFIG
from benchmark_utils import BenchmarkResult, save_results, get_system_info
from bench_matmul import run_matmul_benchmarks
from bench_conv2d import run_conv2d_benchmarks
from bench_nn_layers import run_nn_layer_benchmarks
from bench_training import run_training_benchmarks
from bench_attention import run_attention_benchmarks
from bench_normalization import run_normalization_benchmarks
from bench_mixed_precision import run_mixed_precision_benchmarks
from bench_models import run_model_benchmarks
from bench_embeddings import run_embedding_benchmarks
from bench_rnn import run_rnn_benchmarks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tenzor Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick benchmarks (fewer iterations, smaller sizes)"
    )

    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full benchmarks (more iterations for precision)"
    )

    parser.add_argument(
        "--category", "-c",
        choices=["all", "matmul", "conv2d", "nn_layers", "training",
                 "attention", "normalization", "mixed_precision",
                 "models", "embeddings", "rnn"],
        default="all",
        help="Benchmark category to run"
    )

    parser.add_argument(
        "--device", "-d",
        choices=["all", "cpu", "cuda"],
        default="all",
        help="Device to benchmark on"
    )

    parser.add_argument(
        "--no-pytorch",
        action="store_true",
        help="Skip PyTorch comparison"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=None,
        help="Number of benchmark iterations"
    )

    return parser.parse_args()


def create_config(args) -> BenchmarkConfig:
    """Create configuration based on arguments."""
    if args.quick:
        config = QUICK_CONFIG
    elif args.full:
        config = FULL_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Override devices
    if args.device != "all":
        config.devices = [args.device]

    # Override PyTorch comparison
    if args.no_pytorch:
        config.compare_with_pytorch = False

    # Override iterations
    if args.iterations:
        config.benchmark_iterations = args.iterations

    return config


def run_all_benchmarks(config: BenchmarkConfig, category: str) -> List[BenchmarkResult]:
    """Run benchmarks based on category selection."""
    all_results = []

    if category in ["all", "matmul"]:
        print("\n" + "=" * 80)
        print("  SECTION 1: MATRIX MULTIPLICATION")
        print("=" * 80)
        all_results.extend(run_matmul_benchmarks(config))

    if category in ["all", "conv2d"]:
        print("\n" + "=" * 80)
        print("  SECTION 2: CONVOLUTIONS")
        print("=" * 80)
        all_results.extend(run_conv2d_benchmarks(config))

    if category in ["all", "nn_layers"]:
        print("\n" + "=" * 80)
        print("  SECTION 3: NEURAL NETWORK LAYERS")
        print("=" * 80)
        all_results.extend(run_nn_layer_benchmarks(config))

    if category in ["all", "training"]:
        print("\n" + "=" * 80)
        print("  SECTION 4: TRAINING")
        print("=" * 80)
        all_results.extend(run_training_benchmarks(config))

    if category in ["all", "attention"]:
        print("\n" + "=" * 80)
        print("  SECTION 5: ATTENTION / TRANSFORMERS")
        print("=" * 80)
        all_results.extend(run_attention_benchmarks(config))

    if category in ["all", "normalization"]:
        print("\n" + "=" * 80)
        print("  SECTION 6: NORMALIZATION LAYERS")
        print("=" * 80)
        all_results.extend(run_normalization_benchmarks(config))

    if category in ["all", "mixed_precision"]:
        print("\n" + "=" * 80)
        print("  SECTION 7: MIXED PRECISION")
        print("=" * 80)
        all_results.extend(run_mixed_precision_benchmarks(config))

    if category in ["all", "models"]:
        print("\n" + "=" * 80)
        print("  SECTION 8: END-TO-END MODELS")
        print("=" * 80)
        all_results.extend(run_model_benchmarks(config))

    if category in ["all", "embeddings"]:
        print("\n" + "=" * 80)
        print("  SECTION 9: EMBEDDING LAYERS")
        print("=" * 80)
        all_results.extend(run_embedding_benchmarks(config))

    if category in ["all", "rnn"]:
        print("\n" + "=" * 80)
        print("  SECTION 10: RNN / LSTM / GRU")
        print("=" * 80)
        all_results.extend(run_rnn_benchmarks(config))

    return all_results


def generate_summary(results: List[BenchmarkResult]):
    """Generate summary statistics."""
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"tenzor": [], "pytorch": []}
        # Normalize framework names - treat pytorch variants as pytorch
        framework_key = "tenzor" if r.framework == "tenzor" else "pytorch"
        categories[r.category][framework_key].append(r)

    # Calculate average speedups
    for category, data in categories.items():
        tz_results = data["tenzor"]
        pt_results = data["pytorch"]

        if tz_results and pt_results:
            speedups = []
            for tz_r in tz_results:
                # Find matching PyTorch result
                for pt_r in pt_results:
                    if tz_r.name == pt_r.name and tz_r.device == pt_r.device:
                        speedups.append(tz_r.speedup_vs(pt_r))
                        break

            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                faster_count = sum(1 for s in speedups if s > 1)
                total = len(speedups)

                print(f"\n  {category.upper()}")
                print(f"    Average speedup: {avg_speedup:.2f}x")
                print(f"    Faster in {faster_count}/{total} benchmarks ({100*faster_count/total:.0f}%)")

    # Overall statistics
    tz_total = [r for r in results if r.framework == "tenzor"]
    pt_total = [r for r in results if r.framework != "tenzor"]  # All non-tenzor are pytorch variants

    if tz_total:
        print(f"\n  OVERALL")
        print(f"    Total benchmarks: {len(tz_total)}")
        print(f"    Devices tested: {set(r.device for r in tz_total)}")

        if pt_total:
            all_speedups = []
            for tz_r in tz_total:
                for pt_r in pt_total:
                    if tz_r.name == pt_r.name and tz_r.device == pt_r.device:
                        all_speedups.append(tz_r.speedup_vs(pt_r))
                        break

            if all_speedups:
                overall_avg = sum(all_speedups) / len(all_speedups)
                overall_faster = sum(1 for s in all_speedups if s > 1)
                print(f"    Overall average speedup vs PyTorch: {overall_avg:.2f}x")
                print(f"    Faster than PyTorch: {overall_faster}/{len(all_speedups)} ({100*overall_faster/len(all_speedups):.0f}%)")


def main():
    args = parse_args()
    config = create_config(args)

    # Print header
    print("=" * 80)
    print("  TENZOR BENCHMARK SUITE")
    print("=" * 80)

    # Print system info
    sys_info = get_system_info()
    print(f"\n  Date: {sys_info['timestamp']}")
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  CPU: {sys_info['processor']}")
    if 'gpu_name' in sys_info:
        print(f"  GPU: {sys_info['gpu_name']}")

    # Print config
    print(f"\n  Configuration:")
    print(f"    Warmup iterations: {config.warmup_iterations}")
    print(f"    Benchmark iterations: {config.benchmark_iterations}")
    print(f"    Devices: {config.devices}")
    print(f"    Compare with PyTorch: {config.compare_with_pytorch}")

    # Run benchmarks
    results = run_all_benchmarks(config, args.category)

    # Generate summary
    generate_summary(results)

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        output_file = f"results/benchmark_{timestamp}.json"

    save_results(results, output_file)

    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\n  Results saved to: {output_file}")
    print(f"  Generate report: python generate_report.py {output_file}")


if __name__ == "__main__":
    main()
