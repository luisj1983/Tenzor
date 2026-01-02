#!/usr/bin/env python3
"""
Simple benchmark comparing Tenzor with PyTorch on CPU operations.
"""

import os

# Fix UCX library conflict between PyTorch and Tenzor
# Must be set BEFORE importing either library
os.environ['UCX_TLS'] = 'tcp,cuda_copy,cuda_ipc'  # Disable problematic transports
os.environ['UCX_MEMTYPE_CACHE'] = 'n'  # Disable memory type cache
os.environ['UCX_RNDV_SCHEME'] = 'get_zcopy'  # Use zero-copy get
os.environ['UCX_ERROR_SIGNALS'] = ''  # Disable UCX signal handling
os.environ['UCX_NET_DEVICES'] = ''  # Disable network devices

import sys
import time
import statistics
import ctypes

# Add bin directory to library path before any imports
bin_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'bin')
bin_dir = os.path.abspath(bin_dir)

# Pre-load the libraries in correct order
core_lib = os.path.join(bin_dir, 'libtenzor_core.so')
cpu_backend = os.path.join(bin_dir, 'tenzor_backend_cpu.so')

if os.path.exists(core_lib):
    ctypes.CDLL(core_lib, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(cpu_backend):
    ctypes.CDLL(cpu_backend, mode=ctypes.RTLD_GLOBAL)

# Add tenzor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'tenzor'))

def benchmark(name, func, warmup=5, iterations=50):
    """Run benchmark and return statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        'name': name,
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
    }

def print_result(r, framework):
    print(f"  {framework:10} Mean: {r['mean']:8.4f} ms  Std: {r['std']:8.4f} ms  Min: {r['min']:8.4f} ms")

def print_comparison(tz_result, pt_result):
    speedup = pt_result['mean'] / tz_result['mean']
    if speedup > 1:
        print(f"  {'':10} Tenzor is {speedup:.2f}x FASTER than PyTorch")
    else:
        print(f"  {'':10} PyTorch is {1/speedup:.2f}x faster than Tenzor")

print("=" * 70)
print("  Tenzor vs PyTorch Benchmark (CPU)")
print("=" * 70)

# Import PyTorch first (before tenzor to avoid conflicts)
try:
    import torch
    torch.set_num_threads(os.cpu_count())
    HAS_PYTORCH = True
    print(f"\nPyTorch version: {torch.__version__}")
except ImportError:
    HAS_PYTORCH = False
    print("\nPyTorch not available, running Tenzor-only benchmarks")

# Import Tenzor
import tenzor_core as tz
tz.initialize()  # Initialize backend system
print(f"Tenzor loaded successfully")

# Test sizes
SIZES = [
    (1000, "1K"),
    (10000, "10K"),
    (100000, "100K"),
    (1000000, "1M"),
    (10000000, "10M"),
]

print("\n" + "=" * 70)
print("  Tensor Creation Benchmarks")
print("=" * 70)

for size, label in SIZES:
    print(f"\n--- Size: {label} elements ---")

    # Tenzor zeros
    tz_zeros = benchmark(
        f"zeros({label})",
        lambda s=size: tz.zeros([s]),
    )
    print_result(tz_zeros, "Tenzor")

    if HAS_PYTORCH:
        # PyTorch zeros
        pt_zeros = benchmark(
            f"zeros({label})",
            lambda s=size: torch.zeros(s),
        )
        print_result(pt_zeros, "PyTorch")
        print_comparison(tz_zeros, pt_zeros)

print("\n" + "=" * 70)
print("  Element-wise Operation Benchmarks")
print("=" * 70)

for size, label in SIZES:
    print(f"\n--- Size: {label} elements ---")

    # Create tensors
    tz_a = tz.randn([size])
    tz_b = tz.randn([size])

    if HAS_PYTORCH:
        pt_a = torch.randn(size)
        pt_b = torch.randn(size)

    # Add
    print("\n  ADD:")
    tz_add = benchmark(f"add({label})", lambda: tz_a + tz_b)
    print_result(tz_add, "Tenzor")

    if HAS_PYTORCH:
        pt_add = benchmark(f"add({label})", lambda: pt_a + pt_b)
        print_result(pt_add, "PyTorch")
        print_comparison(tz_add, pt_add)

    # Mul
    print("\n  MUL:")
    tz_mul = benchmark(f"mul({label})", lambda: tz_a * tz_b)
    print_result(tz_mul, "Tenzor")

    if HAS_PYTORCH:
        pt_mul = benchmark(f"mul({label})", lambda: pt_a * pt_b)
        print_result(pt_mul, "PyTorch")
        print_comparison(tz_mul, pt_mul)

print("\n" + "=" * 70)
print("  Reduction Benchmarks")
print("=" * 70)

for size, label in SIZES:
    print(f"\n--- Size: {label} elements ---")

    tz_a = tz.randn([size])
    if HAS_PYTORCH:
        pt_a = torch.randn(size)

    # Sum
    print("\n  SUM:")
    tz_sum = benchmark(f"sum({label})", lambda: tz.sum(tz_a))
    print_result(tz_sum, "Tenzor")

    if HAS_PYTORCH:
        pt_sum = benchmark(f"sum({label})", lambda: torch.sum(pt_a))
        print_result(pt_sum, "PyTorch")
        print_comparison(tz_sum, pt_sum)

    # Mean
    print("\n  MEAN:")
    tz_mean = benchmark(f"mean({label})", lambda: tz.mean(tz_a))
    print_result(tz_mean, "Tenzor")

    if HAS_PYTORCH:
        pt_mean = benchmark(f"mean({label})", lambda: torch.mean(pt_a))
        print_result(pt_mean, "PyTorch")
        print_comparison(tz_mean, pt_mean)

print("\n" + "=" * 70)
print("  Matrix Multiplication Benchmarks")
print("=" * 70)

MATMUL_SIZES = [
    ((128, 128), (128, 128), "128x128"),
    ((256, 256), (256, 256), "256x256"),
    ((512, 512), (512, 512), "512x512"),
    ((1024, 1024), (1024, 1024), "1024x1024"),
]

for shape_a, shape_b, label in MATMUL_SIZES:
    print(f"\n--- {label} matmul ---")

    tz_a = tz.randn(list(shape_a))
    tz_b = tz.randn(list(shape_b))

    if HAS_PYTORCH:
        pt_a = torch.randn(*shape_a)
        pt_b = torch.randn(*shape_b)

    tz_mm = benchmark(f"matmul({label})", lambda: tz.matmul(tz_a, tz_b))
    print_result(tz_mm, "Tenzor")

    if HAS_PYTORCH:
        pt_mm = benchmark(f"matmul({label})", lambda: torch.matmul(pt_a, pt_b))
        print_result(pt_mm, "PyTorch")
        print_comparison(tz_mm, pt_mm)

print("\n" + "=" * 70)
print("  Math Function Benchmarks")
print("=" * 70)

for size, label in SIZES[:4]:  # Skip 10M for math functions
    print(f"\n--- Size: {label} elements ---")

    tz_a = tz.randn([size])
    if HAS_PYTORCH:
        pt_a = torch.randn(size)

    # Exp
    print("\n  EXP:")
    tz_exp = benchmark(f"exp({label})", lambda: tz.exp(tz_a))
    print_result(tz_exp, "Tenzor")

    if HAS_PYTORCH:
        pt_exp = benchmark(f"exp({label})", lambda: torch.exp(pt_a))
        print_result(pt_exp, "PyTorch")
        print_comparison(tz_exp, pt_exp)

    # Sqrt
    print("\n  SQRT:")
    tz_abs_a = tz.abs(tz_a)  # Ensure positive for sqrt
    tz_sqrt = benchmark(f"sqrt({label})", lambda: tz.sqrt(tz_abs_a))
    print_result(tz_sqrt, "Tenzor")

    if HAS_PYTORCH:
        pt_abs_a = torch.abs(pt_a)
        pt_sqrt = benchmark(f"sqrt({label})", lambda: torch.sqrt(pt_abs_a))
        print_result(pt_sqrt, "PyTorch")
        print_comparison(tz_sqrt, pt_sqrt)

    # Tanh
    print("\n  TANH:")
    tz_tanh = benchmark(f"tanh({label})", lambda: tz.tanh(tz_a))
    print_result(tz_tanh, "Tenzor")

    if HAS_PYTORCH:
        pt_tanh = benchmark(f"tanh({label})", lambda: torch.tanh(pt_a))
        print_result(pt_tanh, "PyTorch")
        print_comparison(tz_tanh, pt_tanh)

print("\n" + "=" * 70)
print("  Benchmark Complete")
print("=" * 70)

# Print memory stats
try:
    stats = tz.memory_stats()
    print("\n--- Tenzor Memory Statistics ---")
    for key, val in stats.items():
        print(f"  {key}: {val}")
except Exception as e:
    print(f"\nCould not get memory stats: {e}")
