#!/usr/bin/env python3
"""
PyTorch CPU benchmark for comparison (run separately to avoid UCX conflicts).
"""

import os
import time
import statistics

def benchmark(name, func, warmup=5, iterations=50):
    for _ in range(warmup):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        'name': name,
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
    }

def print_result(r):
    print(f"  Mean: {r['mean']:8.4f} ms  Std: {r['std']:8.4f} ms  Min: {r['min']:8.4f} ms")

import torch
torch.set_num_threads(os.cpu_count())

print("=" * 70)
print(f"  PyTorch {torch.__version__} CPU Benchmark")
print("=" * 70)

SIZES = [
    (1000, "1K"),
    (10000, "10K"),
    (100000, "100K"),
    (1000000, "1M"),
    (10000000, "10M"),
]

print("\n" + "=" * 70)
print("  Tensor Creation")
print("=" * 70)

for size, label in SIZES:
    print(f"\n--- zeros({label}) ---")
    r = benchmark(f"zeros({label})", lambda s=size: torch.zeros(s))
    print_result(r)

print("\n" + "=" * 70)
print("  Element-wise Operations")
print("=" * 70)

for size, label in SIZES:
    a = torch.randn(size)
    b = torch.randn(size)

    print(f"\n--- Size: {label} ---")
    print("  ADD:")
    r = benchmark(f"add({label})", lambda: a + b)
    print_result(r)

    print("  MUL:")
    r = benchmark(f"mul({label})", lambda: a * b)
    print_result(r)

print("\n" + "=" * 70)
print("  Reductions")
print("=" * 70)

for size, label in SIZES:
    a = torch.randn(size)

    print(f"\n--- Size: {label} ---")
    print("  SUM:")
    r = benchmark(f"sum({label})", lambda: torch.sum(a))
    print_result(r)

    print("  MEAN:")
    r = benchmark(f"mean({label})", lambda: torch.mean(a))
    print_result(r)

print("\n" + "=" * 70)
print("  Matrix Multiplication")
print("=" * 70)

MATMUL_SIZES = [
    ((128, 128), (128, 128), "128x128"),
    ((256, 256), (256, 256), "256x256"),
    ((512, 512), (512, 512), "512x512"),
    ((1024, 1024), (1024, 1024), "1024x1024"),
]

for shape_a, shape_b, label in MATMUL_SIZES:
    a = torch.randn(*shape_a)
    b = torch.randn(*shape_b)

    print(f"\n--- {label} ---")
    r = benchmark(f"matmul({label})", lambda: torch.matmul(a, b))
    print_result(r)

print("\n" + "=" * 70)
print("  Math Functions")
print("=" * 70)

for size, label in SIZES[:4]:
    a = torch.randn(size)
    abs_a = torch.abs(a)

    print(f"\n--- Size: {label} ---")
    print("  EXP:")
    r = benchmark(f"exp({label})", lambda: torch.exp(a))
    print_result(r)

    print("  SQRT:")
    r = benchmark(f"sqrt({label})", lambda: torch.sqrt(abs_a))
    print_result(r)

    print("  TANH:")
    r = benchmark(f"tanh({label})", lambda: torch.tanh(a))
    print_result(r)

print("\n" + "=" * 70)
print("  Benchmark Complete")
print("=" * 70)
