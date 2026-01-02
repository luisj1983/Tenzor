#!/usr/bin/env python3
"""
Simple benchmark for Tenzor CPU operations only (no PyTorch).
"""

import sys
import os
import time
import statistics
import ctypes

# Add tenzor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'tenzor'))

def benchmark(name, func, warmup=5, iterations=50):
    """Run benchmark and return statistics."""
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
        'max': max(times),
    }

def print_result(r):
    print(f"  Mean: {r['mean']:8.4f} ms  Std: {r['std']:8.4f} ms  Min: {r['min']:8.4f} ms")

print("=" * 70)
print("  Tenzor CPU Benchmark")
print("=" * 70)

import tenzor_core as tz
tz.initialize()
print("Tenzor initialized\n")

SIZES = [
    (1000, "1K"),
    (10000, "10K"),
    (100000, "100K"),
    (1000000, "1M"),
    (10000000, "10M"),
]

print("=" * 70)
print("  Tensor Creation")
print("=" * 70)

for size, label in SIZES:
    print(f"\n--- zeros({label}) ---")
    r = benchmark(f"zeros({label})", lambda s=size: tz.zeros([s]))
    print_result(r)

print("\n" + "=" * 70)
print("  Element-wise Operations")
print("=" * 70)

for size, label in SIZES:
    a = tz.randn([size])
    b = tz.randn([size])

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
    a = tz.randn([size])

    print(f"\n--- Size: {label} ---")
    print("  SUM:")
    r = benchmark(f"sum({label})", lambda: tz.sum(a))
    print_result(r)

    print("  MEAN:")
    r = benchmark(f"mean({label})", lambda: tz.mean(a))
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
    a = tz.randn(list(shape_a))
    b = tz.randn(list(shape_b))

    print(f"\n--- {label} ---")
    r = benchmark(f"matmul({label})", lambda: tz.matmul(a, b))
    print_result(r)

print("\n" + "=" * 70)
print("  Math Functions")
print("=" * 70)

for size, label in SIZES[:4]:
    a = tz.randn([size])
    abs_a = tz.abs(a)

    print(f"\n--- Size: {label} ---")
    print("  EXP:")
    r = benchmark(f"exp({label})", lambda: tz.exp(a))
    print_result(r)

    print("  SQRT:")
    r = benchmark(f"sqrt({label})", lambda: tz.sqrt(abs_a))
    print_result(r)

    print("  TANH:")
    r = benchmark(f"tanh({label})", lambda: tz.tanh(a))
    print_result(r)

print("\n" + "=" * 70)
print("  Benchmark Complete")
print("=" * 70)

# Print memory stats
try:
    stats = tz.memory_stats()
    print("\n--- Memory Statistics ---")
    for key, val in stats.items():
        print(f"  {key}: {val}")
except Exception as e:
    print(f"\nCould not get memory stats: {e}")
