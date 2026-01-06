#!/usr/bin/env python3
"""Quick test to isolate Linear layer performance."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))

import torch
import tenzor as tz
import time

tz.initialize()

print("=" * 60)
print("Linear Layer Performance Test")
print("=" * 60)

# Test configuration
batch, in_f, out_f = 32, 784, 256
iterations = 1000
warmup = 100

print(f"\nConfiguration: batch={batch}, in={in_f}, out={out_f}")
print(f"Iterations: {iterations}, Warmup: {warmup}")

# Create Tenzor tensors
x = tz.randn([batch, in_f])
w = tz.randn([out_f, in_f])
b = tz.randn([out_f])

# Test 1: Raw matmul + transpose + add
print("\n--- Test 1: Raw matmul + transpose + add ---")
for _ in range(warmup):
    w_t = tz.transpose(w, 0, 1)
    result = tz.matmul(x, w_t) + b

start = time.perf_counter()
for _ in range(iterations):
    w_t = tz.transpose(w, 0, 1)
    result = tz.matmul(x, w_t) + b
elapsed = (time.perf_counter() - start) * 1000 / iterations
print(f"Tenzor (matmul + transpose + add): {elapsed:.3f} ms")

# Test 2: nn.Linear layer
print("\n--- Test 2: nn.Linear layer ---")
layer = tz.nn.Linear(in_f, out_f)
x_var = tz.Variable(x, requires_grad=False)

for _ in range(warmup):
    result = layer(x_var)

start = time.perf_counter()
for _ in range(iterations):
    result = layer(x_var)
elapsed = (time.perf_counter() - start) * 1000 / iterations
print(f"Tenzor nn.Linear: {elapsed:.3f} ms")

# Test 3: PyTorch comparison
print("\n--- Test 3: PyTorch comparison ---")
x_pt = torch.randn(batch, in_f)
w_pt = torch.randn(out_f, in_f)
b_pt = torch.randn(out_f)

for _ in range(warmup):
    result_pt = torch.nn.functional.linear(x_pt, w_pt, b_pt)

start = time.perf_counter()
for _ in range(iterations):
    result_pt = torch.nn.functional.linear(x_pt, w_pt, b_pt)
elapsed = (time.perf_counter() - start) * 1000 / iterations
print(f"PyTorch F.linear: {elapsed:.3f} ms")

# PyTorch nn.Linear
layer_pt = torch.nn.Linear(in_f, out_f)
for _ in range(warmup):
    result_pt = layer_pt(x_pt)

start = time.perf_counter()
for _ in range(iterations):
    result_pt = layer_pt(x_pt)
elapsed = (time.perf_counter() - start) * 1000 / iterations
print(f"PyTorch nn.Linear: {elapsed:.3f} ms")

print("\n" + "=" * 60)
