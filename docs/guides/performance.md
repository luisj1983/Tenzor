# Performance Tuning Guide

Tips for getting the best performance from Tenzor.

## Memory Layout

### Contiguous Tensors
Operations are fastest on contiguous tensors. Check and convert:
```python
if not x.is_contiguous():
    x = x.contiguous()
```

### Channels-Last Format
For CNN workloads, channels-last (NHWC) layout enables Tensor Core utilization:
```python
x = x.to(memory_format=tz.memory_format.channels_last)
```

## Backend Selection

| Backend | Best For | Key Advantage |
|---------|----------|---------------|
| CPU | Development, small models | Universal availability |
| CUDA | NVIDIA GPUs | cuDNN/cuBLAS acceleration |
| ROCm | AMD GPUs | rocBLAS/MIOpen acceleration |
| Vulkan | Cross-platform, mobile | Portable GPU compute |
| OneAPI | Intel GPUs | oneMKL acceleration |

## Memory Management

### Caching Allocator
Tenzor caches freed GPU memory to avoid expensive allocation calls:
```python
# Check memory usage
stats = tz.memory_stats()

# Release cached memory when needed
tz.empty_cache()
```

### Gradient Checkpointing
Trade compute for memory on deep models:
```python
# Checkpoint segments recompute activations during backward
output = tz.autograd.checkpoint(segment_fn, input)
```

## Training Optimization

### Mixed Precision
Use FP16/BF16 for 2x memory savings and faster compute:
```python
scaler = tz.amp.GradScaler()
```

### TF32 (CUDA, Ampere+)
Float32 matmul-based ops (`matmul`, `bmm`, `addmm`, `baddbmm`, and anything built
on them — including attention's Q@K^T and @V GEMMs) can use TF32 tensor cores on
Ampere-class and newer GPUs for roughly a 2x GEMM speedup, at the cost of ~1e-3
relative precision (10-bit mantissa vs FP32's 23-bit). This is the same
precision floor PyTorch's own TF32-enabled tests target (their `tf32_on_and_off`
test decorator uses ~0.001-0.005 tolerance for matmul/conv/attention ops with
TF32 active), and matches measured Tenzor attention error (~1.5e-3 to 2.2e-3
max relative, flat across sequence lengths).

TF32 is **off by default** — Tenzor targets exact CPU/CUDA FP32 parity as a
baseline guarantee, and a silent default-on would break that for anyone who
didn't know to opt out (this mirrors PyTorch's own current default: TF32 is
also off by default there, via `torch.set_float32_matmul_precision('highest')`).
Opt in with:
```bash
TENZOR_ENABLE_TF32=1 python train.py
```
or from C++: `tenzor::cuda::matmul::set_allow_tf32(true)`. This benefits any
Float32 CUDA workload that's GEMM-bound, including `scaled_dot_product_attention`
and `MultiheadAttention` — measured ~30% faster than PyTorch on BERT-base
seq=128 attention with TF32 on, vs ~30% slower with it off.

### Gradient Clipping
Prevent gradient explosions:
```python
optimizer = tz.optim.Adam(params, lr=0.001)
optimizer.clip_grad_norm(max_norm=1.0)
```

### Data Loading
Use multiple workers for data prefetching:
```python
loader = tz.data.DataLoader(dataset, batch_size=64, num_workers=4)
```

## Profiling

### Memory Tracking
```python
tz.reset_memory_stats()
# ... your code ...
stats = tz.memory_stats()
print(f"Peak: {stats.get('peak_allocated_bytes', 0) / 1e9:.2f} GB")
```

## Common Anti-Patterns

1. **Calling `.item()` in loops**: Forces GPU sync. Accumulate losses as tensors.
2. **Unnecessary `.contiguous()`**: Only call when needed (before data pointer access).
3. **Creating tensors in loops**: Pre-allocate and fill instead.
4. **Not using `no_grad()` for inference**: Wastes memory on gradient graphs.

```python
with tz.no_grad():
    predictions = model(test_data)  # No gradient overhead
```
