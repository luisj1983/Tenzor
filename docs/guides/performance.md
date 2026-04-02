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
