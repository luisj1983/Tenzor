# Offload Engine Guide

The Offload Engine enables training of large neural networks that exceed GPU memory by automatically managing parameter and gradient transfers between GPU and CPU.

## Overview

### What is Offloading?

Offloading moves tensors between GPU memory (fast, limited) and CPU RAM (slower, abundant) to enable training models larger than available GPU VRAM.

```
┌─────────────────┐                    ┌─────────────────┐
│    CPU RAM      │                    │    GPU VRAM     │
│   (Storage)     │  ←── Transfer ───→ │   (Compute)     │
│   16-512 GB     │                    │    8-80 GB      │
└─────────────────┘                    └─────────────────┘
```

**Key Insight**: Computations still run on GPU - only idle parameters are stored on CPU.

---

## Quick Start

### Automatic Offloading (Recommended)

```cpp
#include <tenzor/nn/offload.hpp>
#include <tenzor/nn/module.hpp>

// Your model
auto model = MyLargeModel();

// Configure offloading
OffloadContext::Config config;
config.offload_parameters = true;
config.offload_gradients = true;
config.prefetch_depth = 2;

// Create and enable context
OffloadContext ctx(model, config);
ctx.enable();

// Training loop - everything is automatic!
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        auto output = model.forward(batch.input);
        auto loss = criterion(output, batch.target);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}

// Check results
auto stats = ctx.get_stats();
std::cout << "Peak GPU memory: " << stats.peak_gpu_memory_mb << " MB\n";
```

---

## Usage Modes

### 1. OffloadContext (Fully Automatic)

Best for training neural networks. Handles everything automatically via hooks.

```cpp
#include <tenzor/nn/offload.hpp>

OffloadContext::Config config;

// What to offload
config.offload_parameters = true;      // Move params to CPU after use
config.offload_gradients = true;       // Move grads to CPU after backward
config.offload_optimizer_states = false; // Future: optimizer state offloading

// Performance tuning
config.prefetch_depth = 2;             // Prefetch N layers ahead
config.offload_threshold = 1024*1024;  // Only offload tensors > 1 MB

// Memory management
config.pin_first_layer = true;         // Keep first layer on GPU
config.pin_last_layer = true;          // Keep last layer on GPU

// Create context
OffloadContext ctx(model, config);

// Control
ctx.enable();                          // Start automatic offloading
ctx.disable();                         // Pause offloading
ctx.synchronize();                     // Wait for pending transfers

// Statistics
auto stats = ctx.get_stats();
```

### 2. OffloadEngine (Semi-Automatic)

Register tensors for automatic offloading under memory pressure.

```cpp
#include <tenzor/core/offload_engine.hpp>

OffloadEngine::Config config;
config.memory_fraction = 0.9f;         // Trigger at 90% GPU usage
config.enable_auto_monitoring = true;  // Background monitoring thread
config.monitoring_interval_ms = 100;   // Check interval

OffloadEngine engine(config);

// Register tensors with priorities
engine.register_auto_offload(&tensor1, OffloadPriority::LOW);     // Offload first
engine.register_auto_offload(&tensor2, OffloadPriority::NORMAL);
engine.register_auto_offload(&tensor3, OffloadPriority::HIGH);    // Offload last
engine.register_auto_offload(&tensor4, OffloadPriority::CRITICAL); // Never offload

// Tensors are automatically offloaded when GPU memory pressure is high
// LOW priority tensors are offloaded first
```

### 3. Manual Control

For custom pipelines requiring explicit control.

```cpp
#include <tenzor/core/offload_engine.hpp>

OffloadEngine::Config config;
config.pinned_memory_size = 2ULL * 1024 * 1024 * 1024;  // 2 GB pinned pool
config.num_transfer_streams = 4;

OffloadEngine engine(config);

// Synchronous transfers (blocking)
Tensor cpu_tensor = engine.offload_to_cpu(gpu_tensor);
Tensor gpu_tensor = engine.load_to_gpu(cpu_tensor);

// Asynchronous transfers (non-blocking)
auto handle = engine.offload_to_cpu_async(gpu_tensor);

// Do other work while transfer runs...
compute_something_else();

// Wait and get result
handle.wait();
Tensor result = handle.get_tensor();

// Or check without blocking
if (handle.is_ready()) {
    Tensor result = handle.get_tensor();
}

// Prefetch multiple tensors
std::vector<Tensor*> next_params = {&param1, &param2, &param3};
engine.prefetch_to_gpu(next_params);
```

---

## Configuration Reference

### OffloadContext::Config

| Option | Default | Description |
|--------|---------|-------------|
| `offload_parameters` | `true` | Offload model parameters to CPU |
| `offload_gradients` | `true` | Offload gradients to CPU |
| `offload_optimizer_states` | `false` | Offload optimizer states (future) |
| `offload_threshold` | `1 MB` | Minimum tensor size to offload |
| `prefetch_depth` | `2` | Number of layers to prefetch ahead |
| `pin_first_layer` | `true` | Keep first layer on GPU always |
| `pin_last_layer` | `true` | Keep last layer on GPU always |
| `enable_statistics` | `true` | Track offload statistics |
| `cpu_memory_limit` | `16 GB` | Maximum CPU memory for offloaded tensors |
| `gpu_memory_limit` | `8 GB` | GPU memory threshold |

### OffloadEngine::Config

| Option | Default | Description |
|--------|---------|-------------|
| `pinned_memory_size` | `2 GB` | Pinned memory pool for fast DMA |
| `num_transfer_streams` | `4` | CUDA streams for parallel transfers |
| `enable_prefetch` | `true` | Enable prefetch scheduling |
| `prefetch_depth` | `8` | Maximum prefetch queue depth |
| `memory_fraction` | `0.9` | GPU memory threshold for auto-offload |
| `enable_auto_monitoring` | `true` | Background memory monitoring |
| `monitoring_interval_ms` | `100` | Monitoring check interval |

---

## Backend Support

Offloading works with all GPU backends:

| Backend | Transfer Mechanism | Status |
|---------|-------------------|--------|
| CUDA | cudaStreams + cudaEvents | Full support |
| ROCm | hipStreams + hipEvents | Full support |
| OneAPI | SYCL queues | Full support |
| Metal | MTLCommandQueue | Full support |
| Vulkan | VkCommandBuffer + VkFence | Full support |
| WebGPU | wgpuQueue callbacks | Full support |

The API is identical across backends - the engine detects the tensor's device and uses the appropriate transfer mechanism.

```cpp
// Same code works for any backend
Tensor gpu_tensor = model.parameters()[0]->data();  // Could be CUDA, ROCm, etc.
Tensor cpu_copy = engine.offload_to_cpu(gpu_tensor);
```

---

## How It Works

### Training Flow with Offloading

```
Forward Pass:
┌─────────────────────────────────────────────────────────┐
│ Layer 1    │ Layer 2    │ Layer 3    │ Layer 4    │ ... │
├────────────┼────────────┼────────────┼────────────┼─────┤
│ Prefetch   │ Compute    │ (waiting)  │ (on CPU)   │     │
│ to GPU     │ on GPU     │            │            │     │
│            │ Prefetch   │ Compute    │ (waiting)  │     │
│            │ Layer 3    │ on GPU     │            │     │
│ Offload    │            │ Prefetch   │ Compute    │     │
│ to CPU     │ Offload    │ Layer 4    │ on GPU     │     │
└────────────┴────────────┴────────────┴────────────┴─────┘
                    Time →
```

### Memory Lifecycle

1. **Initialization**: All parameters on CPU
2. **Forward Begin**: Prefetch first N layers to GPU
3. **Layer Compute**: Execute forward on GPU
4. **Layer Complete**: Offload params, prefetch next
5. **Backward Begin**: Prefetch last N layers to GPU
6. **Gradient Compute**: Execute backward on GPU
7. **Gradient Complete**: Offload gradients to CPU

---

## Performance Tips

### 1. Tune Prefetch Depth

```cpp
// More prefetch = better overlap but more GPU memory
config.prefetch_depth = 2;  // Conservative
config.prefetch_depth = 4;  // Aggressive (if you have memory)
```

### 2. Set Appropriate Threshold

```cpp
// Don't offload tiny tensors (overhead > benefit)
config.offload_threshold = 1024 * 1024;  // 1 MB minimum
```

### 3. Pin Critical Layers

```cpp
// First/last layers are used most frequently
config.pin_first_layer = true;
config.pin_last_layer = true;
```

### 4. Use Multiple Streams

```cpp
// Parallel transfers for better bandwidth utilization
config.num_transfer_streams = 4;  // or 8 for high-end GPUs
```

### 5. Allocate Enough Pinned Memory

```cpp
// Pinned memory enables fast DMA transfers
config.pinned_memory_size = 4ULL * 1024 * 1024 * 1024;  // 4 GB
```

---

## Monitoring and Statistics

### Get Statistics

```cpp
auto stats = ctx.get_stats();

std::cout << "Peak GPU memory: " << stats.peak_gpu_memory_mb << " MB\n";
std::cout << "Current CPU usage: " << stats.current_cpu_memory_mb << " MB\n";
std::cout << "Parameters offloaded: " << stats.num_parameters_offloaded << "\n";
std::cout << "Gradients offloaded: " << stats.num_gradients_offloaded << "\n";
std::cout << "Avg transfer time: " << stats.avg_transfer_time_ms << " ms\n";
std::cout << "Time saved by prefetch: " << stats.total_time_saved_ms << " ms\n";
```

### Transfer Engine Statistics

```cpp
auto engine_stats = engine.get_statistics();

std::cout << "Total transfers: " << engine_stats.total_transfers << "\n";
std::cout << "Bytes transferred: " << engine_stats.bytes_transferred << "\n";
std::cout << "Average bandwidth: " << engine_stats.avg_bandwidth_gbps << " GB/s\n";
```

---

## Common Patterns

### Training Large Language Models

```cpp
// LLM with billions of parameters
OffloadContext::Config config;
config.offload_parameters = true;
config.offload_gradients = true;
config.prefetch_depth = 3;
config.offload_threshold = 512 * 1024;  // 512 KB (transformers have many small tensors)

OffloadContext ctx(llm_model, config);
ctx.enable();

// Gradient accumulation with offloading
for (int step = 0; step < total_steps; ++step) {
    for (int micro = 0; micro < gradient_accumulation_steps; ++micro) {
        auto output = model.forward(micro_batch[micro]);
        auto loss = criterion(output, targets[micro]) / gradient_accumulation_steps;
        loss.backward();
    }
    optimizer.step();
    optimizer.zero_grad();
}
```

### Inference with Memory Constraints

```cpp
// Large model inference on limited GPU
OffloadEngine::Config config;
config.enable_prefetch = true;
config.memory_fraction = 0.8f;

OffloadEngine engine(config);

// Manual layer-by-layer inference
for (auto& layer : model.layers()) {
    // Load layer to GPU
    for (auto& param : layer->parameters()) {
        *param = engine.load_to_gpu(*param);
    }

    // Run inference
    hidden = layer->forward(hidden);

    // Offload layer
    for (auto& param : layer->parameters()) {
        *param = engine.offload_to_cpu(*param);
    }
}
```

### Mixed Precision with Offloading

```cpp
// Combine FP16 compute with offloading
OffloadContext::Config config;
config.offload_parameters = true;
config.offload_gradients = true;

OffloadContext ctx(model, config);
ctx.enable();

// Parameters offloaded as FP32, computed as FP16
for (auto& batch : dataloader) {
    auto output = model.forward(batch.input.to(DType::Float16));
    auto loss = criterion(output, batch.target);

    // Scale loss for FP16 stability
    (loss * loss_scale).backward();

    // Unscale and step
    optimizer.unscale_gradients(loss_scale);
    optimizer.step();
}
```

---

## Troubleshooting

### Out of CPU Memory

```cpp
// Reduce CPU memory limit
config.cpu_memory_limit = 8ULL * 1024 * 1024 * 1024;  // 8 GB instead of 16

// Or increase offload threshold (offload fewer tensors)
config.offload_threshold = 4 * 1024 * 1024;  // 4 MB minimum
```

### Slow Training

```cpp
// Increase prefetch depth to hide latency
config.prefetch_depth = 4;

// Use more transfer streams
config.num_transfer_streams = 8;

// Increase pinned memory pool
config.pinned_memory_size = 4ULL * 1024 * 1024 * 1024;
```

### GPU Memory Still Full

```cpp
// Lower memory threshold
config.memory_fraction = 0.7f;  // Trigger offload at 70%

// Don't pin layers
config.pin_first_layer = false;
config.pin_last_layer = false;

// Lower offload threshold
config.offload_threshold = 256 * 1024;  // 256 KB
```

---

## API Reference

### OffloadContext

```cpp
class OffloadContext {
    OffloadContext(Module& model, const Config& config);
    ~OffloadContext();

    void enable();                    // Start automatic offloading
    void disable();                   // Pause offloading
    bool is_enabled() const;          // Check if enabled
    void synchronize();               // Wait for pending transfers
    OffloadStats get_stats();         // Get statistics
};
```

### OffloadEngine

```cpp
class OffloadEngine {
    OffloadEngine(const Config& config);
    ~OffloadEngine();

    // Synchronous API
    Tensor offload_to_cpu(const Tensor& gpu_tensor);
    Tensor load_to_gpu(const Tensor& cpu_tensor);
    Tensor load_to_gpu(const Tensor& cpu_tensor, Device gpu_device);

    // Asynchronous API
    TransferHandle offload_to_cpu_async(const Tensor& gpu_tensor);
    TransferHandle load_to_gpu_async(const Tensor& cpu_tensor);
    TransferHandle load_to_gpu_async(const Tensor& cpu_tensor, Device gpu_device);

    // Prefetch
    void prefetch_to_gpu(const std::vector<Tensor*>& tensors);

    // Auto-offload registration
    void register_auto_offload(Tensor* tensor, OffloadPriority priority);
    void unregister_auto_offload(Tensor* tensor);

    // Control
    void synchronize();
    Statistics get_statistics() const;
    void reset_statistics();
};
```

### TransferHandle

```cpp
class TransferHandle {
    bool is_ready() const;    // Check if complete (non-blocking)
    void wait();              // Wait for completion (blocking)
    Tensor get_tensor();      // Get result (waits if needed)
    bool is_valid() const;    // Check if handle is valid
};
```

---

## See Also

- [TransferEngine](transfer_engine.md) - Low-level async transfer API
- [MemoryManager](memory_manager.md) - Memory tracking and pressure management
- [PinnedAllocator](pinned_allocator.md) - Pinned memory pool for fast DMA
