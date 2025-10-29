# ZeRO and CPU Offloading Architecture Design for Tenzor

**Date**: 2025-10-28
**Status**: Design Document
**Goal**: Add DeepSpeed-like ZeRO optimizer and automatic CPU offloading to Tenzor

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [ZeRO Stage 1: Optimizer State Partitioning](#zero-stage-1)
4. [ZeRO Stage 2: Gradient Partitioning](#zero-stage-2)
5. [ZeRO Stage 3: Parameter Partitioning](#zero-stage-3)
6. [CPU Offloading Engine](#cpu-offloading-engine)
7. [Parameter Offloading API](#parameter-offloading-api)
8. [Implementation Roadmap](#implementation-roadmap)
9. [API Examples](#api-examples)

---

## Overview

### Goals
- **Memory Efficiency**: Train models 10-100x larger than GPU memory
- **Performance**: Minimize overhead through smart prefetching and overlap
- **Ease of Use**: Simple API with automatic optimizations
- **Distributed**: Support multi-GPU and multi-node training

### Key Features to Implement

| Feature | Memory Savings | Complexity | Communication Overhead |
|---------|---------------|------------|----------------------|
| **ZeRO Stage 1** | 4x (optimizer states) | Low | Low |
| **ZeRO Stage 2** | 8x (+ gradients) | Medium | Medium |
| **ZeRO Stage 3** | Nx (+ parameters) | High | High |
| **ZeRO-Offload** | ~Infinite (CPU RAM) | Medium | Medium (PCIe) |
| **Parameter Offloading** | Model size | Medium | High (PCIe) |

### Memory Breakdown (BERT-Large, FP32)

| Component | Size | ZeRO-0 | ZeRO-1 | ZeRO-2 | ZeRO-3 | ZeRO+Offload |
|-----------|------|--------|--------|--------|--------|--------------|
| **Parameters** | 1.2 GB | GPU | GPU | GPU | CPU/GPU | CPU |
| **Gradients** | 1.2 GB | GPU | GPU | CPU/GPU | CPU/GPU | CPU |
| **Optimizer (Adam)** | 4.8 GB | GPU | CPU/GPU | CPU/GPU | CPU | CPU |
| **Activations** | 8 GB | GPU | GPU | GPU | GPU | GPU |
| **Total GPU** | 15.2 GB | 15.2 GB | 3.6 GB | 2.4 GB | ~1.5 GB | ~8 GB (activations) |

---

## Architecture Components

### 1. Core Memory Management System

```
┌─────────────────────────────────────────────────────────┐
│                    Memory Manager                        │
│  - Tracks all tensors (CPU/GPU/Pinned)                 │
│  - Memory pressure monitoring                           │
│  - Eviction policies (LRU, priority-based)             │
│  - Prefetch scheduling                                  │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ GPU Memory   │  │ CPU Memory   │  │ Pinned Memory│
│ Pool         │  │ Pool         │  │ Pool         │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 2. Component Hierarchy

```
Application Layer
    │
    ├─ ZeROOptimizer (Wraps standard optimizer)
    │   ├─ Stage 1: Optimizer State Partitioning
    │   ├─ Stage 2: Gradient Partitioning
    │   └─ Stage 3: Parameter Partitioning
    │
    ├─ OffloadEngine
    │   ├─ Async Transfer Manager
    │   ├─ Prefetch Scheduler
    │   └─ Memory Pool Manager
    │
    ├─ DistributedCommunicator
    │   ├─ AllGather/ReduceScatter
    │   ├─ Broadcast/Reduce
    │   └─ P2P Transfers
    │
    └─ ParameterOffloadAPI
        ├─ OffloadContext Manager
        ├─ Parameter Registration
        └─ Auto-Prefetch
```

---

## ZeRO Stage 1: Optimizer State Partitioning

### Concept
Partition optimizer states (momentum, variance for Adam) across data-parallel ranks.
Each GPU stores only 1/N of optimizer states.

### Memory Savings
- **Adam optimizer**: 2 states per parameter (momentum + variance)
- **Storage**: 8 bytes per parameter (FP32 states)
- **Total**: Model has M parameters → 8M bytes
- **With N GPUs**: Each stores 8M/N bytes
- **Savings**: 4x for Adam (from 12 bytes to 3 bytes per param per GPU)

### Architecture

```cpp
// File: include/tenzor/nn/optim/zero_optimizer.hpp

namespace tenzor::nn::optim {

/**
 * @brief ZeRO Stage 1: Optimizer State Partitioning
 *
 * Partitions optimizer states across distributed ranks while keeping
 * parameters and gradients replicated.
 */
class ZeROStage1Optimizer {
public:
    struct Config {
        int world_size;              // Number of GPUs
        int rank;                    // Current GPU rank
        bool offload_to_cpu;         // Offload states to CPU
        size_t cpu_offload_threshold; // Min size to offload (bytes)
        bool overlap_comm;           // Overlap comm with compute
    };

    ZeROStage1Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const Config& config
    );

    // Standard optimizer interface
    auto step() -> void;
    auto zero_grad() -> void;
    auto state_dict() -> std::map<std::string, Tensor>;
    auto load_state_dict(const std::map<std::string, Tensor>&) -> void;

private:
    // Partition optimizer states across ranks
    struct StatePartition {
        int rank;                    // Which rank owns this partition
        std::vector<Tensor*> params; // Parameters in this partition
        std::vector<Tensor> states;  // Optimizer states (momentum, variance)
        Device location;             // CPU or GPU
    };

    std::unique_ptr<Optimizer> base_optimizer_;
    Config config_;
    std::vector<StatePartition> partitions_;

    // Communication group for all-reduce gradients
    std::shared_ptr<DistributedGroup> comm_group_;

    // Offload engine for CPU transfers
    std::shared_ptr<OffloadEngine> offload_engine_;

    // Partition parameters across ranks
    auto partition_parameters() -> void;

    // Update only local partition of optimizer states
    auto update_local_states() -> void;

    // All-reduce gradients across ranks
    auto sync_gradients() -> void;
};

} // namespace tenzor::nn::optim
```

### Algorithm

```
Forward Pass:
    1. Standard forward (parameters replicated on all GPUs)

Backward Pass:
    1. Compute gradients (standard backward)
    2. All-reduce gradients across ranks (sum)

Optimizer Step:
    1. Each rank updates only its partition of optimizer states
       Rank 0: Updates states for params[0:M/N]
       Rank 1: Updates states for params[M/N:2M/N]
       ...
    2. Each rank updates its partition of parameters
    3. All-gather parameters to reconstruct full model
```

### Implementation Details

```cpp
// File: src/nn/optim/zero_stage1.cpp

auto ZeROStage1Optimizer::step() -> void {
    // 1. All-reduce gradients (if not already done in backward hook)
    sync_gradients();

    // 2. Update only local partition of optimizer states
    auto& local_partition = partitions_[config_.rank];

    if (local_partition.location == Device::Type::CPU) {
        // Offload mode: Bring partition to GPU, update, send back
        auto gpu_states = offload_engine_->prefetch_to_gpu(
            local_partition.states
        );
        base_optimizer_->step_partition(local_partition.params, gpu_states);
        offload_engine_->offload_to_cpu_async(gpu_states);
    } else {
        // On-device mode: Update directly on GPU
        base_optimizer_->step_partition(
            local_partition.params,
            local_partition.states
        );
    }

    // 3. All-gather updated parameters
    comm_group_->all_gather_parameters(local_partition.params);
}
```

---

## ZeRO Stage 2: Gradient Partitioning

### Concept
Partition both optimizer states AND gradients across ranks.
Gradients are reduced-scattered during backward pass.

### Memory Savings
- **Stage 1**: 8M bytes (optimizer states partitioned)
- **Stage 2**: 8M + 4M bytes (+ gradients partitioned)
- **Total**: 12M/N per GPU vs 12M per GPU
- **Savings**: 8x reduction for Adam

### Architecture

```cpp
// File: include/tenzor/nn/optim/zero_optimizer.hpp

/**
 * @brief ZeRO Stage 2: Gradient + Optimizer State Partitioning
 *
 * Extends Stage 1 by also partitioning gradients. Uses reduce-scatter
 * in backward pass to compute and partition gradients simultaneously.
 */
class ZeROStage2Optimizer : public ZeROStage1Optimizer {
public:
    ZeROStage2Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const Config& config
    );

    auto step() -> void override;

private:
    // Backward hooks for gradient reduce-scatter
    std::vector<BackwardHook> gradient_hooks_;

    // Bucket gradients for efficient reduce-scatter
    struct GradientBucket {
        std::vector<Tensor*> gradients;
        size_t total_size;
        int target_rank;  // Which rank will own these gradients
    };
    std::vector<GradientBucket> gradient_buckets_;

    // Register backward hooks for gradient partitioning
    auto register_backward_hooks(Module& model) -> void;

    // Reduce-scatter gradients during backward
    auto reduce_scatter_gradients(const GradientBucket& bucket) -> void;

    // Bucket gradients for communication efficiency
    auto create_gradient_buckets() -> void;
};
```

### Algorithm

```
Forward Pass:
    1. Standard forward (parameters replicated)

Backward Pass:
    1. Compute gradients for each layer
    2. As each layer completes backward:
       - Reduce-scatter gradients for that layer's partition
       - Rank i receives sum of gradients for params[i*M/N:(i+1)*M/N]
    3. Free full gradients, keep only local partition

Optimizer Step:
    1. Each rank updates only its partition (local gradients + states)
    2. Each rank updates its partition of parameters
    3. All-gather parameters to reconstruct full model
```

### Key Optimization: Gradient Bucketing

```cpp
auto ZeROStage2Optimizer::reduce_scatter_gradients(
    const GradientBucket& bucket
) -> void {
    // Flatten gradients into contiguous buffer
    Tensor flat_grads = flatten_tensors(bucket.gradients);

    // Reduce-scatter: Each rank gets 1/N of the sum
    // Input:  [grad_0, grad_1, ..., grad_N-1] on each rank
    // Output: Rank i gets sum of all grad_i chunks
    Tensor local_grad_sum = comm_group_->reduce_scatter(
        flat_grads,
        ReduceOp::SUM
    );

    // Unflatten into individual gradient tensors
    unflatten_into(local_grad_sum, bucket.gradients);

    // Free non-local gradients to save memory
    if (bucket.target_rank != config_.rank) {
        for (auto* grad : bucket.gradients) {
            grad->reset();  // Free memory
        }
    }
}
```

---

## ZeRO Stage 3: Parameter Partitioning

### Concept
Partition parameters, gradients, AND optimizer states across ranks.
Parameters are gathered on-demand during forward/backward passes.

### Memory Savings
- **Full Replication**: 4M (params) + 4M (grads) + 8M (states) = 16M per GPU
- **Stage 3 with N GPUs**: 16M/N per GPU
- **Savings**: Linear with world size (8 GPUs = 8x reduction)

### Architecture

```cpp
// File: include/tenzor/nn/optim/zero_optimizer.hpp

/**
 * @brief ZeRO Stage 3: Full Model Partitioning
 *
 * Most aggressive memory savings. Partitions parameters, gradients, AND
 * optimizer states. Parameters are gathered on-demand for computation.
 */
class ZeROStage3Optimizer : public ZeROStage2Optimizer {
public:
    struct Stage3Config : public Config {
        size_t prefetch_bucket_size;  // Size for prefetch buckets (bytes)
        int prefetch_depth;           // How many layers to prefetch ahead
        bool overlap_comm_compute;    // Overlap gather with compute
    };

    ZeROStage3Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const Stage3Config& config
    );

    // Register model for parameter partitioning
    auto register_model(Module& model) -> void;

private:
    Stage3Config config_;

    // Track which parameters are currently gathered
    struct ParameterState {
        Tensor* param;
        Tensor full_param;           // Temporarily gathered full parameter
        bool is_gathered;            // Is full param currently available?
        std::vector<int> users;      // Which layers are using this param
    };
    std::unordered_map<Tensor*, ParameterState> param_states_;

    // Prefetch queue for upcoming parameters
    std::queue<Tensor*> prefetch_queue_;

    // Forward hooks to gather parameters before use
    std::vector<ForwardPreHook> gather_hooks_;

    // Backward hooks to reduce-scatter gradients after use
    std::vector<BackwardPostHook> scatter_hooks_;

    // All-gather parameter before layer forward
    auto gather_parameter(Tensor* param) -> Tensor;

    // Free gathered parameter after use
    auto free_gathered_parameter(Tensor* param) -> void;

    // Prefetch parameters for upcoming layers
    auto prefetch_parameters(const std::vector<Tensor*>& params) -> void;

    // Register hooks for automatic gather/scatter
    auto register_gather_scatter_hooks(Module& model) -> void;
};
```

### Algorithm

```
Forward Pass for Layer i:
    1. Prefetch parameters for layer i+k (k layers ahead)
    2. All-gather parameters for layer i
       Rank 0: Gathers full param from all ranks
       Rank 1: Gathers full param from all ranks
       ...
    3. Compute forward with full parameters
    4. Free full parameters (keep only local partition)

Backward Pass for Layer i:
    1. All-gather parameters for layer i (if not cached)
    2. Compute backward
    3. Reduce-scatter gradients (each rank gets its partition)
    4. Free full parameters

Optimizer Step:
    1. Each rank updates only its partition
    2. Parameters remain partitioned (no all-gather needed)
```

### Key Optimization: Prefetching

```cpp
auto ZeROStage3Optimizer::gather_parameter(Tensor* param) -> Tensor {
    auto& state = param_states_[param];

    if (state.is_gathered) {
        return state.full_param;  // Already gathered
    }

    // Start prefetch for next parameters
    prefetch_parameters(get_next_parameters(param));

    // All-gather this parameter
    Tensor local_partition = param->data();
    Tensor full_param = comm_group_->all_gather(local_partition);

    state.full_param = full_param;
    state.is_gathered = true;

    return full_param;
}

auto ZeROStage3Optimizer::free_gathered_parameter(Tensor* param) -> void {
    auto& state = param_states_[param];

    // Check if any other layer is still using this parameter
    if (!state.users.empty()) {
        return;  // Keep gathered for now
    }

    // Free the gathered full parameter
    state.full_param.reset();
    state.is_gathered = false;
}
```

---

## CPU Offloading Engine

### Concept
Offload optimizer states and gradients to CPU memory to save GPU memory.
Use asynchronous transfers and prefetching to hide CPU<->GPU transfer latency.

### Architecture

```cpp
// File: include/tenzor/core/offload_engine.hpp

namespace tenzor::core {

/**
 * @brief Asynchronous CPU offloading engine
 *
 * Manages tensor movement between CPU and GPU with:
 * - Asynchronous DMA transfers
 * - Pinned memory pooling
 * - Prefetch scheduling
 * - Overlap computation with transfers
 */
class OffloadEngine {
public:
    struct Config {
        size_t pinned_memory_size;    // Size of pinned memory pool (bytes)
        int num_transfer_streams;     // Number of async transfer streams
        bool enable_prefetch;         // Enable automatic prefetching
        int prefetch_depth;           // How many tensors to prefetch ahead
        float memory_fraction;        // Fraction of CPU RAM to use
    };

    explicit OffloadEngine(const Config& config);
    ~OffloadEngine();

    // ========================================================================
    // Synchronous API
    // ========================================================================

    /**
     * @brief Offload tensor to CPU (synchronous)
     */
    auto offload_to_cpu(const Tensor& gpu_tensor) -> Tensor;

    /**
     * @brief Load tensor to GPU (synchronous)
     */
    auto load_to_gpu(const Tensor& cpu_tensor) -> Tensor;

    // ========================================================================
    // Asynchronous API
    // ========================================================================

    /**
     * @brief Offload tensor to CPU (async)
     * @return Handle for checking completion
     */
    auto offload_to_cpu_async(
        const Tensor& gpu_tensor
    ) -> TransferHandle;

    /**
     * @brief Load tensor to GPU (async)
     */
    auto load_to_gpu_async(
        const Tensor& cpu_tensor
    ) -> TransferHandle;

    /**
     * @brief Prefetch tensors to GPU
     *
     * Hints to the engine that these tensors will be needed soon.
     * Engine schedules async transfers.
     */
    auto prefetch_to_gpu(
        const std::vector<Tensor*>& tensors
    ) -> void;

    /**
     * @brief Wait for transfer to complete
     */
    auto wait(const TransferHandle& handle) -> void;

    // ========================================================================
    // Memory Management
    // ========================================================================

    /**
     * @brief Get pinned memory statistics
     */
    auto get_pinned_memory_stats() -> PinnedMemoryStats;

    /**
     * @brief Free unused pinned memory
     */
    auto defragment_pinned_memory() -> void;

    /**
     * @brief Register tensor for automatic offloading
     *
     * Engine will automatically offload this tensor when GPU memory
     * pressure is high.
     */
    auto register_auto_offload(
        Tensor* tensor,
        OffloadPriority priority = OffloadPriority::NORMAL
    ) -> void;

private:
    Config config_;

    // Pinned memory pool for fast CPU<->GPU transfers
    std::unique_ptr<PinnedMemoryAllocator> pinned_allocator_;

    // Transfer streams for async operations
    std::vector<TransferStream> transfer_streams_;

    // Prefetch scheduler
    std::unique_ptr<PrefetchScheduler> prefetch_scheduler_;

    // Memory manager for tracking tensor locations
    std::shared_ptr<MemoryManager> memory_manager_;

    // Queue of pending transfers
    std::queue<TransferRequest> transfer_queue_;

    // Background thread for managing transfers
    std::thread transfer_thread_;
    std::atomic<bool> stop_thread_{false};

    // Execute pending transfers
    auto transfer_worker() -> void;

    // Allocate from pinned memory pool
    auto allocate_pinned(size_t bytes) -> void*;

    // Free pinned memory
    auto free_pinned(void* ptr) -> void;
};

/**
 * @brief Transfer handle for async operations
 */
class TransferHandle {
public:
    auto is_ready() const -> bool;
    auto wait() -> void;
    auto get_result() -> Tensor;

private:
    friend class OffloadEngine;
    std::shared_ptr<TransferState> state_;
};

/**
 * @brief Offload priority for automatic offloading
 */
enum class OffloadPriority {
    CRITICAL,  // Never offload (e.g., frequently used parameters)
    HIGH,      // Offload last
    NORMAL,    // Default priority
    LOW        // Offload first (e.g., optimizer states)
};

} // namespace tenzor::core
```

### Transfer Pipeline

```
CPU Memory                Pinned Memory              GPU Memory
┌──────────┐             ┌──────────┐              ┌──────────┐
│          │   memcpy    │          │  DMA (fast)  │          │
│  Tensor  │────────────>│  Pinned  │─────────────>│  Tensor  │
│  (slow)  │             │  Buffer  │              │          │
└──────────┘             └──────────┘              └──────────┘
     │                         │                         │
     │                         │                         │
     ▼                         ▼                         ▼
  Pageable              Non-pageable              Device Memory
  System RAM            System RAM                (GPU VRAM)

Transfer Path:
1. Allocate pinned (non-pageable) buffer
2. Copy CPU tensor → pinned buffer (memcpy)
3. DMA transfer pinned buffer → GPU (async, ~10 GB/s PCIe 3.0)
4. Free pinned buffer (reuse from pool)
```

### Prefetch Scheduler

```cpp
// File: src/core/prefetch_scheduler.cpp

class PrefetchScheduler {
public:
    /**
     * @brief Schedule tensor for prefetch
     *
     * Predicts when tensor will be needed and starts transfer early
     * to hide latency.
     */
    auto schedule_prefetch(Tensor* tensor, int priority) -> void {
        // Calculate when to start transfer
        auto transfer_time = estimate_transfer_time(tensor);
        auto compute_time = estimate_compute_time_until_use(tensor);

        if (compute_time > transfer_time) {
            // Enough time to hide latency
            auto start_time = compute_time - transfer_time;
            schedule_at(tensor, start_time);
        } else {
            // Not enough time, transfer immediately
            transfer_immediately(tensor);
        }
    }

private:
    // Estimate transfer time based on tensor size and bandwidth
    auto estimate_transfer_time(Tensor* tensor) -> Duration {
        size_t bytes = tensor->numel() * dtype_size(tensor->dtype());
        float bandwidth = 10e9;  // 10 GB/s for PCIe 3.0
        return Duration(bytes / bandwidth);
    }

    // Estimate when tensor will be used (based on execution graph)
    auto estimate_compute_time_until_use(Tensor* tensor) -> Duration;
};
```

---

## Parameter Offloading API

### High-Level API

```cpp
// File: include/tenzor/nn/offload.hpp

namespace tenzor::nn {

/**
 * @brief Automatic parameter offloading context
 *
 * Usage:
 *   OffloadContext ctx(model, config);
 *   // Training loop - parameters automatically offloaded/prefetched
 */
class OffloadContext {
public:
    struct Config {
        bool offload_parameters;      // Offload model parameters
        bool offload_gradients;       // Offload gradients
        bool offload_optimizer_states; // Offload optimizer states
        size_t offload_threshold;     // Min size to offload (bytes)
        int prefetch_depth;           // Layers to prefetch ahead
        bool pin_first_layer;         // Keep first layer on GPU
        bool pin_last_layer;          // Keep last layer on GPU
    };

    OffloadContext(Module& model, const Config& config);
    ~OffloadContext();

    // Manually control offloading
    auto enable() -> void;
    auto disable() -> void;
    auto is_enabled() const -> bool;

    // Get offload statistics
    auto get_stats() -> OffloadStats;

private:
    Module& model_;
    Config config_;
    std::shared_ptr<OffloadEngine> engine_;

    // Track offloaded tensors
    struct TensorInfo {
        Tensor* tensor;
        Tensor cpu_copy;
        bool is_offloaded;
        int use_count;
    };
    std::unordered_map<Tensor*, TensorInfo> tensor_map_;

    // Hooks for automatic offload/prefetch
    std::vector<ForwardPreHook> prefetch_hooks_;
    std::vector<ForwardPostHook> offload_hooks_;

    auto register_hooks() -> void;
    auto offload_layer(Module* layer) -> void;
    auto prefetch_layer(Module* layer) -> void;
};

/**
 * @brief Mark parameter for offloading
 */
auto offload_param(
    Tensor& param,
    OffloadPriority priority = OffloadPriority::NORMAL
) -> void;

/**
 * @brief Context manager for compute region
 *
 * Ensures parameters are on GPU during computation.
 */
class ComputeContext {
public:
    ComputeContext(const std::vector<Tensor*>& tensors);
    ~ComputeContext();  // Auto-offload when scope exits

private:
    std::vector<Tensor*> tensors_;
    std::vector<Tensor> cpu_copies_;
};

} // namespace tenzor::nn
```

---

## Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)
**Goal**: Core infrastructure for memory management and CPU offloading

**Tasks**:
1. ✅ **Memory Manager** (`core/memory_manager.hpp`)
   - Tensor location tracking
   - Memory pressure monitoring
   - Eviction policies (LRU)

2. ✅ **Pinned Memory Allocator** (`core/pinned_allocator.hpp`)
   - Pre-allocate pinned memory pool
   - Fast allocation/deallocation
   - Thread-safe operations

3. ✅ **Transfer Engine** (`core/transfer_engine.hpp`)
   - Async CPU<->GPU transfers
   - Transfer streams
   - Synchronization primitives

4. ✅ **Tests**
   - Memory manager tests
   - Transfer benchmarks
   - Pinned memory tests

**Deliverables**:
- Basic async transfer API working
- 10+ GB/s transfer bandwidth achieved
- Memory manager tracking all tensors

---

### Phase 2: CPU Offloading (3-4 weeks)
**Goal**: Automatic parameter and gradient offloading

**Tasks**:
1. ✅ **Offload Engine** (`core/offload_engine.hpp`)
   - Implement full async API
   - Prefetch scheduler
   - Overlap transfers with compute

2. ✅ **Parameter Offloading API** (`nn/offload.hpp`)
   - OffloadContext for automatic management
   - Manual offload/prefetch functions
   - Forward/backward hooks

3. ✅ **Gradient Offloading**
   - Hook into backward pass
   - Automatic gradient offload after accumulation
   - Prefetch gradients for optimizer step

4. ✅ **Integration**
   - Module hooks for layer-wise offloading
   - Example: Train BERT on 8GB GPU

**Deliverables**:
- Train 13B parameter model on single GPU (8GB)
- Minimal performance overhead (<20%)
- Easy-to-use API

---

### Phase 3: Distributed Communication (3-4 weeks)
**Goal**: NCCL/MPI integration for multi-GPU

**Tasks**:
1. ✅ **Communication Backend** (`distributed/comm_backend.hpp`)
   - NCCL wrapper for CUDA
   - MPI fallback for CPU
   - Communication groups

2. ✅ **Collective Operations** (`distributed/collectives.hpp`)
   - AllReduce
   - AllGather / ReduceScatter
   - Broadcast / Reduce
   - Point-to-point

3. ✅ **Process Group** (`distributed/process_group.hpp`)
   - World/local ranks
   - Device assignment
   - Communication context

4. ✅ **Tests**
   - Multi-GPU collective tests
   - Communication benchmarks
   - Fault tolerance tests

**Deliverables**:
- Working NCCL integration
- Multi-GPU collective operations
- Benchmarked communication bandwidth

---

### Phase 4: ZeRO Stage 1 (2-3 weeks)
**Goal**: Optimizer state partitioning

**Tasks**:
1. ✅ **ZeROStage1Optimizer** (`nn/optim/zero_optimizer.hpp`)
   - Partition optimizer states across ranks
   - All-reduce gradients
   - All-gather parameters after update

2. ✅ **Optimizer Integration**
   - Wrap existing optimizers (Adam, SGD, AdamW)
   - State dict save/load
   - Checkpoint compatibility

3. ✅ **CPU Offload for Stage 1**
   - Offload optimizer states to CPU
   - Overlap optimizer step with prefetch

4. ✅ **Tests & Examples**
   - Multi-GPU training tests
   - Memory profiling
   - Example: GPT-2 with ZeRO-1

**Deliverables**:
- 4x memory reduction for optimizer states
- Train GPT-2 (1.5B) on 4x 16GB GPUs
- <10% performance overhead

---

### Phase 5: ZeRO Stage 2 (2-3 weeks)
**Goal**: Gradient partitioning

**Tasks**:
1. ✅ **ZeROStage2Optimizer** (`nn/optim/zero_optimizer.hpp`)
   - Reduce-scatter gradients in backward
   - Gradient bucketing
   - Memory-efficient gradient accumulation

2. ✅ **Backward Hooks**
   - Register hooks for gradient reduce-scatter
   - Free non-local gradients
   - Overlap backward with communication

3. ✅ **CPU Offload for Stage 2**
   - Offload gradients to CPU
   - Prefetch for optimizer step

4. ✅ **Tests & Examples**
   - Gradient partitioning tests
   - Example: GPT-3 (6.7B) on 4x 24GB GPUs

**Deliverables**:
- 8x memory reduction total
- Train GPT-3 (6.7B) on 4x 24GB GPUs
- <15% performance overhead

---

### Phase 6: ZeRO Stage 3 (4-5 weeks)
**Goal**: Full parameter partitioning

**Tasks**:
1. ✅ **ZeROStage3Optimizer** (`nn/optim/zero_optimizer.hpp`)
   - All-gather parameters before use
   - Free parameters after use
   - Prefetch upcoming parameters

2. ✅ **Forward/Backward Hooks**
   - Automatic gather before forward
   - Automatic scatter after backward
   - Handle nested modules

3. ✅ **Advanced Optimizations**
   - Parameter prefetching scheduler
   - Communication/compute overlap
   - Memory-aware bucket sizing

4. ✅ **CPU Offload for Stage 3**
   - Keep parameters on CPU
   - Gather to GPU on-demand
   - Stream prefetching

5. ✅ **Tests & Examples**
   - Full parameter partitioning tests
   - Example: GPT-3 (175B) on 8x 40GB GPUs

**Deliverables**:
- Nx memory reduction (N = world size)
- Train GPT-3 (175B) on 8x A100 (40GB)
- <25% performance overhead
- Production-ready API

---

### Phase 7: Optimizations & Polish (3-4 weeks)
**Goal**: Performance tuning and production readiness

**Tasks**:
1. ✅ **Performance Profiling**
   - Identify bottlenecks
   - Communication/compute overlap analysis
   - Memory access patterns

2. ✅ **Optimizations**
   - Better prefetch heuristics
   - Dynamic bucket sizing
   - Adaptive offloading based on memory pressure

3. ✅ **Fault Tolerance**
   - Checkpoint/restore for ZeRO
   - Handle GPU failures
   - Elastic training support

4. ✅ **Documentation**
   - API documentation
   - Best practices guide
   - Performance tuning guide
   - Migration guide from PyTorch/DeepSpeed

5. ✅ **Examples & Tutorials**
   - BERT training with ZeRO
   - GPT training with ZeRO
   - Custom model integration

**Deliverables**:
- Complete documentation
- 10+ working examples
- Performance on par with DeepSpeed
- Production-ready release

---

## API Examples

### Example 1: Basic CPU Offloading

```cpp
#include <tenzor/nn/offload.hpp>

// Create model
auto model = BertModel(config);

// Configure offloading
OffloadContext::Config offload_cfg;
offload_cfg.offload_parameters = true;
offload_cfg.offload_gradients = true;
offload_cfg.prefetch_depth = 2;  // Prefetch 2 layers ahead

// Enable automatic offloading
OffloadContext ctx(model, offload_cfg);

// Training loop - parameters automatically managed
for (auto& batch : dataloader) {
    auto output = model.forward(batch.input);
    auto loss = criterion(output, batch.target);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}

// Print statistics
auto stats = ctx.get_stats();
std::cout << "Peak GPU memory: " << stats.peak_gpu_memory_mb << " MB\n";
std::cout << "Average transfer overhead: " << stats.avg_transfer_time_ms << " ms\n";
```

### Example 2: ZeRO Stage 1 (Multi-GPU)

```cpp
#include <tenzor/distributed/launcher.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>

int main(int argc, char** argv) {
    // Initialize distributed training
    auto world = DistributedWorld::init(argc, argv);

    // Create model and move to GPU
    auto model = GPT2Model(config);
    model.to(Device::cuda(world.local_rank()));

    // Create optimizer with ZeRO Stage 1
    auto base_optimizer = std::make_unique<Adam>(model.parameters(), 1e-4);

    ZeROStage1Optimizer::Config zero_cfg;
    zero_cfg.world_size = world.size();
    zero_cfg.rank = world.rank();
    zero_cfg.offload_to_cpu = true;  // Offload optimizer states to CPU

    auto optimizer = ZeROStage1Optimizer(
        std::move(base_optimizer),
        zero_cfg
    );

    // Training loop
    for (auto& batch : dataloader) {
        auto output = model.forward(batch.input);
        auto loss = criterion(output, batch.target);
        loss.backward();

        // Optimizer automatically handles:
        // 1. Gradient all-reduce
        // 2. Update only local partition of states
        // 3. All-gather updated parameters
        optimizer.step();
        optimizer.zero_grad();
    }

    return 0;
}
```

### Example 3: ZeRO Stage 2 (Gradient Partitioning)

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>

// Create model
auto model = GPT2Model(config);
model.to(Device::cuda());

// Create optimizer with ZeRO Stage 2
auto base_optimizer = std::make_unique<AdamW>(model.parameters(), 2e-5);

ZeROStage2Optimizer::Config zero_cfg;
zero_cfg.world_size = world.size();
zero_cfg.rank = world.rank();
zero_cfg.offload_to_cpu = true;
zero_cfg.overlap_comm = true;  // Overlap communication with computation

auto optimizer = ZeROStage2Optimizer(
    std::move(base_optimizer),
    zero_cfg
);

// Register backward hooks for gradient reduce-scatter
optimizer.register_backward_hooks(model);

// Training loop - gradients automatically partitioned
for (auto& batch : dataloader) {
    auto output = model.forward(batch.input);
    auto loss = criterion(output, batch.target);

    // Backward automatically:
    // 1. Reduces and scatters gradients
    // 2. Frees non-local gradients
    loss.backward();

    optimizer.step();  // Each rank updates its partition
    optimizer.zero_grad();
}
```

### Example 4: ZeRO Stage 3 (Full Partitioning)

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>

// Create model
auto model = GPT3Model(config);  // 175B parameters

// Create optimizer with ZeRO Stage 3
auto base_optimizer = std::make_unique<AdamW>(model.parameters(), 1e-4);

ZeROStage3Optimizer::Stage3Config zero_cfg;
zero_cfg.world_size = world.size();
zero_cfg.rank = world.rank();
zero_cfg.offload_to_cpu = false;  // Keep on GPU for speed
zero_cfg.prefetch_bucket_size = 100 * 1024 * 1024;  // 100 MB buckets
zero_cfg.prefetch_depth = 2;  // Prefetch 2 layers ahead
zero_cfg.overlap_comm_compute = true;

auto optimizer = ZeROStage3Optimizer(
    std::move(base_optimizer),
    zero_cfg
);

// Register model for parameter partitioning
optimizer.register_model(model);

// Training loop - parameters automatically gathered/scattered
for (auto& batch : dataloader) {
    // Forward automatically:
    // 1. Gathers parameters before each layer
    // 2. Frees parameters after use
    // 3. Prefetches upcoming layers
    auto output = model.forward(batch.input);

    auto loss = criterion(output, batch.target);

    // Backward automatically:
    // 1. Gathers parameters for gradient computation
    // 2. Reduces and scatters gradients
    // 3. Frees parameters
    loss.backward();

    optimizer.step();  // Updates local partition only
    optimizer.zero_grad();
}

// Save checkpoint (automatically gathers full model)
optimizer.save_checkpoint("checkpoint.pt");
```

### Example 5: Hybrid ZeRO Stage 3 + CPU Offload

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>

// Maximum memory efficiency: ZeRO Stage 3 + CPU offload
ZeROStage3Optimizer::Stage3Config zero_cfg;
zero_cfg.world_size = world.size();
zero_cfg.rank = world.rank();

// Offload everything to CPU
zero_cfg.offload_to_cpu = true;
zero_cfg.offload_gradients = true;
zero_cfg.offload_optimizer_states = true;

// Aggressive prefetching to hide CPU<->GPU latency
zero_cfg.prefetch_depth = 4;
zero_cfg.prefetch_bucket_size = 200 * 1024 * 1024;  // 200 MB
zero_cfg.overlap_comm_compute = true;

auto optimizer = ZeROStage3Optimizer(
    std::make_unique<AdamW>(model.parameters(), 1e-4),
    zero_cfg
);

optimizer.register_model(model);

// Can now train MASSIVE models with limited GPU memory
// Example: Train 70B model on 8x 16GB GPUs
for (auto& batch : dataloader) {
    auto output = model.forward(batch.input);
    auto loss = criterion(output, batch.target);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

### Example 6: Manual Fine-Grained Control

```cpp
#include <tenzor/core/offload_engine.hpp>
#include <tenzor/nn/offload.hpp>

// Create offload engine
OffloadEngine::Config cfg;
cfg.pinned_memory_size = 4ULL * 1024 * 1024 * 1024;  // 4 GB pinned
cfg.num_transfer_streams = 4;
cfg.enable_prefetch = true;

auto engine = std::make_shared<OffloadEngine>(cfg);

// Manually offload model layers
for (auto& layer : model.transformer_layers) {
    // Offload weights to CPU
    layer.attention.q_weight = engine->offload_to_cpu(
        layer.attention.q_weight
    );
    layer.attention.k_weight = engine->offload_to_cpu(
        layer.attention.k_weight
    );
    // ... offload other weights
}

// Training with manual prefetch
for (int i = 0; i < num_layers; ++i) {
    // Prefetch next layer (async)
    if (i < num_layers - 1) {
        engine->prefetch_to_gpu({
            &layers[i+1].attention.q_weight,
            &layers[i+1].attention.k_weight,
            // ... other weights
        });
    }

    // Current layer computation
    {
        // Load current layer to GPU
        ComputeContext ctx({
            &layers[i].attention.q_weight,
            &layers[i].attention.k_weight,
            // ... other weights
        });

        // Compute (weights automatically on GPU)
        auto output = layers[i].forward(input);
    }  // Auto-offload when ctx goes out of scope
}
```

---

## Performance Targets

### Memory Efficiency

| Model | Parameters | ZeRO-0 | ZeRO-1 | ZeRO-2 | ZeRO-3 | ZeRO-3 + Offload |
|-------|-----------|--------|--------|--------|--------|------------------|
| **BERT-Base** | 110M | 2.5 GB | 0.8 GB | 0.4 GB | 0.2 GB | 0.1 GB |
| **BERT-Large** | 340M | 7.5 GB | 2.2 GB | 1.2 GB | 0.5 GB | 0.2 GB |
| **GPT-2** | 1.5B | 35 GB | 10 GB | 5 GB | 2 GB | 0.5 GB |
| **GPT-3** | 6.7B | 150 GB | 42 GB | 22 GB | 8 GB | 2 GB |
| **GPT-3** | 175B | 3.9 TB | 1.1 TB | 550 GB | 190 GB | 48 GB |

**Target**: Train 175B parameter model on 8x A100 (40GB) = 320 GB total

### Performance Overhead

| Feature | Target Overhead | Mitigation |
|---------|----------------|------------|
| **ZeRO Stage 1** | <10% | Efficient all-reduce, overlap with compute |
| **ZeRO Stage 2** | <15% | Gradient bucketing, reduce-scatter overlap |
| **ZeRO Stage 3** | <25% | Aggressive prefetching, large buckets |
| **CPU Offload** | <20% | Async transfers, prefetch scheduler |
| **ZeRO-3 + Offload** | <40% | Optimize prefetch depth, increase bucket size |

### Bandwidth Requirements

| Operation | Bandwidth | Frequency | Bottleneck |
|-----------|-----------|-----------|------------|
| **Parameter All-Gather** | ~100-300 GB/s | Per layer forward | NVLink/PCIe |
| **Gradient Reduce-Scatter** | ~100-300 GB/s | Per layer backward | NVLink/PCIe |
| **CPU Offload** | ~10-25 GB/s | Optimizer step | PCIe 3.0/4.0 |
| **Optimizer Update** | ~100 GB/s | Once per step | GPU bandwidth |

---

## Testing Strategy

### Unit Tests

```cpp
// Test ZeRO Stage 1
TEST(ZeROStage1, OptimzerStatePartitioning) {
    // Create distributed context
    auto world = MockDistributedWorld(4);  // 4 GPUs

    // Create model and optimizer
    auto model = SimpleModel();
    auto optimizer = ZeROStage1Optimizer(
        std::make_unique<Adam>(model.parameters()),
        {.world_size=4, .rank=0}
    );

    // Train for one step
    auto loss = model.forward(input);
    loss.backward();
    optimizer.step();

    // Verify optimizer states are partitioned
    EXPECT_EQ(optimizer.get_state_size(),
              model.num_parameters() / 4);
}

// Test CPU offloading
TEST(OffloadEngine, AsyncTransfer) {
    OffloadEngine engine({
        .pinned_memory_size = 1024*1024*100,  // 100 MB
        .num_transfer_streams = 2
    });

    // Create GPU tensor
    Tensor gpu_tensor({1000, 1000}, DType::Float32, Device::cuda());

    // Async offload
    auto handle = engine.offload_to_cpu_async(gpu_tensor);

    // Verify async
    EXPECT_FALSE(handle.is_ready());  // Should still be transferring

    // Wait and verify
    auto cpu_tensor = handle.get_result();
    EXPECT_EQ(cpu_tensor.device().type(), Device::Type::CPU);
    EXPECT_TRUE(tensors_equal(gpu_tensor, cpu_tensor));
}
```

### Integration Tests

```cpp
// Test full training pipeline with ZeRO Stage 3
TEST(Integration, GPT2WithZeROStage3) {
    // Setup distributed (4 GPUs)
    auto world = DistributedWorld::init_for_test(4);

    // Create model
    auto model = GPT2Model(GPT2Config::gpt2_small());

    // Create ZeRO Stage 3 optimizer
    auto optimizer = ZeROStage3Optimizer(
        std::make_unique<AdamW>(model.parameters(), 1e-4),
        {.world_size=4, .rank=world.rank()}
    );
    optimizer.register_model(model);

    // Train for 100 steps
    for (int step = 0; step < 100; ++step) {
        auto batch = generate_random_batch();
        auto loss = model.forward(batch);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    // Verify memory usage is reduced
    auto memory_used = get_gpu_memory_used();
    auto expected_memory = model.memory_footprint() / 4;  // Divided by world size
    EXPECT_LT(memory_used, expected_memory * 1.3);  // Allow 30% overhead
}
```

### Benchmark Suite

```cpp
// Benchmark ZeRO overhead
BENCHMARK(ZeROStage1_vs_Standard) {
    // Standard training
    auto time_standard = benchmark_training(model, standard_optimizer, 100);

    // ZeRO Stage 1 training
    auto time_zero = benchmark_training(model, zero_stage1_optimizer, 100);

    // Measure overhead
    auto overhead = (time_zero - time_standard) / time_standard * 100;

    // Report
    std::cout << "ZeRO Stage 1 overhead: " << overhead << "%\n";
    EXPECT_LT(overhead, 10.0);  // Target: <10% overhead
}
```

---

## Conclusion

This design document provides a comprehensive architecture for implementing:
1. ✅ **ZeRO Stage 1-3** - Memory-efficient distributed training
2. ✅ **Automatic CPU Offloading** - Train massive models on limited GPU memory
3. ✅ **Parameter Offloading API** - Fine-grained control over memory management

**Memory Efficiency**:
- ZeRO Stage 1: 4x reduction
- ZeRO Stage 2: 8x reduction
- ZeRO Stage 3: Nx reduction (N = number of GPUs)
- CPU Offload: ~Infinite (limited by system RAM)

**Performance**:
- Target <10-40% overhead depending on stage
- Achieved through: prefetching, overlap, bucketing, async transfers

**Implementation Timeline**:
- **Total**: ~20-25 weeks (5-6 months)
- Can be done incrementally with each phase providing value

**Next Steps**:
1. Approve design document
2. Set up development infrastructure
3. Begin Phase 1: Foundation
4. Iterate with continuous testing and benchmarking

This would make Tenzor competitive with PyTorch + DeepSpeed for large-scale model training!

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Status**: ✅ Design Complete, Ready for Implementation
