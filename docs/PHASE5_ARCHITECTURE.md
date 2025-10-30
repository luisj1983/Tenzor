# Phase 5: ZeRO Stage 2 (Gradient Partitioning) Architecture

## Executive Summary

This document defines the complete architecture for ZeRO Stage 2 implementation, extending the existing ZeRO Stage 1 optimizer with gradient partitioning capabilities. ZeRO Stage 2 reduces memory usage by 8x for Adam optimizer (4x for optimizer states + 2x for gradients) through reduce-scatter operations during backward pass.

**Memory Savings:**
- **Stage 1**: 8M bytes optimizer states partitioned → 8M/N per GPU
- **Stage 2**: 8M + 4M bytes (optimizer states + gradients partitioned) → 12M/N per GPU
- **Total Reduction**: 8x for Adam (from 12M to 12M/N where N = world_size)

**Key Innovation:** Gradients are reduced and partitioned during backward pass using reduce-scatter, eliminating the need for full gradient storage on each rank.

---

## 1. Class Hierarchy and Design

### 1.1 Class Structure

```cpp
// File: include/tenzor/nn/optim/zero_optimizer.hpp

/**
 * @brief ZeRO Stage 2: Gradient + Optimizer State Partitioning
 *
 * Extends ZeRO Stage 1 by partitioning gradients during backward pass.
 * Uses reduce-scatter to compute and partition gradients simultaneously,
 * eliminating the need for full gradient storage on each rank.
 *
 * **Algorithm:**
 * 1. Forward pass: Standard computation with replicated parameters
 * 2. Backward pass: As each layer completes:
 *    - Bucket gradients for efficient communication
 *    - Reduce-scatter gradients (each rank gets 1/N of sum)
 *    - Free non-local gradients to save memory
 * 3. Optimizer step: Update local parameter partition only
 * 4. All-gather: Reconstruct full parameters across ranks
 *
 * **Memory Savings:**
 * - Adam: 8x reduction (4x optimizer states + 2x gradients)
 * - SGD with momentum: 6x reduction (2x states + 2x gradients)
 *
 * **Communication Patterns:**
 * - Backward: Reduce-scatter gradients (bucketed for efficiency)
 * - Optimizer: All-gather updated parameters
 *
 * **Bucketing Strategy:**
 * Gradients are bucketed by size to amortize communication overhead.
 * Typical bucket size: 25MB (configurable). Reduce-scatter is triggered
 * when bucket is full or backward pass completes.
 */
class ZeROStage2Optimizer : public ZeROStage1Optimizer {
public:
    /**
     * @brief Configuration for ZeRO Stage 2
     */
    struct Config : public ZeROStage1Config {
        size_t gradient_bucket_size{25 * 1024 * 1024};  ///< Bucket size in bytes (default: 25MB)
        bool overlap_grad_reduce{true};                  ///< Overlap reduce-scatter with backward
        bool free_gradients_after_reduce{true};          ///< Free non-local grads after reduce
        size_t num_gradient_buckets{0};                  ///< Number of buckets (0 = auto)

        Config() = default;

        /**
         * @brief Convert from Stage 1 config
         */
        explicit Config(const ZeROStage1Config& base)
            : ZeROStage1Config(base) {}
    };

    /**
     * @brief Construct ZeRO Stage 2 optimizer
     *
     * @param base_optimizer Base optimizer (Adam, SGD, etc.)
     * @param config ZeRO Stage 2 configuration
     * @throws std::invalid_argument if config is invalid
     */
    ZeROStage2Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const Config& config
    );

    /**
     * @brief Destructor - cleanup hooks and resources
     */
    ~ZeROStage2Optimizer() override;

    /**
     * @brief Perform optimizer step with gradient partitioning
     *
     * Algorithm:
     * 1. Ensure all gradient reduce-scatters complete
     * 2. Fetch optimizer states from CPU (if offloaded)
     * 3. Update local parameter partition only
     * 4. Offload states back to CPU (if enabled)
     * 5. All-gather updated parameters across ranks
     *
     * Note: All-reduce is NOT needed as gradients were already
     *       reduced during backward via reduce-scatter.
     */
    auto step() -> void override;

    /**
     * @brief Zero all parameter gradients
     */
    auto zero_grad() -> void override;

    /**
     * @brief Register model for gradient hooks
     *
     * Must be called after optimizer construction to enable
     * automatic gradient reduce-scatter during backward pass.
     *
     * @param model Neural network module
     */
    auto register_model(Module& model) -> void;

    /**
     * @brief Get gradient communication statistics
     */
    struct GradientStats {
        size_t num_buckets{0};                  ///< Number of gradient buckets
        size_t total_gradient_elements{0};      ///< Total gradient elements
        size_t avg_bucket_size{0};              ///< Average bucket size (bytes)
        size_t num_reduce_scatter_calls{0};     ///< Communication call count
        double reduce_scatter_time_ms{0.0};     ///< Total reduce-scatter time
    };

    auto get_gradient_stats() const -> GradientStats;

private:
    // Configuration
    Config config_;

    // Gradient Bucketing

    /**
     * @brief Bucket of gradients for efficient reduce-scatter
     */
    struct GradientBucket {
        std::vector<std::shared_ptr<Variable>> params;  ///< Parameters in bucket
        std::vector<size_t> param_offsets;              ///< Offsets in flat buffer
        std::vector<size_t> param_sizes;                ///< Size of each param
        Tensor flat_buffer;                              ///< Flattened gradient buffer
        size_t total_bytes{0};                           ///< Total bucket size
        int target_rank{-1};                             ///< Rank that owns these grads
        bool is_ready{false};                            ///< Ready for reduce-scatter

        // Synchronization for async communication
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<bool> reduce_scatter_complete{false};
    };

    std::vector<GradientBucket> gradient_buckets_;

    /**
     * @brief Hook IDs for unregistration
     */
    struct HookHandle {
        size_t backward_hook_id;
        std::weak_ptr<Variable> param;
    };
    std::vector<HookHandle> registered_hooks_;

    // Communication state
    std::atomic<size_t> pending_reduce_scatters_{0};
    std::mutex communication_mutex_;
    std::condition_variable communication_cv_;

    // Statistics
    mutable std::atomic<size_t> total_reduce_scatter_calls_{0};
    mutable std::atomic<double> total_reduce_scatter_time_{0.0};

    // Initialization

    /**
     * @brief Create gradient buckets from parameters
     *
     * Groups parameters into buckets based on size and memory layout
     * to minimize communication overhead. Uses reverse order matching
     * typical backward pass execution order.
     */
    auto create_gradient_buckets() -> void;

    /**
     * @brief Register backward hooks for gradient reduce-scatter
     *
     * Hooks are triggered when gradients are ready during backward.
     * Each hook adds gradient to bucket and triggers reduce-scatter
     * when bucket is full.
     */
    auto register_backward_hooks(Module& model) -> void;

    /**
     * @brief Assign parameter to bucket and rank
     *
     * @param param Parameter variable
     * @return Pair of (bucket_index, rank)
     */
    auto assign_param_to_bucket_and_rank(
        const std::shared_ptr<Variable>& param
    ) -> std::pair<size_t, int>;

    // Communication

    /**
     * @brief Reduce-scatter gradients in bucket
     *
     * Algorithm:
     * 1. Flatten gradients into contiguous buffer
     * 2. Perform reduce-scatter: Each rank gets 1/N of sum
     * 3. Unflatten reduced gradient into local buffer
     * 4. Free non-local gradients to save memory
     *
     * @param bucket Gradient bucket to reduce-scatter
     */
    auto reduce_scatter_gradients(GradientBucket& bucket) -> void;

    /**
     * @brief Wait for all reduce-scatter operations to complete
     */
    auto wait_for_gradient_communication() -> void;

    /**
     * @brief Flatten gradients into bucket buffer
     *
     * @param bucket Bucket to flatten into
     */
    auto flatten_gradients_to_bucket(GradientBucket& bucket) -> void;

    /**
     * @brief Unflatten reduced gradients from bucket buffer
     *
     * @param bucket Bucket to unflatten from
     */
    auto unflatten_gradients_from_bucket(const GradientBucket& bucket) -> void;

    /**
     * @brief Free non-local gradients to save memory
     *
     * After reduce-scatter, each rank only needs its local partition.
     * This frees gradients not owned by current rank.
     *
     * @param bucket Bucket whose non-local grads to free
     */
    auto free_non_local_gradients(const GradientBucket& bucket) -> void;

    // Backward Hook Callbacks

    /**
     * @brief Callback when parameter gradient is ready
     *
     * Called automatically during backward pass. Adds gradient to
     * bucket and triggers reduce-scatter when bucket is full.
     *
     * @param param Parameter whose gradient is ready
     */
    auto on_gradient_ready(const std::shared_ptr<Variable>& param) -> void;

    /**
     * @brief Process bucket when ready (full or backward complete)
     *
     * @param bucket_idx Index of bucket to process
     */
    auto process_bucket_when_ready(size_t bucket_idx) -> void;
};
```

### 1.2 Integration with ZeROStage1Optimizer

**Inheritance Rationale:**
- ZeROStage2Optimizer inherits all optimizer state partitioning from Stage 1
- Adds gradient partitioning as an additional optimization layer
- Reuses communication infrastructure (ProcessGroup, OffloadEngine)
- Overrides `step()` to skip all-reduce (already done in backward)

**Shared Components:**
- `StatePartition`: Reused for optimizer state management
- `partition_parameters()`: Reused for parameter assignment
- `update_local_partition()`: Reused for optimizer step
- `all_gather_parameters()`: Reused for parameter reconstruction

**New Components:**
- `GradientBucket`: Bucket management for efficient communication
- `register_backward_hooks()`: Automatic gradient capture
- `reduce_scatter_gradients()`: Gradient reduce-scatter logic

---

## 2. Gradient Flow Architecture

### 2.1 Forward Pass

```
┌─────────────────────────────────────────────────────────────┐
│                     FORWARD PASS                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Rank 0, 1, 2, ..., N-1: All ranks have full parameters   │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│  │  Layer1 │ -> │  Layer2 │ -> │  Layer3 │ -> ... -> Loss│
│  └─────────┘    └─────────┘    └─────────┘               │
│                                                             │
│  Parameters: Replicated across all ranks                   │
│  Memory: P (parameter memory) per rank                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Backward Pass with Gradient Partitioning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BACKWARD PASS (Stage 2)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Loss.backward() triggers reverse topological order traversal:         │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Layer3  │  │  Layer2  │  │  Layer1  │  │  Input   │              │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘              │
│       │             │             │             │                      │
│       v             v             v             v                      │
│  [Compute Grad] [Compute Grad] [Compute Grad] [Compute Grad]          │
│       │             │             │             │                      │
│       v             v             v             v                      │
│  ┌────────────────────────────────────────────────┐                   │
│  │  Backward Hook: on_gradient_ready(param)      │                   │
│  ├────────────────────────────────────────────────┤                   │
│  │  1. Add gradient to GradientBucket            │                   │
│  │  2. If bucket full:                            │                   │
│  │     - flatten_gradients_to_bucket()           │                   │
│  │     - reduce_scatter_gradients()              │                   │
│  │     - free_non_local_gradients()              │                   │
│  └────────────────────────────────────────────────┘                   │
│       │             │             │             │                      │
│       v             v             v             v                      │
│  ┌─────────────────────────────────────────────────────────┐          │
│  │          REDUCE-SCATTER OPERATION                        │          │
│  ├─────────────────────────────────────────────────────────┤          │
│  │  Input:  [grad_full] on each rank                       │          │
│  │  Output: Rank i receives sum of gradients for           │          │
│  │          params[i*M/N : (i+1)*M/N] (its partition)      │          │
│  │                                                          │          │
│  │  Rank 0: ┌─────┐                                        │          │
│  │          │ G0  │ -> reduced to Rank 0                   │          │
│  │          └─────┘                                        │          │
│  │  Rank 1: ┌─────┐                                        │          │
│  │          │ G1  │ -> reduced to Rank 1                   │          │
│  │          └─────┘                                        │          │
│  │  Rank 2: ┌─────┐                                        │          │
│  │          │ G2  │ -> reduced to Rank 2                   │          │
│  │          └─────┘                                        │          │
│  └─────────────────────────────────────────────────────────┘          │
│                                                                         │
│  Memory: After backward, each rank has only 1/N of gradients           │
│          Gradient memory reduced from G to G/N per rank                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Gradient Bucketing Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                     GRADIENT BUCKETING                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Parameters sorted in REVERSE order (matching backward):       │
│                                                                │
│  Bucket 0 (Rank 0):  [Layer_N.weight, Layer_N.bias, ...]     │
│  Bucket 1 (Rank 1):  [Layer_N-1.weight, Layer_N-1.bias, ...] │
│  Bucket 2 (Rank 2):  [Layer_N-2.weight, Layer_N-2.bias, ...] │
│  ...                                                           │
│                                                                │
│  Bucketing Rules:                                              │
│  1. Sort parameters by backward execution order (reverse)      │
│  2. Group consecutive parameters into buckets of ~25MB         │
│  3. Assign buckets to ranks in round-robin fashion             │
│  4. Trigger reduce-scatter when:                               │
│     - Bucket reaches size threshold (25MB)                     │
│     - Backward pass completes (flush remaining buckets)        │
│                                                                │
│  Benefits:                                                     │
│  - Amortizes communication overhead                            │
│  - Overlaps communication with backward computation            │
│  - Reduces number of small messages                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.4 Optimizer Step

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZER STEP (Stage 2)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  step() {                                                       │
│                                                                 │
│    1. wait_for_gradient_communication()                        │
│       └─> Ensure all reduce-scatter ops complete               │
│                                                                 │
│    2. fetch_states_to_gpu() [if CPU offload enabled]          │
│       └─> Transfer optimizer states from CPU to GPU            │
│                                                                 │
│    3. update_local_partition()                                 │
│       ├─> Rank 0: Update params[0:M/N] with local grads       │
│       ├─> Rank 1: Update params[M/N:2M/N] with local grads    │
│       └─> Rank i: Update params[i*M/N:(i+1)*M/N]              │
│                                                                 │
│    4. offload_states_to_cpu() [if CPU offload enabled]        │
│       └─> Transfer optimizer states back to CPU                │
│                                                                 │
│    5. all_gather_parameters()                                  │
│       ├─> Each rank broadcasts its updated partition           │
│       └─> Reconstruct full parameter set on all ranks          │
│                                                                 │
│    Result: All ranks have synchronized, updated parameters     │
│  }                                                              │
│                                                                 │
│  NOTE: No all_reduce_gradients() needed!                       │
│        Gradients were already reduced during backward          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Communication Strategy

### 3.1 Reduce-Scatter Operation

**Mathematical Definition:**
```
Input:  Each rank i has tensor T_i of shape [N, D]
Output: Each rank i receives tensor R_i of shape [D] where:
        R_i = sum(T_0[i], T_1[i], ..., T_{N-1}[i])
```

**Gradient Partitioning Application:**
```cpp
// Each rank has full gradients: [g_0, g_1, g_2, ..., g_{N-1}]
// After reduce-scatter:
//   Rank 0 receives: sum(g_0 from all ranks)
//   Rank 1 receives: sum(g_1 from all ranks)
//   Rank i receives: sum(g_i from all ranks)

auto reduce_scatter_gradients(GradientBucket& bucket) -> void {
    // Step 1: Flatten gradients into contiguous buffer
    flatten_gradients_to_bucket(bucket);

    // Step 2: Determine chunk size for each rank
    size_t total_elements = bucket.flat_buffer.numel();
    size_t chunk_size = (total_elements + world_size_ - 1) / world_size_;

    // Step 3: Perform reduce-scatter via ProcessGroup
    // Each rank will receive chunk[rank*chunk_size : (rank+1)*chunk_size]
    Tensor local_reduced = process_group_->reduce_scatter(
        bucket.flat_buffer,
        ReduceOp::SUM,
        chunk_size
    );

    // Step 4: Store reduced gradient in local partition
    unflatten_gradients_from_bucket(bucket);

    // Step 5: Free non-local gradients
    if (config_.free_gradients_after_reduce) {
        free_non_local_gradients(bucket);
    }

    // Step 6: Mark bucket as complete
    bucket.reduce_scatter_complete.store(true);
    bucket.cv.notify_all();
}
```

### 3.2 Communication/Compute Overlap

**Timeline with Overlap:**
```
Time ──────────────────────────────────────────────────────────>

Rank 0:
  ┌─────────┬────────────┬────────────┬─────────────┐
  │ Layer 3 │ Layer 2    │ Layer 1    │ Wait Comm   │
  │ Compute │ Compute    │ Compute    │ Complete    │
  └─────────┴────────────┴────────────┴─────────────┘
       │         │            │              │
       └─ RS0 ───┘─ RS1 ──────┘─ RS2 ────────┘
          (Background communication)

Where:
  - RS0, RS1, RS2: Reduce-scatter operations for buckets 0, 1, 2
  - Backward compute for Layer i overlaps with RS for Layer i+1
```

**Implementation:**
```cpp
// Async reduce-scatter with overlap
auto reduce_scatter_gradients(GradientBucket& bucket) -> void {
    std::lock_guard<std::mutex> lock(communication_mutex_);

    // Increment pending counter
    pending_reduce_scatters_.fetch_add(1);

    // Launch async reduce-scatter
    std::thread([this, &bucket]() {
        auto start = std::chrono::high_resolution_clock::now();

        // Flatten and reduce-scatter
        flatten_gradients_to_bucket(bucket);
        Tensor local_reduced = process_group_->reduce_scatter(
            bucket.flat_buffer,
            ReduceOp::SUM
        );
        unflatten_gradients_from_bucket(bucket);
        free_non_local_gradients(bucket);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        // Update statistics
        total_reduce_scatter_calls_.fetch_add(1);
        total_reduce_scatter_time_.fetch_add(elapsed);

        // Mark complete
        bucket.reduce_scatter_complete.store(true);
        pending_reduce_scatters_.fetch_sub(1);
        communication_cv_.notify_all();
    }).detach();
}

auto wait_for_gradient_communication() -> void {
    std::unique_lock<std::mutex> lock(communication_mutex_);
    communication_cv_.wait(lock, [this]() {
        return pending_reduce_scatters_.load() == 0;
    });
}
```

### 3.3 Bucketing Algorithm

**Pseudo-code:**
```python
def create_gradient_buckets():
    # Sort parameters in reverse order (backward execution order)
    params_reversed = reversed(model.parameters())

    buckets = []
    current_bucket = GradientBucket()
    current_size = 0
    rank_idx = 0

    for param in params_reversed:
        param_size = param.numel() * dtype_size(param.dtype)

        # Check if adding this param would exceed bucket size
        if current_size + param_size > config.gradient_bucket_size:
            # Finalize current bucket
            current_bucket.target_rank = rank_idx % world_size
            buckets.append(current_bucket)

            # Start new bucket
            current_bucket = GradientBucket()
            current_size = 0
            rank_idx += 1

        # Add param to current bucket
        current_bucket.params.append(param)
        current_size += param_size

    # Don't forget last bucket
    if current_bucket.params:
        current_bucket.target_rank = rank_idx % world_size
        buckets.append(current_bucket)

    return buckets
```

### 3.4 Communication Cost Analysis

**Stage 1 (All-Reduce):**
```
Forward:  0 communication
Backward: 0 communication (gradients computed locally)
Step:
  - All-reduce gradients: O(2 * G * (N-1) / N)  [latency: log(N)]
  - All-gather params:    O(2 * P * (N-1) / N)  [latency: log(N)]
Total: O(2 * (G + P) * (N-1) / N)
```

**Stage 2 (Reduce-Scatter + All-Gather):**
```
Forward:  0 communication
Backward:
  - Reduce-scatter grads: O(2 * G * (N-1) / N)  [latency: log(N)]
Step:
  - All-gather params:    O(2 * P * (N-1) / N)  [latency: log(N)]
Total: O(2 * (G + P) * (N-1) / N)
```

**Key Insight:** Stage 2 has same communication volume as Stage 1, but:
1. Reduce-scatter overlaps with backward compute
2. Gradients are freed immediately after reduce-scatter
3. Memory savings: G/N per rank (vs G per rank in Stage 1)

---

## 4. File Organization

### 4.1 Header File Additions

**File: `/home/lee/Projects/Tenzor/include/tenzor/nn/optim/zero_optimizer.hpp`**

Add after `ZeROStage1Optimizer` class definition (before closing namespace):

```cpp
// Location: After line 362 in current file

/**
 * @brief ZeRO Stage 2: Gradient + Optimizer State Partitioning
 *
 * [Full class definition from Section 1.1]
 */
class ZeROStage2Optimizer : public ZeROStage1Optimizer {
    // ... (as defined in Section 1.1)
};

} // namespace optim
} // namespace tenzor
```

**Additions Summary:**
- New `ZeROStage2Optimizer` class
- New `Config` struct extending `ZeROStage1Config`
- New `GradientBucket` struct for bucketing
- New `HookHandle` struct for hook management
- New `GradientStats` struct for statistics

### 4.2 Implementation File Structure

**File: `/home/lee/Projects/Tenzor/src/nn/optim/zero_optimizer.cpp`**

Append to existing file (after line 751):

```cpp
// Location: After ZeROStage1Optimizer implementation

// =============================================================================
// ZeRO Stage 2 Implementation
// =============================================================================

namespace tenzor {
namespace optim {

// -----------------------------------------------------------------------------
// Constructor & Destructor
// -----------------------------------------------------------------------------

ZeROStage2Optimizer::ZeROStage2Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const Config& config
) : ZeROStage1Optimizer(std::move(base_optimizer), config),
    config_(config) {

    // Gradient bucketing is deferred until register_model() is called
    // This allows the optimizer to be constructed before the model is finalized
}

ZeROStage2Optimizer::~ZeROStage2Optimizer() {
    // Unregister backward hooks to prevent dangling references
    for (auto& handle : registered_hooks_) {
        if (auto param = handle.param.lock()) {
            param->remove_backward_hook(handle.backward_hook_id);
        }
    }
}

// -----------------------------------------------------------------------------
// Optimizer Interface
// -----------------------------------------------------------------------------

auto ZeROStage2Optimizer::step() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Step 1: Wait for all gradient reduce-scatters to complete
    wait_for_gradient_communication();

    // Step 2: Fetch optimizer states from CPU if offloaded
    if (config_.offload_to_cpu && offload_engine_) {
        fetch_states_to_gpu();
    }

    // Step 3: Update local partition of parameters
    // Note: Gradients are already reduced, no need for all_reduce_gradients()
    update_local_partition();

    // Step 4: Offload states back to CPU if enabled
    if (config_.offload_to_cpu && offload_engine_) {
        offload_states_to_cpu();
    }

    // Step 5: All-gather updated parameters across ranks
    if (config_.world_size > 1) {
        all_gather_parameters();
    }
}

auto ZeROStage2Optimizer::zero_grad() -> void {
    // Clear all gradient buckets
    for (auto& bucket : gradient_buckets_) {
        std::lock_guard<std::mutex> lock(bucket.mutex);
        bucket.is_ready = false;
        bucket.reduce_scatter_complete.store(false);

        // Clear flat buffer
        if (bucket.flat_buffer.defined()) {
            bucket.flat_buffer = Tensor();
        }
    }

    // Reset pending counter
    pending_reduce_scatters_.store(0);

    // Call parent zero_grad
    ZeROStage1Optimizer::zero_grad();
}

auto ZeROStage2Optimizer::register_model(Module& model) -> void {
    // Create gradient buckets based on model parameters
    create_gradient_buckets();

    // Register backward hooks for automatic reduce-scatter
    register_backward_hooks(model);
}

auto ZeROStage2Optimizer::get_gradient_stats() const -> GradientStats {
    GradientStats stats;

    stats.num_buckets = gradient_buckets_.size();
    stats.num_reduce_scatter_calls = total_reduce_scatter_calls_.load();
    stats.reduce_scatter_time_ms = total_reduce_scatter_time_.load();

    for (const auto& bucket : gradient_buckets_) {
        stats.total_gradient_elements += bucket.total_bytes / sizeof(float);
    }

    if (stats.num_buckets > 0) {
        size_t total_bucket_bytes = 0;
        for (const auto& bucket : gradient_buckets_) {
            total_bucket_bytes += bucket.total_bytes;
        }
        stats.avg_bucket_size = total_bucket_bytes / stats.num_buckets;
    }

    return stats;
}

// -----------------------------------------------------------------------------
// Private: Initialization
// -----------------------------------------------------------------------------

auto ZeROStage2Optimizer::create_gradient_buckets() -> void {
    // Implementation in Section 3.3
}

auto ZeROStage2Optimizer::register_backward_hooks(Module& model) -> void {
    // Implementation in Section 2.2
}

auto ZeROStage2Optimizer::assign_param_to_bucket_and_rank(
    const std::shared_ptr<Variable>& param
) -> std::pair<size_t, int> {
    // Implementation detail
}

// -----------------------------------------------------------------------------
// Private: Communication
// -----------------------------------------------------------------------------

auto ZeROStage2Optimizer::reduce_scatter_gradients(
    GradientBucket& bucket
) -> void {
    // Implementation in Section 3.1
}

auto ZeROStage2Optimizer::wait_for_gradient_communication() -> void {
    // Implementation in Section 3.2
}

auto ZeROStage2Optimizer::flatten_gradients_to_bucket(
    GradientBucket& bucket
) -> void {
    // Implementation detail
}

auto ZeROStage2Optimizer::unflatten_gradients_from_bucket(
    const GradientBucket& bucket
) -> void {
    // Implementation detail
}

auto ZeROStage2Optimizer::free_non_local_gradients(
    const GradientBucket& bucket
) -> void {
    // Implementation detail
}

// -----------------------------------------------------------------------------
// Private: Backward Hook Callbacks
// -----------------------------------------------------------------------------

auto ZeROStage2Optimizer::on_gradient_ready(
    const std::shared_ptr<Variable>& param
) -> void {
    // Implementation detail
}

auto ZeROStage2Optimizer::process_bucket_when_ready(size_t bucket_idx) -> void {
    // Implementation detail
}

} // namespace optim
} // namespace tenzor
```

**Implementation Structure:**
1. Constructor/Destructor (hook cleanup)
2. Optimizer Interface (`step()`, `zero_grad()`, `register_model()`)
3. Initialization (`create_gradient_buckets()`, `register_backward_hooks()`)
4. Communication (reduce-scatter, flatten/unflatten, free gradients)
5. Backward Hooks (gradient ready callbacks)

### 4.3 Test File Organization

**File: `/home/lee/Projects/Tenzor/tests/nn/optim/test_zero_stage2_optimizer.cpp`**

```cpp
/**
 * @file test_zero_stage2_optimizer.cpp
 * @brief Unit tests for ZeRO Stage 2 Optimizer
 */

#include <gtest/gtest.h>
#include "tenzor/nn/optim/zero_optimizer.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/distributed/distributed.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::nn;

// Test fixture for Stage 2 tests
class ZeROStage2OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize distributed if not already
        if (!distributed::is_initialized()) {
            distributed::init_process_group(distributed::Backend::GLOO);
        }

        rank_ = distributed::get_rank();
        world_size_ = distributed::get_world_size();
    }

    void TearDown() override {
        // Cleanup
    }

    int rank_;
    int world_size_;
};

// Test Cases:
// 1. test_basic_construction
// 2. test_gradient_bucketing
// 3. test_reduce_scatter_correctness
// 4. test_backward_hook_registration
// 5. test_memory_savings
// 6. test_convergence_parity_with_stage1
// 7. test_cpu_offload_with_gradient_partitioning
// 8. test_communication_overlap
// 9. test_mixed_precision_gradients
// 10. test_checkpoint_save_load

// ... (test implementations)
```

**Test Structure:**
1. Construction and configuration tests
2. Gradient bucketing correctness
3. Reduce-scatter operation correctness
4. Backward hook mechanism
5. Memory usage verification
6. Convergence parity with Stage 1
7. CPU offload integration
8. Communication overlap validation
9. Mixed precision support
10. Checkpointing with partitioned gradients

---

## 5. Key Design Decisions

### 5.1 Why Inherit from ZeROStage1Optimizer?

**Decision:** `ZeROStage2Optimizer` inherits from `ZeROStage1Optimizer`

**Rationale:**
1. **Code Reuse:** Stage 2 reuses all optimizer state partitioning logic from Stage 1
2. **Layered Optimization:** Stage 2 adds gradient partitioning on top of state partitioning
3. **Shared Infrastructure:** Both stages use same communication backend and offload engine
4. **Incremental Complexity:** Stage 2 is conceptually "Stage 1 + gradient partitioning"

**Trade-offs:**
- **Pro:** Minimal code duplication, easier maintenance
- **Pro:** Users can switch between Stage 1 and Stage 2 easily
- **Con:** Tight coupling between stages (but acceptable given similarity)

### 5.2 Bucketing Strategy

**Decision:** Bucket gradients in reverse order (backward execution order)

**Rationale:**
1. **Backward Order:** Gradients become available in reverse layer order during backward
2. **Early Triggering:** Buckets can be triggered as soon as they're full
3. **Overlap Opportunity:** Reduce-scatter for bucket i can overlap with backward compute for bucket i-1
4. **Memory Pressure:** Earlier reduce-scatter means earlier memory freeing

**Configuration:**
- Default bucket size: 25MB (empirically optimal per DeepSpeed paper)
- Configurable via `gradient_bucket_size` parameter
- Auto-tuning: Number of buckets can be auto-determined

### 5.3 Asynchronous Communication

**Decision:** Use asynchronous reduce-scatter with overlap

**Rationale:**
1. **Performance:** Hide communication latency behind backward computation
2. **Throughput:** Maximize GPU utilization by avoiding idle time
3. **Scalability:** Critical for large models with deep layers

**Implementation:**
- Detached threads for async reduce-scatter
- Atomic counters for pending operations
- Condition variables for synchronization

**Trade-offs:**
- **Pro:** Significant speedup (up to 1.5x in backward pass)
- **Con:** Increased code complexity (thread safety)
- **Con:** Requires careful synchronization at step()

### 5.4 Gradient Memory Management

**Decision:** Free non-local gradients immediately after reduce-scatter

**Rationale:**
1. **Memory Savings:** Each rank only needs 1/N of gradients after reduce-scatter
2. **Peak Memory:** Reduces peak memory usage during backward pass
3. **Scalability:** Enables training larger models

**Configuration:**
- Controlled by `free_gradients_after_reduce` flag
- Default: `true` (aggressive memory savings)
- Can disable for debugging/profiling

### 5.5 Hook-Based Gradient Capture

**Decision:** Use backward hooks on Variables for automatic gradient capture

**Rationale:**
1. **Automatic:** No manual intervention in training loop required
2. **Flexible:** Works with any model architecture
3. **Efficient:** Gradients captured as soon as they're ready

**Integration:**
- Hooks registered via `Module::register_backward_hook()`
- Callbacks triggered automatically by autograd engine
- Cleanup handled in destructor

**Trade-offs:**
- **Pro:** Transparent to user, zero boilerplate
- **Pro:** Guaranteed to capture all gradients
- **Con:** Requires hook infrastructure in autograd system
- **Con:** Slight overhead per gradient computation

### 5.6 Compatibility with CPU Offload

**Decision:** Stage 2 fully compatible with CPU offload from Stage 1

**Rationale:**
1. **Orthogonal Features:** Gradient partitioning and state offload are independent
2. **Cumulative Savings:** Users get memory savings from both features
3. **Unified API:** Single config controls both features

**Implementation:**
- Reuses `OffloadEngine` from Stage 1
- Gradient reduce-scatter happens on GPU
- Optimizer states offloaded to CPU as in Stage 1

---

## 6. Implementation Phases

### Phase 5.1: Basic Infrastructure (Week 1)
- [ ] Implement `GradientBucket` struct
- [ ] Implement `create_gradient_buckets()` (basic version, no optimization)
- [ ] Implement `flatten_gradients_to_bucket()` and `unflatten_gradients_from_bucket()`
- [ ] Add unit tests for bucketing logic

### Phase 5.2: Communication (Week 2)
- [ ] Implement `reduce_scatter_gradients()` (synchronous version)
- [ ] Implement `free_non_local_gradients()`
- [ ] Add reduce-scatter correctness tests
- [ ] Verify memory savings in tests

### Phase 5.3: Backward Hooks (Week 3)
- [ ] Implement `register_backward_hooks()`
- [ ] Implement `on_gradient_ready()` callback
- [ ] Implement `process_bucket_when_ready()`
- [ ] Add hook integration tests

### Phase 5.4: Optimizer Integration (Week 4)
- [ ] Implement `ZeROStage2Optimizer::step()`
- [ ] Implement `zero_grad()` with bucket reset
- [ ] Add convergence tests (parity with Stage 1)
- [ ] Add end-to-end training tests

### Phase 5.5: Asynchronous Communication (Week 5)
- [ ] Add async reduce-scatter with threads
- [ ] Implement `wait_for_gradient_communication()`
- [ ] Add synchronization primitives (mutex, cv)
- [ ] Add communication overlap tests

### Phase 5.6: Optimization & Tuning (Week 6)
- [ ] Optimize bucketing algorithm (balance load)
- [ ] Add statistics tracking (`GradientStats`)
- [ ] Add profiling hooks for performance analysis
- [ ] Tune default bucket size

### Phase 5.7: CPU Offload Integration (Week 7)
- [ ] Test Stage 2 with CPU offload enabled
- [ ] Verify memory savings with both features
- [ ] Add combined feature tests
- [ ] Benchmark performance

### Phase 5.8: Documentation & Examples (Week 8)
- [ ] Write API documentation
- [ ] Create usage examples
- [ ] Write performance tuning guide
- [ ] Create migration guide from Stage 1

---

## 7. API Usage Example

### 7.1 Basic Usage

```cpp
#include "tenzor/nn/optim/zero_optimizer.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/distributed/distributed.hpp"

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::nn;

int main() {
    // Initialize distributed training
    distributed::init_process_group("nccl");
    int rank = distributed::get_rank();
    int world_size = distributed::get_world_size();

    // Create model
    class MyModel : public Module {
    public:
        MyModel() {
            fc1_ = std::make_shared<Linear>(1024, 4096);
            fc2_ = std::make_shared<Linear>(4096, 4096);
            fc3_ = std::make_shared<Linear>(4096, 1024);
            register_module("fc1", fc1_);
            register_module("fc2", fc2_);
            register_module("fc3", fc3_);
        }

        auto forward(const Variable& x) -> Variable override {
            auto h1 = fc1_->forward(x).relu();
            auto h2 = fc2_->forward(h1).relu();
            return fc3_->forward(h2);
        }

    private:
        std::shared_ptr<Linear> fc1_, fc2_, fc3_;
    };

    auto model = std::make_shared<MyModel>();
    model->to(Device::cuda(rank));

    // Create base optimizer
    auto base_adam = std::make_unique<Adam>(model->parameters(), 1e-3);

    // Wrap with ZeRO Stage 2
    ZeROStage2Optimizer::Config config;
    config.world_size = world_size;
    config.rank = rank;
    config.offload_to_cpu = true;           // Enable CPU offload
    config.gradient_bucket_size = 25 * 1024 * 1024;  // 25MB buckets
    config.overlap_grad_reduce = true;       // Enable overlap

    auto optimizer = ZeROStage2Optimizer(std::move(base_adam), config);

    // IMPORTANT: Register model for gradient hooks
    optimizer.register_model(*model);

    // Training loop
    model->train();
    for (int epoch = 0; epoch < 10; ++epoch) {
        for (auto& batch : dataloader) {
            // Zero gradients
            optimizer.zero_grad();

            // Forward pass
            auto output = model->forward(batch.input);
            auto loss = mse_loss(output, batch.target);

            // Backward pass (automatic reduce-scatter via hooks)
            loss.backward();

            // Optimizer step
            optimizer.step();
        }

        // Print statistics (rank 0 only)
        if (rank == 0) {
            auto stats = optimizer.get_gradient_stats();
            auto mem_stats = optimizer.get_memory_stats();

            std::cout << "Epoch " << epoch << ":\n";
            std::cout << "  Gradient buckets: " << stats.num_buckets << "\n";
            std::cout << "  Avg bucket size: " << stats.avg_bucket_size / 1024 / 1024 << " MB\n";
            std::cout << "  GPU optimizer memory: " << mem_stats.gpu_optimizer_memory / 1024 / 1024 << " MB\n";
            std::cout << "  CPU optimizer memory: " << mem_stats.cpu_optimizer_memory / 1024 / 1024 << " MB\n";
        }
    }

    // Cleanup
    distributed::destroy_process_group();
    return 0;
}
```

### 7.2 Advanced: Custom Bucketing

```cpp
// Custom bucketing for heterogeneous parameters
ZeROStage2Optimizer::Config config;
config.gradient_bucket_size = 50 * 1024 * 1024;  // 50MB buckets for large model
config.num_gradient_buckets = 16;  // Force 16 buckets

auto optimizer = ZeROStage2Optimizer(std::move(base_adam), config);
optimizer.register_model(*model);
```

### 7.3 Mixed with CPU Offload

```cpp
// Maximum memory savings: Gradient partitioning + CPU offload
ZeROStage2Optimizer::Config config;
config.offload_to_cpu = true;
config.cpu_offload_threshold = 512;  // Offload states > 512 bytes
config.pin_memory = true;             // Use pinned memory for fast transfers

auto optimizer = ZeROStage2Optimizer(std::move(base_adam), config);
optimizer.register_model(*model);

// Memory savings:
// - Gradients: 2x reduction (partitioned across ranks)
// - Optimizer states: 4x reduction (partitioned) + moved to CPU
// - Total: ~12x memory savings for Adam optimizer
```

---

## 8. Performance Expectations

### 8.1 Memory Savings

| Component | Stage 1 | Stage 2 | Reduction |
|-----------|---------|---------|-----------|
| Parameters (P) | P | P | 1x (replicated) |
| Gradients (G) | G | G/N | N x |
| Optimizer States (O) | O/N | O/N | N x |
| **Total** | P + G + O/N | P + G/N + O/N | ~1.5x for P=G=O |

**For Adam optimizer (O = 2P, G = P):**
- Stage 1: P + P + 2P/N = P(1 + 1 + 2/N)
- Stage 2: P + P/N + 2P/N = P(1 + 3/N)
- Reduction: ~1.7x for N=4, ~2.3x for N=8

### 8.2 Communication Overhead

| Stage | Forward | Backward | Step | Total |
|-------|---------|----------|------|-------|
| Stage 1 | 0 | 0 | 2(G+P)(N-1)/N | 2(G+P)(N-1)/N |
| Stage 2 | 0 | 2G(N-1)/N | 2P(N-1)/N | 2(G+P)(N-1)/N |

**Key Insight:** Same total communication volume, but Stage 2 overlaps gradient communication with backward compute!

### 8.3 Expected Speedup

Assuming 40% of time in backward and 10% overlap efficiency:
- Stage 1: 100% time
- Stage 2: 100% - (40% × 10%) = 96% time
- **Speedup: ~4% faster** (theoretical)

In practice, with optimized bucketing and overlap:
- **Speedup: 10-15% faster** (empirical, from DeepSpeed paper)

---

## 9. Testing Strategy

### 9.1 Unit Tests

1. **Gradient Bucketing:**
   - Verify bucket sizes are balanced
   - Verify parameters assigned to correct ranks
   - Verify reverse order (backward execution order)

2. **Reduce-Scatter Correctness:**
   - Verify each rank receives correct gradient partition
   - Verify gradient values are correctly summed
   - Verify non-local gradients are freed

3. **Backward Hooks:**
   - Verify hooks are triggered during backward
   - Verify all parameters have hooks registered
   - Verify hook cleanup on destructor

4. **Memory Management:**
   - Verify memory usage reduces after reduce-scatter
   - Verify peak memory is lower than Stage 1
   - Verify no memory leaks

### 9.2 Integration Tests

1. **Convergence Parity:**
   - Train same model with Stage 1 and Stage 2
   - Verify loss curves match
   - Verify final accuracy matches

2. **Multi-GPU Consistency:**
   - Train with different world sizes (2, 4, 8 GPUs)
   - Verify results are consistent
   - Verify scaling efficiency

3. **CPU Offload Integration:**
   - Test Stage 2 + CPU offload
   - Verify memory savings are additive
   - Verify correctness with both features

### 9.3 Performance Tests

1. **Communication Overhead:**
   - Measure reduce-scatter time vs backward time
   - Verify overlap is achieved
   - Compare with Stage 1 all-reduce time

2. **Scalability:**
   - Test with 1, 2, 4, 8, 16 GPUs
   - Measure weak scaling efficiency
   - Measure strong scaling efficiency

3. **Memory Benchmarks:**
   - Measure peak memory usage
   - Compare with Stage 1 and baseline
   - Verify memory savings scale with world size

---

## 10. Migration Guide (Stage 1 → Stage 2)

### 10.1 Code Changes

**Before (Stage 1):**
```cpp
auto optimizer = ZeROStage1Optimizer(std::move(base_adam), config);

// Training loop
for (auto& batch : dataloader) {
    optimizer.zero_grad();
    auto output = model->forward(batch.input);
    auto loss = criterion(output, batch.target);
    loss.backward();
    optimizer.step();
}
```

**After (Stage 2):**
```cpp
// Change 1: Use Stage 2 optimizer
ZeROStage2Optimizer::Config config2(config);  // Convert from Stage 1 config
auto optimizer = ZeROStage2Optimizer(std::move(base_adam), config2);

// Change 2: Register model ONCE after optimizer creation
optimizer.register_model(*model);

// Training loop (UNCHANGED)
for (auto& batch : dataloader) {
    optimizer.zero_grad();
    auto output = model->forward(batch.input);
    auto loss = criterion(output, batch.target);
    loss.backward();
    optimizer.step();
}
```

**Required Changes:**
1. Change optimizer type from `ZeROStage1Optimizer` to `ZeROStage2Optimizer`
2. Call `optimizer.register_model(*model)` once after construction
3. No other changes to training loop!

### 10.2 Configuration Changes

```cpp
// Stage 1 config
ZeROStage1Config config1;
config1.world_size = 4;
config1.rank = rank;
config1.offload_to_cpu = true;

// Stage 2 config (extends Stage 1)
ZeROStage2Optimizer::Config config2(config1);  // Convert from Stage 1
config2.gradient_bucket_size = 25 * 1024 * 1024;  // New: bucket size
config2.overlap_grad_reduce = true;               // New: overlap control
```

### 10.3 Checkpoint Compatibility

**Important:** Stage 1 and Stage 2 checkpoints are **NOT directly compatible** due to different gradient storage.

**Solution:** Use separate checkpoint paths for Stage 1 and Stage 2:
```cpp
// Stage 1 checkpoint
optimizer_stage1.save_checkpoint("checkpoint_stage1");

// Stage 2 checkpoint
optimizer_stage2.save_checkpoint("checkpoint_stage2");
```

To migrate checkpoints between stages, use the base optimizer state dict:
```cpp
// Extract base optimizer states (parameter values + optimizer states)
auto state = optimizer_stage1.base_optimizer().state_dict();

// Load into Stage 2 optimizer's base optimizer
optimizer_stage2.base_optimizer().load_state_dict(state);
```

---

## 11. Summary

This architecture document defines a complete ZeRO Stage 2 implementation that:

1. **Extends Stage 1** with minimal code duplication through inheritance
2. **Partitions gradients** during backward using reduce-scatter operations
3. **Reduces memory** by 8x for Adam optimizer (vs baseline), 2x vs Stage 1
4. **Overlaps communication** with backward compute for better performance
5. **Maintains compatibility** with CPU offload and other Stage 1 features
6. **Provides simple API** requiring only one additional call: `register_model()`

**Key Files:**
- Header: `/home/lee/Projects/Tenzor/include/tenzor/nn/optim/zero_optimizer.hpp` (append)
- Implementation: `/home/lee/Projects/Tenzor/src/nn/optim/zero_optimizer.cpp` (append)
- Tests: `/home/lee/Projects/Tenzor/tests/nn/optim/test_zero_stage2_optimizer.cpp` (new)

**Implementation Timeline:** 8 weeks (phased approach)

**Expected Impact:**
- **Memory:** 2x reduction vs Stage 1 (8x vs baseline)
- **Performance:** 10-15% faster than Stage 1 (with overlap)
- **Scalability:** Enables training models 8x larger on same hardware
