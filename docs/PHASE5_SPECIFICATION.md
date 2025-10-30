# Phase 5: ZeRO Stage 2 (Gradient Partitioning) - Technical Specification

**Date**: 2025-10-30
**Status**: Design Specification
**Phase**: 5 - ZeRO Stage 2 Implementation
**Dependencies**: Phase 4 (ZeRO Stage 1) Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Detailed Class Design](#detailed-class-design)
3. [Backward Hook System](#backward-hook-system)
4. [Communication Protocols](#communication-protocols)
5. [CPU Offload Extension](#cpu-offload-extension)
6. [Test Requirements](#test-requirements)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Performance Targets](#performance-targets)

---

## Overview

### Goals

ZeRO Stage 2 extends ZeRO Stage 1 by additionally partitioning gradients across distributed ranks, reducing memory consumption from **8x to 12x** for Adam optimizer.

**Key Differences from Stage 1**:
- **Stage 1**: Partitions optimizer states only; gradients are replicated and all-reduced
- **Stage 2**: Partitions both optimizer states AND gradients; uses reduce-scatter instead of all-reduce

### Memory Savings

| Component | ZeRO Stage 1 | ZeRO Stage 2 | Savings |
|-----------|--------------|--------------|---------|
| **Parameters** | Replicated | Replicated | Same |
| **Gradients** | Replicated (4M) | Partitioned (4M/N) | 4M - 4M/N |
| **Optimizer States** | Partitioned (8M/N) | Partitioned (8M/N) | Same |
| **Total per GPU** | 4M + 8M/N | 4M/N + 8M/N | 4M - 4M/N |

**Example (4 GPUs, 1.5B parameters, FP32)**:
- Model size: 6 GB parameters
- Stage 1: 6 GB (params) + 6 GB (grads) + 3 GB (states) = **15 GB per GPU**
- Stage 2: 6 GB (params) + 1.5 GB (grads) + 3 GB (states) = **10.5 GB per GPU**
- **Savings: 4.5 GB per GPU (30% reduction)**

### Algorithm Overview

```
Forward Pass:
    1. Standard forward (parameters replicated on all ranks)

Backward Pass:
    1. Compute gradients for each layer (reverse topological order)
    2. For each gradient bucket (grouped by rank ownership):
       a. Reduce-scatter: Sum gradients across ranks and partition result
       b. Rank i receives sum of gradients for its parameter partition
       c. Free non-local gradients immediately to save memory
    3. Continue backward to next layer

Optimizer Step:
    1. Each rank updates only its local partition
       - Uses locally reduced gradients
       - Updates local optimizer states
       - Updates local parameters
    2. All-gather updated parameters across ranks
    3. Parameters are now synchronized and replicated
```

---

## Detailed Class Design

### 1. ZeROStage2Optimizer Class

```cpp
/**
 * @file zero_optimizer.hpp (additions)
 * @brief ZeRO Stage 2: Gradient + Optimizer State Partitioning
 *
 * Extends ZeROStage1Optimizer by adding gradient partitioning via
 * reduce-scatter operations during backward pass.
 */

namespace tenzor {
namespace optim {

/**
 * @brief Configuration for ZeRO Stage 2 Optimizer
 *
 * Extends ZeROStage1Config with gradient bucketing parameters.
 */
struct ZeROStage2Config : public ZeROStage1Config {
    size_t gradient_bucket_size{25 * 1024 * 1024};  ///< Target bucket size (25 MB default)
    bool reduce_bucket_on_completion{true};          ///< Trigger reduce-scatter when bucket full
    bool overlap_grad_comm{true};                    ///< Overlap reduce-scatter with backward
    bool free_non_local_grads{true};                 ///< Free gradients not owned by this rank
    bool offload_gradients{false};                   ///< Offload local gradients to CPU
    int max_gradient_buckets{100};                   ///< Max number of gradient buckets

    ZeROStage2Config() = default;

    /**
     * @brief Copy constructor from Stage 1 config
     */
    explicit ZeROStage2Config(const ZeROStage1Config& base)
        : ZeROStage1Config(base) {}
};

/**
 * @brief ZeRO Stage 2: Gradient + Optimizer State Partitioning
 *
 * Partitions both optimizer states and gradients across distributed ranks.
 * Uses reduce-scatter in backward pass to compute and partition gradients
 * simultaneously, reducing memory consumption.
 *
 * **Key Features**:
 * - Gradient bucketing for efficient communication
 * - Reduce-scatter triggers on bucket completion
 * - Automatic freeing of non-local gradients
 * - Optional gradient offload to CPU
 * - Overlapped communication and computation
 *
 * **Algorithm**:
 * 1. Register backward hooks on all parameters
 * 2. During backward, accumulate gradients into buckets
 * 3. When bucket is full, trigger reduce-scatter
 * 4. Each rank receives its partition of reduced gradients
 * 5. Free gradients not owned by this rank
 * 6. Optimizer step uses local gradients only
 *
 * @code
 * // Example: Distributed training with ZeRO Stage 2
 * distributed::init_process_group("nccl");
 *
 * auto params = model.parameters();
 * auto adam = std::make_unique<Adam>(params, 1e-3);
 *
 * ZeROStage2Config config;
 * config.world_size = distributed::get_world_size();
 * config.rank = distributed::get_rank();
 * config.offload_to_cpu = true;
 * config.offload_gradients = true;
 * config.gradient_bucket_size = 25 * 1024 * 1024;  // 25 MB buckets
 *
 * auto optimizer = ZeROStage2Optimizer(std::move(adam), config);
 * optimizer.register_backward_hooks(model);  // Register hooks
 *
 * // Training loop
 * for (auto& batch : dataloader) {
 *     optimizer.zero_grad();
 *     auto loss = model.forward(batch).backward();  // Hooks auto-trigger
 *     optimizer.step();  // Updates local partition only
 * }
 * @endcode
 */
class ZeROStage2Optimizer : public ZeROStage1Optimizer {
public:
    /**
     * @brief Construct ZeRO Stage 2 optimizer
     *
     * @param base_optimizer Base optimizer (Adam, SGD, etc.)
     * @param config ZeRO Stage 2 configuration
     * @throws std::invalid_argument if config is invalid
     */
    ZeROStage2Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const ZeROStage2Config& config
    );

    /**
     * @brief Destructor - cleanup hooks and buckets
     */
    ~ZeROStage2Optimizer() override;

    /**
     * @brief Perform optimizer step with gradient partitioning
     *
     * Algorithm:
     * 1. Flush any remaining gradient buckets (reduce-scatter)
     * 2. If gradient offload: Fetch local gradients to GPU
     * 3. If optimizer state offload: Fetch local states to GPU
     * 4. Update local parameter partition with base optimizer
     * 5. If offload: Send states/gradients back to CPU
     * 6. All-gather updated parameters across ranks
     *
     * @throws std::runtime_error if distributed not initialized
     */
    auto step() -> void override;

    /**
     * @brief Register backward hooks on module parameters
     *
     * Registers hooks that trigger reduce-scatter operations
     * during backward pass when gradients are ready.
     *
     * @param module Module to register hooks on
     * @throws std::runtime_error if hooks already registered
     */
    auto register_backward_hooks(nn::Module& module) -> void;

    /**
     * @brief Manually trigger reduce-scatter for gradient bucket
     *
     * Useful for debugging or manual control flow.
     *
     * @param bucket_id Bucket ID to reduce-scatter
     */
    auto reduce_scatter_bucket(size_t bucket_id) -> void;

    /**
     * @brief Flush all pending gradient buckets
     *
     * Forces reduce-scatter for all buckets with pending gradients.
     * Called automatically by step() but can be manually invoked.
     */
    auto flush_gradient_buckets() -> void;

    /**
     * @brief Get gradient bucket statistics
     */
    struct BucketStats {
        size_t num_buckets{0};           ///< Total number of buckets
        size_t total_gradients{0};       ///< Total gradient tensors
        size_t avg_bucket_size{0};       ///< Average bucket size (bytes)
        size_t max_bucket_size{0};       ///< Largest bucket size (bytes)
        size_t min_bucket_size{0};       ///< Smallest bucket size (bytes)
        size_t num_reduce_scatters{0};   ///< Number of reduce-scatter operations
    };

    /**
     * @brief Get bucket statistics
     */
    auto get_bucket_stats() const -> BucketStats;

    /**
     * @brief Check if gradients are offloaded to CPU
     */
    auto is_gradient_offload_enabled() const -> bool {
        return config_stage2_.offload_gradients;
    }

protected:
    /**
     * @brief Gradient bucket for efficient communication
     *
     * Groups gradients by owning rank to minimize communication overhead.
     * Buckets are flushed when full or at optimizer step.
     */
    struct GradientBucket {
        int owner_rank{0};                              ///< Rank that owns these gradients
        std::vector<std::shared_ptr<Variable>> params;  ///< Parameters in bucket
        std::vector<Tensor*> gradients;                 ///< Gradient tensors (GPU)
        std::vector<Tensor> local_gradients;            ///< Local reduced gradients (after reduce-scatter)
        size_t total_bytes{0};                          ///< Total bucket size (bytes)
        bool is_reduced{false};                         ///< Has reduce-scatter completed?
        Device storage_device{Device::cpu()};           ///< Where to store reduced gradients
        std::mutex bucket_mutex;                        ///< Thread safety
    };

    /**
     * @brief Backward hook handle for cleanup
     */
    struct BackwardHookHandle {
        std::shared_ptr<Variable> param;                ///< Parameter this hook is attached to
        size_t bucket_id{0};                            ///< Gradient bucket ID
        std::function<void()> cleanup_fn;               ///< Cleanup function
    };

    // Member variables
    ZeROStage2Config config_stage2_;                    ///< Stage 2 configuration
    std::vector<GradientBucket> gradient_buckets_;      ///< All gradient buckets
    std::vector<BackwardHookHandle> backward_hooks_;    ///< Registered backward hooks
    std::atomic<bool> hooks_registered_{false};         ///< Are hooks registered?
    std::atomic<size_t> reduce_scatter_count_{0};       ///< Number of reduce-scatter ops

    // Bucket management
    mutable std::mutex buckets_mutex_;                  ///< Bucket synchronization

    /**
     * @brief Create gradient buckets from parameters
     *
     * Groups parameters into buckets based on:
     * 1. Owning rank (parameters in same partition)
     * 2. Target bucket size (config_.gradient_bucket_size)
     * 3. Reverse topological order (for backward pass)
     *
     * @param params All parameters to bucket
     */
    auto create_gradient_buckets(
        const std::vector<std::shared_ptr<Variable>>& params
    ) -> void;

    /**
     * @brief Determine which rank owns a parameter
     *
     * Uses same partitioning strategy as ZeROStage1Optimizer.
     *
     * @param param_index Parameter index
     * @return Rank that owns this parameter
     */
    auto get_parameter_owner_rank(size_t param_index) const -> int;

    /**
     * @brief Reduce-scatter gradients in bucket
     *
     * Algorithm:
     * 1. Flatten all gradients in bucket into contiguous buffer
     * 2. Reduce-scatter: Each rank gets 1/N of the reduced result
     * 3. Unflatten local partition into local_gradients
     * 4. Free non-local gradients (if config_.free_non_local_grads)
     *
     * @param bucket Gradient bucket to reduce-scatter
     */
    auto reduce_scatter_gradients(GradientBucket& bucket) -> void;

    /**
     * @brief Backward hook function (called when gradient ready)
     *
     * @param param Parameter that received gradient
     * @param bucket_id Bucket this parameter belongs to
     */
    auto on_gradient_ready(
        std::shared_ptr<Variable> param,
        size_t bucket_id
    ) -> void;

    /**
     * @brief Check if bucket should be flushed
     *
     * @param bucket Bucket to check
     * @return True if bucket is full or all gradients are ready
     */
    auto should_flush_bucket(const GradientBucket& bucket) const -> bool;

    /**
     * @brief Offload local gradients to CPU
     */
    auto offload_gradients_to_cpu() -> void;

    /**
     * @brief Fetch local gradients from CPU to GPU
     */
    auto fetch_gradients_to_gpu() -> void;

    /**
     * @brief Free non-local gradients to save memory
     *
     * @param bucket Bucket to free gradients from
     */
    auto free_non_local_gradients(GradientBucket& bucket) -> void;

    /**
     * @brief Get total size of all gradients (bytes)
     */
    auto compute_total_gradient_memory() const -> size_t;

    /**
     * @brief Validate bucket integrity (debugging)
     */
    auto validate_buckets() const -> bool;
};

} // namespace optim
} // namespace tenzor
```

### 2. Key Data Structures

#### GradientBucket Structure

```cpp
/**
 * @brief Gradient bucket implementation details
 */
struct GradientBucket {
    // Identity
    int owner_rank{0};                              // Rank [0, world_size) that owns this partition
    size_t bucket_id{0};                            // Unique bucket identifier

    // Parameters and gradients
    std::vector<std::shared_ptr<Variable>> params;  // Parameters in this bucket
    std::vector<Tensor*> gradients;                 // Raw gradient pointers (fast access)
    std::vector<Tensor> local_gradients;            // Local reduced gradients (after reduce-scatter)

    // Memory tracking
    size_t total_bytes{0};                          // Total size of all gradients (bytes)
    size_t num_params{0};                           // Number of parameters
    std::vector<size_t> param_sizes;                // Size of each parameter (bytes)
    std::vector<int64_t> gradient_offsets;          // Offset in flattened buffer

    // State tracking
    std::atomic<size_t> gradients_ready{0};         // How many gradients have been computed
    bool is_reduced{false};                         // Has reduce-scatter completed?
    bool is_flushed{false};                         // Has bucket been flushed?

    // Storage
    Device storage_device{Device::cpu()};           // CPU or GPU storage for local gradients
    std::unique_ptr<Tensor> flattened_buffer;       // Temporary buffer for communication

    // Synchronization
    std::mutex bucket_mutex;                        // Thread-safe access
    std::condition_variable ready_cv;               // Wait for gradients ready

    /**
     * @brief Check if all gradients in bucket are ready
     */
    auto all_gradients_ready() const -> bool {
        return gradients_ready.load() == params.size();
    }

    /**
     * @brief Mark gradient as ready
     */
    auto mark_gradient_ready() -> void {
        gradients_ready.fetch_add(1, std::memory_order_release);
    }

    /**
     * @brief Reset bucket state for next backward pass
     */
    auto reset() -> void {
        gradients_ready.store(0, std::memory_order_release);
        is_reduced = false;
        is_flushed = false;
        gradients.clear();
        local_gradients.clear();
    }
};
```

#### BackwardHookHandle Structure

```cpp
/**
 * @brief Handle for backward hook management
 */
struct BackwardHookHandle {
    std::shared_ptr<Variable> param;                // Parameter this hook is attached to
    size_t bucket_id{0};                            // Which gradient bucket
    size_t hook_id{0};                              // Unique hook ID
    std::function<void()> cleanup_fn;               // Cleanup function (unregister hook)
    bool is_active{true};                           // Is hook still active?

    /**
     * @brief Disable this hook
     */
    auto disable() -> void {
        if (is_active && cleanup_fn) {
            cleanup_fn();
            is_active = false;
        }
    }

    ~BackwardHookHandle() {
        disable();
    }
};
```

---

## Backward Hook System

### 1. Hook Registration

```cpp
/**
 * @brief Register backward hooks on module
 *
 * Walks the module tree and registers hooks on all parameters.
 * Hooks are triggered when gradients are computed during backward pass.
 */
auto ZeROStage2Optimizer::register_backward_hooks(nn::Module& module) -> void {
    if (hooks_registered_.load()) {
        throw std::runtime_error("Backward hooks already registered");
    }

    // Get all parameters
    auto params = module.parameters();

    // Create gradient buckets
    create_gradient_buckets(params);

    // Register hook for each parameter
    for (size_t i = 0; i < params.size(); ++i) {
        auto& param = params[i];

        // Determine which bucket this parameter belongs to
        size_t bucket_id = param_to_bucket_map_[param.get()];

        // Create hook function
        auto hook_fn = [this, param, bucket_id]() {
            this->on_gradient_ready(param, bucket_id);
        };

        // Register hook on parameter
        // Hook is called when param.grad() is computed
        auto cleanup_fn = param->register_backward_hook(hook_fn);

        // Store handle for cleanup
        backward_hooks_.push_back({
            .param = param,
            .bucket_id = bucket_id,
            .hook_id = i,
            .cleanup_fn = cleanup_fn,
            .is_active = true
        });
    }

    hooks_registered_.store(true);
}
```

### 2. Gradient Ready Callback

```cpp
/**
 * @brief Called when gradient is ready for a parameter
 *
 * This is the core of ZeRO Stage 2. When a parameter's gradient is computed:
 * 1. Add gradient to bucket
 * 2. Check if bucket is full
 * 3. If full, trigger reduce-scatter immediately
 * 4. Otherwise, wait for more gradients
 */
auto ZeROStage2Optimizer::on_gradient_ready(
    std::shared_ptr<Variable> param,
    size_t bucket_id
) -> void {
    auto& bucket = gradient_buckets_[bucket_id];

    {
        std::lock_guard<std::mutex> lock(bucket.bucket_mutex);

        // Add gradient to bucket
        if (param->grad()) {
            bucket.gradients.push_back(&param->grad());
        }

        // Mark gradient as ready
        bucket.mark_gradient_ready();
    }

    // Check if bucket should be flushed
    if (should_flush_bucket(bucket)) {
        // Trigger reduce-scatter immediately
        reduce_scatter_bucket(bucket_id);
    }
}
```

### 3. Bucket Creation Strategy

```cpp
/**
 * @brief Create gradient buckets from parameters
 *
 * Strategy:
 * 1. Group parameters by owning rank
 * 2. Within each rank, group by target bucket size
 * 3. Order buckets in reverse topological order (backward pass order)
 */
auto ZeROStage2Optimizer::create_gradient_buckets(
    const std::vector<std::shared_ptr<Variable>>& params
) -> void {

    // Step 1: Group parameters by owning rank
    std::vector<std::vector<std::shared_ptr<Variable>>> params_by_rank(config_.world_size);

    for (size_t i = 0; i < params.size(); ++i) {
        int owner_rank = get_parameter_owner_rank(i);
        params_by_rank[owner_rank].push_back(params[i]);
    }

    // Step 2: Create buckets within each rank
    for (int rank = 0; rank < config_.world_size; ++rank) {
        auto& rank_params = params_by_rank[rank];

        // Reverse order (backward pass processes in reverse)
        std::reverse(rank_params.begin(), rank_params.end());

        // Create buckets of target size
        GradientBucket current_bucket;
        current_bucket.owner_rank = rank;
        current_bucket.bucket_id = gradient_buckets_.size();

        for (auto& param : rank_params) {
            size_t param_bytes = param->numel() * dtype_size(param->dtype());

            // If adding this param exceeds bucket size, flush current bucket
            if (current_bucket.total_bytes > 0 &&
                current_bucket.total_bytes + param_bytes > config_stage2_.gradient_bucket_size) {

                gradient_buckets_.push_back(std::move(current_bucket));

                // Start new bucket
                current_bucket = GradientBucket{};
                current_bucket.owner_rank = rank;
                current_bucket.bucket_id = gradient_buckets_.size();
            }

            // Add parameter to bucket
            current_bucket.params.push_back(param);
            current_bucket.param_sizes.push_back(param_bytes);
            current_bucket.total_bytes += param_bytes;
            current_bucket.num_params++;

            // Track parameter to bucket mapping
            param_to_bucket_map_[param.get()] = current_bucket.bucket_id;
        }

        // Add final bucket
        if (current_bucket.num_params > 0) {
            gradient_buckets_.push_back(std::move(current_bucket));
        }
    }

    // Validate bucket count
    if (gradient_buckets_.size() > config_stage2_.max_gradient_buckets) {
        throw std::runtime_error(
            "Too many gradient buckets: " + std::to_string(gradient_buckets_.size()) +
            " > max " + std::to_string(config_stage2_.max_gradient_buckets)
        );
    }
}
```

### 4. Memory Management

```cpp
/**
 * @brief Free non-local gradients to save memory
 *
 * After reduce-scatter, we only need gradients owned by this rank.
 * Free all other gradients to save GPU memory.
 */
auto ZeROStage2Optimizer::free_non_local_gradients(GradientBucket& bucket) -> void {
    if (!config_stage2_.free_non_local_grads) {
        return;  // Feature disabled
    }

    if (bucket.owner_rank == config_.rank) {
        return;  // This rank owns these gradients, keep them
    }

    // Free all gradients in bucket
    for (auto* grad_ptr : bucket.gradients) {
        if (grad_ptr && grad_ptr->defined()) {
            // Free GPU memory by resetting tensor
            grad_ptr->reset();
        }
    }

    bucket.gradients.clear();
}
```

---

## Communication Protocols

### 1. Reduce-Scatter Algorithm

```cpp
/**
 * @brief Reduce-scatter gradients in bucket
 *
 * Reduce-scatter is the key operation in ZeRO Stage 2.
 * It combines reduction (sum across ranks) with scattering (partition result).
 *
 * Example (4 ranks, 8 parameters):
 * Before:
 *   Rank 0: [g0, g1, g2, g3, g4, g5, g6, g7]
 *   Rank 1: [g0, g1, g2, g3, g4, g5, g6, g7]
 *   Rank 2: [g0, g1, g2, g3, g4, g5, g6, g7]
 *   Rank 3: [g0, g1, g2, g3, g4, g5, g6, g7]
 *
 * After reduce-scatter:
 *   Rank 0: [sum(g0), sum(g1)]           <- Receives first 1/4
 *   Rank 1: [sum(g2), sum(g3)]           <- Receives second 1/4
 *   Rank 2: [sum(g4), sum(g5)]           <- Receives third 1/4
 *   Rank 3: [sum(g6), sum(g7)]           <- Receives fourth 1/4
 */
auto ZeROStage2Optimizer::reduce_scatter_gradients(GradientBucket& bucket) -> void {
    std::lock_guard<std::mutex> lock(bucket.bucket_mutex);

    if (bucket.is_reduced) {
        return;  // Already reduced
    }

    // Step 1: Flatten all gradients into contiguous buffer
    Tensor flattened = flatten_gradients(bucket);

    // Step 2: Perform reduce-scatter collective
    // Each rank receives 1/N of the reduced result
    Tensor local_reduced;

    if (config_.process_group && config_.world_size > 1) {
        // Distributed reduce-scatter
        local_reduced = config_.process_group->reduce_scatter(
            flattened,
            distributed::ReduceOp::SUM
        );
    } else {
        // Single-process mode: just copy
        local_reduced = flattened;
    }

    // Step 3: Unflatten local partition into individual gradient tensors
    bucket.local_gradients = unflatten_gradients(local_reduced, bucket);

    // Step 4: Update parameter gradients with local reduced values
    if (bucket.owner_rank == config_.rank) {
        // This rank owns these gradients
        for (size_t i = 0; i < bucket.params.size(); ++i) {
            auto& param = bucket.params[i];
            if (param->grad().defined()) {
                // Replace gradient with reduced value
                param->grad() = bucket.local_gradients[i];
            }
        }
    }

    // Step 5: Free non-local gradients
    free_non_local_gradients(bucket);

    // Step 6: Optionally offload local gradients to CPU
    if (config_stage2_.offload_gradients && bucket.owner_rank == config_.rank) {
        for (auto& grad : bucket.local_gradients) {
            grad = grad.to(Device::cpu());
        }
        bucket.storage_device = Device::cpu();
    }

    bucket.is_reduced = true;
    bucket.is_flushed = true;
    reduce_scatter_count_.fetch_add(1, std::memory_order_release);
}
```

### 2. Gradient Flattening

```cpp
/**
 * @brief Flatten bucket gradients into contiguous buffer
 *
 * Creates a single contiguous tensor from all gradients in bucket.
 * Required for efficient collective communication.
 */
auto ZeROStage2Optimizer::flatten_gradients(const GradientBucket& bucket) -> Tensor {
    if (bucket.gradients.empty()) {
        return Tensor{};
    }

    // Calculate total elements
    size_t total_elements = 0;
    for (auto* grad : bucket.gradients) {
        if (grad && grad->defined()) {
            total_elements += grad->numel();
        }
    }

    // Allocate contiguous buffer
    auto dtype = bucket.gradients[0]->dtype();
    auto device = bucket.gradients[0]->device();
    Tensor flattened = zeros({static_cast<int64_t>(total_elements)}, dtype, device);

    // Copy gradients into buffer
    size_t offset = 0;
    bucket.gradient_offsets.clear();

    for (auto* grad : bucket.gradients) {
        if (grad && grad->defined()) {
            size_t numel = grad->numel();
            bucket.gradient_offsets.push_back(offset);

            // Copy data
            auto grad_flat = grad->view({-1});
            flattened.slice(0, offset, offset + numel).copy_(grad_flat);

            offset += numel;
        }
    }

    return flattened;
}
```

### 3. Gradient Unflattening

```cpp
/**
 * @brief Unflatten local reduced gradients
 *
 * Splits the reduced buffer back into individual gradient tensors.
 */
auto ZeROStage2Optimizer::unflatten_gradients(
    const Tensor& flattened,
    const GradientBucket& bucket
) -> std::vector<Tensor> {

    std::vector<Tensor> gradients;
    gradients.reserve(bucket.params.size());

    size_t offset = 0;
    for (size_t i = 0; i < bucket.params.size(); ++i) {
        auto& param = bucket.params[i];
        size_t numel = param->numel();

        // Extract this gradient's slice
        Tensor grad_flat = flattened.slice(0, offset, offset + numel);

        // Reshape to parameter shape
        Tensor grad = grad_flat.view(param->shape());

        gradients.push_back(grad);
        offset += numel;
    }

    return gradients;
}
```

### 4. Bucket Flushing Strategy

```cpp
/**
 * @brief Determine if bucket should be flushed
 *
 * Flush conditions:
 * 1. All gradients in bucket are ready
 * 2. Bucket size exceeds threshold
 * 3. Optimizer step() is called (flush all pending)
 */
auto ZeROStage2Optimizer::should_flush_bucket(const GradientBucket& bucket) const -> bool {
    // Condition 1: All gradients ready
    if (bucket.all_gradients_ready()) {
        return true;
    }

    // Condition 2: Bucket full (if reduce_bucket_on_completion enabled)
    if (config_stage2_.reduce_bucket_on_completion &&
        bucket.total_bytes >= config_stage2_.gradient_bucket_size) {
        return true;
    }

    return false;
}

/**
 * @brief Flush all pending gradient buckets
 *
 * Called by step() to ensure all gradients are reduced before optimizer update.
 */
auto ZeROStage2Optimizer::flush_gradient_buckets() -> void {
    for (size_t i = 0; i < gradient_buckets_.size(); ++i) {
        auto& bucket = gradient_buckets_[i];

        if (!bucket.is_flushed && bucket.gradients_ready > 0) {
            reduce_scatter_bucket(i);
        }
    }
}
```

### 5. Communication Overlap

```cpp
/**
 * @brief Overlap reduce-scatter with backward computation
 *
 * Optimization: Start reduce-scatter for completed buckets while
 * backward pass is still running for later layers.
 */
auto ZeROStage2Optimizer::reduce_scatter_bucket(size_t bucket_id) -> void {
    auto& bucket = gradient_buckets_[bucket_id];

    if (config_stage2_.overlap_grad_comm) {
        // Async reduce-scatter
        std::thread([this, bucket_id]() {
            this->reduce_scatter_gradients(this->gradient_buckets_[bucket_id]);
        }).detach();
    } else {
        // Synchronous reduce-scatter
        reduce_scatter_gradients(bucket);
    }
}
```

---

## CPU Offload Extension

### 1. Gradient Offload

```cpp
/**
 * @brief Offload local gradients to CPU
 *
 * After reduce-scatter, gradients are partitioned. Offload local partition
 * to CPU to save GPU memory.
 */
auto ZeROStage2Optimizer::offload_gradients_to_cpu() -> void {
    if (!config_stage2_.offload_gradients) {
        return;
    }

    if (!offload_engine_) {
        throw std::runtime_error("Offload engine not initialized");
    }

    for (auto& bucket : gradient_buckets_) {
        if (bucket.owner_rank != config_.rank) {
            continue;  // Not our gradients
        }

        if (bucket.storage_device.type() == Device::Type::CPU) {
            continue;  // Already on CPU
        }

        // Offload local gradients asynchronously
        std::vector<core::TransferHandle> handles;

        for (auto& grad : bucket.local_gradients) {
            if (grad.defined() && grad.device().type() == Device::Type::CUDA) {
                auto handle = offload_engine_->offload_to_cpu_async(grad);
                handles.push_back(handle);
            }
        }

        // Wait for all transfers
        for (auto& handle : handles) {
            handle.wait();
        }

        bucket.storage_device = Device::cpu();
    }
}
```

### 2. Gradient Prefetch

```cpp
/**
 * @brief Fetch local gradients from CPU to GPU
 *
 * Before optimizer step, prefetch gradients from CPU to GPU.
 */
auto ZeROStage2Optimizer::fetch_gradients_to_gpu() -> void {
    if (!config_stage2_.offload_gradients) {
        return;
    }

    if (!offload_engine_) {
        throw std::runtime_error("Offload engine not initialized");
    }

    for (auto& bucket : gradient_buckets_) {
        if (bucket.owner_rank != config_.rank) {
            continue;  // Not our gradients
        }

        if (bucket.storage_device.type() != Device::Type::CPU) {
            continue;  // Already on GPU
        }

        // Prefetch local gradients asynchronously
        std::vector<Tensor*> grad_ptrs;
        for (auto& grad : bucket.local_gradients) {
            if (grad.defined()) {
                grad_ptrs.push_back(&grad);
            }
        }

        if (!grad_ptrs.empty()) {
            offload_engine_->prefetch_to_gpu(grad_ptrs);
        }

        bucket.storage_device = Device::cuda();
    }
}
```

### 3. Combined State + Gradient Offload

```cpp
/**
 * @brief Optimizer step with combined offload
 *
 * Orchestrates CPU offload for both optimizer states and gradients.
 */
auto ZeROStage2Optimizer::step() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Step 1: Flush any pending gradient buckets
    flush_gradient_buckets();

    // Step 2: Fetch gradients from CPU (if offloaded)
    if (config_stage2_.offload_gradients && offload_engine_) {
        fetch_gradients_to_gpu();
    }

    // Step 3: Fetch optimizer states from CPU (if offloaded)
    if (config_.offload_to_cpu && offload_engine_) {
        fetch_states_to_gpu();  // Inherited from Stage 1
    }

    // Step 4: Update local partition
    update_local_partition();

    // Step 5: Offload states back to CPU
    if (config_.offload_to_cpu && offload_engine_) {
        offload_states_to_cpu();  // Inherited from Stage 1
    }

    // Step 6: Offload gradients back to CPU
    if (config_stage2_.offload_gradients && offload_engine_) {
        offload_gradients_to_cpu();
    }

    // Step 7: All-gather updated parameters
    if (config_.world_size > 1) {
        all_gather_parameters();  // Inherited from Stage 1
    }

    step_count_++;
}
```

---

## Test Requirements

### 1. Unit Tests

#### 1.1 Gradient Partitioning Tests

```cpp
/**
 * @file test_zero_stage2_gradient_partition.cpp
 * @brief Test gradient partitioning correctness
 */

// Test: Gradients are correctly partitioned across ranks
TEST(ZeROStage2, GradientPartitioning) {
    // Setup: 4 ranks, 100 parameters
    auto params = create_test_params(100);
    auto optimizer = create_stage2_optimizer(params, 4, 0);

    // Execute: Backward pass
    compute_dummy_gradients(params);
    optimizer.flush_gradient_buckets();

    // Verify: Each rank has 1/4 of gradients
    EXPECT_EQ(count_local_gradients(optimizer), 25);
    EXPECT_TRUE(verify_gradient_partitioning(optimizer, 4));
}

// Test: Reduce-scatter produces correct results
TEST(ZeROStage2, ReduceScatterCorrectness) {
    // Setup: 2 ranks with known gradient values
    auto params = create_test_params(10);
    auto optimizer = create_stage2_optimizer(params, 2, 0);

    // Set gradients to known values
    // Rank 0: all gradients = 1.0
    // Rank 1: all gradients = 2.0
    set_gradients(params, 1.0);

    // Execute reduce-scatter
    optimizer.flush_gradient_buckets();

    // Verify: Sum = 3.0, each rank has half
    auto local_grads = get_local_gradients(optimizer);
    for (auto& grad : local_grads) {
        EXPECT_FLOAT_EQ(grad.mean().item<float>(), 3.0f);
    }
}
```

#### 1.2 Gradient Bucketing Tests

```cpp
/**
 * @file test_zero_stage2_bucketing.cpp
 * @brief Test gradient bucketing strategy
 */

// Test: Buckets are created with target size
TEST(ZeROStage2, BucketSizeTargeting) {
    auto params = create_test_params(100, {1024, 1024});  // ~4MB per param

    ZeROStage2Config config;
    config.gradient_bucket_size = 25 * 1024 * 1024;  // 25 MB target

    auto optimizer = ZeROStage2Optimizer(
        std::make_unique<Adam>(params, 0.001),
        config
    );

    auto stats = optimizer.get_bucket_stats();

    // Verify: Bucket sizes are close to target
    EXPECT_LE(stats.max_bucket_size, config.gradient_bucket_size * 1.5);
    EXPECT_GE(stats.min_bucket_size, config.gradient_bucket_size * 0.5);
}

// Test: Parameters are grouped by owning rank
TEST(ZeROStage2, BucketRankGrouping) {
    auto params = create_test_params(100);
    auto optimizer = create_stage2_optimizer(params, 4, 0);

    // Verify: Each bucket contains parameters from single rank
    auto buckets = optimizer.get_gradient_buckets();
    for (auto& bucket : buckets) {
        auto ranks = get_parameter_ranks(bucket);
        EXPECT_EQ(ranks.size(), 1);  // All params in bucket owned by same rank
    }
}
```

#### 1.3 Backward Hook Tests

```cpp
/**
 * @file test_zero_stage2_hooks.cpp
 * @brief Test backward hook registration and triggering
 */

// Test: Hooks are registered on all parameters
TEST(ZeROStage2, HookRegistration) {
    auto model = create_test_model();
    auto optimizer = create_stage2_optimizer(model.parameters(), 2, 0);

    EXPECT_NO_THROW(optimizer.register_backward_hooks(model));

    // Verify: Hooks registered
    EXPECT_EQ(optimizer.get_num_hooks(), model.parameters().size());
}

// Test: Hooks trigger reduce-scatter when gradient ready
TEST(ZeROStage2, HookTriggering) {
    auto model = create_test_model();
    auto optimizer = create_stage2_optimizer(model.parameters(), 2, 0);
    optimizer.register_backward_hooks(model);

    // Execute backward
    auto output = model.forward(create_test_input());
    output.backward();

    // Verify: Reduce-scatter was triggered
    auto stats = optimizer.get_bucket_stats();
    EXPECT_GT(stats.num_reduce_scatters, 0);
}

// Test: Hooks can be cleaned up
TEST(ZeROStage2, HookCleanup) {
    auto model = create_test_model();
    auto optimizer = create_stage2_optimizer(model.parameters(), 2, 0);
    optimizer.register_backward_hooks(model);

    // Cleanup
    optimizer.~ZeROStage2Optimizer();

    // Verify: No crashes, hooks unregistered
    SUCCEED();
}
```

### 2. Integration Tests

#### 2.1 Multi-Rank Training Test

```cpp
/**
 * @file test_zero_stage2_multirank.cpp
 * @brief Test multi-rank training with ZeRO Stage 2
 */

// Test: Train model with 4 GPUs
TEST(ZeROStage2Integration, FourGPUTraining) {
    // Setup distributed
    distributed::init_process_group("nccl");
    auto rank = distributed::get_rank();
    auto world_size = distributed::get_world_size();
    ASSERT_EQ(world_size, 4);

    // Create model and optimizer
    auto model = GPT2Small();
    model.to(Device::cuda(rank));

    auto optimizer = ZeROStage2Optimizer(
        std::make_unique<AdamW>(model.parameters(), 1e-4),
        create_stage2_config(world_size, rank)
    );
    optimizer.register_backward_hooks(model);

    // Train for 100 steps
    for (int step = 0; step < 100; ++step) {
        optimizer.zero_grad();
        auto loss = model.forward(get_batch()).backward();
        optimizer.step();
    }

    // Verify: Memory usage reduced compared to Stage 1
    auto memory_used = get_gpu_memory_mb();
    auto expected = compute_stage2_memory(model);
    EXPECT_LT(memory_used, expected * 1.2);  // Allow 20% overhead
}
```

#### 2.2 Gradient Accumulation Test

```cpp
// Test: Gradient accumulation works correctly
TEST(ZeROStage2Integration, GradientAccumulation) {
    auto model = create_test_model();
    auto optimizer = create_stage2_optimizer(model.parameters(), 2, 0);
    optimizer.register_backward_hooks(model);

    // Accumulate gradients over 4 steps
    for (int i = 0; i < 4; ++i) {
        auto loss = model.forward(get_batch());
        loss.backward();  // Accumulate
    }

    // Optimizer step
    optimizer.step();

    // Verify: Gradients accumulated correctly
    EXPECT_TRUE(verify_gradient_accumulation(optimizer, 4));
}
```

### 3. Memory Reduction Verification Tests

```cpp
/**
 * @file test_zero_stage2_memory.cpp
 * @brief Verify memory reduction targets
 */

// Test: Stage 2 uses less memory than Stage 1
TEST(ZeROStage2Memory, MemoryReduction) {
    auto model = BertBase();

    // Baseline: Stage 1
    auto stage1_optimizer = create_stage1_optimizer(model.parameters(), 4, 0);
    auto stage1_memory = measure_memory_usage(stage1_optimizer, model);

    // Test: Stage 2
    auto stage2_optimizer = create_stage2_optimizer(model.parameters(), 4, 0);
    auto stage2_memory = measure_memory_usage(stage2_optimizer, model);

    // Verify: Stage 2 uses less memory
    EXPECT_LT(stage2_memory, stage1_memory);

    // Verify: Memory reduction is close to theoretical (4M reduction in gradients)
    auto gradient_memory = compute_gradient_memory(model);
    auto expected_savings = gradient_memory * (1 - 1.0/4);  // 3/4 of gradients freed
    auto actual_savings = stage1_memory - stage2_memory;

    EXPECT_NEAR(actual_savings, expected_savings, expected_savings * 0.2);
}

// Test: CPU offload saves GPU memory
TEST(ZeROStage2Memory, CPUOffloadMemory) {
    auto model = BertBase();

    ZeROStage2Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = true;
    config.offload_gradients = true;

    auto optimizer = ZeROStage2Optimizer(
        std::make_unique<Adam>(model.parameters(), 1e-3),
        config
    );

    // Train for a few steps
    for (int i = 0; i < 10; ++i) {
        optimizer.zero_grad();
        auto loss = model.forward(get_batch()).backward();
        optimizer.step();
    }

    // Verify: GPU memory is minimal (only activations)
    auto gpu_memory = get_gpu_memory_mb();
    auto activation_memory = compute_activation_memory(model);
    EXPECT_LT(gpu_memory, activation_memory * 1.5);
}
```

### 4. Performance Tests

```cpp
/**
 * @file test_zero_stage2_performance.cpp
 * @brief Performance and overhead tests
 */

// Test: Measure communication overhead
TEST(ZeROStage2Performance, CommunicationOverhead) {
    auto model = GPT2Small();

    // Baseline: Standard training (no ZeRO)
    auto baseline_time = benchmark_training(model, standard_optimizer, 100);

    // Test: ZeRO Stage 2
    auto zero_time = benchmark_training(model, stage2_optimizer, 100);

    // Verify: Overhead is within target (<15%)
    auto overhead = (zero_time - baseline_time) / baseline_time * 100;
    EXPECT_LT(overhead, 15.0);
}

// Test: Bucket size affects performance
TEST(ZeROStage2Performance, BucketSizeImpact) {
    auto model = BertBase();

    // Test different bucket sizes
    std::vector<size_t> bucket_sizes = {
        10 * 1024 * 1024,   // 10 MB
        25 * 1024 * 1024,   // 25 MB (default)
        50 * 1024 * 1024,   // 50 MB
        100 * 1024 * 1024   // 100 MB
    };

    for (auto bucket_size : bucket_sizes) {
        auto time = benchmark_with_bucket_size(model, bucket_size);
        std::cout << "Bucket size: " << bucket_size << " time: " << time << "ms\n";
    }

    // Generally: Larger buckets = fewer communication ops = better performance
    // But too large = higher memory pressure
}
```

### 5. Test Suite Summary

| Test Category | Test Count | Purpose |
|---------------|-----------|---------|
| **Unit Tests** | ~25 | Correctness of individual components |
| - Gradient Partitioning | 5 | Verify reduce-scatter correctness |
| - Bucketing Strategy | 6 | Verify bucket creation logic |
| - Backward Hooks | 8 | Verify hook registration and triggering |
| - Memory Management | 6 | Verify gradient freeing |
| **Integration Tests** | ~15 | End-to-end workflows |
| - Multi-Rank Training | 5 | 2-8 GPU training scenarios |
| - Gradient Accumulation | 3 | Verify accumulation works |
| - Model Compatibility | 7 | Test with BERT, GPT2, ResNet |
| **Memory Tests** | ~8 | Memory reduction verification |
| - Stage 1 vs Stage 2 | 3 | Measure memory savings |
| - CPU Offload | 3 | Verify GPU memory reduction |
| - Memory Leaks | 2 | Check for leaks |
| **Performance Tests** | ~6 | Overhead measurement |
| - Communication Overhead | 3 | Measure reduce-scatter overhead |
| - Bucket Size Tuning | 3 | Optimal bucket size |
| **Total** | **~54 tests** | Comprehensive coverage |

---

## Implementation Roadmap

### Phase 5.1: Core Gradient Partitioning (Week 1-2)

**Tasks**:
1. **Implement ZeROStage2Optimizer class** (3-4 days)
   - Extend ZeROStage1Optimizer
   - Add gradient bucket data structures
   - Implement basic reduce-scatter logic

2. **Implement gradient bucketing** (2-3 days)
   - Parameter grouping by rank
   - Bucket size optimization
   - Reverse topological ordering

3. **Unit tests for partitioning** (2 days)
   - Test reduce-scatter correctness
   - Test bucket creation
   - Test gradient distribution

**Deliverable**: Basic gradient partitioning working in single-node multi-GPU setup.

---

### Phase 5.2: Backward Hook System (Week 2-3)

**Tasks**:
1. **Implement backward hook registration** (2-3 days)
   - Hook registration on parameters
   - Hook cleanup mechanism
   - Thread-safe hook management

2. **Implement gradient ready callback** (2-3 days)
   - Bucket filling logic
   - Trigger reduce-scatter on bucket full
   - Handle edge cases (empty gradients, etc.)

3. **Unit tests for hooks** (2 days)
   - Test hook registration
   - Test hook triggering
   - Test hook cleanup

**Deliverable**: Automatic reduce-scatter triggering during backward pass.

---

### Phase 5.3: Communication Optimization (Week 3-4)

**Tasks**:
1. **Implement gradient flattening/unflattening** (2 days)
   - Efficient tensor flattening
   - Unflatten with correct shapes
   - Handle different dtypes

2. **Optimize reduce-scatter** (2-3 days)
   - Overlap communication with computation
   - Async reduce-scatter
   - Stream prioritization

3. **Performance tests** (2 days)
   - Measure communication overhead
   - Benchmark different bucket sizes
   - Profile bottlenecks

**Deliverable**: Optimized communication with <15% overhead target.

---

### Phase 5.4: CPU Offload Extension (Week 4-5)

**Tasks**:
1. **Implement gradient offload** (2-3 days)
   - Offload after reduce-scatter
   - Prefetch before optimizer step
   - Async transfers

2. **Integrate with Stage 1 offload** (2 days)
   - Combined state + gradient offload
   - Coordinated prefetching
   - Memory management

3. **Memory tests** (2 days)
   - Verify GPU memory reduction
   - Test with large models
   - Profile memory usage

**Deliverable**: Full CPU offload for states and gradients.

---

### Phase 5.5: Integration and Testing (Week 5-6)

**Tasks**:
1. **Multi-rank integration tests** (3 days)
   - 2-8 GPU training tests
   - Different model architectures
   - Long training runs

2. **Memory verification tests** (2 days)
   - Measure memory reduction vs Stage 1
   - Compare to theoretical targets
   - Test memory leak detection

3. **Documentation** (2 days)
   - API documentation
   - Usage examples
   - Performance tuning guide

**Deliverable**: Production-ready ZeRO Stage 2 with full test coverage.

---

### Timeline Summary

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| 5.1 Core Partitioning | 2 weeks | Basic reduce-scatter working |
| 5.2 Backward Hooks | 1 week | Automatic triggering |
| 5.3 Communication Optimization | 1 week | <15% overhead |
| 5.4 CPU Offload | 1 week | Full offload support |
| 5.5 Integration & Testing | 1 week | Production ready |
| **Total** | **6 weeks** | **ZeRO Stage 2 Complete** |

---

## Performance Targets

### 1. Memory Reduction

| Model | Parameters | Stage 1 (4 GPUs) | Stage 2 (4 GPUs) | Savings |
|-------|-----------|------------------|------------------|---------|
| BERT-Base | 110M | 800 MB | 550 MB | 31% |
| BERT-Large | 340M | 2.2 GB | 1.5 GB | 32% |
| GPT-2 | 1.5B | 10 GB | 7 GB | 30% |
| GPT-3 (6.7B) | 6.7B | 42 GB | 29 GB | 31% |

**Target**: 30-35% memory reduction over Stage 1 for Adam optimizer.

### 2. Communication Overhead

| Operation | Target Overhead | Mitigation Strategy |
|-----------|----------------|---------------------|
| Reduce-scatter | <10% | Large buckets (25-50 MB) |
| Bucket filling | <2% | Lock-free data structures |
| Hook triggering | <1% | Minimal per-gradient overhead |
| Gradient flattening | <3% | Pre-allocated buffers |
| **Total** | **<15%** | **Combined optimizations** |

### 3. Throughput

| Benchmark | Target Performance | Notes |
|-----------|-------------------|-------|
| BERT-Base (4 GPUs) | >90% of baseline | Baseline = no ZeRO |
| GPT-2 (8 GPUs) | >85% of baseline | Larger model = higher comm overhead |
| ResNet-50 (4 GPUs) | >95% of baseline | CNN = less communication |

**Target**: >85% throughput retention compared to standard distributed training.

### 4. Scalability

| World Size | Expected Overhead | Memory per GPU (GPT-2) |
|------------|------------------|------------------------|
| 2 GPUs | <10% | 7.5 GB |
| 4 GPUs | <15% | 5 GB |
| 8 GPUs | <20% | 3.5 GB |
| 16 GPUs | <25% | 2.5 GB |

**Target**: Near-linear memory scaling with world size.

---

## Validation Criteria

### Functional Requirements

- [ ] ZeROStage2Optimizer extends ZeROStage1Optimizer correctly
- [ ] Gradient buckets created with target size (±20%)
- [ ] Reduce-scatter produces correct gradient sums
- [ ] Non-local gradients freed after reduce-scatter
- [ ] Backward hooks registered and triggered correctly
- [ ] CPU offload works for states + gradients
- [ ] Multi-rank training converges to same loss as baseline
- [ ] State dict save/load works across ranks
- [ ] Graceful handling of gradient accumulation

### Non-Functional Requirements

- [ ] Memory reduction: 30-35% vs Stage 1
- [ ] Communication overhead: <15%
- [ ] Throughput: >85% of baseline
- [ ] Scalability: Linear memory scaling
- [ ] No memory leaks detected
- [ ] No deadlocks in multi-GPU training
- [ ] Clean shutdown and resource cleanup

### Test Coverage

- [ ] Unit tests: >25 tests passing
- [ ] Integration tests: >15 tests passing
- [ ] Memory tests: >8 tests passing
- [ ] Performance tests: >6 tests passing
- [ ] Total coverage: >90% of code

---

## Conclusion

This specification provides a complete blueprint for implementing ZeRO Stage 2 (Gradient Partitioning) in Tenzor. Key components include:

1. **ZeROStage2Optimizer class** - Extends Stage 1 with gradient partitioning
2. **Gradient bucketing** - Efficient grouping for communication
3. **Backward hooks** - Automatic reduce-scatter triggering
4. **Reduce-scatter protocol** - Core communication primitive
5. **CPU offload** - Extended to include gradients
6. **Comprehensive tests** - 54+ tests for full validation

**Expected Outcomes**:
- 30-35% memory reduction over ZeRO Stage 1
- <15% performance overhead
- Production-ready implementation in 6 weeks

**Next Phase**: ZeRO Stage 3 (Parameter Partitioning) for Nx memory reduction.

---

**Document Status**: ✅ Complete and Ready for Implementation
**Estimated Effort**: 6 weeks (1 engineer)
**Risk Level**: Medium (complex distributed coordination)
**Dependencies**: Phase 4 (ZeRO Stage 1) must be complete
