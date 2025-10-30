# Phase 6: ZeRO Stage 3 (Parameter Partitioning) - Technical Specification

**Date**: 2025-10-30
**Status**: Design Specification
**Version**: 1.0
**Goal**: Full parameter, gradient, and optimizer state partitioning with on-demand gathering

---

## Table of Contents

1. [Overview](#overview)
2. [Complete API Definition](#complete-api-definition)
3. [Algorithm Details](#algorithm-details)
4. [Data Structures](#data-structures)
5. [Memory Management](#memory-management)
6. [Integration Points](#integration-points)
7. [Implementation Examples](#implementation-examples)
8. [Performance Targets](#performance-targets)
9. [Testing Requirements](#testing-requirements)
10. [Validation Checklist](#validation-checklist)

---

## Overview

### Concept

ZeRO Stage 3 represents the most aggressive memory optimization strategy in the ZeRO family. It partitions:
- **Parameters**: Each rank stores only 1/N of model parameters
- **Gradients**: Each rank stores only 1/N of gradients (inherited from Stage 2)
- **Optimizer States**: Each rank stores only 1/N of optimizer states (inherited from Stage 1)

### Memory Savings Formula

For a model with **M** parameters (FP32):
- **Standard Training**: 16M bytes per GPU (4M params + 4M grads + 8M states)
- **ZeRO Stage 3 with N GPUs**: 16M/N bytes per GPU
- **Savings**: Linear with world size (8 GPUs = 8x reduction)

### Key Challenge

Parameters must be **gathered on-demand** during forward/backward passes, introducing communication overhead. Success depends on:
1. **Efficient all-gather** operations before layer computation
2. **Aggressive prefetching** to hide communication latency
3. **Smart memory management** to free gathered parameters immediately after use
4. **Communication/compute overlap** to maximize throughput

---

## Complete API Definition

### 1. ZeROStage3Optimizer Class

```cpp
// File: include/tenzor/nn/optim/zero_optimizer.hpp

namespace tenzor::nn::optim {

/**
 * @brief ZeRO Stage 3: Full Model Partitioning
 *
 * Most aggressive memory savings. Partitions parameters, gradients, AND
 * optimizer states across distributed ranks. Parameters are gathered
 * on-demand for computation and freed immediately after use.
 *
 * Memory Usage Per Rank (N ranks, M parameters):
 *   - Parameters: M/N (only local partition)
 *   - Gradients: M/N (only local partition)
 *   - Optimizer States: 2M/N (only local partition for Adam)
 *   - Temporary: M (gathered parameters during forward/backward)
 *
 * Communication Pattern:
 *   - Forward: All-gather parameters before each layer
 *   - Backward: All-gather parameters, reduce-scatter gradients
 *   - Optimizer: No communication (operates on local partition)
 */
class ZeROStage3Optimizer : public ZeROStage2Optimizer {
public:
    /**
     * @brief Configuration for ZeRO Stage 3
     */
    struct Stage3Config : public Config {
        // ====================================================================
        // Prefetching Configuration
        // ====================================================================

        /** Size of buckets for parameter gathering (bytes).
         *  Larger buckets = fewer all-gather calls but more memory.
         *  Recommended: 100-500 MB based on model size. */
        size_t prefetch_bucket_size = 100 * 1024 * 1024;  // 100 MB default

        /** Number of layers to prefetch ahead during forward/backward.
         *  Larger depth = better latency hiding but more GPU memory.
         *  Recommended: 1-4 based on available memory. */
        int prefetch_depth = 2;

        /** Enable overlap of communication with computation.
         *  Uses separate CUDA streams for gather/compute. */
        bool overlap_comm_compute = true;

        // ====================================================================
        // Memory Management
        // ====================================================================

        /** Maximum number of gathered parameters to cache simultaneously.
         *  Limits peak memory usage during forward/backward passes. */
        int max_cached_params = 10;

        /** Enable parameter caching across forward/backward passes.
         *  Avoids re-gathering parameters if they're used in both passes. */
        bool cache_params_across_passes = true;

        /** Threshold for small parameters (bytes).
         *  Parameters smaller than this are not partitioned. */
        size_t partition_threshold = 1024;  // 1 KB

        // ====================================================================
        // CPU Offload Integration
        // ====================================================================

        /** Offload partitioned parameters to CPU when not in use. */
        bool offload_params_to_cpu = false;

        /** Offload gathered parameters to CPU after use. */
        bool offload_gathered_to_cpu = false;

        /** CPU offload priority for parameters. */
        OffloadPriority param_offload_priority = OffloadPriority::NORMAL;

        // ====================================================================
        // Advanced Optimizations
        // ====================================================================

        /** Use NCCL groups for parallel gather operations.
         *  Experimental: may improve bandwidth utilization. */
        bool use_nccl_groups = false;

        /** Enable gradient checkpointing integration.
         *  Manages parameter gathering during recomputation. */
        bool gradient_checkpointing_aware = false;

        /** Align parameter partitions to this byte boundary.
         *  Improves memory coalescing. Recommended: 128 or 256. */
        size_t partition_alignment = 128;
    };

    // ========================================================================
    // Constructor
    // ========================================================================

    /**
     * @brief Construct ZeRO Stage 3 optimizer
     *
     * @param base_optimizer The underlying optimizer (Adam, AdamW, SGD, etc.)
     * @param config Stage 3 configuration
     * @param comm_group Communication group for distributed operations
     */
    ZeROStage3Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const Stage3Config& config,
        std::shared_ptr<DistributedGroup> comm_group = nullptr
    );

    ~ZeROStage3Optimizer() override;

    // ========================================================================
    // Model Registration
    // ========================================================================

    /**
     * @brief Register model for parameter partitioning
     *
     * This must be called before training begins. It:
     *   1. Partitions all model parameters across ranks
     *   2. Registers forward/backward hooks for automatic gather/scatter
     *   3. Builds execution graph for prefetch scheduling
     *   4. Initializes parameter state tracking
     *
     * @param model The model to partition
     * @throws std::runtime_error if model is already registered
     */
    auto register_model(Module& model) -> void;

    /**
     * @brief Unregister model (for cleanup or re-registration)
     */
    auto unregister_model() -> void;

    // ========================================================================
    // Optimizer Interface (Override)
    // ========================================================================

    /**
     * @brief Perform optimizer step
     *
     * Stage 3 step workflow:
     *   1. Wait for all gradient reduce-scatter to complete
     *   2. Update only local partition of parameters
     *   3. NO all-gather needed (parameters remain partitioned)
     */
    auto step() -> void override;

    /**
     * @brief Zero gradients
     *
     * Only zeros local partition of gradients.
     */
    auto zero_grad() -> void override;

    // ========================================================================
    // State Management
    // ========================================================================

    /**
     * @brief Get optimizer state dictionary
     *
     * Returns state for only the local partition. To get full state,
     * use gather_full_state().
     */
    auto state_dict() -> std::map<std::string, Tensor> override;

    /**
     * @brief Load optimizer state dictionary
     *
     * Expects partitioned state. To load from full checkpoint,
     * use load_full_state().
     */
    auto load_state_dict(const std::map<std::string, Tensor>& state) -> void override;

    /**
     * @brief Gather full optimizer state from all ranks
     *
     * Used for checkpointing. Expensive operation that should only
     * be called periodically.
     */
    auto gather_full_state() -> std::map<std::string, Tensor>;

    /**
     * @brief Load from full (non-partitioned) checkpoint
     *
     * Automatically partitions the state across ranks.
     */
    auto load_full_state(const std::map<std::string, Tensor>& full_state) -> void;

    // ========================================================================
    // Manual Control API
    // ========================================================================

    /**
     * @brief Manually gather a parameter
     *
     * Useful for inference or fine-grained control.
     * Returns the full (gathered) parameter.
     *
     * @param param Parameter to gather (must be registered)
     * @return Full parameter (replicated across all ranks)
     */
    auto gather_parameter(Tensor* param) -> Tensor;

    /**
     * @brief Manually free a gathered parameter
     *
     * Releases the full parameter, keeping only the local partition.
     *
     * @param param Parameter to free
     */
    auto free_gathered_parameter(Tensor* param) -> void;

    /**
     * @brief Prefetch parameters for upcoming layers
     *
     * Manually trigger prefetch for specific parameters.
     *
     * @param params Parameters to prefetch
     */
    auto prefetch_parameters(const std::vector<Tensor*>& params) -> void;

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /**
     * @brief Get performance statistics
     */
    struct Stats {
        // Communication stats
        size_t total_all_gather_calls = 0;
        size_t total_all_gather_bytes = 0;
        double avg_all_gather_time_ms = 0.0;

        // Memory stats
        size_t peak_gathered_memory_bytes = 0;
        size_t current_gathered_memory_bytes = 0;
        int num_cached_params = 0;

        // Prefetch efficiency
        double prefetch_hit_rate = 0.0;  // % of gathers satisfied by prefetch
        int prefetch_queue_depth = 0;

        // Performance metrics
        double forward_comm_time_ms = 0.0;
        double backward_comm_time_ms = 0.0;
        double overlap_efficiency = 0.0;  // % of comm hidden by compute
    };

    auto get_stats() -> Stats;
    auto reset_stats() -> void;

private:
    // ========================================================================
    // Internal State
    // ========================================================================

    Stage3Config config_;

    /** Module being managed (weak reference to avoid ownership) */
    Module* registered_model_ = nullptr;

    /** Parameter state tracking map */
    std::unordered_map<Tensor*, ParameterState> param_states_;

    /** Prefetch queue and scheduler */
    std::unique_ptr<PrefetchScheduler> prefetch_scheduler_;

    /** Cache for gathered parameters */
    std::unique_ptr<ParameterCache> parameter_cache_;

    /** Hook handles for cleanup */
    std::vector<ForwardPreHook> forward_hooks_;
    std::vector<BackwardPostHook> backward_hooks_;

    /** Statistics */
    Stats stats_;
    std::mutex stats_mutex_;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /** Partition all model parameters across ranks */
    auto partition_model_parameters(Module& model) -> void;

    /** Register gather/scatter hooks on all modules */
    auto register_gather_scatter_hooks(Module& model) -> void;

    /** Build execution graph for prefetch scheduling */
    auto build_execution_graph(Module& model) -> void;

    /** All-gather parameter (internal implementation) */
    auto gather_parameter_impl(ParameterState& state) -> void;

    /** Free gathered parameter (internal implementation) */
    auto free_gathered_parameter_impl(ParameterState& state) -> void;

    /** Check if parameter should be partitioned */
    auto should_partition_parameter(const Tensor& param) const -> bool;
};

} // namespace tenzor::nn::optim
```

### 2. ParameterState Structure

```cpp
/**
 * @brief Tracks state of a single parameter across its lifecycle
 */
struct ParameterState {
    // ====================================================================
    // Parameter Identity
    // ====================================================================

    /** Pointer to the parameter tensor (original reference) */
    Tensor* param;

    /** Parameter name (for debugging/logging) */
    std::string name;

    /** Parameter size in bytes */
    size_t size_bytes;

    // ====================================================================
    // Partitioning Information
    // ====================================================================

    /** Rank that owns this partition */
    int owner_rank;

    /** Local partition of the parameter (1/N of full parameter) */
    Tensor local_partition;

    /** Partition start offset in the full parameter */
    size_t partition_offset;

    /** Partition size (may be smaller than size_bytes/N for last rank) */
    size_t partition_size;

    // ====================================================================
    // Gathered State
    // ====================================================================

    /** Full parameter (temporarily gathered during forward/backward) */
    Tensor full_param;

    /** Is the full parameter currently available? */
    bool is_gathered = false;

    /** Reference count: how many layers are currently using this param */
    std::atomic<int> ref_count{0};

    /** Last access timestamp (for LRU eviction) */
    std::chrono::steady_clock::time_point last_access_time;

    // ====================================================================
    // Communication Handles
    // ====================================================================

    /** Handle for ongoing all-gather operation (if any) */
    std::shared_ptr<CommHandle> gather_handle;

    /** Handle for ongoing reduce-scatter operation (if any) */
    std::shared_ptr<CommHandle> scatter_handle;

    // ====================================================================
    // Prefetch State
    // ====================================================================

    /** Is this parameter currently being prefetched? */
    bool is_prefetching = false;

    /** Priority for prefetch scheduling (higher = prefetch earlier) */
    int prefetch_priority = 0;

    /** Estimated time until this parameter is needed (microseconds) */
    int64_t time_until_use_us = 0;

    // ====================================================================
    // CPU Offload State
    // ====================================================================

    /** Is the local partition currently on CPU? */
    bool partition_on_cpu = false;

    /** Is the gathered parameter currently on CPU? */
    bool gathered_on_cpu = false;

    /** Handle for CPU offload operation */
    std::shared_ptr<TransferHandle> offload_handle;

    // ====================================================================
    // Module Dependency Tracking
    // ====================================================================

    /** List of module indices that use this parameter */
    std::vector<int> dependent_modules;

    /** Layer index in execution graph (for prefetch ordering) */
    int layer_index = -1;

    // ====================================================================
    // Methods
    // ====================================================================

    /** Increment reference count */
    auto acquire() -> void {
        ref_count.fetch_add(1, std::memory_order_relaxed);
        last_access_time = std::chrono::steady_clock::now();
    }

    /** Decrement reference count */
    auto release() -> int {
        return ref_count.fetch_sub(1, std::memory_order_relaxed) - 1;
    }

    /** Check if parameter can be freed */
    auto can_free() const -> bool {
        return is_gathered && ref_count.load(std::memory_order_relaxed) == 0;
    }

    /** Get age since last access (for LRU eviction) */
    auto age_ms() const -> int64_t {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_access_time
        ).count();
    }
};
```

### 3. Hook Interfaces

```cpp
/**
 * @brief Forward pre-hook: Gather parameters before layer forward
 */
class ForwardPreHook {
public:
    using HookFunction = std::function<void(Module*, const std::vector<Tensor>&)>;

    ForwardPreHook(Module* module, HookFunction hook)
        : module_(module), hook_(std::move(hook)) {}

    /** Execute hook before module forward */
    auto execute(const std::vector<Tensor>& inputs) -> void {
        hook_(module_, inputs);
    }

    /** Unregister this hook */
    auto remove() -> void;

private:
    Module* module_;
    HookFunction hook_;
    int hook_id_;
};

/**
 * @brief Backward post-hook: Free parameters and scatter gradients after layer backward
 */
class BackwardPostHook {
public:
    using HookFunction = std::function<void(
        Module*,
        const std::vector<Tensor>&,  // inputs
        const std::vector<Tensor>&   // grad_outputs
    )>;

    BackwardPostHook(Module* module, HookFunction hook)
        : module_(module), hook_(std::move(hook)) {}

    /** Execute hook after module backward */
    auto execute(
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& grad_outputs
    ) -> void {
        hook_(module_, inputs, grad_outputs);
    }

    /** Unregister this hook */
    auto remove() -> void;

private:
    Module* module_;
    HookFunction hook_;
    int hook_id_;
};
```

### 4. Prefetch Scheduler Interface

```cpp
/**
 * @brief Intelligent prefetch scheduler for parameter gathering
 *
 * Predicts when parameters will be needed and schedules async
 * all-gather operations to hide communication latency.
 */
class PrefetchScheduler {
public:
    struct Config {
        /** Maximum depth of prefetch queue */
        int max_queue_depth = 4;

        /** Minimum compute time to consider prefetch worthwhile (us) */
        int64_t min_compute_time_us = 100;

        /** Safety margin: start prefetch this much earlier (us) */
        int64_t prefetch_margin_us = 500;

        /** Enable learning from execution history */
        bool adaptive_scheduling = true;
    };

    explicit PrefetchScheduler(const Config& config);

    /**
     * @brief Schedule parameter for prefetch
     *
     * Analyzes execution graph and timing to determine optimal
     * prefetch start time.
     *
     * @param param Parameter to prefetch
     * @param priority Prefetch priority (higher = sooner)
     */
    auto schedule_prefetch(ParameterState& param, int priority = 0) -> void;

    /**
     * @brief Cancel pending prefetch
     */
    auto cancel_prefetch(ParameterState& param) -> void;

    /**
     * @brief Update timing statistics for adaptive scheduling
     */
    auto record_timing(
        const ParameterState& param,
        int64_t gather_time_us,
        int64_t compute_time_us
    ) -> void;

    /**
     * @brief Execute pending prefetch operations
     *
     * Should be called regularly (e.g., before each layer forward).
     */
    auto execute_pending_prefetches() -> void;

private:
    Config config_;

    /** Priority queue of pending prefetches */
    struct PrefetchItem {
        ParameterState* param;
        int priority;
        int64_t scheduled_time_us;

        bool operator<(const PrefetchItem& other) const {
            return priority < other.priority;  // Higher priority first
        }
    };
    std::priority_queue<PrefetchItem> prefetch_queue_;

    /** Execution history for adaptive scheduling */
    struct TimingHistory {
        int64_t avg_gather_time_us = 0;
        int64_t avg_compute_time_us = 0;
        int sample_count = 0;
    };
    std::unordered_map<Tensor*, TimingHistory> timing_history_;

    /** Estimate transfer time for a parameter */
    auto estimate_gather_time(const ParameterState& param) const -> int64_t;

    /** Estimate compute time until parameter is used */
    auto estimate_compute_time_until_use(const ParameterState& param) const -> int64_t;
};
```

---

## Algorithm Details

### 1. Parameter Partitioning Scheme

```cpp
/**
 * @brief Partition model parameters across ranks
 *
 * Algorithm:
 *   1. Flatten all parameters into a single buffer
 *   2. Divide buffer into N equal chunks (one per rank)
 *   3. Each rank keeps only its chunk in GPU memory
 *   4. Track partition boundaries for reconstruction
 *
 * Example: 4 ranks, parameter with 1000 elements
 *   Rank 0: elements [0:250)
 *   Rank 1: elements [250:500)
 *   Rank 2: elements [500:750)
 *   Rank 3: elements [750:1000)
 */
auto ZeROStage3Optimizer::partition_model_parameters(Module& model) -> void {
    auto params = model.parameters();

    // Skip empty models
    if (params.empty()) {
        return;
    }

    // Calculate total parameter count
    size_t total_params = 0;
    for (auto* param : params) {
        total_params += param->numel();
    }

    // Calculate partition size for this rank
    size_t partition_size = (total_params + config_.world_size - 1) / config_.world_size;
    size_t partition_start = config_.rank * partition_size;
    size_t partition_end = std::min(partition_start + partition_size, total_params);

    // Partition each parameter
    size_t current_offset = 0;
    for (auto* param : params) {
        auto param_size = param->numel();

        // Skip tiny parameters (not worth partitioning)
        if (param_size * param->dtype_size() < config_.partition_threshold) {
            continue;
        }

        // Calculate this parameter's partition boundaries
        size_t param_start = current_offset;
        size_t param_end = current_offset + param_size;

        // Find overlap with this rank's partition
        size_t overlap_start = std::max(param_start, partition_start);
        size_t overlap_end = std::min(param_end, partition_end);

        ParameterState state;
        state.param = param;
        state.name = param->name();
        state.size_bytes = param_size * param->dtype_size();
        state.owner_rank = config_.rank;
        state.partition_offset = overlap_start - param_start;
        state.partition_size = overlap_end - overlap_start;

        if (overlap_end > overlap_start) {
            // This rank owns part of this parameter
            // Extract local partition
            state.local_partition = param->slice(0, overlap_start - param_start,
                                                  overlap_end - param_start);

            // Replace full parameter with partition
            *param = state.local_partition;
        } else {
            // This rank owns no part of this parameter
            // Free the parameter entirely
            state.local_partition = Tensor();  // Empty
            *param = Tensor();  // Free GPU memory
        }

        // Store state
        param_states_[param] = std::move(state);

        current_offset += param_size;
    }
}
```

### 2. All-Gather Timing (Before Forward/Backward)

```cpp
/**
 * @brief Gather full parameter before use
 *
 * Algorithm:
 *   1. Check if parameter is already gathered (cache hit)
 *   2. If not, check if prefetch is in progress (wait for it)
 *   3. If neither, start all-gather synchronously
 *   4. Increment reference count to prevent premature freeing
 *   5. Return full parameter for computation
 *
 * Communication: All-gather
 *   - Input: Local partition (M/N elements) on each rank
 *   - Output: Full parameter (M elements) on all ranks
 *   - Bandwidth: N * M * dtype_size / time
 */
auto ZeROStage3Optimizer::gather_parameter_impl(ParameterState& state) -> void {
    // Fast path: parameter already gathered
    if (state.is_gathered) {
        state.acquire();  // Increment ref count
        return;
    }

    // Check if prefetch is in progress
    if (state.is_prefetching && state.gather_handle) {
        // Wait for prefetch to complete
        state.gather_handle->wait();
        state.is_prefetching = false;
        state.is_gathered = true;
        state.acquire();
        return;
    }

    // Start prefetch for next parameters (speculation)
    if (prefetch_scheduler_) {
        auto next_params = get_next_parameters_in_execution_order(state);
        for (auto* next_param : next_params) {
            prefetch_scheduler_->schedule_prefetch(param_states_[next_param]);
        }
    }

    // Perform synchronous all-gather
    auto start_time = std::chrono::steady_clock::now();

    // Allocate buffer for full parameter
    state.full_param = Tensor(
        state.param->shape(),
        state.param->dtype(),
        state.param->device()
    );

    // All-gather: collect partitions from all ranks
    comm_group_->all_gather(
        state.local_partition,  // input: local partition
        state.full_param        // output: full parameter
    );

    // Update state
    state.is_gathered = true;
    state.acquire();  // First reference

    // Update statistics
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_all_gather_calls++;
        stats_.total_all_gather_bytes += state.size_bytes;
        stats_.avg_all_gather_time_ms =
            (stats_.avg_all_gather_time_ms * (stats_.total_all_gather_calls - 1) +
             duration_ms) / stats_.total_all_gather_calls;

        stats_.current_gathered_memory_bytes += state.size_bytes;
        stats_.peak_gathered_memory_bytes = std::max(
            stats_.peak_gathered_memory_bytes,
            stats_.current_gathered_memory_bytes
        );
    }
}
```

### 3. Parameter Freeing Logic

```cpp
/**
 * @brief Free gathered parameter after use
 *
 * Algorithm:
 *   1. Decrement reference count
 *   2. If ref count > 0, parameter still in use - return
 *   3. If ref count == 0, check cache policy
 *   4. If cache enabled and space available, keep in cache
 *   5. Otherwise, free full parameter, keep only partition
 *
 * Memory Freed: M * dtype_size bytes (full parameter)
 * Memory Kept: M/N * dtype_size bytes (local partition)
 */
auto ZeROStage3Optimizer::free_gathered_parameter_impl(ParameterState& state) -> void {
    // Decrement reference count
    int remaining_refs = state.release();

    // Parameter still in use by other modules
    if (remaining_refs > 0) {
        return;
    }

    // Check cache policy
    if (config_.cache_params_across_passes && parameter_cache_) {
        // Try to keep in cache for reuse in backward pass
        if (parameter_cache_->can_cache(state.size_bytes)) {
            parameter_cache_->add(state.param, state.full_param);
            return;  // Keep gathered parameter in cache
        }
    }

    // Free the gathered parameter
    state.full_param.reset();  // Release GPU memory
    state.is_gathered = false;

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.current_gathered_memory_bytes -= state.size_bytes;
    }

    // Optionally offload local partition to CPU
    if (config_.offload_params_to_cpu && !state.partition_on_cpu) {
        if (offload_engine_) {
            state.offload_handle = offload_engine_->offload_to_cpu_async(
                state.local_partition
            );
            state.partition_on_cpu = true;
        }
    }
}
```

### 4. Prefetch Scheduling Algorithm

```cpp
/**
 * @brief Schedule prefetch to hide all-gather latency
 *
 * Algorithm:
 *   1. Estimate all-gather time (T_gather) based on parameter size
 *   2. Estimate compute time until use (T_compute)
 *   3. If T_compute > T_gather: schedule prefetch at (T_compute - T_gather)
 *   4. If T_compute <= T_gather: start prefetch immediately
 *   5. Track prefetch handle for later synchronization
 *
 * Goal: Prefetch completes exactly when parameter is needed
 */
auto PrefetchScheduler::schedule_prefetch(ParameterState& param, int priority) -> void {
    // Estimate timing
    auto gather_time_us = estimate_gather_time(param);
    auto compute_time_us = estimate_compute_time_until_use(param);

    // Add safety margin
    compute_time_us = std::max(0L, compute_time_us - config_.prefetch_margin_us);

    // Decide when to start prefetch
    if (compute_time_us > gather_time_us) {
        // Enough time to hide latency - schedule for later
        auto start_time_us = compute_time_us - gather_time_us;

        PrefetchItem item;
        item.param = &param;
        item.priority = priority;
        item.scheduled_time_us = get_current_time_us() + start_time_us;

        prefetch_queue_.push(item);
    } else {
        // Not enough time - prefetch immediately
        start_async_gather(param);
    }
}

auto PrefetchScheduler::execute_pending_prefetches() -> void {
    auto current_time = get_current_time_us();

    // Execute all prefetches whose scheduled time has arrived
    while (!prefetch_queue_.empty()) {
        auto& item = prefetch_queue_.top();

        if (item.scheduled_time_us > current_time) {
            break;  // Not yet time for this prefetch
        }

        // Start async gather
        start_async_gather(*item.param);

        prefetch_queue_.pop();
    }
}

auto PrefetchScheduler::start_async_gather(ParameterState& param) -> void {
    if (param.is_gathered || param.is_prefetching) {
        return;  // Already gathered or being gathered
    }

    // Allocate buffer for full parameter
    param.full_param = Tensor(
        param.param->shape(),
        param.param->dtype(),
        param.param->device()
    );

    // Start async all-gather
    param.gather_handle = comm_group_->all_gather_async(
        param.local_partition,  // input
        param.full_param        // output
    );

    param.is_prefetching = true;
}
```

### 5. Communication/Compute Overlap Strategy

```cpp
/**
 * @brief Overlap all-gather with computation using separate CUDA streams
 *
 * Architecture:
 *   - Stream 0: Computation (forward/backward)
 *   - Stream 1: Communication (all-gather for layer i+1)
 *   - Stream 2: Communication (all-gather for layer i+2)
 *
 * Timeline:
 *   |--- Compute Layer i ---||--- Compute Layer i+1 ---|
 *        |-- Gather i+1 --||-- Gather i+2 --|
 *
 * Synchronization:
 *   - Before computing layer i: wait for gather i to complete
 *   - After computing layer i: start gather i+2 (if not started)
 */
class OverlapScheduler {
public:
    /**
     * @brief Execute layer forward with overlap
     */
    auto forward_with_overlap(
        Module* layer,
        const std::vector<Tensor>& inputs,
        int layer_idx
    ) -> std::vector<Tensor> {
        // Get parameters for this layer
        auto params = layer->parameters();

        // Wait for all-gather to complete for this layer
        for (auto* param : params) {
            auto& state = param_states_[param];
            if (state.gather_handle) {
                // Synchronize gather stream with compute stream
                compute_stream_.wait_for(gather_stream_);
                state.gather_handle->wait();
                state.is_gathered = true;
                state.is_prefetching = false;
            }
        }

        // Start prefetch for layer i+prefetch_depth
        if (layer_idx + config_.prefetch_depth < num_layers_) {
            auto future_layer = layers_[layer_idx + config_.prefetch_depth];
            auto future_params = future_layer->parameters();

            for (auto* param : future_params) {
                auto& state = param_states_[param];
                if (!state.is_gathered && !state.is_prefetching) {
                    // Start async gather on gather stream
                    gather_stream_.execute([&]() {
                        start_async_gather(state);
                    });
                }
            }
        }

        // Compute layer forward on compute stream
        auto outputs = compute_stream_.execute([&]() {
            return layer->forward(inputs);
        });

        return outputs;
    }

private:
    CUDAStream compute_stream_;
    CUDAStream gather_stream_;
};
```

---

## Data Structures

### 1. Parameter State Tracking Map

```cpp
/**
 * @brief Map from parameter pointer to its state
 *
 * Key: Tensor* (parameter pointer)
 * Value: ParameterState (partition info, gather state, etc.)
 *
 * Thread Safety: Protected by optimizer mutex for modifications,
 *                lock-free reads for state queries
 */
std::unordered_map<Tensor*, ParameterState> param_states_;

// Example usage:
auto& state = param_states_[param_ptr];
if (state.is_gathered) {
    // Use full parameter
    compute_with(state.full_param);
} else {
    // Gather first
    gather_parameter_impl(state);
    compute_with(state.full_param);
}
```

### 2. Prefetch Queue Design

```cpp
/**
 * @brief Priority queue for prefetch scheduling
 *
 * Elements are ordered by prefetch priority (higher = earlier).
 * Used to determine which parameters to gather next.
 */
struct PrefetchQueue {
    struct Item {
        ParameterState* param;
        int priority;
        int64_t scheduled_time_us;

        // Higher priority = higher in queue
        bool operator<(const Item& other) const {
            if (priority != other.priority) {
                return priority < other.priority;
            }
            return scheduled_time_us > other.scheduled_time_us;
        }
    };

    std::priority_queue<Item> queue;
    std::mutex mutex;

    auto push(ParameterState* param, int priority) -> void {
        std::lock_guard<std::mutex> lock(mutex);

        Item item;
        item.param = param;
        item.priority = priority;
        item.scheduled_time_us = calculate_scheduled_time(param);

        queue.push(item);
    }

    auto pop() -> std::optional<Item> {
        std::lock_guard<std::mutex> lock(mutex);

        if (queue.empty()) {
            return std::nullopt;
        }

        auto item = queue.top();
        queue.pop();
        return item;
    }

    auto size() const -> size_t {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};
```

### 3. Gathered Parameter Cache

```cpp
/**
 * @brief LRU cache for gathered parameters
 *
 * Keeps recently gathered parameters in GPU memory to avoid
 * re-gathering in backward pass. Size-limited to prevent OOM.
 */
class ParameterCache {
public:
    struct Config {
        size_t max_cache_size_bytes;  // Maximum cache size
        int max_num_params;           // Maximum number of parameters
    };

    explicit ParameterCache(const Config& config)
        : config_(config) {}

    /**
     * @brief Add parameter to cache
     */
    auto add(Tensor* key, const Tensor& full_param) -> void {
        // Evict LRU items if necessary
        while (current_size_bytes_ + full_param.nbytes() > config_.max_cache_size_bytes) {
            evict_lru();
        }

        // Add to cache
        CacheEntry entry;
        entry.key = key;
        entry.full_param = full_param;
        entry.last_access = std::chrono::steady_clock::now();
        entry.size_bytes = full_param.nbytes();

        cache_[key] = entry;
        lru_list_.push_front(key);
        current_size_bytes_ += entry.size_bytes;
    }

    /**
     * @brief Get parameter from cache
     */
    auto get(Tensor* key) -> std::optional<Tensor> {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return std::nullopt;  // Cache miss
        }

        // Update LRU
        it->second.last_access = std::chrono::steady_clock::now();
        lru_list_.remove(key);
        lru_list_.push_front(key);

        return it->second.full_param;
    }

    /**
     * @brief Check if parameter can be added without eviction
     */
    auto can_cache(size_t size_bytes) const -> bool {
        return current_size_bytes_ + size_bytes <= config_.max_cache_size_bytes;
    }

private:
    struct CacheEntry {
        Tensor* key;
        Tensor full_param;
        std::chrono::steady_clock::time_point last_access;
        size_t size_bytes;
    };

    Config config_;
    std::unordered_map<Tensor*, CacheEntry> cache_;
    std::list<Tensor*> lru_list_;  // Most recent at front
    size_t current_size_bytes_ = 0;

    auto evict_lru() -> void {
        if (lru_list_.empty()) {
            return;
        }

        // Remove least recently used
        auto* lru_key = lru_list_.back();
        lru_list_.pop_back();

        auto it = cache_.find(lru_key);
        if (it != cache_.end()) {
            current_size_bytes_ -= it->second.size_bytes;
            cache_.erase(it);
        }
    }
};
```

### 4. Hook Registration System

```cpp
/**
 * @brief Module hook registry for automatic gather/scatter
 */
class HookRegistry {
public:
    /**
     * @brief Register gather hook on module
     *
     * Hook is called before module forward to gather parameters.
     */
    auto register_forward_pre_hook(
        Module* module,
        ForwardPreHook::HookFunction hook
    ) -> ForwardPreHook {
        auto hook_obj = ForwardPreHook(module, std::move(hook));
        int hook_id = next_hook_id_++;
        forward_hooks_[module].emplace_back(hook_id, hook_obj);
        return hook_obj;
    }

    /**
     * @brief Register scatter hook on module
     *
     * Hook is called after module backward to free parameters
     * and reduce-scatter gradients.
     */
    auto register_backward_post_hook(
        Module* module,
        BackwardPostHook::HookFunction hook
    ) -> BackwardPostHook {
        auto hook_obj = BackwardPostHook(module, std::move(hook));
        int hook_id = next_hook_id_++;
        backward_hooks_[module].emplace_back(hook_id, hook_obj);
        return hook_obj;
    }

    /**
     * @brief Unregister all hooks for a module
     */
    auto unregister_module_hooks(Module* module) -> void {
        forward_hooks_.erase(module);
        backward_hooks_.erase(module);
    }

    /**
     * @brief Execute all forward hooks for a module
     */
    auto execute_forward_hooks(
        Module* module,
        const std::vector<Tensor>& inputs
    ) -> void {
        auto it = forward_hooks_.find(module);
        if (it != forward_hooks_.end()) {
            for (auto& [id, hook] : it->second) {
                hook.execute(inputs);
            }
        }
    }

    /**
     * @brief Execute all backward hooks for a module
     */
    auto execute_backward_hooks(
        Module* module,
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& grad_outputs
    ) -> void {
        auto it = backward_hooks_.find(module);
        if (it != backward_hooks_.end()) {
            for (auto& [id, hook] : it->second) {
                hook.execute(inputs, grad_outputs);
            }
        }
    }

private:
    std::unordered_map<Module*, std::vector<std::pair<int, ForwardPreHook>>> forward_hooks_;
    std::unordered_map<Module*, std::vector<std::pair<int, BackwardPostHook>>> backward_hooks_;
    std::atomic<int> next_hook_id_{0};
};
```

---

## Memory Management

### 1. When to Gather Parameters

**Forward Pass:**
```cpp
// Gather before each layer's forward computation
auto gather_for_forward(Module* layer) -> void {
    auto params = layer->parameters();

    for (auto* param : params) {
        auto& state = param_states_[param];

        // Check cache first
        if (auto cached = parameter_cache_->get(param)) {
            state.full_param = *cached;
            state.is_gathered = true;
            stats_.cache_hits++;
        } else {
            // Gather from distributed ranks
            gather_parameter_impl(state);
            stats_.cache_misses++;
        }

        // Replace parameter with full version for computation
        *param = state.full_param;
    }
}
```

**Backward Pass:**
```cpp
// Gather before each layer's backward computation (if not cached)
auto gather_for_backward(Module* layer) -> void {
    auto params = layer->parameters();

    for (auto* param : params) {
        auto& state = param_states_[param];

        // In backward, parameter might still be cached from forward
        if (!state.is_gathered) {
            gather_parameter_impl(state);
        } else {
            // Reuse from cache - just increment ref count
            state.acquire();
        }

        *param = state.full_param;
    }
}
```

### 2. When to Free Gathered Parameters

**After Forward:**
```cpp
// Free after layer forward completes
auto free_after_forward(Module* layer) -> void {
    auto params = layer->parameters();

    for (auto* param : params) {
        auto& state = param_states_[param];

        // Check if parameter will be reused in backward
        if (config_.cache_params_across_passes) {
            // Keep in cache for backward pass
            parameter_cache_->add(param, state.full_param);
        } else {
            // Free immediately
            free_gathered_parameter_impl(state);
        }

        // Restore parameter to partition
        *param = state.local_partition;
    }
}
```

**After Backward:**
```cpp
// Free after layer backward completes
auto free_after_backward(Module* layer) -> void {
    auto params = layer->parameters();

    for (auto* param : params) {
        auto& state = param_states_[param];

        // Always free after backward (no longer needed)
        free_gathered_parameter_impl(state);

        // Restore parameter to partition
        *param = state.local_partition;
    }
}
```

### 3. Reference Counting for Shared Parameters

```cpp
/**
 * @brief Handle shared parameters (used by multiple modules)
 *
 * Example: Embedding layer weights shared with output layer
 */
class SharedParameterManager {
public:
    /**
     * @brief Register that a module uses a parameter
     */
    auto register_usage(Tensor* param, Module* module) -> void {
        auto& state = param_states_[param];

        // Add module to dependents list
        auto module_id = get_module_id(module);
        if (std::find(state.dependent_modules.begin(),
                     state.dependent_modules.end(),
                     module_id) == state.dependent_modules.end()) {
            state.dependent_modules.push_back(module_id);
        }
    }

    /**
     * @brief Gather parameter with reference counting
     */
    auto gather_shared_parameter(Tensor* param, Module* requesting_module) -> void {
        auto& state = param_states_[param];

        // Gather if not already gathered
        if (!state.is_gathered) {
            gather_parameter_impl(state);
        }

        // Increment reference count for this module
        state.acquire();
    }

    /**
     * @brief Free parameter when module is done
     */
    auto release_shared_parameter(Tensor* param, Module* releasing_module) -> void {
        auto& state = param_states_[param];

        // Decrement reference count
        int remaining = state.release();

        // Free only when all users are done
        if (remaining == 0) {
            free_gathered_parameter_impl(state);
        }
    }

private:
    std::unordered_map<Module*, int> module_ids_;
    int next_module_id_ = 0;

    auto get_module_id(Module* module) -> int {
        auto it = module_ids_.find(module);
        if (it != module_ids_.end()) {
            return it->second;
        }
        int id = next_module_id_++;
        module_ids_[module] = id;
        return id;
    }
};
```

### 4. CPU Offload Integration

```cpp
/**
 * @brief Integrate CPU offload with parameter partitioning
 *
 * Strategy:
 *   - Keep partitions on CPU by default (minimal GPU memory)
 *   - Prefetch to GPU before gathering
 *   - Offload gathered parameters to CPU after use
 */
auto gather_parameter_with_cpu_offload(ParameterState& state) -> void {
    // Step 1: Bring local partition to GPU (if on CPU)
    if (state.partition_on_cpu) {
        // Wait for any pending offload to complete
        if (state.offload_handle) {
            state.offload_handle->wait();
        }

        // Load partition to GPU
        state.local_partition = offload_engine_->load_to_gpu(
            state.local_partition
        );
        state.partition_on_cpu = false;
    }

    // Step 2: All-gather to reconstruct full parameter
    gather_parameter_impl(state);

    // Step 3: Optionally offload partition back to CPU
    if (config_.offload_params_to_cpu) {
        state.offload_handle = offload_engine_->offload_to_cpu_async(
            state.local_partition
        );
        state.partition_on_cpu = true;
    }
}

auto free_parameter_with_cpu_offload(ParameterState& state) -> void {
    // Step 1: Free gathered parameter
    free_gathered_parameter_impl(state);

    // Step 2: Optionally offload gathered param to CPU instead of freeing
    if (config_.offload_gathered_to_cpu && state.full_param.defined()) {
        state.offload_handle = offload_engine_->offload_to_cpu_async(
            state.full_param
        );
        state.gathered_on_cpu = true;
    }
}
```

---

## Integration Points

### 1. How It Extends ZeROStage2Optimizer

```cpp
/**
 * @brief Inheritance hierarchy
 *
 * ZeROStage1Optimizer (Optimizer state partitioning)
 *     └─> ZeROStage2Optimizer (+ Gradient partitioning)
 *           └─> ZeROStage3Optimizer (+ Parameter partitioning)
 *
 * Stage 3 inherits all functionality from Stage 2:
 *   - Optimizer state partitioning (from Stage 1)
 *   - Gradient reduce-scatter (from Stage 2)
 *
 * Stage 3 adds:
 *   - Parameter partitioning
 *   - On-demand parameter gathering
 *   - Prefetch scheduling
 *   - Parameter caching
 */
class ZeROStage3Optimizer : public ZeROStage2Optimizer {
public:
    /**
     * @brief Step function extends Stage 2 behavior
     */
    auto step() -> void override {
        // Stage 2 behavior (inherited):
        //   1. Wait for gradient reduce-scatter to complete
        //   2. Update local partition of optimizer states
        //   3. Update local partition of parameters

        // Stage 3 difference: NO all-gather needed!
        //   - Parameters remain partitioned after step
        //   - Will be gathered on-demand in next forward pass

        // Call Stage 2 step (which calls Stage 1 step internally)
        // But override the all-gather behavior
        update_local_partition_only();
    }

private:
    /**
     * @brief Update only local partition (no all-gather)
     */
    auto update_local_partition_only() -> void {
        // Get local partition of gradients (already reduce-scattered)
        auto& local_partition = partitions_[config_.rank];

        // Update local partition of optimizer states and parameters
        base_optimizer_->step_partition(
            local_partition.params,
            local_partition.states
        );

        // NOTE: Unlike Stage 1/2, we do NOT all-gather parameters here
        // They will be gathered on-demand during next forward pass
    }
};
```

### 2. Backward Compatibility with Stage 1/2

```cpp
/**
 * @brief Unified ZeRO optimizer interface
 *
 * Users can switch between stages by changing config:
 */
enum class ZeROStage {
    STAGE_1,  // Optimizer state partitioning only
    STAGE_2,  // + Gradient partitioning
    STAGE_3   // + Parameter partitioning
};

/**
 * @brief Factory function for ZeRO optimizers
 */
auto create_zero_optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    ZeROStage stage,
    const ZeROConfig& config
) -> std::unique_ptr<ZeROOptimizer> {
    switch (stage) {
        case ZeROStage::STAGE_1:
            return std::make_unique<ZeROStage1Optimizer>(
                std::move(base_optimizer), config
            );

        case ZeROStage::STAGE_2:
            return std::make_unique<ZeROStage2Optimizer>(
                std::move(base_optimizer), config
            );

        case ZeROStage::STAGE_3:
            return std::make_unique<ZeROStage3Optimizer>(
                std::move(base_optimizer),
                static_cast<const ZeROStage3Optimizer::Stage3Config&>(config)
            );

        default:
            throw std::invalid_argument("Invalid ZeRO stage");
    }
}

/**
 * @brief Example: Switch from Stage 2 to Stage 3
 */
void upgrade_from_stage2_to_stage3() {
    // Start with Stage 2
    auto optimizer = create_zero_optimizer(
        std::make_unique<AdamW>(model.parameters()),
        ZeROStage::STAGE_2,
        config
    );

    // Train for a while...
    for (int epoch = 0; epoch < 10; ++epoch) {
        train_epoch(model, optimizer);
    }

    // Upgrade to Stage 3 for even more memory savings
    ZeROStage3Optimizer::Stage3Config stage3_config = config;
    stage3_config.prefetch_depth = 2;

    auto stage3_optimizer = create_zero_optimizer(
        std::make_unique<AdamW>(model.parameters()),
        ZeROStage::STAGE_3,
        stage3_config
    );

    // Load state from Stage 2 optimizer
    stage3_optimizer->load_state_dict(optimizer->state_dict());

    // Continue training with Stage 3
    for (int epoch = 10; epoch < 20; ++epoch) {
        train_epoch(model, stage3_optimizer);
    }
}
```

### 3. Module Hook Registration

```cpp
/**
 * @brief Register hooks on all modules in the model
 */
auto ZeROStage3Optimizer::register_gather_scatter_hooks(Module& model) -> void {
    // Recursively traverse module hierarchy
    auto modules = model.modules();  // Get all submodules

    for (auto* module : modules) {
        // Skip modules without parameters
        if (module->parameters().empty()) {
            continue;
        }

        // Register forward pre-hook: gather parameters before forward
        auto forward_hook = [this, module](
            Module* m,
            const std::vector<Tensor>& inputs
        ) {
            // Gather all parameters for this module
            for (auto* param : module->parameters()) {
                if (param_states_.count(param)) {
                    auto& state = param_states_[param];
                    gather_parameter_impl(state);
                    *param = state.full_param;  // Replace with full param
                }
            }

            // Prefetch parameters for next module
            auto next_module = get_next_module_in_execution_order(module);
            if (next_module) {
                for (auto* param : next_module->parameters()) {
                    if (param_states_.count(param)) {
                        prefetch_scheduler_->schedule_prefetch(
                            param_states_[param],
                            /* priority */ 1
                        );
                    }
                }
            }
        };

        forward_hooks_.push_back(
            module->register_forward_pre_hook(std::move(forward_hook))
        );

        // Register backward post-hook: free parameters and scatter gradients
        auto backward_hook = [this, module](
            Module* m,
            const std::vector<Tensor>& inputs,
            const std::vector<Tensor>& grad_outputs
        ) {
            // Free all parameters for this module
            for (auto* param : module->parameters()) {
                if (param_states_.count(param)) {
                    auto& state = param_states_[param];
                    free_gathered_parameter_impl(state);
                    *param = state.local_partition;  // Restore partition
                }
            }
        };

        backward_hooks_.push_back(
            module->register_backward_post_hook(std::move(backward_hook))
        );
    }
}
```

### 4. Optimizer Step Workflow

```cpp
/**
 * @brief Complete training step workflow with ZeRO Stage 3
 */
auto training_step_with_zero_stage3() -> void {
    // ========================================================================
    // Forward Pass
    // ========================================================================

    // For each layer in model:
    //   1. Forward pre-hook triggered:
    //      - All-gather parameters for current layer
    //      - Prefetch parameters for next layer
    //   2. Compute layer forward with full parameters
    //   3. Forward post-hook triggered:
    //      - Free gathered parameters (or cache for backward)

    auto output = model.forward(input);

    // ========================================================================
    // Loss Computation
    // ========================================================================

    auto loss = criterion(output, target);

    // ========================================================================
    // Backward Pass
    // ========================================================================

    // For each layer in reverse order:
    //   1. Backward pre-hook triggered:
    //      - All-gather parameters for current layer (or get from cache)
    //   2. Compute layer backward with full parameters
    //   3. Backward post-hook triggered:
    //      - Reduce-scatter gradients (from Stage 2)
    //      - Free gathered parameters

    loss.backward();

    // ========================================================================
    // Optimizer Step
    // ========================================================================

    // Stage 3 step:
    //   1. Wait for all gradient reduce-scatter operations to complete
    //   2. Update only local partition of parameters using local gradients
    //   3. NO communication needed (parameters remain partitioned)

    optimizer.step();

    // ========================================================================
    // Zero Gradients
    // ========================================================================

    // Zero only local partition of gradients
    optimizer.zero_grad();
}
```

---

## Implementation Examples

### Example 1: Basic ZeRO Stage 3 Setup

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/distributed/distributed.hpp>

int main(int argc, char** argv) {
    // Initialize distributed environment
    auto world = tenzor::distributed::init(argc, argv);

    // Create model
    auto model = GPT2Model(GPT2Config::gpt2_medium());  // 350M parameters
    model.to(Device::cuda(world.local_rank()));

    // Configure ZeRO Stage 3
    ZeROStage3Optimizer::Stage3Config config;
    config.world_size = world.size();
    config.rank = world.rank();
    config.prefetch_bucket_size = 100 * 1024 * 1024;  // 100 MB
    config.prefetch_depth = 2;  // Prefetch 2 layers ahead
    config.overlap_comm_compute = true;
    config.cache_params_across_passes = true;

    // Create optimizer
    auto base_optimizer = std::make_unique<AdamW>(
        model.parameters(),
        /* lr */ 1e-4,
        /* betas */ std::make_pair(0.9, 0.999),
        /* eps */ 1e-8,
        /* weight_decay */ 0.01
    );

    auto optimizer = ZeROStage3Optimizer(
        std::move(base_optimizer),
        config
    );

    // Register model for parameter partitioning
    optimizer.register_model(model);

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto& batch : dataloader) {
            // Forward pass (parameters automatically gathered)
            auto outputs = model.forward(batch.input_ids);

            // Compute loss
            auto loss = cross_entropy(outputs, batch.labels);

            // Backward pass (parameters gathered, gradients scattered)
            loss.backward();

            // Optimizer step (operates on local partition only)
            optimizer.step();
            optimizer.zero_grad();

            // Print statistics periodically
            if (step % 100 == 0) {
                auto stats = optimizer.get_stats();
                std::cout << "Step " << step << ":\n"
                          << "  All-gather calls: " << stats.total_all_gather_calls << "\n"
                          << "  Avg gather time: " << stats.avg_all_gather_time_ms << " ms\n"
                          << "  Peak gathered memory: "
                          << stats.peak_gathered_memory_bytes / 1024 / 1024 << " MB\n"
                          << "  Prefetch hit rate: "
                          << stats.prefetch_hit_rate * 100 << "%\n";
            }
        }
    }

    return 0;
}
```

### Example 2: ZeRO Stage 3 with CPU Offload

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/core/offload_engine.hpp>

int main(int argc, char** argv) {
    auto world = tenzor::distributed::init(argc, argv);

    // Create large model (13B parameters)
    auto model = GPT2Model(GPT2Config::gpt2_xl());
    model.to(Device::cuda(world.local_rank()));

    // Configure ZeRO Stage 3 with CPU offload
    ZeROStage3Optimizer::Stage3Config config;
    config.world_size = world.size();
    config.rank = world.rank();

    // Enable CPU offload for maximum memory savings
    config.offload_params_to_cpu = true;
    config.offload_gathered_to_cpu = false;  // Keep gathered on GPU for speed
    config.param_offload_priority = OffloadPriority::HIGH;

    // Aggressive prefetching to hide CPU<->GPU transfer latency
    config.prefetch_bucket_size = 200 * 1024 * 1024;  // 200 MB
    config.prefetch_depth = 4;  // Prefetch 4 layers ahead
    config.overlap_comm_compute = true;

    // Create offload engine
    OffloadEngine::Config offload_config;
    offload_config.pinned_memory_size = 4ULL * 1024 * 1024 * 1024;  // 4 GB pinned
    offload_config.num_transfer_streams = 4;
    offload_config.enable_prefetch = true;

    auto offload_engine = std::make_shared<OffloadEngine>(offload_config);

    // Create optimizer
    auto optimizer = ZeROStage3Optimizer(
        std::make_unique<AdamW>(model.parameters(), 1e-4),
        config
    );

    // Set offload engine
    optimizer.set_offload_engine(offload_engine);

    // Register model
    optimizer.register_model(model);

    // Training loop
    for (auto& batch : dataloader) {
        auto outputs = model.forward(batch.input_ids);
        auto loss = cross_entropy(outputs, batch.labels);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    return 0;
}
```

### Example 3: Manual Parameter Management

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>

// Fine-grained control over parameter gathering
void manual_parameter_control() {
    // Setup (same as Example 1)
    auto optimizer = ZeROStage3Optimizer(/* ... */);
    optimizer.register_model(model);

    // Disable automatic hooks for manual control
    optimizer.disable_automatic_hooks();

    // Manually control gathering for each layer
    for (int i = 0; i < model.num_layers(); ++i) {
        auto* layer = model.get_layer(i);
        auto params = layer->parameters();

        // Manually gather parameters for this layer
        std::vector<Tensor> full_params;
        for (auto* param : params) {
            full_params.push_back(optimizer.gather_parameter(param));
        }

        // Manually prefetch next layer
        if (i + 1 < model.num_layers()) {
            auto* next_layer = model.get_layer(i + 1);
            optimizer.prefetch_parameters(next_layer->parameters());
        }

        // Compute layer forward with full parameters
        auto output = layer->forward(input, full_params);

        // Manually free parameters
        for (auto* param : params) {
            optimizer.free_gathered_parameter(param);
        }

        input = output;
    }
}
```

### Example 4: Checkpoint Saving and Loading

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>

// Save checkpoint with ZeRO Stage 3
void save_checkpoint(
    const ZeROStage3Optimizer& optimizer,
    const std::string& path
) {
    // Gather full optimizer state from all ranks
    // This is expensive but necessary for checkpointing
    auto full_state = optimizer.gather_full_state();

    // Only rank 0 saves to disk
    if (optimizer.config().rank == 0) {
        save_state_dict(path, full_state);
    }
}

// Load checkpoint with ZeRO Stage 3
void load_checkpoint(
    ZeROStage3Optimizer& optimizer,
    const std::string& path
) {
    // Load full state on rank 0
    std::map<std::string, Tensor> full_state;
    if (optimizer.config().rank == 0) {
        full_state = load_state_dict(path);
    }

    // Broadcast to all ranks
    broadcast_state_dict(full_state, /* root */ 0);

    // Automatically partition and load
    optimizer.load_full_state(full_state);
}
```

---

## Performance Targets

### 1. Memory Reduction

| Model Size | GPUs | ZeRO-2 Memory | ZeRO-3 Memory | Reduction |
|------------|------|---------------|---------------|-----------|
| 350M (GPT-2 Medium) | 4 | 2.4 GB | 0.6 GB | 4x |
| 1.5B (GPT-2 XL) | 8 | 2.8 GB | 0.35 GB | 8x |
| 6.7B | 16 | 3.5 GB | 0.22 GB | 16x |
| 13B | 32 | 3.8 GB | 0.12 GB | 32x |
| 175B (GPT-3) | 64 | 24 GB | 0.38 GB | 64x |

**Target**: Train 175B parameter model on 64x 40GB A100s (2.5TB total).

### 2. Communication Overhead

| Operation | Frequency | Target Time (ms) | Bandwidth (GB/s) |
|-----------|-----------|------------------|------------------|
| All-gather (100MB param) | Per layer forward | <5 ms | >20 |
| Reduce-scatter (100MB grad) | Per layer backward | <5 ms | >20 |
| Prefetch (200MB) | Async | <10 ms | >20 |

**Mitigation Strategies**:
- Large bucket sizes (100-500 MB) to amortize latency
- Aggressive prefetching (2-4 layers ahead)
- Communication/compute overlap (separate CUDA streams)
- NCCL optimization (ring all-gather, tree reduce-scatter)

### 3. End-to-End Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Stage 3 Overhead** | <25% | (Time_Stage3 - Time_ZeRO2) / Time_ZeRO2 |
| **Prefetch Hit Rate** | >80% | Prefetch_hits / Total_gathers |
| **Overlap Efficiency** | >60% | Hidden_comm_time / Total_comm_time |
| **Peak Memory Overhead** | <2x partition size | Peak_gathered / Partition_size |

**Example**: GPT-2 Medium (350M params), 4 GPUs
- Baseline (ZeRO-2): 100 samples/sec, 2.4 GB memory
- Target (ZeRO-3): >80 samples/sec, 0.6 GB memory
- Acceptable overhead: 20% slowdown for 4x memory savings

### 4. Scalability Targets

| World Size | Memory per GPU | Throughput Efficiency |
|------------|----------------|----------------------|
| 8 GPUs | 16M/8 bytes | >85% of ZeRO-2 |
| 16 GPUs | 16M/16 bytes | >80% of ZeRO-2 |
| 32 GPUs | 16M/32 bytes | >75% of ZeRO-2 |
| 64 GPUs | 16M/64 bytes | >70% of ZeRO-2 |

**Goal**: Linear memory scaling with sublinear performance degradation.

---

## Testing Requirements

### 1. Unit Tests

```cpp
// File: tests/nn/optim/test_zero_stage3.cpp

/**
 * Test 1: Parameter partitioning correctness
 */
TEST(ZeROStage3, ParameterPartitioning) {
    auto world = MockDistributedWorld(4);
    auto model = SimpleModel(/* 1000 parameters */);

    auto optimizer = ZeROStage3Optimizer(
        std::make_unique<Adam>(model.parameters()),
        {.world_size=4, .rank=0}
    );

    optimizer.register_model(model);

    // Verify each rank has 1/4 of parameters
    size_t local_param_count = 0;
    for (auto* param : model.parameters()) {
        local_param_count += param->numel();
    }

    EXPECT_EQ(local_param_count, 1000 / 4);
}

/**
 * Test 2: Parameter gathering and freeing
 */
TEST(ZeROStage3, GatherAndFree) {
    auto world = MockDistributedWorld(4);
    auto model = SimpleModel();
    auto optimizer = ZeROStage3Optimizer(/* ... */);

    auto* param = model.parameters()[0];

    // Initially, parameter is partitioned
    EXPECT_EQ(param->numel(), total_size / 4);

    // Gather parameter
    auto full_param = optimizer.gather_parameter(param);
    EXPECT_EQ(full_param.numel(), total_size);

    // Free parameter
    optimizer.free_gathered_parameter(param);
    EXPECT_EQ(param->numel(), total_size / 4);
}

/**
 * Test 3: Prefetch scheduler
 */
TEST(ZeROStage3, PrefetchScheduler) {
    PrefetchScheduler scheduler({.max_queue_depth = 4});

    ParameterState param1, param2, param3;

    // Schedule prefetches
    scheduler.schedule_prefetch(param1, /* priority */ 2);
    scheduler.schedule_prefetch(param2, /* priority */ 1);
    scheduler.schedule_prefetch(param3, /* priority */ 3);

    // Execute in priority order (3 > 2 > 1)
    scheduler.execute_pending_prefetches();

    EXPECT_TRUE(param3.is_prefetching);
    EXPECT_TRUE(param1.is_prefetching);
    EXPECT_TRUE(param2.is_prefetching);
}

/**
 * Test 4: Reference counting for shared parameters
 */
TEST(ZeROStage3, SharedParameterRefCounting) {
    auto model = ModelWithSharedParams();  // Embedding shared with output
    auto optimizer = ZeROStage3Optimizer(/* ... */);

    auto* shared_param = model.embedding.weight;

    // Gather for embedding layer
    optimizer.gather_parameter(shared_param);
    EXPECT_EQ(param_state.ref_count, 1);

    // Gather for output layer (same param)
    optimizer.gather_parameter(shared_param);
    EXPECT_EQ(param_state.ref_count, 2);

    // Free from embedding
    optimizer.free_gathered_parameter(shared_param);
    EXPECT_TRUE(param_state.is_gathered);  // Still gathered (ref_count=1)

    // Free from output
    optimizer.free_gathered_parameter(shared_param);
    EXPECT_FALSE(param_state.is_gathered);  // Now freed (ref_count=0)
}
```

### 2. Integration Tests

```cpp
/**
 * Test 5: Full training loop with ZeRO Stage 3
 */
TEST(ZeROStage3Integration, GPT2Training) {
    auto world = DistributedWorld::init_for_test(8);

    // Create GPT-2 model
    auto model = GPT2Model(GPT2Config::gpt2_small());

    // Create ZeRO Stage 3 optimizer
    auto optimizer = ZeROStage3Optimizer(
        std::make_unique<AdamW>(model.parameters(), 1e-4),
        {.world_size=8, .rank=world.rank(), .prefetch_depth=2}
    );

    optimizer.register_model(model);

    // Train for 100 steps
    for (int step = 0; step < 100; ++step) {
        auto batch = generate_random_batch();

        // Forward
        auto outputs = model.forward(batch.input_ids);
        auto loss = cross_entropy(outputs, batch.labels);

        // Backward
        loss.backward();

        // Step
        optimizer.step();
        optimizer.zero_grad();
    }

    // Verify memory usage
    auto memory_used = get_gpu_memory_used();
    auto expected_memory = model.memory_footprint() / 8;  // Divided by world size
    EXPECT_LT(memory_used, expected_memory * 1.5);  // Allow 50% overhead

    // Verify optimizer statistics
    auto stats = optimizer.get_stats();
    EXPECT_GT(stats.prefetch_hit_rate, 0.8);  // >80% hit rate
    EXPECT_GT(stats.overlap_efficiency, 0.6);  // >60% overlap
}

/**
 * Test 6: Checkpoint save and load
 */
TEST(ZeROStage3Integration, CheckpointSaveLoad) {
    auto world = DistributedWorld::init_for_test(4);

    // Train model
    auto model1 = SimpleModel();
    auto optimizer1 = ZeROStage3Optimizer(/* ... */);
    train_for_steps(model1, optimizer1, 100);

    // Save checkpoint
    auto state = optimizer1.gather_full_state();
    save_state_dict("checkpoint.pt", state);

    // Create new model and optimizer
    auto model2 = SimpleModel();
    auto optimizer2 = ZeROStage3Optimizer(/* ... */);

    // Load checkpoint
    auto loaded_state = load_state_dict("checkpoint.pt");
    optimizer2.load_full_state(loaded_state);

    // Verify state matches
    EXPECT_TRUE(states_equal(optimizer1.state_dict(), optimizer2.state_dict()));
}
```

### 3. Performance Benchmarks

```cpp
/**
 * Benchmark 1: Overhead vs ZeRO Stage 2
 */
BENCHMARK(ZeROStage3_vs_Stage2) {
    auto model = GPT2Model(GPT2Config::gpt2_medium());

    // Baseline: ZeRO Stage 2
    auto stage2_optimizer = ZeROStage2Optimizer(/* ... */);
    auto time_stage2 = benchmark_training(model, stage2_optimizer, 100);

    // Test: ZeRO Stage 3
    auto stage3_optimizer = ZeROStage3Optimizer(/* ... */);
    auto time_stage3 = benchmark_training(model, stage3_optimizer, 100);

    // Measure overhead
    auto overhead = (time_stage3 - time_stage2) / time_stage2 * 100;

    std::cout << "ZeRO Stage 3 overhead: " << overhead << "%\n";
    EXPECT_LT(overhead, 25.0);  // Target: <25% overhead
}

/**
 * Benchmark 2: Prefetch efficiency
 */
BENCHMARK(PrefetchEfficiency) {
    auto optimizer = ZeROStage3Optimizer(/* ... */);

    // Train with prefetching
    auto time_with_prefetch = benchmark_with_config(
        {.prefetch_depth = 2, .overlap_comm_compute = true}
    );

    // Train without prefetching
    auto time_without_prefetch = benchmark_with_config(
        {.prefetch_depth = 0, .overlap_comm_compute = false}
    );

    // Measure speedup
    auto speedup = time_without_prefetch / time_with_prefetch;

    std::cout << "Prefetch speedup: " << speedup << "x\n";
    EXPECT_GT(speedup, 1.5);  // Target: >1.5x speedup
}

/**
 * Benchmark 3: Scalability test
 */
BENCHMARK(Scalability) {
    for (int world_size : {4, 8, 16, 32}) {
        auto world = DistributedWorld::init_for_test(world_size);

        auto model = GPT2Model(GPT2Config::gpt2_medium());
        auto optimizer = ZeROStage3Optimizer(
            std::make_unique<AdamW>(model.parameters()),
            {.world_size=world_size, .rank=0}
        );

        auto throughput = measure_throughput(model, optimizer);
        auto memory_per_gpu = measure_memory(model, optimizer);

        std::cout << "World size " << world_size << ":\n"
                  << "  Throughput: " << throughput << " samples/sec\n"
                  << "  Memory: " << memory_per_gpu << " GB\n";

        // Verify linear memory scaling
        EXPECT_NEAR(memory_per_gpu, baseline_memory / world_size,
                    baseline_memory / world_size * 0.3);  // 30% tolerance
    }
}
```

---

## Validation Checklist

Before marking Phase 6 as complete, verify:

### API Completeness
- [ ] ZeROStage3Optimizer class fully implemented
- [ ] Stage3Config with all required fields
- [ ] ParameterState structure tracking all state
- [ ] ForwardPreHook and BackwardPostHook interfaces
- [ ] PrefetchScheduler with adaptive scheduling
- [ ] ParameterCache with LRU eviction
- [ ] HookRegistry for automatic hook management

### Algorithm Correctness
- [ ] Parameter partitioning produces correct partitions
- [ ] All-gather reconstructs full parameters correctly
- [ ] Reduce-scatter distributes gradients correctly
- [ ] Reference counting prevents premature freeing
- [ ] Shared parameters handled correctly
- [ ] Prefetch scheduler hides latency effectively

### Memory Management
- [ ] Parameters gathered before forward/backward
- [ ] Parameters freed immediately after use
- [ ] Cache respects memory limits
- [ ] No memory leaks in gather/free cycles
- [ ] Peak memory stays within bounds
- [ ] CPU offload integration works correctly

### Integration
- [ ] Extends ZeROStage2Optimizer correctly
- [ ] Backward compatible with Stage 1/2 checkpoints
- [ ] Hooks registered on all modules
- [ ] Works with custom modules
- [ ] State dict save/load works
- [ ] Checkpoint gather/partition works

### Performance
- [ ] <25% overhead vs ZeRO Stage 2
- [ ] >80% prefetch hit rate
- [ ] >60% communication/compute overlap
- [ ] Linear memory scaling with world size
- [ ] Sublinear performance degradation

### Testing
- [ ] Unit tests for all components
- [ ] Integration tests for full training
- [ ] Benchmark suite for performance
- [ ] Scalability tests (4-64 GPUs)
- [ ] Checkpoint compatibility tests
- [ ] Edge case tests (shared params, tiny params, etc.)

### Documentation
- [ ] API documentation complete
- [ ] Algorithm explanations clear
- [ ] Code examples working
- [ ] Performance tuning guide
- [ ] Migration guide from PyTorch/DeepSpeed
- [ ] Troubleshooting guide

---

## Conclusion

This specification provides a complete, unambiguous blueprint for implementing ZeRO Stage 3 (Parameter Partitioning) in Tenzor. Key deliverables:

1. **Complete API**: All classes, structures, and interfaces defined
2. **Detailed Algorithms**: Step-by-step implementation guidance
3. **Data Structures**: Efficient designs for state tracking
4. **Memory Management**: Clear policies for gather/free cycles
5. **Integration Points**: Seamless extension of Stage 2
6. **Code Examples**: Working examples for all use cases
7. **Performance Targets**: Clear benchmarks to achieve
8. **Testing Requirements**: Comprehensive test coverage

**Success Criteria**:
- Train 175B parameter model on 64x 40GB GPUs
- <25% performance overhead vs ZeRO Stage 2
- >80% prefetch hit rate
- >60% communication/compute overlap
- Production-ready API with full documentation

**Estimated Implementation Time**: 4-5 weeks

**Dependencies**:
- Phase 1: Memory Manager, Transfer Engine (✅ Complete)
- Phase 2: Offload Engine, Parameter Offloading API (✅ Complete)
- Phase 3: Distributed Communication (✅ Complete)
- Phase 4: ZeRO Stage 1 (✅ Complete)
- Phase 5: ZeRO Stage 2 (✅ Complete)

This specification is ready for implementation. All ambiguities have been resolved, and developers have everything needed to build ZeRO Stage 3 successfully.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Ready for Implementation
