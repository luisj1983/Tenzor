# Phase 6: ZeRO Stage 3 Architecture Design

**Date**: 2025-10-30
**Status**: Architecture Design
**Goal**: Full Parameter Partitioning with Automatic Gather/Scatter

---

## Table of Contents

1. [Overview](#overview)
2. [Class Hierarchy](#class-hierarchy)
3. [Hook System Design](#hook-system-design)
4. [Parameter State Machine](#parameter-state-machine)
5. [Prefetch Scheduler Design](#prefetch-scheduler-design)
6. [Communication Strategy](#communication-strategy)
7. [File Structure](#file-structure)
8. [Implementation Flow](#implementation-flow)
9. [Testing Strategy](#testing-strategy)

---

## Overview

### What is ZeRO Stage 3?

ZeRO Stage 3 is the most aggressive memory optimization in the ZeRO family. It partitions **parameters**, **gradients**, AND **optimizer states** across distributed ranks. Parameters are gathered on-demand during forward and backward passes, enabling training of massive models.

### Key Differences from Stage 2

| Aspect | Stage 2 | Stage 3 |
|--------|---------|---------|
| **Parameters** | Replicated on all ranks | Partitioned (gathered on-demand) |
| **Gradients** | Partitioned (reduce-scatter) | Partitioned (reduce-scatter) |
| **Optimizer States** | Partitioned | Partitioned |
| **Memory Savings** | 8x | Nx (N = world size) |
| **Complexity** | Medium | High |
| **Communication** | Reduce-scatter (backward) | All-gather (forward/backward) + Reduce-scatter (backward) |

### Memory Analysis

For a model with M parameters (FP32):

```
ZeRO Stage 2 (per GPU):
  Parameters:       4M bytes (replicated)
  Gradients:        4M/N bytes (partitioned)
  Optimizer States: 8M/N bytes (partitioned)
  Total:            4M + 12M/N bytes

ZeRO Stage 3 (per GPU):
  Parameters:       4M/N bytes (partitioned)
  Gradients:        4M/N bytes (partitioned)
  Optimizer States: 8M/N bytes (partitioned)
  Total:            16M/N bytes

Example (GPT-3 175B, 8 GPUs):
  Stage 2: 700 GB + 88 GB = 788 GB per GPU (doesn't fit!)
  Stage 3: 88 GB per GPU (fits in 8x A100 80GB!)
```

---

## Class Hierarchy

### UML Class Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         Optimizer                               │
│                        (Base Class)                             │
├────────────────────────────────────────────────────────────────┤
│ + step() -> void                                               │
│ + zero_grad() -> void                                          │
│ + state_dict() -> map                                          │
│ + load_state_dict(map) -> void                                 │
└────────────────────────────────────────────────────────────────┘
                              △
                              │ inherits
                              │
┌────────────────────────────────────────────────────────────────┐
│                    ZeROStage1Optimizer                         │
│                 (Optimizer State Partition)                     │
├────────────────────────────────────────────────────────────────┤
│ # base_optimizer_: unique_ptr<Optimizer>                       │
│ # config_: ZeROStage1Config                                    │
│ # partitions_: vector<StatePartition>                          │
│ # offload_engine_: shared_ptr<OffloadEngine>                   │
├────────────────────────────────────────────────────────────────┤
│ + step() -> void                        [all-reduce grads]     │
│ # partition_parameters() -> void                               │
│ # all_reduce_gradients() -> void                               │
│ # all_gather_parameters() -> void                              │
│ # update_local_partition() -> void                             │
└────────────────────────────────────────────────────────────────┘
                              △
                              │ inherits
                              │
┌────────────────────────────────────────────────────────────────┐
│                    ZeROStage2Optimizer                         │
│              (Gradient + State Partition)                       │
├────────────────────────────────────────────────────────────────┤
│ # gradient_buckets_: vector<GradientBucket>                    │
│ # hooks_registered_: bool                                      │
├────────────────────────────────────────────────────────────────┤
│ + step() -> void                [skip all-reduce, use r-s]     │
│ + register_backward_hooks() -> void                            │
│ # create_gradient_buckets() -> void                            │
│ # reduce_scatter_gradients(bucket) -> void                     │
│ # gradient_hook(bucket_idx, param_idx) -> void                 │
└────────────────────────────────────────────────────────────────┘
                              △
                              │ inherits
                              │
┌────────────────────────────────────────────────────────────────┐
│                    ZeROStage3Optimizer                         │
│            (Full Model Partition - NEW)                        │
├────────────────────────────────────────────────────────────────┤
│ # stage3_config_: Stage3Config                                 │
│ # param_states_: map<Tensor*, ParameterState>                  │
│ # prefetch_scheduler_: unique_ptr<PrefetchScheduler>           │
│ # forward_hooks_: vector<ForwardPreHook>                       │
│ # backward_hooks_: vector<BackwardPostHook>                    │
│ # gather_stream_: CUDAStream                                   │
│ # scatter_stream_: CUDAStream                                  │
├────────────────────────────────────────────────────────────────┤
│ + register_model(Module&) -> void                              │
│ + step() -> void               [no gather needed]              │
│ # gather_parameter(Tensor*) -> Tensor                          │
│ # scatter_parameter(Tensor*) -> void                           │
│ # register_gather_scatter_hooks(Module&) -> void               │
│ # forward_pre_hook(Module*, inputs) -> void                    │
│ # backward_post_hook(Module*, grad_outputs) -> void            │
│ # prefetch_next_parameters(Module*) -> void                    │
│ # free_gathered_parameter(Tensor*) -> void                     │
└────────────────────────────────────────────────────────────────┘
```

### New Member Variables for Stage 3

```cpp
class ZeROStage3Optimizer : public ZeROStage2Optimizer {
private:
    // =========================================================================
    // Configuration
    // =========================================================================

    Stage3Config stage3_config_;  // Extends Stage2Config with prefetch settings

    // =========================================================================
    // Parameter State Tracking
    // =========================================================================

    /**
     * @brief State information for each parameter
     *
     * Tracks whether a parameter is currently gathered, who is using it,
     * and manages the lifecycle of temporary gathered buffers.
     */
    struct ParameterState {
        Tensor* param;                      ///< Original parameter (partitioned)
        Tensor full_param;                  ///< Gathered full parameter (temporary)
        Tensor partition;                   ///< Local partition backup

        // State tracking
        enum class State {
            PARTITIONED,    ///< Only local partition exists
            GATHERING,      ///< All-gather in progress (async)
            GATHERED,       ///< Full parameter available
            SCATTERING,     ///< Reduce-scatter in progress (async)
        };
        State state{State::PARTITIONED};

        // Reference counting
        std::vector<Module*> active_users;  ///< Modules currently using this param
        int gather_count{0};                ///< Number of pending gathers
        int use_count{0};                   ///< Number of active users

        // Async handles
        std::shared_ptr<AsyncHandle> gather_handle;   ///< Handle for async all-gather
        std::shared_ptr<AsyncHandle> scatter_handle;  ///< Handle for async reduce-scatter

        // Memory management
        bool pinned_in_memory{false};       ///< Keep gathered (e.g., first/last layer)
        size_t memory_bytes{0};             ///< Size of full parameter
    };

    std::unordered_map<Tensor*, ParameterState> param_states_;
    mutable std::mutex param_states_mutex_;

    // =========================================================================
    // Prefetch Scheduling
    // =========================================================================

    /**
     * @brief Prefetch scheduler for predictive parameter gathering
     *
     * Analyzes the computation graph to predict which parameters will be
     * needed next and starts gathering them early to hide latency.
     */
    class PrefetchScheduler {
    public:
        struct PrefetchRequest {
            Tensor* param;                  ///< Parameter to prefetch
            int priority;                   ///< Higher = more urgent
            std::chrono::time_point<std::chrono::steady_clock> deadline;  ///< When needed
            Module* requester;              ///< Which module needs it
        };

        PrefetchScheduler(int max_concurrent, size_t prefetch_buffer_size);

        auto schedule(Tensor* param, int priority, Module* module) -> void;
        auto execute_pending() -> void;
        auto cancel(Tensor* param) -> void;
        auto get_stats() -> PrefetchStats;

    private:
        std::priority_queue<PrefetchRequest> queue_;
        std::unordered_set<Tensor*> in_flight_;
        std::mutex queue_mutex_;
        int max_concurrent_prefetches_;
        size_t max_prefetch_buffer_bytes_;
    };

    std::unique_ptr<PrefetchScheduler> prefetch_scheduler_;

    // =========================================================================
    // Hook System
    // =========================================================================

    /**
     * @brief Forward pre-hook: Gather parameters before layer execution
     */
    struct ForwardPreHook {
        Module* module;                     ///< Module this hook is attached to
        std::vector<Tensor*> params;        ///< Parameters used by this module
        std::function<void()> hook_fn;      ///< Hook callback
        int hook_id;                        ///< Unique hook identifier
    };

    /**
     * @brief Backward post-hook: Reduce-scatter gradients after layer backward
     */
    struct BackwardPostHook {
        Module* module;                     ///< Module this hook is attached to
        std::vector<Tensor*> params;        ///< Parameters used by this module
        std::function<void()> hook_fn;      ///< Hook callback
        int hook_id;                        ///< Unique hook identifier
    };

    std::vector<ForwardPreHook> forward_hooks_;
    std::vector<BackwardPostHook> backward_hooks_;
    int next_hook_id_{0};

    // =========================================================================
    // Communication Streams
    // =========================================================================

    /**
     * @brief Separate CUDA streams for overlapping communication with compute
     */
    CUDAStream gather_stream_;      ///< Stream for all-gather operations
    CUDAStream scatter_stream_;     ///< Stream for reduce-scatter operations
    CUDAStream compute_stream_;     ///< Main computation stream

    // =========================================================================
    // Performance Monitoring
    // =========================================================================

    struct PerformanceStats {
        size_t total_gathers{0};
        size_t total_scatters{0};
        size_t prefetch_hits{0};
        size_t prefetch_misses{0};
        double avg_gather_time_ms{0.0};
        double avg_scatter_time_ms{0.0};
        double communication_overhead_pct{0.0};
    };

    PerformanceStats perf_stats_;
    mutable std::mutex stats_mutex_;
};
```

### New Member Functions for Stage 3

```cpp
class ZeROStage3Optimizer : public ZeROStage2Optimizer {
public:
    // =========================================================================
    // Constructor & Setup
    // =========================================================================

    /**
     * @brief Construct ZeRO Stage 3 optimizer
     *
     * @param base_optimizer Base optimizer (Adam, SGD, etc.)
     * @param config Stage 3 configuration
     */
    ZeROStage3Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const Stage3Config& config
    );

    /**
     * @brief Register model for parameter partitioning
     *
     * This MUST be called after model creation to:
     * - Partition model parameters across ranks
     * - Register forward/backward hooks on all modules
     * - Initialize parameter state tracking
     * - Set up prefetch scheduler
     *
     * @param model Model to partition
     * @throws std::runtime_error if already registered
     */
    auto register_model(Module& model) -> void;

    // =========================================================================
    // Optimizer Interface (Overrides)
    // =========================================================================

    /**
     * @brief Perform optimizer step (parameters already partitioned)
     *
     * Algorithm:
     * 1. No all-gather needed (parameters already partitioned)
     * 2. Gradients already reduced-scattered via backward hooks
     * 3. Fetch optimizer states from CPU if offloaded
     * 4. Update local partition of parameters
     * 5. Offload states back to CPU if enabled
     * 6. NO all-gather (parameters stay partitioned!)
     *
     * @throws std::runtime_error if distributed not initialized
     */
    auto step() -> void override;

    // =========================================================================
    // Parameter Management
    // =========================================================================

    /**
     * @brief All-gather parameter from all ranks
     *
     * Gathers the full parameter by collecting partitions from all ranks.
     * Uses async communication if configured for overlap.
     *
     * @param param Parameter to gather
     * @return Full (gathered) parameter tensor
     * @throws std::runtime_error if communication fails
     */
    auto gather_parameter(Tensor* param) -> Tensor;

    /**
     * @brief All-gather parameter asynchronously
     *
     * @param param Parameter to gather
     * @return AsyncHandle for checking completion
     */
    auto gather_parameter_async(Tensor* param) -> std::shared_ptr<AsyncHandle>;

    /**
     * @brief Wait for async gather to complete
     *
     * @param handle Handle from gather_parameter_async()
     * @return Full (gathered) parameter tensor
     */
    auto wait_gather(std::shared_ptr<AsyncHandle> handle) -> Tensor;

    /**
     * @brief Free gathered parameter (keep only local partition)
     *
     * Frees the temporary gathered buffer and returns to partitioned state.
     * Only frees if no other modules are using the parameter (ref counting).
     *
     * @param param Parameter to free
     */
    auto free_gathered_parameter(Tensor* param) -> void;

    /**
     * @brief Reduce-scatter parameter gradients
     *
     * Reduces gradients across ranks and scatters result (each rank gets 1/N).
     * This is called automatically by backward hooks.
     *
     * @param param Parameter whose gradient to reduce-scatter
     */
    auto scatter_parameter_gradient(Tensor* param) -> void;

    // =========================================================================
    // Hook Management
    // =========================================================================

    /**
     * @brief Register gather/scatter hooks on all modules
     *
     * Called internally by register_model(). Walks the module tree and
     * registers hooks for automatic parameter management.
     *
     * @param model Root module
     */
    auto register_gather_scatter_hooks(Module& model) -> void;

    /**
     * @brief Forward pre-hook: Gather parameters before layer forward
     *
     * This hook is called before each module's forward pass to:
     * 1. Prefetch parameters for next N layers
     * 2. Wait for this layer's parameters to finish gathering
     * 3. Replace partitioned parameters with full gathered versions
     *
     * @param module Module about to execute forward
     * @param inputs Input tensors (unused)
     */
    auto forward_pre_hook(Module* module, const std::vector<Tensor>& inputs) -> void;

    /**
     * @brief Backward post-hook: Scatter gradients after layer backward
     *
     * This hook is called after each module's backward pass to:
     * 1. Reduce-scatter gradients for this layer's parameters
     * 2. Free gathered parameters (if not pinned)
     * 3. Free non-local gradients
     *
     * @param module Module that just completed backward
     * @param grad_outputs Gradient outputs (used for reduce-scatter)
     */
    auto backward_post_hook(Module* module, const std::vector<Tensor>& grad_outputs) -> void;

    // =========================================================================
    // Prefetching
    // =========================================================================

    /**
     * @brief Prefetch parameters for upcoming modules
     *
     * Analyzes the computation graph to predict which parameters will be
     * needed in the next K layers and starts gathering them early.
     *
     * @param current_module Module currently executing
     */
    auto prefetch_next_parameters(Module* current_module) -> void;

    /**
     * @brief Get prefetch statistics
     */
    struct PrefetchStats {
        size_t prefetch_queue_size{0};
        size_t prefetched_bytes{0};
        double hit_rate{0.0};           ///< Fraction of gathers that hit prefetch
        double avg_prefetch_time_ms{0.0};
    };

    auto get_prefetch_stats() const -> PrefetchStats;

    // =========================================================================
    // State Management
    // =========================================================================

    /**
     * @brief Get current state of a parameter
     *
     * @param param Parameter to query
     * @return Current state (PARTITIONED, GATHERING, GATHERED, SCATTERING)
     */
    auto get_parameter_state(Tensor* param) const -> ParameterState::State;

    /**
     * @brief Check if parameter is currently gathered
     *
     * @param param Parameter to check
     * @return true if full parameter is available
     */
    auto is_parameter_gathered(Tensor* param) const -> bool;

    /**
     * @brief Pin parameter in memory (keep gathered)
     *
     * Useful for frequently used parameters (e.g., first/last layer).
     * Pinned parameters are never freed after gathering.
     *
     * @param param Parameter to pin
     */
    auto pin_parameter(Tensor* param) -> void;

    /**
     * @brief Unpin parameter (allow freeing)
     *
     * @param param Parameter to unpin
     */
    auto unpin_parameter(Tensor* param) -> void;

    // =========================================================================
    // Performance Monitoring
    // =========================================================================

    /**
     * @brief Get performance statistics
     */
    auto get_performance_stats() const -> PerformanceStats;

    /**
     * @brief Reset performance counters
     */
    auto reset_performance_stats() -> void;

private:
    // Internal implementation methods...
};
```

### Stage3Config Structure

```cpp
/**
 * @brief Configuration for ZeRO Stage 3 Optimizer
 */
struct Stage3Config : public ZeROStage2Config {
    // Prefetch settings
    size_t prefetch_bucket_size{100 * 1024 * 1024};  ///< Target prefetch size (default: 100MB)
    int prefetch_depth{2};                            ///< Layers to prefetch ahead (default: 2)
    int max_concurrent_prefetches{4};                 ///< Max simultaneous prefetches
    bool overlap_comm_compute{true};                  ///< Overlap gather with compute

    // Memory management
    bool pin_first_layer{true};                       ///< Keep first layer gathered
    bool pin_last_layer{true};                        ///< Keep last layer gathered
    size_t max_gathered_buffer_size{500 * 1024 * 1024}; ///< Max memory for gathered params (500MB)

    // Communication settings
    bool use_async_gather{true};                      ///< Use async all-gather
    bool use_separate_streams{true};                  ///< Separate CUDA streams for comm
    int gather_stream_priority{-1};                   ///< CUDA stream priority (higher = more urgent)

    Stage3Config() = default;
};
```

---

## Hook System Design

### Hook Registration Flow

```
register_model(model)
    │
    ├─> Walk module tree (DFS)
    │
    ├─> For each module:
    │   ├─> Extract parameters used by module
    │   ├─> Create ForwardPreHook
    │   │   └─> Callback: gather_parameters_for_module()
    │   ├─> Create BackwardPostHook
    │   │   └─> Callback: scatter_gradients_for_module()
    │   └─> Register hooks with autograd system
    │
    └─> Initialize parameter states
        └─> Create ParameterState for each parameter
```

### Hook Execution Order

#### Forward Pass (Layer i)

```
┌─────────────────────────────────────────────────────────────┐
│ FORWARD PRE-HOOK (Before Layer i Forward)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Prefetch layer i+k parameters (k=prefetch_depth)        │
│    - Check prefetch_scheduler queue                         │
│    - Start async all-gather for upcoming layers            │
│                                                             │
│ 2. Wait for layer i parameters to finish gathering         │
│    - Check if already gathered (prefetch hit!)             │
│    - If not, synchronous all-gather now (prefetch miss)    │
│    - Update perf_stats (hit/miss)                          │
│                                                             │
│ 3. Replace partitioned params with gathered params         │
│    - Swap module.param.data = full_param                   │
│    - Increment use_count (reference counting)              │
│    - Mark state as GATHERED                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER i FORWARD                                             │
│ - Uses full (gathered) parameters                           │
│ - Computes output = forward(input)                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ POST FORWARD (After Layer i Forward)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Decrement use_count for layer i parameters              │
│                                                             │
│ 2. If use_count == 0 and not pinned:                       │
│    - Free gathered parameter                                │
│    - Keep only local partition                              │
│    - Mark state as PARTITIONED                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Backward Pass (Layer i)

```
┌─────────────────────────────────────────────────────────────┐
│ BACKWARD PRE-HOOK (Before Layer i Backward)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Check if parameters are gathered                        │
│    - If not gathered, gather now (needed for grad calc)    │
│    - If already gathered from forward, reuse               │
│                                                             │
│ 2. Increment use_count                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER i BACKWARD                                            │
│ - Computes gradients w.r.t. parameters                      │
│ - Gradients stored in param.grad()                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ BACKWARD POST-HOOK (After Layer i Backward)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Reduce-scatter gradients for layer i parameters         │
│    - All-reduce gradients across ranks (SUM)               │
│    - Each rank keeps only its partition (1/N)              │
│    - Free non-local gradients                               │
│    - Mark state as SCATTERING (if async)                    │
│                                                             │
│ 2. Free gathered parameters (if use_count == 0)            │
│    - Check reference count                                  │
│    - If no users, free full param, keep partition          │
│    - Mark state as PARTITIONED                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Hook Implementation

```cpp
auto ZeROStage3Optimizer::forward_pre_hook(
    Module* module,
    const std::vector<Tensor>& inputs
) -> void {
    // Find the hook for this module
    auto hook_it = std::find_if(
        forward_hooks_.begin(),
        forward_hooks_.end(),
        [module](const ForwardPreHook& h) { return h.module == module; }
    );

    if (hook_it == forward_hooks_.end()) {
        return;  // No hook registered for this module
    }

    const auto& hook = *hook_it;

    // Step 1: Prefetch parameters for upcoming modules
    prefetch_next_parameters(module);

    // Step 2: Gather parameters for this module
    for (auto* param : hook.params) {
        auto& state = param_states_[param];

        // Check if already gathered (prefetch hit)
        if (state.state == ParameterState::State::GATHERED) {
            perf_stats_.prefetch_hits++;
            state.use_count++;
            continue;
        }

        // Check if gathering in progress (async)
        if (state.state == ParameterState::State::GATHERING) {
            // Wait for async gather to complete
            if (state.gather_handle) {
                state.full_param = wait_gather(state.gather_handle);
                state.gather_handle = nullptr;
            }
            state.state = ParameterState::State::GATHERED;
            state.use_count++;
            continue;
        }

        // Not gathered yet - prefetch miss
        perf_stats_.prefetch_misses++;

        // Synchronous gather now
        state.full_param = gather_parameter(param);
        state.state = ParameterState::State::GATHERED;
        state.use_count++;

        // Replace module's parameter with gathered version
        // This is a pointer swap in the module's parameter storage
        param->set_data(state.full_param);
    }
}

auto ZeROStage3Optimizer::backward_post_hook(
    Module* module,
    const std::vector<Tensor>& grad_outputs
) -> void {
    // Find the hook for this module
    auto hook_it = std::find_if(
        backward_hooks_.begin(),
        backward_hooks_.end(),
        [module](const BackwardPostHook& h) { return h.module == module; }
    );

    if (hook_it == backward_hooks_.end()) {
        return;
    }

    const auto& hook = *hook_it;

    // Step 1: Reduce-scatter gradients
    for (auto* param : hook.params) {
        if (!param->has_grad()) {
            continue;
        }

        // Reduce-scatter gradient (inherited from Stage 2)
        scatter_parameter_gradient(param);
    }

    // Step 2: Free gathered parameters
    for (auto* param : hook.params) {
        free_gathered_parameter(param);
    }
}
```

---

## Parameter State Machine

### State Diagram

```
                    ┌──────────────────┐
                    │   PARTITIONED    │ ◄─────────────┐
                    │ (Initial State)  │               │
                    └──────────────────┘               │
                            │                          │
                            │ gather_parameter()       │
                            │ (or prefetch)            │
                            ▼                          │
                    ┌──────────────────┐               │
                    │    GATHERING     │               │
                    │ (All-gather I/O) │               │
                    └──────────────────┘               │
                            │                          │
                            │ async completes          │
                            ▼                          │
                    ┌──────────────────┐               │
                    │    GATHERED      │               │
                    │ (Full param avail)│              │
                    └──────────────────┘               │
                            │                          │
                            │ use_count == 0 &&        │
                            │ !pinned                  │
                            ▼                          │
                    ┌──────────────────┐               │
                    │   SCATTERING     │               │
                    │ (Reduce-scatter  │               │
                    │  gradients)      │               │
                    └──────────────────┘               │
                            │                          │
                            │ scatter completes        │
                            └──────────────────────────┘
```

### State Transitions

| Current State | Trigger | Next State | Action |
|--------------|---------|------------|--------|
| **PARTITIONED** | `gather_parameter()` | GATHERING | Start all-gather communication |
| **GATHERING** | Async completes | GATHERED | Mark parameter ready, store full_param |
| **GATHERED** | `use_count == 0` | SCATTERING | Start reduce-scatter for gradients |
| **SCATTERING** | Async completes | PARTITIONED | Free full_param, keep partition only |
| **GATHERED** | `pinned_in_memory` | GATHERED | Never transition (stay gathered) |

### Reference Counting

```cpp
struct ParameterState {
    int use_count{0};  // Number of active users

    // Increment when:
    // - forward_pre_hook() starts using parameter
    // - backward_pre_hook() starts using parameter

    // Decrement when:
    // - forward pass completes (end of forward)
    // - backward_post_hook() completes

    // Free when:
    // - use_count == 0
    // - !pinned_in_memory
    // - state == GATHERED
};

auto ZeROStage3Optimizer::free_gathered_parameter(Tensor* param) -> void {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto& state = param_states_[param];

    // Decrement reference count
    state.use_count--;

    if (state.use_count < 0) {
        throw std::runtime_error("Parameter use_count went negative!");
    }

    // Only free if no users and not pinned
    if (state.use_count == 0 &&
        !state.pinned_in_memory &&
        state.state == ParameterState::State::GATHERED) {

        // Free the gathered buffer
        state.full_param.reset();

        // Restore parameter to partition
        param->set_data(state.partition);

        // Update state
        state.state = ParameterState::State::PARTITIONED;

        perf_stats_.total_frees++;
    }
}
```

### Concurrent Access Handling

Multiple modules might try to use the same parameter simultaneously (e.g., shared weights). The state machine uses:

1. **Mutex protection**: `param_states_mutex_` guards all state transitions
2. **Reference counting**: `use_count` tracks active users
3. **State checks**: Only transition when safe (e.g., all users finished)

```cpp
// Thread-safe gather with concurrent access support
auto ZeROStage3Optimizer::gather_parameter(Tensor* param) -> Tensor {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto& state = param_states_[param];

    // Case 1: Already gathered - just increment ref count
    if (state.state == ParameterState::State::GATHERED) {
        state.use_count++;
        return state.full_param;
    }

    // Case 2: Gathering in progress - wait for it
    if (state.state == ParameterState::State::GATHERING) {
        // Unlock while waiting (allow other threads to make progress)
        lock.unlock();
        Tensor result = wait_gather(state.gather_handle);
        lock.lock();

        state.use_count++;
        return result;
    }

    // Case 3: Partitioned - start new gather
    state.state = ParameterState::State::GATHERING;
    state.gather_handle = gather_parameter_async(param);

    // Wait for gather to complete
    lock.unlock();
    state.full_param = wait_gather(state.gather_handle);
    lock.lock();

    state.gather_handle = nullptr;
    state.state = ParameterState::State::GATHERED;
    state.use_count = 1;

    return state.full_param;
}
```

---

## Prefetch Scheduler Design

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   PrefetchScheduler                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Priority Queue (Heap)                     │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │ Request 1: param_5, priority=10, t=100ms │    │    │
│  │  │ Request 2: param_6, priority=9,  t=150ms │    │    │
│  │  │ Request 3: param_7, priority=8,  t=200ms │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────┘    │
│                        │                                   │
│                        ▼                                   │
│  ┌──────────────────────────────────────────────────┐    │
│  │         In-Flight Set (Currently Gathering)       │    │
│  │  {param_1, param_2, param_3}                     │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
│  Decision Logic:                                          │
│  - If in_flight.size() < max_concurrent: Start next      │
│  - Else: Wait for one to complete                        │
│  - Evict oldest if buffer full                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Scheduling Algorithm

```cpp
class PrefetchScheduler {
public:
    /**
     * @brief Schedule parameter for prefetch
     *
     * Adds parameter to priority queue. Higher priority = prefetched sooner.
     * Priority calculated based on:
     * - Distance to module (closer = higher priority)
     * - Parameter size (smaller = higher priority for quick wins)
     * - Historical access patterns
     */
    auto schedule(Tensor* param, int priority, Module* requester) -> void {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        // Check if already in queue or in-flight
        if (in_flight_.count(param) > 0) {
            return;  // Already being gathered
        }

        // Check buffer size limit
        size_t param_size = param->numel() * dtype_size(param->dtype());
        if (current_buffer_size_ + param_size > max_prefetch_buffer_bytes_) {
            // Buffer full - evict lowest priority item
            evict_lowest_priority();
        }

        // Add to priority queue
        PrefetchRequest request{
            .param = param,
            .priority = priority,
            .deadline = std::chrono::steady_clock::now() +
                        estimate_gather_time(param),
            .requester = requester
        };

        queue_.push(request);
        current_buffer_size_ += param_size;

        // Execute if capacity available
        execute_pending();
    }

    /**
     * @brief Execute pending prefetch requests
     *
     * Starts async gathers for up to max_concurrent_prefetches_ parameters.
     */
    auto execute_pending() -> void {
        // Already locked by caller

        while (in_flight_.size() < max_concurrent_prefetches_ && !queue_.empty()) {
            auto request = queue_.top();
            queue_.pop();

            // Check if deadline already passed
            if (std::chrono::steady_clock::now() > request.deadline) {
                // Too late to prefetch - will be gathered on-demand
                stats_.missed_deadlines++;
                continue;
            }

            // Start async gather
            auto handle = optimizer_->gather_parameter_async(request.param);
            in_flight_.insert(request.param);
            in_flight_handles_[request.param] = handle;

            stats_.total_prefetches++;
        }
    }

    /**
     * @brief Check for completed prefetches
     *
     * Polls in-flight gathers and moves completed ones to cache.
     */
    auto check_completions() -> void {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        auto it = in_flight_.begin();
        while (it != in_flight_.end()) {
            auto* param = *it;
            auto& handle = in_flight_handles_[param];

            if (handle->is_ready()) {
                // Gather complete - update parameter state
                // (The ParameterState is updated by the gather operation)

                it = in_flight_.erase(it);
                in_flight_handles_.erase(param);
                stats_.completed_prefetches++;
            } else {
                ++it;
            }
        }

        // Execute more if capacity freed up
        execute_pending();
    }

private:
    // Priority queue (max-heap by priority)
    std::priority_queue<
        PrefetchRequest,
        std::vector<PrefetchRequest>,
        PrefetchRequestComparator
    > queue_;

    // Currently gathering parameters
    std::unordered_set<Tensor*> in_flight_;
    std::unordered_map<Tensor*, std::shared_ptr<AsyncHandle>> in_flight_handles_;

    // Configuration
    int max_concurrent_prefetches_;
    size_t max_prefetch_buffer_bytes_;
    size_t current_buffer_size_{0};

    // Synchronization
    std::mutex queue_mutex_;

    // Statistics
    struct Stats {
        size_t total_prefetches{0};
        size_t completed_prefetches{0};
        size_t missed_deadlines{0};
        size_t evictions{0};
    } stats_;
};
```

### Priority Calculation

```cpp
/**
 * @brief Calculate prefetch priority for a parameter
 *
 * Higher priority = prefetch sooner
 *
 * Factors:
 * - Layer distance: 1 / (layer_idx_diff + 1)
 * - Parameter size: Smaller params get slight boost (quick wins)
 * - Access frequency: Frequently used params get boost
 */
auto calculate_priority(
    Tensor* param,
    Module* current_module,
    const std::vector<Module*>& module_execution_order
) -> int {
    // Find current module's index
    auto current_it = std::find(
        module_execution_order.begin(),
        module_execution_order.end(),
        current_module
    );

    if (current_it == module_execution_order.end()) {
        return 0;  // Unknown module
    }

    size_t current_idx = std::distance(module_execution_order.begin(), current_it);

    // Find module that owns this parameter
    Module* owner = find_parameter_owner(param, module_execution_order);
    if (!owner) {
        return 0;
    }

    auto owner_it = std::find(
        module_execution_order.begin(),
        module_execution_order.end(),
        owner
    );

    size_t owner_idx = std::distance(module_execution_order.begin(), owner_it);

    // Distance-based priority
    int distance = static_cast<int>(owner_idx) - static_cast<int>(current_idx);
    if (distance <= 0) {
        return 0;  // Already passed or current
    }

    // Base priority: Inverse of distance
    int priority = 1000 / (distance + 1);

    // Size bonus: Smaller params get slight boost (can gather quickly)
    size_t param_bytes = param->numel() * dtype_size(param->dtype());
    if (param_bytes < 1024 * 1024) {  // < 1MB
        priority += 100;
    }

    // Frequency bonus: Check access history
    int access_count = get_parameter_access_count(param);
    priority += access_count * 10;

    return priority;
}
```

### Background Thread vs Inline Prefetch

Two implementation strategies:

#### Strategy 1: Inline Prefetch (Simpler)

```cpp
auto ZeROStage3Optimizer::forward_pre_hook(...) -> void {
    // Prefetch synchronously in hook
    prefetch_next_parameters(module);  // Starts async gathers

    // Then proceed with current layer
    gather_parameters_for_module(module);

    // Prefetches happen in background while this layer computes
}
```

**Pros**: Simple, no threads, easier debugging
**Cons**: Prefetch timing depends on hook execution, less control

#### Strategy 2: Background Thread (More Control)

```cpp
class PrefetchScheduler {
private:
    std::thread prefetch_thread_;
    std::atomic<bool> stop_{false};

    auto prefetch_worker() -> void {
        while (!stop_) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                check_completions();  // Move completed gathers to cache
                execute_pending();    // Start new gathers if capacity available
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

public:
    PrefetchScheduler() {
        prefetch_thread_ = std::thread(&PrefetchScheduler::prefetch_worker, this);
    }

    ~PrefetchScheduler() {
        stop_ = true;
        if (prefetch_thread_.joinable()) {
            prefetch_thread_.join();
        }
    }
};
```

**Pros**: Fine-grained control, can optimize timing, continuous monitoring
**Cons**: Thread overhead, synchronization complexity

**Recommendation**: Start with inline (Strategy 1), migrate to background thread (Strategy 2) if profiling shows benefits.

---

## Communication Strategy

### All-Gather for Parameters

```cpp
/**
 * @brief All-gather parameter from all ranks
 *
 * Each rank has a partition (1/N of the parameter).
 * All-gather collects all partitions to reconstruct the full parameter.
 *
 * Example (4 ranks, parameter size = 1000 elements):
 * - Rank 0: [0:250]
 * - Rank 1: [250:500]
 * - Rank 2: [500:750]
 * - Rank 3: [750:1000]
 *
 * After all-gather, all ranks have [0:1000].
 */
auto ZeROStage3Optimizer::gather_parameter(Tensor* param) -> Tensor {
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }

    auto& state = param_states_[param];

    // Get local partition
    Tensor local_partition = state.partition;

    // Allocate buffer for full parameter
    std::vector<int64_t> full_shape = param->shape();
    Tensor full_param = zeros(full_shape, param->dtype(), param->device());

    // All-gather operation
    // This is a collective: all ranks must call it
    config_.process_group->all_gather(
        local_partition,   // Input: local partition
        full_param,        // Output: full parameter
        gather_stream_     // CUDA stream for async
    );

    // Synchronize stream if not overlapping
    if (!stage3_config_.overlap_comm_compute) {
        gather_stream_.synchronize();
    }

    return full_param;
}
```

### Reduce-Scatter for Gradients

```cpp
/**
 * @brief Reduce-scatter parameter gradients
 *
 * Combines reduce (sum gradients across ranks) and scatter (partition result).
 * Each rank ends up with the sum of gradients for its partition.
 *
 * Example (4 ranks, gradient size = 1000):
 * Before:
 * - Rank 0: grad[0:1000] = [g0_0, g0_1, ..., g0_999]
 * - Rank 1: grad[0:1000] = [g1_0, g1_1, ..., g1_999]
 * - Rank 2: grad[0:1000] = [g2_0, g2_1, ..., g2_999]
 * - Rank 3: grad[0:1000] = [g3_0, g3_1, ..., g3_999]
 *
 * After:
 * - Rank 0: grad[0:250] = [sum(g*_0), sum(g*_1), ..., sum(g*_249)]
 * - Rank 1: grad[250:500] = [sum(g*_250), ..., sum(g*_499)]
 * - Rank 2: grad[500:750] = [sum(g*_500), ..., sum(g*_749)]
 * - Rank 3: grad[750:1000] = [sum(g*_750), ..., sum(g*_999)]
 */
auto ZeROStage3Optimizer::scatter_parameter_gradient(Tensor* param) -> void {
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }

    if (!param->has_grad()) {
        return;
    }

    auto& grad_opt = param->grad();
    if (!grad_opt.has_value()) {
        return;
    }

    Tensor full_grad = grad_opt.value();

    // Allocate buffer for local partition of gradient
    auto& state = param_states_[param];
    Tensor local_grad = zeros_like(state.partition);

    // Reduce-scatter operation
    // This is a collective: all ranks must call it
    config_.process_group->reduce_scatter(
        full_grad,          // Input: full gradient
        local_grad,         // Output: local partition of summed gradients
        distributed::ReduceOp::SUM,
        scatter_stream_     // CUDA stream for async
    );

    // Replace full gradient with local partition
    param->grad() = local_grad;

    // Free full gradient buffer (now only have local partition)
    full_grad.reset();
}
```

### Bucket Optimization

Group small parameters into buckets to reduce communication overhead:

```cpp
/**
 * @brief Gather multiple parameters in one all-gather operation
 *
 * Instead of:
 *   all_gather(param1)  // 10KB
 *   all_gather(param2)  // 15KB
 *   all_gather(param3)  // 20KB
 *
 * Do:
 *   all_gather([param1, param2, param3])  // 45KB (one comm op)
 */
auto ZeROStage3Optimizer::gather_parameters_bucketed(
    const std::vector<Tensor*>& params
) -> std::vector<Tensor> {
    // Flatten all local partitions into one buffer
    std::vector<Tensor> local_partitions;
    for (auto* param : params) {
        local_partitions.push_back(param_states_[param].partition);
    }

    Tensor flat_partitions = flatten_tensors(local_partitions);

    // All-gather the flattened buffer
    Tensor flat_full = zeros(
        {flat_partitions.numel() * config_.world_size},
        flat_partitions.dtype(),
        flat_partitions.device()
    );

    config_.process_group->all_gather(flat_partitions, flat_full, gather_stream_);

    // Unflatten into individual parameters
    std::vector<Tensor> full_params;
    size_t offset = 0;
    for (auto* param : params) {
        size_t param_size = param->numel();
        Tensor full_param = flat_full.slice(0, offset, offset + param_size);
        full_param = full_param.reshape(param->shape());
        full_params.push_back(full_param);
        offset += param_size;
    }

    return full_params;
}
```

### Communication/Compute Overlap

```
Timeline without overlap:
┌──────────┬──────────┬──────────┬──────────┐
│  Gather  │  Compute │  Gather  │  Compute │
│ Layer 1  │ Layer 1  │ Layer 2  │ Layer 2  │
└──────────┴──────────┴──────────┴──────────┘
    50ms       100ms      50ms       100ms
Total: 300ms

Timeline with overlap (prefetch):
┌──────────┬──────────┐
│  Gather  │  Compute │
│ Layer 1  │ Layer 1  │
└──────────┴─┬────────┘
             │  Gather  │
             │ Layer 2  │ (concurrent)
             └──────────┴──────────┐
                         │ Compute │
                         │ Layer 2  │
                         └──────────┘
Total: 200ms (33% faster!)
```

Implementation:

```cpp
auto ZeROStage3Optimizer::forward_pre_hook(...) -> void {
    // Step 1: Start prefetch for layer i+2 (non-blocking)
    if (i + 2 < num_layers) {
        prefetch_scheduler_->schedule(
            layers[i+2].params,
            priority = 100 - (i+2),  // Higher priority for closer layers
            layers[i+2]
        );
    }

    // Step 2: Wait for layer i parameters (hopefully already gathered by prefetch)
    gather_parameters_for_module(layers[i]);

    // Step 3: Layer i forward computes while layer i+2 gathers in background
    // (happens automatically via async CUDA streams)
}
```

---

## File Structure

### Header File Organization

**File**: `include/tenzor/nn/optim/zero_optimizer.hpp`

```cpp
#pragma once

#include "optimizer.hpp"
#include "../../distributed/distributed.hpp"
#include "../../core/offload_engine.hpp"
#include "../../core/cuda_stream.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>

namespace tenzor {
namespace optim {

// Forward declarations
class Module;
struct AsyncHandle;

// =============================================================================
// ZeRO Stage 1 (Existing)
// =============================================================================
struct ZeROStage1Config { /* ... */ };
class ZeROStage1Optimizer : public Optimizer { /* ... */ };

// =============================================================================
// ZeRO Stage 2 (Existing)
// =============================================================================
struct ZeROStage2Config : public ZeROStage1Config { /* ... */ };
class ZeROStage2Optimizer : public ZeROStage1Optimizer { /* ... */ };

// =============================================================================
// ZeRO Stage 3 (NEW)
// =============================================================================

/**
 * @brief Configuration for ZeRO Stage 3
 */
struct Stage3Config : public ZeROStage2Config {
    // Prefetch settings
    size_t prefetch_bucket_size{100 * 1024 * 1024};
    int prefetch_depth{2};
    int max_concurrent_prefetches{4};
    bool overlap_comm_compute{true};

    // Memory management
    bool pin_first_layer{true};
    bool pin_last_layer{true};
    size_t max_gathered_buffer_size{500 * 1024 * 1024};

    // Communication settings
    bool use_async_gather{true};
    bool use_separate_streams{true};
    int gather_stream_priority{-1};

    Stage3Config() = default;
};

/**
 * @brief ZeRO Stage 3: Full Model Partitioning
 */
class ZeROStage3Optimizer : public ZeROStage2Optimizer {
public:
    // Constructor & Destructor
    ZeROStage3Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const Stage3Config& config
    );
    ~ZeROStage3Optimizer() override;

    // Model registration
    auto register_model(Module& model) -> void;

    // Optimizer interface
    auto step() -> void override;

    // Parameter management
    auto gather_parameter(Tensor* param) -> Tensor;
    auto gather_parameter_async(Tensor* param) -> std::shared_ptr<AsyncHandle>;
    auto free_gathered_parameter(Tensor* param) -> void;
    auto scatter_parameter_gradient(Tensor* param) -> void;

    // State queries
    auto is_parameter_gathered(Tensor* param) const -> bool;
    auto pin_parameter(Tensor* param) -> void;
    auto unpin_parameter(Tensor* param) -> void;

    // Statistics
    struct PerformanceStats { /* ... */ };
    auto get_performance_stats() const -> PerformanceStats;
    auto reset_performance_stats() -> void;

private:
    // Parameter state
    struct ParameterState { /* ... */ };
    std::unordered_map<Tensor*, ParameterState> param_states_;
    mutable std::mutex param_states_mutex_;

    // Hooks
    struct ForwardPreHook { /* ... */ };
    struct BackwardPostHook { /* ... */ };
    std::vector<ForwardPreHook> forward_hooks_;
    std::vector<BackwardPostHook> backward_hooks_;

    // Prefetching
    class PrefetchScheduler;
    std::unique_ptr<PrefetchScheduler> prefetch_scheduler_;

    // Communication
    CUDAStream gather_stream_;
    CUDAStream scatter_stream_;

    // Statistics
    PerformanceStats perf_stats_;
    mutable std::mutex stats_mutex_;

    // Configuration
    Stage3Config stage3_config_;

    // Internal methods
    auto register_gather_scatter_hooks(Module& model) -> void;
    auto forward_pre_hook(Module* module, const std::vector<Tensor>& inputs) -> void;
    auto backward_post_hook(Module* module, const std::vector<Tensor>& grad_outputs) -> void;
    auto prefetch_next_parameters(Module* current_module) -> void;
    auto wait_gather(std::shared_ptr<AsyncHandle> handle) -> Tensor;
};

} // namespace optim
} // namespace tenzor
```

### Source File Organization

**File**: `src/nn/optim/zero_stage3.cpp`

```cpp
#include "tenzor/nn/optim/zero_optimizer.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace optim {

// =============================================================================
// PrefetchScheduler Implementation
// =============================================================================

class ZeROStage3Optimizer::PrefetchScheduler {
    // Implementation details...
};

// =============================================================================
// Constructor & Destructor
// =============================================================================

ZeROStage3Optimizer::ZeROStage3Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const Stage3Config& config
) : ZeROStage2Optimizer(std::move(base_optimizer), config),
    stage3_config_(config) {
    // Initialize prefetch scheduler
    // Initialize CUDA streams
    // ...
}

ZeROStage3Optimizer::~ZeROStage3Optimizer() {
    // Cleanup
}

// =============================================================================
// Model Registration
// =============================================================================

auto ZeROStage3Optimizer::register_model(Module& model) -> void {
    // Walk module tree and register hooks
    // Initialize parameter states
    // ...
}

// =============================================================================
// Optimizer Step
// =============================================================================

auto ZeROStage3Optimizer::step() -> void {
    // Update partitions (no gather needed)
    // ...
}

// =============================================================================
// Parameter Gathering
// =============================================================================

auto ZeROStage3Optimizer::gather_parameter(Tensor* param) -> Tensor {
    // Synchronous all-gather
    // ...
}

auto ZeROStage3Optimizer::gather_parameter_async(Tensor* param)
    -> std::shared_ptr<AsyncHandle> {
    // Asynchronous all-gather
    // ...
}

// =============================================================================
// Hook Implementation
// =============================================================================

auto ZeROStage3Optimizer::forward_pre_hook(...) -> void {
    // Gather parameters before forward
    // ...
}

auto ZeROStage3Optimizer::backward_post_hook(...) -> void {
    // Reduce-scatter gradients after backward
    // ...
}

// =============================================================================
// Prefetching
// =============================================================================

auto ZeROStage3Optimizer::prefetch_next_parameters(...) -> void {
    // Schedule prefetch for upcoming layers
    // ...
}

} // namespace optim
} // namespace tenzor
```

### Test File Organization

**File**: `tests/nn/optim/test_zero_stage3.cpp`

```cpp
#include "tenzor/nn/optim/zero_optimizer.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/distributed/distributed.hpp"
#include <gtest/gtest.h>

namespace tenzor {
namespace optim {
namespace test {

// =============================================================================
// Test Fixtures
// =============================================================================

class ZeROStage3Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize distributed environment
        // Create test model
        // ...
    }

    void TearDown() override {
        // Cleanup
    }

    // Test utilities
    auto create_simple_model() -> std::unique_ptr<Module>;
    auto create_test_optimizer() -> std::unique_ptr<ZeROStage3Optimizer>;
};

// =============================================================================
// Parameter State Machine Tests
// =============================================================================

TEST_F(ZeROStage3Test, ParameterStateTransitions) {
    // Test PARTITIONED -> GATHERING -> GATHERED -> PARTITIONED
}

TEST_F(ZeROStage3Test, ReferenceCountingWorks) {
    // Test use_count increment/decrement
}

TEST_F(ZeROStage3Test, PinnedParametersNeverFreed) {
    // Test pinned_in_memory flag
}

// =============================================================================
// Hook System Tests
// =============================================================================

TEST_F(ZeROStage3Test, ForwardPreHookGathersParameters) {
    // Test hook gathers params before forward
}

TEST_F(ZeROStage3Test, BackwardPostHookScattersGradients) {
    // Test hook reduce-scatters grads after backward
}

TEST_F(ZeROStage3Test, HooksExecuteInCorrectOrder) {
    // Test hook execution sequence
}

// =============================================================================
// Prefetch Scheduler Tests
// =============================================================================

TEST_F(ZeROStage3Test, PrefetchReducesGatherLatency) {
    // Test prefetch improves performance
}

TEST_F(ZeROStage3Test, PrefetchQueuePrioritizesCorrectly) {
    // Test priority queue ordering
}

TEST_F(ZeROStage3Test, PrefetchRespectsMemoryLimit) {
    // Test max_prefetch_buffer_size
}

// =============================================================================
// Communication Tests
// =============================================================================

TEST_F(ZeROStage3Test, AllGatherReconstructsFullParameter) {
    // Test gather_parameter()
}

TEST_F(ZeROStage3Test, ReduceScatterPartitionsGradients) {
    // Test scatter_parameter_gradient()
}

TEST_F(ZeROStage3Test, AsyncCommunicationWorks) {
    // Test async gather/scatter
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(ZeROStage3Test, TrainSimpleModelEnd2End) {
    // Full training loop with Stage 3
}

TEST_F(ZeROStage3Test, MemoryUsageLowerThanStage2) {
    // Verify memory savings
}

TEST_F(ZeROStage3Test, CheckpointSaveLoadWorks) {
    // Test save/load with partitioned params
}

} // namespace test
} // namespace optim
} // namespace tenzor
```

---

## Implementation Flow

### Phase 6 Implementation Steps

```
Week 1-2: Core Infrastructure
├─> Implement ParameterState structure
├─> Implement parameter state machine transitions
├─> Implement gather_parameter() (synchronous)
├─> Implement scatter_parameter_gradient()
├─> Write unit tests for state machine
└─> Write unit tests for gather/scatter

Week 3-4: Hook System
├─> Implement ForwardPreHook registration
├─> Implement BackwardPostHook registration
├─> Implement forward_pre_hook() logic
├─> Implement backward_post_hook() logic
├─> Implement reference counting
└─> Write unit tests for hooks

Week 5-6: Prefetch Scheduler
├─> Implement PrefetchScheduler class
├─> Implement priority queue logic
├─> Implement prefetch_next_parameters()
├─> Implement async gather (gather_parameter_async)
├─> Tune prefetch heuristics
└─> Write unit tests for prefetching

Week 7: Integration & Testing
├─> Integrate with existing Stage 1/2
├─> Test on real models (BERT, GPT-2)
├─> Memory profiling
├─> Performance benchmarking
└─> Fix bugs and optimize

Week 8: Documentation & Polish
├─> API documentation
├─> Usage examples
├─> Best practices guide
└─> Performance tuning guide
```

### Critical Path Items

1. **Parameter State Machine** - Must be rock-solid (reference counting bugs = memory leaks)
2. **Hook Registration** - Must work with autograd system
3. **Async Communication** - Must overlap correctly with compute
4. **Prefetch Scheduler** - Performance depends heavily on this

---

## Testing Strategy

### Unit Tests

```cpp
// Test 1: Parameter state transitions
TEST(ZeROStage3, ParameterStateTransitions) {
    auto param = create_test_parameter();
    auto optimizer = create_stage3_optimizer();

    // Initial state: PARTITIONED
    EXPECT_EQ(optimizer.get_parameter_state(param), State::PARTITIONED);

    // Gather -> GATHERING -> GATHERED
    auto full_param = optimizer.gather_parameter(param);
    EXPECT_EQ(optimizer.get_parameter_state(param), State::GATHERED);

    // Free -> PARTITIONED
    optimizer.free_gathered_parameter(param);
    EXPECT_EQ(optimizer.get_parameter_state(param), State::PARTITIONED);
}

// Test 2: Reference counting
TEST(ZeROStage3, ReferenceCountingWorks) {
    auto param = create_test_parameter();
    auto optimizer = create_stage3_optimizer();

    // Gather twice (two users)
    optimizer.gather_parameter(param);  // use_count = 1
    optimizer.gather_parameter(param);  // use_count = 2

    // Free once - should not actually free (use_count = 1)
    optimizer.free_gathered_parameter(param);
    EXPECT_TRUE(optimizer.is_parameter_gathered(param));

    // Free again - should free now (use_count = 0)
    optimizer.free_gathered_parameter(param);
    EXPECT_FALSE(optimizer.is_parameter_gathered(param));
}

// Test 3: Prefetch improves performance
TEST(ZeROStage3, PrefetchReducesLatency) {
    auto model = create_test_model();
    auto optimizer = create_stage3_optimizer_with_prefetch();

    // Measure gather time WITHOUT prefetch
    auto start1 = std::chrono::steady_clock::now();
    optimizer.gather_parameter(model.layer1.weight);
    auto end1 = std::chrono::steady_clock::now();
    auto no_prefetch_time = end1 - start1;

    // Measure gather time WITH prefetch
    optimizer.prefetch_next_parameters(model.layer0);  // Prefetch layer1
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Let it gather

    auto start2 = std::chrono::steady_clock::now();
    optimizer.gather_parameter(model.layer1.weight);  // Should hit cache
    auto end2 = std::chrono::steady_clock::now();
    auto with_prefetch_time = end2 - start2;

    // Prefetch should be faster (or at least not slower)
    EXPECT_LE(with_prefetch_time, no_prefetch_time * 1.1);
}
```

### Integration Tests

```cpp
// Test: Full training loop with Stage 3
TEST(ZeROStage3Integration, TrainBERTModel) {
    // Setup distributed (4 GPUs)
    auto world = DistributedWorld::init_for_test(4);

    // Create BERT model
    auto model = BERTModel(BERTConfig::bert_base());

    // Create ZeRO Stage 3 optimizer
    Stage3Config config;
    config.world_size = 4;
    config.rank = world.rank();
    config.prefetch_depth = 2;
    config.overlap_comm_compute = true;

    auto optimizer = ZeROStage3Optimizer(
        std::make_unique<AdamW>(model.parameters(), 1e-4),
        config
    );

    // Register model for partitioning
    optimizer.register_model(model);

    // Verify parameters are partitioned
    for (auto& param : model.parameters()) {
        EXPECT_FALSE(optimizer.is_parameter_gathered(param));
    }

    // Train for 100 steps
    for (int step = 0; step < 100; ++step) {
        optimizer.zero_grad();

        auto batch = generate_random_batch(32);
        auto output = model.forward(batch.input);

        // Verify parameters were gathered during forward
        // (This is checked inside the hooks)

        auto loss = cross_entropy(output, batch.labels);
        loss.backward();

        optimizer.step();

        // Verify parameters are partitioned again after step
        for (auto& param : model.parameters()) {
            if (!optimizer.is_pinned(param)) {
                EXPECT_FALSE(optimizer.is_parameter_gathered(param));
            }
        }
    }

    // Verify memory usage
    auto stats = optimizer.get_memory_stats();
    size_t expected_memory = model.memory_footprint() / 4;  // Divided by world size
    EXPECT_LT(stats.gpu_optimizer_memory, expected_memory * 1.5);  // Allow 50% overhead
}
```

### Performance Benchmarks

```cpp
// Benchmark: Stage 3 vs Stage 2 memory usage
BENCHMARK(CompareMemoryStage2VsStage3) {
    auto model = GPT2Model(GPT2Config::gpt2_medium());

    // Measure Stage 2 memory
    auto stage2_opt = create_stage2_optimizer(model);
    size_t stage2_memory = measure_gpu_memory();

    // Measure Stage 3 memory
    auto stage3_opt = create_stage3_optimizer(model);
    stage3_opt.register_model(model);
    size_t stage3_memory = measure_gpu_memory();

    // Stage 3 should use significantly less memory
    double reduction = (stage2_memory - stage3_memory) / (double)stage2_memory;
    EXPECT_GT(reduction, 0.5);  // At least 50% reduction

    std::cout << "Stage 2 memory: " << stage2_memory << " bytes\n";
    std::cout << "Stage 3 memory: " << stage3_memory << " bytes\n";
    std::cout << "Reduction: " << (reduction * 100) << "%\n";
}

// Benchmark: Communication overhead
BENCHMARK(MeasureCommunicationOverhead) {
    auto model = create_test_model();
    auto optimizer = create_stage3_optimizer(model);

    // Measure total time with Stage 3
    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < 100; ++step) {
        train_step(model, optimizer);
    }
    auto end = std::chrono::steady_clock::now();
    auto stage3_time = end - start;

    // Measure baseline time (no ZeRO)
    auto baseline_opt = create_baseline_optimizer(model);
    start = std::chrono::steady_clock::now();
    for (int step = 0; step < 100; ++step) {
        train_step(model, baseline_opt);
    }
    end = std::chrono::steady_clock::now();
    auto baseline_time = end - start;

    // Calculate overhead
    double overhead = (stage3_time - baseline_time) / baseline_time * 100;

    std::cout << "Baseline time: " << baseline_time.count() << " ms\n";
    std::cout << "Stage 3 time: " << stage3_time.count() << " ms\n";
    std::cout << "Overhead: " << overhead << "%\n";

    // Target: <25% overhead
    EXPECT_LT(overhead, 25.0);
}
```

---

## Conclusion

This architecture document provides a complete blueprint for implementing ZeRO Stage 3:

1. **Class Hierarchy**: ZeROStage3Optimizer extends Stage 2 with parameter partitioning
2. **Hook System**: Forward/backward hooks automatically gather/scatter parameters
3. **State Machine**: Robust parameter lifecycle with reference counting
4. **Prefetch Scheduler**: Predictive prefetching to hide communication latency
5. **Communication**: Efficient all-gather and reduce-scatter with overlap
6. **File Structure**: Clean separation of header, implementation, and tests

### Key Implementation Priorities

1. **Correctness First**: State machine and reference counting must be bulletproof
2. **Performance Second**: Prefetching and overlap optimize after correctness verified
3. **Testing Throughout**: Unit tests, integration tests, and benchmarks at every stage

### Expected Outcomes

- **Memory Savings**: Nx reduction (8x for 8 GPUs)
- **Performance Overhead**: <25% vs baseline
- **Scalability**: Train 175B models on 8x A100 GPUs

**Ready for implementation!**

