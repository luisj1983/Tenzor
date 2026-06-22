/**
 * @file fsdp.hpp
 * @brief Fully Sharded Data Parallel (FSDP) training wrapper
 *
 * Implements ZeRO-3 style parameter sharding where each rank holds only a
 * shard of the model parameters. Parameters are all-gathered before each
 * forward pass and gradients are reduce-scattered during the backward pass,
 * enabling training of models that exceed single-GPU memory.
 */

#pragma once

#include "distributed.hpp"
#include "../nn/module.hpp"
#include "../autograd/variable.hpp"
#include "../core/dtype.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

namespace tenzor::distributed {

/**
 * @brief Sharding strategy controlling what gets sharded across ranks.
 */
enum class ShardingStrategy {
    /** @brief Full ZeRO-3: shard parameters, gradients, and optimizer states */
    FULL_SHARD,

    /** @brief ZeRO-2: shard only gradients and optimizer states (keep full params) */
    SHARD_GRAD_OP,

    /** @brief No sharding: equivalent to DDP (for debugging/comparison) */
    NO_SHARD
};

/**
 * @brief Configuration for FSDP wrapping.
 */
struct FSDPConfig {
    /** @brief Sharding strategy (default: FULL_SHARD) */
    ShardingStrategy strategy{ShardingStrategy::FULL_SHARD};

    /** @brief Offload parameters to CPU when not in use */
    bool cpu_offload{false};

    /** @brief Minimum parameter count for auto-wrapping a submodule as an FSDP unit */
    size_t auto_wrap_min_params{100000};

    /** @brief Whether to use mixed precision for communication (future) */
    bool mixed_precision{false};

    /**
     * @brief Reduced-precision dtype for cross-rank communication.
     *
     * Audit-4 W.11: previously the FSDP all-gather hardcoded Float16 when
     * mixed_precision was enabled. BFloat16 is the dominant AMP dtype for
     * large-language-model training (matches Float32 exponent range — F16
     * overflows for activations ≥ 65k) and was unreachable. This field
     * lets the caller pick the comm dtype; the default is BFloat16, which
     * is the safer choice for the modern transformer workloads FSDP
     * targets.
     */
    DType comm_dtype{DType::BFloat16};

    /** @brief Limit all-gather to this many FSDP units in flight at once (0 = no limit) */
    size_t forward_prefetch_limit{2};

    /** @brief Whether to prefetch parameters for the next FSDP unit during forward */
    bool forward_prefetch{true};

    /** @brief Whether to prefetch parameters for backward during forward */
    bool backward_prefetch{true};
};

/**
 * @brief An FSDP unit wrapping a subtree of the model.
 *
 * Each FSDPUnit manages a flat parameter buffer that contains all parameters
 * from its wrapped module subtree. The buffer is sharded across ranks, with
 * each rank holding a contiguous chunk of size total_params / world_size.
 *
 * Lifecycle during training:
 * 1. Pre-forward: all-gather to reconstruct full parameters
 * 2. Forward: compute with full parameters
 * 3. Post-forward: free non-local shards (FULL_SHARD only)
 * 4. Pre-backward: all-gather parameters again (FULL_SHARD only)
 * 5. Post-backward: reduce-scatter gradients, re-shard parameters
 */
class FSDPUnit {
public:
    /**
     * @brief Construct an FSDP unit for a module subtree.
     *
     * Flattens all parameters in the module into a contiguous buffer,
     * then shards the buffer across ranks.
     *
     * @param module The module subtree to wrap
     * @param pg Process group for communication
     * @param config FSDP configuration
     */
    FSDPUnit(nn::Module& module, ProcessGroup& pg, const FSDPConfig& config,
             std::vector<std::shared_ptr<Variable>> explicit_params = {});

    ~FSDPUnit();

    // Non-copyable
    FSDPUnit(const FSDPUnit&) = delete;
    FSDPUnit& operator=(const FSDPUnit&) = delete;

    /**
     * @brief All-gather parameters from all ranks to reconstruct full buffer.
     *
     * After this call, all parameters in the wrapped module have their full
     * (unsharded) values and can be used for computation.
     */
    auto all_gather_params() -> void;

    /**
     * @brief Free non-local parameter shards to reclaim memory.
     *
     * After forward, we only need to keep our local shard. This frees
     * the full parameter buffer and restores parameters to their sharded state.
     * Only effective when strategy is FULL_SHARD.
     */
    auto free_full_params() -> void;

    /**
     * @brief Reduce-scatter gradients and re-shard parameters.
     *
     * Called during the backward pass. Performs reduce-scatter on the
     * gradient buffer (each rank gets its shard of the averaged gradient)
     * and optionally divides by world_size.
     */
    auto reduce_scatter_grads() -> void;

    /**
     * @brief Offload local shard to CPU (if cpu_offload is enabled).
     */
    auto offload_to_cpu() -> void;

    /**
     * @brief Reload local shard from CPU back to GPU.
     */
    auto reload_from_cpu() -> void;

    /**
     * @brief Get the wrapped module.
     */
    auto module() -> nn::Module& { return module_; }
    auto module() const -> const nn::Module& { return module_; }

    /**
     * @brief Check if parameters are currently in the full (all-gathered) state.
     */
    auto is_full() const -> bool { return params_full_; }

    /**
     * @brief Get total number of elements across all parameters in this unit.
     */
    auto total_numel() const -> size_t { return total_numel_; }

    /**
     * @brief Get the local shard size (elements this rank is responsible for).
     */
    auto shard_numel() const -> size_t { return shard_numel_; }

    /** @brief A rank's contiguous range within the flattened parameter buffer. */
    struct ShardRange {
        size_t shard_numel{0};   ///< padded shard length (same for every rank)
        size_t shard_offset{0};  ///< start offset of this rank's shard
        size_t valid_numel{0};   ///< elements actually backed by real data
                                 ///< (0 when the shard is entirely padding)
    };

    /**
     * @brief Compute a rank's shard range over `total_numel` elements split
     *        across `world_size` ranks (ceil division; trailing ranks padded).
     *
     * `valid_numel` is clamped so that a high rank whose offset lands at or
     * beyond `total_numel` (e.g. total_numel=5, world_size=4, rank 3 -> offset 6)
     * reports 0 valid elements rather than underflowing the unsigned length
     * subtraction, which would otherwise request a ~1.8e19-element slice.
     */
    static auto compute_shard_range(size_t total_numel, int world_size, int rank)
        -> ShardRange {
        ShardRange r{};
        if (world_size <= 0 || rank < 0) {
            return r;
        }
        r.shard_numel = (total_numel + static_cast<size_t>(world_size) - 1) /
                        static_cast<size_t>(world_size);
        r.shard_offset = static_cast<size_t>(rank) * r.shard_numel;
        if (r.shard_offset < total_numel) {
            const size_t remaining = total_numel - r.shard_offset;
            r.valid_numel = remaining < r.shard_numel ? remaining : r.shard_numel;
        }
        return r;
    }

    /**
     * @brief Get the flat parameter buffer (full or sharded depending on state).
     */
    auto flat_param() const -> const Tensor& { return flat_param_; }

    /**
     * @brief Get the flat gradient buffer.
     */
    auto flat_grad() const -> const Tensor& { return flat_grad_; }

    // audit-9 JJ.4: accessors needed by FullyShardedDataParallel::state_dict /
    // load_state_dict to serialise per-rank shards.

    /** @brief Get the local shard tensor (this rank's slice of flat_param_). */
    auto local_shard() const -> const Tensor& { return local_shard_; }

    /** @brief Get the byte offset into the flat buffer at which this shard starts. */
    auto shard_offset() const -> size_t { return shard_offset_; }

    /** @brief Get original (unflattened) parameter shapes, in flatten order. */
    auto param_shapes() const -> const std::vector<std::vector<int64_t>>& { return param_shapes_; }

    /** @brief Get original parameter element counts, in flatten order. */
    auto param_numels() const -> const std::vector<size_t>& { return param_numels_; }

    /** @brief Names of original parameters (matching param_shapes order). */
    auto param_names() const -> std::vector<std::string>;

    /**
     * @brief Replace the local shard tensor in-place.
     *
     * audit-9 JJ.4: load_state_dict path needs to copy a saved shard into
     * the live local_shard_.  Validates numel and dtype; throws on
     * mismatch.  Does not adjust shard_offset_ — caller (FSDP::load_state_dict)
     * is responsible for that.
     */
    auto copy_local_shard_from(const Tensor& src) -> void;

private:
    nn::Module& module_;
    ProcessGroup* pg_;
    FSDPConfig config_;

    /** @brief Flat contiguous buffer holding all parameters */
    Tensor flat_param_;

    /** @brief Flat contiguous buffer holding all gradients */
    Tensor flat_grad_;

    /** @brief Local shard of the flat parameter buffer */
    Tensor local_shard_;

    /** @brief CPU copy of local shard (used when cpu_offload is enabled) */
    Tensor cpu_shard_;

    /** @brief Original parameter shapes for unflattening */
    std::vector<std::vector<int64_t>> param_shapes_;

    /** @brief Original parameter numels for offset computation */
    std::vector<size_t> param_numels_;

    /** @brief Shared pointers to original parameters (for writing back) */
    std::vector<std::shared_ptr<Variable>> original_params_;

    // When non-empty, the explicit parameter set this unit owns. The recursive
    // auto-wrap policy uses this to partition the module tree so that every
    // parameter is sharded by exactly one unit (including the root module's own
    // parameters and the parameters of descendants of a wrapped submodule).
    // When empty, flatten_params() falls back to module_.own_parameters().
    std::vector<std::shared_ptr<Variable>> explicit_params_;

    /** @brief Total number of elements across all parameters */
    size_t total_numel_{0};

    /** @brief Number of elements in this rank's shard */
    size_t shard_numel_{0};

    /** @brief Offset into flat buffer where this rank's shard starts */
    size_t shard_offset_{0};

    /** @brief Whether parameters are currently in full (all-gathered) state */
    bool params_full_{false};

    /** @brief Whether local shard is currently on CPU */
    bool offloaded_{false};

    /**
     * @brief NN.18: re-entry guard for summon_full_params().
     *
     * Incremented by FSDP::summon_full_params() before triggering forwards,
     * decremented by FSDP::release_full_params() afterwards.  The forward
     * post-hook (which would normally call free_full_params() to release
     * the all-gathered shards immediately after each module forward) sees
     * a non-zero depth and early-returns, leaving the full params resident
     * for the duration of the summon window.  Without this guard, a
     * forward executed *inside* summon_full_params() would free the very
     * params summon was trying to keep around, and the next layer would
     * read a sharded tensor.
     */
    int summon_depth_{0};

public:
    /** @brief NN.18: bump/drop the summon re-entry guard.  Internal — used
     *         by FullyShardedDataParallel::summon_full_params() /
     *         release_full_params(). */
    auto enter_summon() -> void { ++summon_depth_; }
    auto exit_summon()  -> void { if (summon_depth_ > 0) --summon_depth_; }
    auto in_summon() const -> bool { return summon_depth_ > 0; }
private:

    // ---- GPU communication resources ----

    /** @brief Whether the process group uses GPU backend */
    bool use_gpu_comm_{false};

    /** @brief Dedicated communication stream (void* to avoid cuda header) */
    void* comm_stream_{nullptr};

    /** @brief CUDA event for synchronization */
    void* comm_event_{nullptr};

    /**
     * @brief Flatten all module parameters into a single contiguous buffer.
     */
    auto flatten_params() -> void;

    /**
     * @brief Shard the flat buffer: each rank keeps its contiguous chunk.
     */
    auto shard_params() -> void;

    /**
     * @brief Write values from flat_param_ back into original parameter Variables.
     */
    auto unflatten_params() -> void;

    /**
     * @brief Collect gradients from original parameters into flat_grad_.
     */
    auto collect_grads() -> void;

    /**
     * @brief Write reduced gradient shard back to original parameter Variables.
     */
    auto scatter_grads_to_params() -> void;

    /**
     * @brief Initialize GPU communication resources.
     */
    auto init_comm_resources() -> void;

    /**
     * @brief Destroy GPU communication resources.
     */
    auto destroy_comm_resources() -> void;
};

/**
 * @brief Fully Sharded Data Parallel wrapper for memory-efficient distributed training.
 *
 * Wraps an nn::Module and automatically shards parameters across processes
 * using the ZeRO-3 algorithm. Each rank holds only 1/N of the parameters
 * (where N is world_size), with parameters being all-gathered on demand
 * for computation and freed immediately after.
 *
 * Memory savings: approximately world_size reduction in parameter memory,
 * enabling training of models that don't fit on a single GPU.
 *
 * Usage:
 * @code
 * // Initialize distributed context
 * distributed::init_process_group("nccl", rank, world_size);
 * auto pg = DistributedContext::get_process_group();
 *
 * // Create model and wrap with FSDP
 * auto model = std::make_shared<LargeModel>();
 * FSDPConfig config;
 * config.strategy = ShardingStrategy::FULL_SHARD;
 * config.cpu_offload = true;
 * FullyShardedDataParallel fsdp(*model, *pg, config);
 *
 * // Training loop
 * for (auto& batch : dataloader) {
 *     auto output = fsdp.forward(input);
 *     auto loss = criterion(output, target);
 *     loss.backward();
 *     fsdp.finalize_backward();  // reduce-scatter gradients
 *     optimizer.step();
 *     model->zero_grad();
 * }
 * @endcode
 */
class FullyShardedDataParallel {
public:
    /**
     * @brief Construct FSDP wrapper.
     *
     * Applies auto-wrap policy to identify FSDP units (submodules with
     * parameter count >= auto_wrap_min_params), flattens and shards
     * parameters for each unit, and registers forward/backward hooks.
     *
     * @param module The neural network module to wrap
     * @param pg The process group for communication
     * @param config FSDP configuration
     */
    FullyShardedDataParallel(nn::Module& module, ProcessGroup& pg,
                             const FSDPConfig& config = FSDPConfig{});

    ~FullyShardedDataParallel();

    // Non-copyable
    FullyShardedDataParallel(const FullyShardedDataParallel&) = delete;
    FullyShardedDataParallel& operator=(const FullyShardedDataParallel&) = delete;

    /**
     * @brief Forward pass with automatic parameter gathering.
     *
     * For each FSDP unit in execution order:
     * 1. All-gather parameters (restore full params from shards)
     * 2. Execute the unit's forward computation
     * 3. Free non-local shards (FULL_SHARD only)
     *
     * @param input Input variable
     * @return Output variable from the wrapped module
     */
    auto forward(const Variable& input) -> Variable;

    /**
     * @brief Finalize backward pass: reduce-scatter all gradients.
     *
     * Must be called after loss.backward() to perform the reduce-scatter
     * on each FSDP unit's gradient buffer. After this call, each rank holds
     * only its shard of the averaged gradient.
     */
    auto finalize_backward() -> void;

    /**
     * @brief Get reference to the wrapped module.
     */
    auto module() -> nn::Module& { return module_; }
    auto module() const -> const nn::Module& { return module_; }

    /**
     * @brief Get the FSDP units.
     */
    auto units() const -> const std::vector<std::unique_ptr<FSDPUnit>>& { return units_; }

    /**
     * @brief Get the FSDP configuration.
     */
    auto config() const -> const FSDPConfig& { return config_; }

    /**
     * @brief Summon full parameters temporarily for saving/inspection.
     *
     * All-gathers all parameters across all FSDP units. The caller
     * should call release_full_params() when done.
     */
    auto summon_full_params() -> void;

    /**
     * @brief Release full parameters and return to sharded state.
     */
    auto release_full_params() -> void;

    /**
     * @brief Get total number of parameters across all FSDP units.
     */
    auto total_params() const -> size_t;

    /**
     * @brief Get total sharded parameter memory (bytes on this rank).
     */
    auto sharded_param_bytes() const -> size_t;

    /**
     * @brief Replace the process group for elastic training recovery.
     *
     * After a worker failure, the training group is rebuilt with a new
     * world_size. This re-shards all parameters for the new group.
     *
     * @param new_pg New process group
     */
    auto reset_process_group(ProcessGroup& new_pg) -> void {
        pg_ = &new_pg;
        // Re-shard: all-gather current shards, then re-partition for new world_size
        summon_full_params();
        release_full_params();
    }

    /**
     * @brief Per-rank sharded checkpoint dictionary.
     *
     * audit-9 JJ.4: returns the per-rank shard of every flat-parameter
     * tensor across all FSDP units, plus metadata (`world_size`, `rank`,
     * unit/parameter names, original shapes) needed to reassemble at load
     * time.  Round-trips with `load_state_dict()`.  Each rank's dict
     * contains only its shard slice; checkpoint orchestration on the
     * caller side concatenates across ranks (or uses an
     * IO-format-aware sharded saver like distcp).
     *
     * Key layout: `unit_<i>/<param_name>/shard` -> Tensor (1-D, this
     * rank's slice of the flat parameter).  Plus scalar metadata keys:
     * `world_size`, `rank`, `<param>/orig_shape`,
     * `<param>/flat_numel`.
     */
    auto state_dict() const -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Restore from a per-rank sharded checkpoint dictionary.
     *
     * audit-9 JJ.4: validates that `world_size` and `rank` match the
     * current process group; validates flat-param numel + original shape
     * for each unit; copies the shard tensors into the FSDP unit's
     * flat-storage slice for this rank.  Throws on mismatch (changed
     * world_size, mismatched unit count, mismatched parameter shape).
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void;

private:
    nn::Module& module_;
    ProcessGroup* pg_;
    FSDPConfig config_;

    /** @brief FSDP units (one per wrapped submodule or the root) */
    std::vector<std::unique_ptr<FSDPUnit>> units_;

    /** @brief Mutex for thread-safe operations */
    std::mutex mutex_;

    /**
     * @brief Apply auto-wrap policy to identify submodules that should
     *        become FSDP units based on parameter count threshold.
     */
    auto apply_auto_wrap() -> void;

    /**
     * @brief Wrap a single module as an FSDP unit.
     *
     * @param module Module to wrap
     */
    auto wrap_module(nn::Module& module) -> void;

    /**
     * @brief Recursively partition a module subtree into FSDP units.
     *
     * Bottom-up: each child subtree whose (not-yet-wrapped) trainable parameter
     * count meets the auto-wrap threshold becomes its own unit owning exactly the
     * parameters not already claimed by a deeper unit; everything else bubbles up
     * to the caller. Guarantees every parameter is sharded by exactly one unit.
     *
     * @param module Subtree root to partition.
     * @return Parameters of this subtree not placed into a descendant unit.
     */
    auto collect_units(nn::Module& module)
        -> std::vector<std::shared_ptr<Variable>>;

    /**
     * @brief Count parameters in a module (non-recursive: only own params).
     *
     * @param module Module to count
     * @return Number of parameters
     */
    auto count_params(nn::Module& module) const -> size_t;

    /**
     * @brief Register forward pre-hooks for all-gathering.
     */
    auto register_hooks() -> void;
};

} // namespace tenzor::distributed
