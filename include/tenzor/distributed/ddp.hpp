/**
 * @file ddp.hpp
 * @brief Distributed Data Parallel training wrapper
 *
 * Provides DistributedDataParallel (DDP) which wraps an nn::Module and
 * automatically synchronizes gradients across processes during the backward
 * pass. Uses gradient bucketing for efficient communication/computation overlap.
 */

#pragma once

#include "distributed.hpp"
#include "gradient_compression.hpp"
#include "../nn/module.hpp"
#include "../autograd/variable.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <unordered_set>

namespace tenzor::distributed {

/**
 * @brief Gradient bucket for efficient all-reduce communication.
 *
 * Groups multiple parameter gradients together into buckets of configurable
 * size (default 25MB) to amortize communication overhead and improve bandwidth
 * utilization. Parameters are assigned to buckets in reverse order of their
 * usage in the forward pass, matching PyTorch DDP behavior.
 */
struct GradBucket {
    /** @brief Parameters whose gradients belong to this bucket */
    std::vector<std::shared_ptr<Variable>> params;

    /** @brief Total size of gradients in this bucket in bytes */
    size_t size_bytes{0};

    /** @brief Whether all gradients in this bucket are ready for all-reduce */
    bool ready{false};

    /** @brief Number of parameters whose gradients have been computed */
    size_t pending_count{0};
};

/**
 * @brief Distributed Data Parallel wrapper for multi-process training.
 *
 * Wraps an nn::Module and automatically synchronizes gradients across
 * processes during the backward pass using gradient bucketing for
 * communication/computation overlap.
 *
 * Usage:
 * @code
 * // Initialize distributed context
 * distributed::init_process_group("nccl", rank, world_size);
 *
 * // Create model and wrap with DDP
 * auto model = std::make_shared<MyModel>();
 * auto pg = DistributedContext::get_process_group();
 * DistributedDataParallel ddp(*model, *pg);
 *
 * // Training loop
 * for (auto& batch : dataloader) {
 *     auto output = ddp.forward(input);
 *     auto loss = criterion(output, target);
 *     loss.backward();
 *     ddp.synchronize_gradients();  // All-reduce gradients
 *     optimizer.step();
 *     model->zero_grad();
 * }
 * @endcode
 *
 * Design Notes:
 * - Parameters are grouped into ~25MB buckets (configurable)
 * - Buckets are ordered in reverse parameter order for overlap with backward
 * - After all-reduce, gradients are divided by world_size to compute average
 * - Initial parameter broadcast ensures all ranks start with same weights
 */
class DistributedDataParallel {
public:
    /** @brief Default bucket size: 25MB */
    static constexpr size_t DEFAULT_BUCKET_SIZE = 25 * 1024 * 1024;

    /**
     * @brief Construct DDP wrapper.
     *
     * Broadcasts parameters from rank 0 to all other ranks to ensure
     * all processes start with identical model weights. If the process
     * group uses a GPU backend (e.g., NCCLProcessGroup), creates a
     * dedicated CUDA communication stream for async all-reduce overlap.
     *
     * @param module The neural network module to wrap
     * @param pg The process group for communication
     * @param bucket_size_bytes Maximum size of each gradient bucket in bytes
     */
    /**
     * @param find_unused_parameters If true, detect parameters that did not
     *        receive gradients and skip their all-reduce. Logs warnings on
     *        first detection. Detection is cached after first iteration;
     *        call invalidate_unused_cache() for dynamic graphs.
     */
    DistributedDataParallel(nn::Module& module, ProcessGroup& pg,
                            size_t bucket_size_bytes = DEFAULT_BUCKET_SIZE,
                            bool find_unused_parameters = false);

    /**
     * @brief Destructor - cleans up CUDA stream and event resources.
     */
    ~DistributedDataParallel();

    // Non-copyable (owns CUDA resources)
    DistributedDataParallel(const DistributedDataParallel&) = delete;
    DistributedDataParallel& operator=(const DistributedDataParallel&) = delete;

    /**
     * @brief Forward pass (delegates to wrapped module).
     *
     * @param input Input variable
     * @return Output variable from the wrapped module
     */
    auto forward(const Variable& input) -> Variable;

    /**
     * @brief Get reference to the wrapped module.
     *
     * @return Reference to the underlying nn::Module
     */
    auto module() -> nn::Module& { return module_; }

    /**
     * @brief Get const reference to the wrapped module.
     *
     * @return Const reference to the underlying nn::Module
     */
    auto module() const -> const nn::Module& { return module_; }

    /**
     * @brief Synchronize all gradients across processes.
     *
     * Performs all-reduce on each gradient bucket and divides by
     * world_size to compute the average gradient across all processes.
     *
     * Should be called after loss.backward() and before optimizer.step().
     */
    auto synchronize_gradients() -> void;

    /**
     * @brief Get the gradient buckets (for inspection/debugging).
     *
     * @return Const reference to the vector of gradient buckets
     */
    auto buckets() const -> const std::vector<GradBucket>& { return buckets_; }

    /**
     * @brief Synchronize all pending asynchronous all-reduce operations.
     *
     * When using GPU backends (NCCL), all-reduce operations are launched
     * asynchronously on a dedicated communication stream. This method
     * ensures all pending all-reduce operations have completed by making
     * the compute stream wait on the communication stream's events.
     *
     * Must be called after loss.backward() completes and before
     * optimizer.step() to ensure gradients are fully synchronized.
     *
     * On CPU backends, this is a no-op since all-reduce is synchronous.
     */
    auto sync_comm() -> void;

    /**
     * @brief Enable or disable automatic gradient synchronization.
     *
     * When enabled, gradient hooks registered on each parameter automatically
     * mark their bucket as gradient-ready after backward. When all parameters
     * in a bucket have their gradients computed, all-reduce is fired
     * immediately, overlapping communication with backward computation.
     *
     * When disabled, the caller must explicitly call synchronize_gradients()
     * after the backward pass completes (useful for gradient accumulation
     * across multiple micro-batches).
     *
     * @param enabled true to enable auto-sync (default), false to disable
     */
    auto auto_sync_gradients(bool enabled) -> void;

    /**
     * @brief Check if auto-sync is currently enabled.
     *
     * @return true if gradient hooks will auto-trigger all-reduce
     */
    auto is_auto_sync_enabled() const -> bool { return auto_sync_enabled_; }

    /**
     * @brief Communication statistics for profiling.
     */
    struct CommStats {
        size_t total_bytes_transferred{0};  ///< Total bytes sent in all-reduce ops
        size_t num_all_reduces{0};          ///< Number of all-reduce operations
        double total_comm_time_ms{0.0};     ///< Total communication wall time in ms
        auto avg_comm_time_ms() const -> double {
            return num_all_reduces > 0 ? total_comm_time_ms / num_all_reduces : 0.0;
        }
    };

    /**
     * @brief RAII guard that disables gradient sync for gradient accumulation.
     *
     * Usage:
     * @code
     * {
     *     auto guard = ddp.no_sync();
     *     for (int i = 0; i < num_micro_batches - 1; ++i) {
     *         loss = model.forward(micro_batch[i]);
     *         loss.backward();
     *     }
     * }
     * // Guard destructor re-enables sync and synchronizes gradients
     * @endcode
     */
    class GradAccumulationGuard {
    public:
        explicit GradAccumulationGuard(DistributedDataParallel& ddp)
            : ddp_(ddp), was_enabled_(ddp.is_auto_sync_enabled()) {
            ddp_.auto_sync_gradients(false);
        }
        ~GradAccumulationGuard() {
            ddp_.auto_sync_gradients(was_enabled_);
            if (was_enabled_) {
                ddp_.synchronize_gradients();
            }
        }
        GradAccumulationGuard(const GradAccumulationGuard&) = delete;
        auto operator=(const GradAccumulationGuard&) -> GradAccumulationGuard& = delete;
    private:
        DistributedDataParallel& ddp_;
        bool was_enabled_;
    };

    /**
     * @brief Create a guard that disables gradient sync for accumulation.
     *
     * @return RAII guard that re-enables sync on destruction
     */
    auto no_sync() -> GradAccumulationGuard { return GradAccumulationGuard(*this); }

    /** @brief Get communication statistics */
    auto comm_stats() const -> const CommStats& { return comm_stats_; }

    /** @brief Reset communication statistics */
    auto reset_comm_stats() -> void { comm_stats_ = CommStats{}; }

    /**
     * @brief Set gradient compressor for bandwidth-efficient all-reduce.
     *
     * When set, gradients are compressed before all-reduce and decompressed
     * afterwards. Pass nullptr to disable compression.
     *
     * @param compressor Compressor instance (DDP takes ownership)
     */
    auto set_gradient_compressor(std::unique_ptr<GradientCompressor> compressor) -> void {
        compressor_ = std::move(compressor);
    }

    /**
     * @brief Reset all bucket ready states for the next iteration.
     *
     * Called automatically at the start of forward() when auto-sync is
     * enabled. Can also be called manually if needed.
     */
    auto reset_buckets() -> void;

    /**
     * @brief Replace the process group for elastic training recovery.
     *
     * Called after a worker failure when the training group is rebuilt
     * with a different world_size. Re-initializes buckets for the new group.
     *
     * @param new_pg New process group
     */
    auto reset_process_group(ProcessGroup& new_pg) -> void {
        pg_ = &new_pg;
        reset_buckets();
    }

    /**
     * @brief Invalidate the cached unused parameter detection.
     *
     * Call this when using dynamic graphs where unused parameters may change
     * between iterations. By default, unused parameter detection is cached
     * after the first iteration for performance.
     */
    auto invalidate_unused_cache() -> void {
        unused_detection_cached_ = false;
        cached_unused_indices_.clear();
    }

    /**
     * @brief Get the current process group.
     */
    auto process_group() const -> ProcessGroup& { return *pg_; }

private:
    nn::Module& module_;
    ProcessGroup* pg_;
    std::vector<GradBucket> buckets_;
    bool auto_sync_enabled_{true};
    bool find_unused_parameters_{false};
    bool logged_unused_warning_{false};
    bool unused_detection_cached_{false};
    std::unordered_set<size_t> cached_unused_indices_;
    CommStats comm_stats_;

    /** @brief Optional gradient compressor for bandwidth reduction */
    std::unique_ptr<GradientCompressor> compressor_;

    /** @brief Maps parameter data_ptr to its bucket index for O(1) hook lookup */
    std::unordered_map<const void*, size_t> param_to_bucket_;

    /** @brief Mutex for thread-safe bucket ready marking from hooks */
    std::mutex bucket_mutex_;

    // ---- Async communication stream/event members ----

    /** @brief Whether the process group uses a GPU backend (NCCL) */
    bool use_gpu_comm_{false};

    /**
     * @brief Dedicated CUDA stream for communication (separate from compute).
     *
     * Stored as void* to avoid including cuda_runtime.h in the header.
     * This is a cudaStream_t when use_gpu_comm_ is true, nullptr otherwise.
     * All NCCL all-reduce operations are launched on this stream so they
     * can overlap with backward computation on the default compute stream.
     */
    void* comm_stream_{nullptr};

    /**
     * @brief CUDA event recorded after each bucket's all-reduce completes.
     *
     * Stored as void* (cudaEvent_t). One event per bucket. Before the
     * optimizer step, the compute stream waits on these events to ensure
     * all gradients are fully reduced.
     */
    std::vector<void*> bucket_events_;

    /**
     * @brief Number of async all-reduce operations currently in flight.
     *
     * Incremented when an async all-reduce is launched, decremented
     * when sync_comm() completes. Used to skip sync when no operations
     * are pending.
     */
    std::atomic<size_t> pending_async_ops_{0};

    /**
     * @brief Initialize the communication CUDA stream and events.
     *
     * Creates a dedicated non-blocking CUDA stream for NCCL all-reduce
     * and one CUDA event per bucket for synchronization. No-op on CPU.
     */
    auto init_comm_resources() -> void;

    /**
     * @brief Destroy the communication stream and events.
     *
     * Called from the destructor. Cleans up CUDA stream/event resources.
     */
    auto destroy_comm_resources() -> void;

    /**
     * @brief Build gradient buckets by grouping parameters.
     *
     * Parameters are added in reverse order (matching backward pass order)
     * into buckets that don't exceed bucket_size_bytes.
     *
     * @param bucket_size_bytes Maximum bucket size in bytes
     */
    auto build_buckets(size_t bucket_size_bytes) -> void;

    /**
     * @brief Register backward hooks on all parameters for auto-sync.
     *
     * Each parameter gets a hook via Variable::register_hook() that marks
     * its bucket as gradient-ready. When all parameters in a bucket have
     * their gradients, all-reduce is fired immediately.
     */
    auto register_grad_hooks() -> void;

    /**
     * @brief Called by gradient hooks when a parameter's gradient is ready.
     *
     * Marks the parameter's bucket and fires all-reduce if the bucket is full.
     *
     * @param param_ptr data_ptr() of the parameter whose gradient arrived
     */
    auto mark_param_ready(const void* param_ptr) -> void;

    /**
     * @brief All-reduce a single gradient bucket (synchronous fallback).
     *
     * Performs in-place SUM all-reduce on each gradient tensor in the bucket,
     * then divides by world_size to compute the average. Used for CPU
     * backends where async overlap is not possible.
     *
     * @param bucket The bucket to all-reduce
     */
    auto all_reduce_bucket(GradBucket& bucket) -> void;

    /**
     * @brief All-reduce a single gradient bucket asynchronously on the comm stream.
     *
     * Launches NCCL all-reduce on the dedicated communication stream,
     * then records a CUDA event so the compute stream can later wait.
     * This allows communication to overlap with backward computation
     * of earlier layers.
     *
     * Falls back to synchronous all_reduce_bucket() on CPU backends.
     *
     * @param bucket The bucket to all-reduce
     * @param bucket_idx Index of the bucket (used to select the correct event)
     */
    auto all_reduce_bucket_async(GradBucket& bucket, size_t bucket_idx) -> void;

    /**
     * @brief Broadcast parameters from rank 0 to all processes.
     *
     * Ensures all processes start with identical model parameters.
     */
    auto broadcast_parameters() -> void;
};

} // namespace tenzor::distributed
