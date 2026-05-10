/**
 * @file offload.hpp
 * @brief High-level Parameter Offloading API for ZeRO Phase 2
 *
 * Provides automatic parameter and gradient offloading for neural network models,
 * enabling training of large models that don't fit in GPU memory.
 *
 * Key Features:
 * - Automatic parameter/gradient offloading via hooks
 * - Layer-wise prefetching for overlapping transfers with compute
 * - RAII-based compute contexts for manual control
 * - Memory pressure monitoring and statistics
 * - Integration with Module system
 */

#pragma once

#include "tenzor/nn/module.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/transfer_engine.hpp"
#include "tenzor/core/memory_manager.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>

namespace tenzor {
namespace nn {

/**
 * @brief Priority levels for offload operations
 */
enum class OffloadPriority {
    LOW,      ///< Low priority - offload last
    NORMAL,   ///< Normal priority
    HIGH      ///< High priority - offload first
};

/**
 * @brief Statistics for offload operations
 */
struct OffloadStats {
    double peak_gpu_memory_mb{0.0};       ///< Peak GPU memory usage in MB
    double current_cpu_memory_mb{0.0};    ///< Current CPU memory usage in MB
    double avg_transfer_time_ms{0.0};     ///< Average transfer time in milliseconds
    size_t num_parameters_offloaded{0};   ///< Number of parameters currently offloaded
    size_t num_gradients_offloaded{0};    ///< Number of gradients currently offloaded
    size_t total_prefetch_count{0};       ///< Total number of prefetch operations
    size_t total_offload_count{0};        ///< Total number of offload operations
    double total_time_saved_ms{0.0};      ///< Estimated time saved via prefetching
};

/**
 * @brief Automatic parameter offloading context for entire model
 *
 * OffloadContext provides automatic management of parameter and gradient offloading
 * for an entire neural network model. It registers hooks on the model to automatically
 * transfer parameters to GPU before forward pass and offload them to CPU after.
 *
 * Features:
 * - Layer-wise parameter offloading/prefetching
 * - Gradient offloading after backward pass
 * - Configurable prefetch depth for overlapping transfers
 * - Optional pinning of first/last layers for performance
 * - Automatic memory pressure management
 *
 * @code
 * // Setup offload context
 * OffloadContext::Config config;
 * config.offload_parameters = true;
 * config.offload_gradients = true;
 * config.prefetch_depth = 2;  // Prefetch 2 layers ahead
 *
 * OffloadContext ctx(model, config);
 * ctx.enable();
 *
 * // Training loop - parameters automatically managed
 * for (int epoch = 0; epoch < 10; ++epoch) {
 *     auto output = model.forward(input);
 *     auto loss = criterion(output, target);
 *     loss.backward();
 *     optimizer.step();
 * }
 *
 * auto stats = ctx.get_stats();
 * std::cout << "Offloaded " << stats.num_parameters_offloaded << " parameters\n";
 * @endcode
 */
class OffloadContext {
public:
    /**
     * @brief Configuration for offload context
     */
    struct Config {
        bool offload_parameters{true};         ///< Offload model parameters to CPU
        bool offload_gradients{true};          ///< Offload gradients to CPU
        bool offload_optimizer_states{false};  ///< Offload optimizer states (future)
        size_t offload_threshold{1024 * 1024}; ///< Min tensor size to offload (bytes)
        int prefetch_depth{2};                 ///< Number of layers to prefetch ahead
        bool pin_first_layer{true};            ///< Keep first layer on GPU
        bool pin_last_layer{true};             ///< Keep last layer on GPU
        bool enable_statistics{true};          ///< Track offload statistics
        size_t cpu_memory_limit{16ULL * 1024 * 1024 * 1024};  ///< CPU memory limit (16 GB)
        size_t gpu_memory_limit{8ULL * 1024 * 1024 * 1024};   ///< GPU memory limit (8 GB)
        Device target_device{Device::cuda(0)}; ///< Target device for computation (for CPU-start models)

        /** Optional pre-built TransferEngine to share with cooperating subsystems
         *  (typically a `core::OffloadEngine` doing parameter / optimizer-state offload).
         *  When set, OffloadContext adopts this engine instead of constructing its own,
         *  so the host-side pinned buffer pool is shared rather than duplicated.
         *
         *  See review item #17: a training run with both parameter offload and
         *  activation offload pinned ~2.5 GB of host RAM in two separate pools by
         *  default; sharing collapses that to one pool.
         */
        std::shared_ptr<core::TransferEngine> shared_transfer_engine{nullptr};

        Config() = default;
    };

    /**
     * @brief Construct offload context for model
     *
     * @param model Neural network model to manage
     * @param config Configuration settings
     * @throws std::runtime_error if model has no parameters
     *
     * @code
     * OffloadContext::Config config;
     * config.prefetch_depth = 3;
     * OffloadContext ctx(my_model, config);
     * @endcode
     */
    OffloadContext(Module& model, const Config& config);

    /**
     * @brief Destructor - cleanup and synchronization
     *
     * Waits for pending transfers and removes all hooks.
     */
    ~OffloadContext();

    // Disable copy/move to prevent hook management issues
    OffloadContext(const OffloadContext&) = delete;
    auto operator=(const OffloadContext&) = delete;
    OffloadContext(OffloadContext&&) = delete;
    auto operator=(OffloadContext&&) = delete;

    // ========================================================================
    // Control Methods
    // ========================================================================

    /**
     * @brief Enable automatic offloading
     *
     * Activates all registered hooks for automatic parameter/gradient management.
     */
    auto enable() -> void;

    /**
     * @brief Disable automatic offloading
     *
     * Deactivates hooks but keeps all structures intact for re-enabling.
     */
    auto disable() -> void;

    /**
     * @brief Check if offloading is enabled
     *
     * @return true if hooks are active
     */
    auto is_enabled() const -> bool;

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /**
     * @brief Get current offload statistics
     *
     * @return Statistics structure with current metrics
     */
    auto get_stats() -> OffloadStats;

    /**
     * @brief Reset statistics counters
     */
    auto reset_stats() -> void;

    /**
     * @brief Get current GPU memory usage
     *
     * @return GPU memory used in bytes
     */
    auto get_gpu_memory_usage() const -> size_t;

    /**
     * @brief Get current CPU memory usage for offloaded tensors
     *
     * @return CPU memory used in bytes
     */
    auto get_cpu_memory_usage() const -> size_t;

private:
    // Reference to managed model
    Module& model_;

    // Configuration
    Config config_;

    // Enabled state
    std::atomic<bool> enabled_{false};

    // Transfer and memory management
    std::shared_ptr<core::TransferEngine> transfer_engine_;
    std::shared_ptr<core::MemoryManager> memory_manager_;

    /**
     * @brief Information about tracked tensor
     */
    struct TensorInfo {
        Tensor* tensor;                  ///< Pointer to original tensor
        Tensor cpu_copy;                 ///< Copy on CPU (if offloaded)
        Device original_device;          ///< Original GPU device before offloading
        bool is_offloaded{false};        ///< Currently offloaded to CPU
        bool is_pinned{false};           ///< Pinned to GPU (don't offload)
        bool is_gradient{false};         ///< True if this is a gradient tensor
        int use_count{0};                ///< Number of times accessed
        OffloadPriority priority{OffloadPriority::NORMAL};  ///< Offload priority
        size_t size_bytes{0};            ///< Size in bytes
        Module* owning_layer{nullptr};   ///< Layer that owns this parameter

        // --- Async transfer plumbing ---
        // When non-empty, an async PCIe transfer is in flight for this tensor. The next
        // access (offload_tensor / prefetch_tensor / forward_pre_hook) waits on it before
        // reading or issuing a new transfer. This is the mechanism that lets a layer's
        // offload-to-CPU overlap with the *next* layer's compute on the GPU stream.
        tenzor::core::TransferHandle pending_handle;
        bool pending_is_offload{false};  ///< Direction of pending_handle (only valid when handle is_valid)
    };

    // Map of tracked tensors (parameter/gradient pointers -> info)
    std::unordered_map<Tensor*, TensorInfo> tensor_map_;
    mutable std::mutex tensor_map_mutex_;

    // Layer ordering for sequential offload/prefetch
    std::vector<Module*> layer_order_;
    std::unordered_map<Module*, int> layer_indices_;

    // Statistics
    struct {
        std::atomic<size_t> total_prefetches{0};
        std::atomic<size_t> total_offloads{0};
        std::atomic<size_t> peak_gpu_memory{0};
        std::atomic<size_t> current_cpu_memory{0};
        std::atomic<double> total_transfer_time_ms{0.0};
        std::atomic<size_t> transfer_count{0};
    } stats_;

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Register hooks on all modules
     */
    auto register_hooks() -> void;

    /**
     * @brief Recursively register hooks on a module and its submodules
     */
    auto register_hooks_recursive(Module* module) -> void;

    /**
     * @brief Build layer ordering for sequential processing
     */
    auto build_layer_order() -> void;

    /**
     * @brief Collect all parameters and gradients from model
     */
    auto collect_tensors() -> void;

    /**
     * @brief Initialize tensor info for a parameter
     */
    auto initialize_tensor_info(Tensor* tensor, Module* layer) -> void;

    // ========================================================================
    // Offload/Prefetch Operations
    // ========================================================================

    /**
     * @brief Offload layer parameters to CPU
     *
     * @param layer Module to offload
     */
    auto offload_layer(Module* layer) -> void;

    /**
     * @brief Prefetch layer parameters to GPU
     *
     * @param layer Module to prefetch
     */
    auto prefetch_layer(Module* layer) -> void;

    /**
     * @brief Offload single tensor to CPU
     *
     * @param tensor_ptr Pointer to tensor to offload
     * @return true if offload succeeded
     */
    auto offload_tensor(Tensor* tensor_ptr) -> bool;

    /**
     * @brief Prefetch single tensor to GPU
     *
     * @param tensor_ptr Pointer to tensor to prefetch
     * @return true if prefetch succeeded
     */
    auto prefetch_tensor(Tensor* tensor_ptr) -> bool;

    /**
     * @brief Check if tensor should be offloaded
     *
     * @param info Tensor info structure
     * @return true if tensor meets offload criteria
     */
    auto should_offload(const TensorInfo& info) const -> bool;

    /**
     * @brief Drain a pending async transfer for `info` and finalize the tensor swap.
     *
     * If `info.pending_handle` is valid, blocks until the transfer completes, then commits
     * the result: for an offload, replaces *tensor_ptr with the resulting CPU tensor and
     * marks is_offloaded=true; for a prefetch, replaces *tensor_ptr with the GPU tensor and
     * marks is_offloaded=false. Idempotent — no-op when no transfer is pending.
     *
     * Caller must hold tensor_map_mutex_.
     */
    auto finalize_pending(TensorInfo& info, Tensor* tensor_ptr) -> void;

    /**
     * @brief Walk tensor_map_ and finalize every pending async transfer.
     *
     * Used by enable() (so the bulk-offload path remains synchronous from the outside —
     * the per-tensor async benefit is only useful during the per-layer hook path inside a
     * training loop). Caller must hold tensor_map_mutex_.
     */
    auto drain_all_pending() -> void;

    /**
     * @brief Walk tensor_map_ and finalize every pending offload whose handle is
     * already complete (TransferHandle::is_ready() == true). Never blocks.
     *
     * This is the non-blocking sibling of drain_all_pending(). Call from the top
     * of forward_pre_hook / backward_pre_hook so the GPU memory of any layer
     * whose forward_post_hook-issued offload has finished in the background
     * gets actually released, instead of sitting until that same tensor is
     * re-touched next training step. Without this, peak GPU residency stays
     * close to "all params resident" — defeating the point of layer offload.
     *
     * Takes tensor_map_mutex_ itself; caller must NOT hold it. Returns the
     * number of transfers committed.
     */
    auto finalize_completed_offloads() -> size_t;

    // ========================================================================
    // Hook Callbacks
    // ========================================================================

    /**
     * @brief Forward pre-hook: prefetch parameters before forward pass
     */
    auto forward_pre_hook(Module* layer) -> void;

    /**
     * @brief Forward post-hook: offload parameters after forward pass
     */
    auto forward_post_hook(Module* layer) -> void;

    /**
     * @brief Backward pre-hook: prefetch parameters before backward pass
     */
    auto backward_pre_hook(Module* layer) -> void;

    /**
     * @brief Backward post-hook: offload gradients after backward pass
     */
    auto backward_post_hook(Module* layer) -> void;

    // ========================================================================
    // Memory Management
    // ========================================================================

    /**
     * @brief Update statistics
     */
    auto update_stats() -> void;

    /**
     * @brief Check memory pressure and evict if needed
     */
    auto check_memory_pressure() -> void;
};

/**
 * @brief RAII context manager for compute region
 *
 * ComputeContext ensures that specified tensors are on GPU during the scope
 * and automatically offloaded when the scope exits. Useful for manual control
 * over specific tensor lifetimes.
 *
 * @code
 * Tensor param1 = get_offloaded_param();
 * Tensor param2 = get_offloaded_param();
 *
 * {
 *     ComputeContext ctx({&param1, &param2});
 *     // Parameters are on GPU here
 *     auto result = compute(param1, param2);
 * }  // Parameters automatically offloaded to CPU
 * @endcode
 */
class ComputeContext {
public:
    /**
     * @brief Construct compute context and load tensors to GPU
     *
     * @param tensors Vector of tensor pointers to manage
     *
     * @code
     * ComputeContext ctx({&weight, &bias});
     * auto output = forward(weight, bias);
     * @endcode
     */
    explicit ComputeContext(const std::vector<Tensor*>& tensors);

    /**
     * @brief Destructor - offload tensors back to CPU
     *
     * Automatically transfers tensors back to CPU when scope exits.
     */
    ~ComputeContext();

    // Disable copy/move to prevent double-free
    ComputeContext(const ComputeContext&) = delete;
    auto operator=(const ComputeContext&) = delete;
    ComputeContext(ComputeContext&&) = delete;
    auto operator=(ComputeContext&&) = delete;

    /**
     * @brief Synchronize all transfers
     *
     * Blocks until all GPU operations complete.
     */
    auto synchronize() -> void;

private:
    // Tensors being managed
    std::vector<Tensor*> tensors_;

    // CPU copies for restoration
    std::vector<Tensor> cpu_copies_;

    // Original devices for each tensor
    std::vector<Device> original_devices_;

    // Transfer engine for async operations
    std::shared_ptr<core::TransferEngine> transfer_engine_;
};

/**
 * @brief Mark parameter for offloading with specified priority
 *
 * Manually marks a tensor for offloading. The tensor will be offloaded to CPU
 * when memory pressure is high, with priority determining eviction order.
 *
 * @param param Tensor to mark for offloading
 * @param priority Offload priority (higher priority = offload sooner)
 *
 * @code
 * Tensor large_weight({1000, 1000}, DType::Float32, Device::cuda());
 * offload_param(large_weight, OffloadPriority::HIGH);
 * @endcode
 */
auto offload_param(
    Tensor& param,
    OffloadPriority priority = OffloadPriority::NORMAL
) -> void;

/**
 * @brief Get global offload context (if exists)
 *
 * Returns the currently active global offload context.
 *
 * @return Pointer to global context, or nullptr if none active
 */
auto get_global_offload_context() -> OffloadContext*;

/**
 * @brief Set global offload context
 *
 * Sets the global offload context for use by offload_param().
 *
 * @param ctx Pointer to offload context
 */
auto set_global_offload_context(OffloadContext* ctx) -> void;

} // namespace nn
} // namespace tenzor
