/**
 * @file offload_engine.hpp
 * @brief High-level offload engine for ZeRO optimizer Phase 2
 *
 * Provides synchronous/asynchronous API for GPU<->CPU tensor offloading with:
 * - Automatic memory management using pinned memory pool
 * - Prefetch scheduling to hide latency
 * - Auto-offload registry for memory pressure management
 * - Integration with TransferEngine, PinnedAllocator, and MemoryManager
 *
 * Part of ZeRO Offload Phase 2 implementation.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/transfer_engine.hpp"
#include "tenzor/core/pinned_allocator.hpp"
#include "tenzor/core/memory_manager.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

namespace tenzor {
namespace core {

/**
 * @brief Priority level for auto-offload registry
 */
enum class OffloadPriority {
    CRITICAL,  ///< Never offload unless absolutely necessary
    HIGH,      ///< Offload only under severe memory pressure
    NORMAL,    ///< Default offload priority
    LOW        ///< Offload first when memory pressure increases
};

/**
 * @brief High-level offload engine for ZeRO optimizer
 *
 * Builds on TransferEngine to provide:
 * 1. Simple sync/async API for tensor offloading
 * 2. Automatic pinned memory management
 * 3. Prefetch scheduling to overlap compute and transfers
 * 4. Auto-offload registry integrated with MemoryManager
 *
 * Architecture:
 * - Uses TransferEngine for low-level async transfers
 * - Uses PinnedAllocator for fast pinned memory pool
 * - Uses MemoryManager for memory pressure tracking
 * - Background prefetch thread for scheduled transfers
 *
 * Usage:
 * @code
 * OffloadEngine::Config config;
 * config.pinned_memory_size = 2ULL * 1024 * 1024 * 1024;  // 2 GB
 * config.num_transfer_streams = 4;
 * config.enable_prefetch = true;
 *
 * OffloadEngine engine(config);
 *
 * // Synchronous offload
 * Tensor cpu_tensor = engine.offload_to_cpu(gpu_tensor);
 *
 * // Asynchronous offload
 * auto handle = engine.offload_to_cpu_async(gpu_tensor);
 * // ... do other work ...
 * Tensor cpu_tensor = handle.get_tensor();
 *
 * // Prefetch (start transfer early)
 * std::vector<Tensor*> tensors = {&param1, &param2, &param3};
 * engine.prefetch_to_gpu(tensors);
 *
 * // Auto-offload registration
 * engine.register_auto_offload(&large_tensor, OffloadPriority::LOW);
 * @endcode
 */
class OffloadEngine {
public:
    /**
     * @brief Configuration for offload engine
     */
    struct Config {
        size_t pinned_memory_size{2ULL * 1024 * 1024 * 1024};  ///< Pinned memory pool size (2 GB)
        int num_transfer_streams{4};                            ///< Number of CUDA streams for transfers
        bool enable_prefetch{true};                             ///< Enable prefetch scheduling
        int prefetch_depth{8};                                  ///< Max prefetch queue depth
        float memory_fraction{0.9f};                            ///< GPU memory threshold for auto-offload (0.0-1.0)
        size_t cpu_memory_limit{16ULL * 1024 * 1024 * 1024};   ///< CPU memory limit (16 GB)
        size_t gpu_memory_limit{8ULL * 1024 * 1024 * 1024};    ///< GPU memory limit (8 GB)
        bool enable_auto_monitoring{true};                      ///< Enable automatic memory monitoring
        int monitoring_interval_ms{100};                        ///< Monitoring check interval in milliseconds

        /** Optional pre-built TransferEngine to share across cooperating components.
         *
         *  When set, OffloadEngine adopts this engine instead of constructing its own
         *  (the pinned_memory_size / num_transfer_streams knobs are then ignored). The
         *  intended use case is wiring the same TransferEngine into both an
         *  OffloadEngine (for parameter / optimizer-state offload) and an
         *  OffloadContext (for activation offload) so the host-side pinned buffer pool
         *  is shared rather than duplicated — see review item #17. A typical training
         *  setup with both kinds of offload pinned ~2.5 GB of host RAM in two separate
         *  pools; sharing cuts that to one ~2 GB pool.
         */
        std::shared_ptr<TransferEngine> shared_transfer_engine{nullptr};

        Config() = default;
    };

    /**
     * @brief Construct offload engine with configuration
     *
     * @param config Engine configuration
     * @throws std::runtime_error if initialization fails
     */
    explicit OffloadEngine(const Config& config);

    /**
     * @brief Destructor - waits for pending operations and cleanup
     */
    ~OffloadEngine();

    // Disable copy/move to prevent resource management issues
    OffloadEngine(const OffloadEngine&) = delete;
    OffloadEngine& operator=(const OffloadEngine&) = delete;
    OffloadEngine(OffloadEngine&&) = delete;
    OffloadEngine& operator=(OffloadEngine&&) = delete;

    // ========================================================================
    // Synchronous Offload API
    // ========================================================================

    /**
     * @brief Offload GPU tensor to CPU (synchronous)
     *
     * Transfers tensor from GPU to CPU memory, blocking until complete.
     * Uses pinned memory for fast DMA transfer if available.
     *
     * @param gpu_tensor Source tensor on GPU
     * @return New tensor on CPU with copied data
     * @throws std::runtime_error if gpu_tensor is not on GPU
     */
    auto offload_to_cpu(const Tensor& gpu_tensor) -> Tensor;

    /**
     * @brief Load CPU tensor to GPU (synchronous)
     *
     * Transfers tensor from CPU to GPU memory, blocking until complete.
     * Uses pinned memory for fast DMA transfer if available.
     *
     * @param cpu_tensor Source tensor on CPU
     * @return New tensor on GPU with copied data
     * @throws std::runtime_error if cpu_tensor is not on CPU
     */
    auto load_to_gpu(const Tensor& cpu_tensor) -> Tensor;

    /**
     * @brief Load CPU tensor to specific GPU device (synchronous)
     *
     * @param cpu_tensor Source tensor on CPU
     * @param gpu_device Target GPU device
     * @return New tensor on specified GPU with copied data
     * @throws std::runtime_error if cpu_tensor is not on CPU
     */
    auto load_to_gpu(const Tensor& cpu_tensor, Device gpu_device) -> Tensor;

    // ========================================================================
    // Asynchronous Offload API
    // ========================================================================

    /**
     * @brief Offload GPU tensor to CPU (asynchronous)
     *
     * Issues async transfer and returns immediately.
     * Use returned handle to check completion and get result.
     *
     * @param gpu_tensor Source tensor on GPU
     * @return Handle for tracking transfer progress
     * @throws std::runtime_error if gpu_tensor is not on GPU
     */
    auto offload_to_cpu_async(const Tensor& gpu_tensor) -> TransferHandle;

    /**
     * @brief Load CPU tensor to GPU (asynchronous)
     *
     * Issues async transfer and returns immediately.
     * Use returned handle to check completion and get result.
     *
     * @param cpu_tensor Source tensor on CPU
     * @return Handle for tracking transfer progress
     * @throws std::runtime_error if cpu_tensor is not on CPU
     */
    auto load_to_gpu_async(const Tensor& cpu_tensor) -> TransferHandle;

    /**
     * @brief Load CPU tensor to specific GPU device (asynchronous)
     *
     * @param cpu_tensor Source tensor on CPU
     * @param gpu_device Target GPU device
     * @return Handle for tracking transfer progress
     * @throws std::runtime_error if cpu_tensor is not on CPU
     */
    auto load_to_gpu_async(const Tensor& cpu_tensor, Device gpu_device) -> TransferHandle;

    /**
     * @brief Prefetch tensors to GPU asynchronously
     *
     * Starts async transfers for multiple tensors to hide latency.
     * Returns immediately - transfers happen in background.
     *
     * Useful for preloading parameters before forward pass:
     * @code
     * // Prefetch next batch of parameters
     * std::vector<Tensor*> next_params = {&layer1_weight, &layer1_bias, &layer2_weight};
     * engine.prefetch_to_gpu(next_params);
     * // ... compute with current params ...
     * // By the time we need next_params, they're already on GPU
     * @endcode
     *
     * @param tensors Vector of tensor pointers to prefetch
     * @note Tensors must be on CPU, silently skips GPU tensors
     */
    auto prefetch_to_gpu(const std::vector<Tensor*>& tensors) -> void;

    /**
     * @brief Prefetch single tensor to GPU asynchronously
     *
     * @param tensor Pointer to tensor to prefetch
     * @note Tensor must be on CPU, silently skips if on GPU
     */
    auto prefetch_to_gpu(Tensor* tensor) -> void;

    // ========================================================================
    // Memory Management
    // ========================================================================

    /**
     * @brief Get pinned memory statistics
     *
     * @return Statistics snapshot of pinned memory pool
     */
    auto get_pinned_memory_stats() -> core::PinnedMemoryStats;

    /**
     * @brief Register tensor for automatic offloading
     *
     * When GPU memory pressure exceeds threshold (memory_fraction),
     * registered tensors are automatically offloaded to CPU based on priority.
     * Lower priority tensors are offloaded first.
     *
     * @param tensor Pointer to tensor to register
     * @param priority Offload priority level
     *
     * @code
     * // Gradients can be offloaded aggressively
     * engine.register_auto_offload(&gradient_tensor, OffloadPriority::LOW);
     *
     * // Optimizer states need to stay on GPU longer
     * engine.register_auto_offload(&momentum_tensor, OffloadPriority::HIGH);
     * @endcode
     */
    auto register_auto_offload(Tensor* tensor, OffloadPriority priority) -> void;

    /**
     * @brief Unregister tensor from automatic offloading
     *
     * @param tensor Pointer to tensor to unregister
     */
    auto unregister_auto_offload(Tensor* tensor) -> void;

    /**
     * @brief Mark a registered tensor as recently accessed (LRU bump).
     *
     * Phase C (C2): the auto-offload eviction sort key uses last_access_tick to
     * keep recently-used tensors GPU-resident. Internal sites bump the tick on
     * register and on the offload/load path; callers performing other
     * accesses (e.g. their own custom kernel reads) can call this to declare
     * "this tensor is in use; please don't evict it next." No-op when the
     * tensor isn't in the registry.
     */
    auto mark_accessed(Tensor* tensor) -> void;

    /**
     * @brief Manually trigger auto-offload check
     *
     * Checks GPU memory pressure and offloads registered tensors if needed.
     * Called automatically by offload operations, but can be called manually.
     *
     * @return Number of tensors offloaded
     */
    auto check_and_offload() -> size_t;

    /**
     * @brief Get current GPU memory pressure (0.0-1.0)
     *
     * Uses the default GPU device type for pressure calculation.
     *
     * @return Memory pressure value (0.0 = empty, 1.0 = full)
     */
    auto get_gpu_memory_pressure() const -> float;

    /**
     * @brief Get GPU memory pressure for a specific device type (0.0-1.0)
     *
     * @param device_type The device type to check pressure for
     * @return Memory pressure value (0.0 = empty, 1.0 = full)
     */
    auto get_gpu_memory_pressure(Device::Type device_type) const -> float;

    /**
     * @brief Check if GPU memory pressure is over threshold
     *
     * Checks every GPU backend (CUDA, ROCm, Vulkan, OneAPI, MPS), not just the
     * default-detected one, so auto-offload triggers correctly on a
     * combined-backend build where tensors are registered on a non-default GPU.
     *
     * @return true if auto-offload should trigger
     */
    auto is_over_threshold() const -> bool;

    // ========================================================================
    // Synchronization
    // ========================================================================

    /**
     * @brief Synchronize all pending operations
     *
     * Blocks until all transfers (including prefetches) complete.
     */
    auto synchronize() -> void;

    /**
     * @brief Wait for all prefetch operations to complete
     *
     * Blocks until prefetch queue is empty and all prefetches finish.
     */
    auto wait_for_prefetch() -> void;

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /**
     * @brief Get total number of offload operations
     */
    auto get_offload_count() const -> size_t {
        return offload_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get total number of load operations
     */
    auto get_load_count() const -> size_t {
        return load_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get total number of prefetch operations
     */
    auto get_prefetch_count() const -> size_t {
        return prefetch_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get total number of auto-offload operations
     */
    auto get_auto_offload_count() const -> size_t {
        return auto_offload_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get number of registered tensors
     */
    auto get_registered_tensor_count() const -> size_t;

    /**
     * @brief Reset statistics counters
     */
    auto reset_statistics() -> void;

    /** @brief Borrow this engine's underlying TransferEngine so other subsystems
     *         (e.g. OffloadContext for activation offload) can share the same pinned
     *         host buffer pool. See review item #17 for the rationale.
     */
    auto transfer_engine() const -> std::shared_ptr<TransferEngine> { return transfer_engine_; }

private:
    // Configuration
    Config config_;

    // Core components
    std::shared_ptr<TransferEngine> transfer_engine_;      ///< Low-level async transfer engine (shared so OffloadContext can adopt the same pool)
    // Phase C (C1): pinned_alloc_ was here -- a duplicate 2 GB host-pinned pool that
    // was never used outside the constructor and the stats getter. TransferEngine
    // owns the actual pinned-buffer pool used for DMAs. Field removed; the include
    // of pinned_allocator.hpp is kept because PinnedMemoryStats is still our public
    // return type for get_pinned_memory_stats().
    std::unique_ptr<MemoryManager> memory_manager_;        ///< Memory pressure tracking

    // Default GPU device for transfers
    Device default_gpu_device_;

    // Auto-offload registry
    struct AutoOffloadEntry {
        Tensor* tensor;
        OffloadPriority priority;
        size_t registration_time;  ///< For FIFO within same priority
        // Phase C (C2): added for LRU + size-aware eviction. last_access_tick is
        // bumped on load_to_gpu / offload_to_cpu for the same registered pointer;
        // size_bytes is captured at registration time (assumed roughly stable for
        // the lifetime of the registered tensor). Sort key now uses (priority,
        // -size_bytes, last_access_tick) so high-priority + large + stale evict
        // first; eliminates the "evict 10 small tensors instead of 1 large" and
        // the "evict the tensor used every step" thrash classes.
        size_t last_access_tick{0};
        size_t size_bytes{0};
    };

    std::vector<AutoOffloadEntry> auto_offload_registry_;
    mutable std::mutex registry_mutex_;  // mutable to allow locking in const methods
    std::atomic<size_t> registration_counter_{0};

    // Prefetch queue and thread
    struct PrefetchRequest {
        Tensor* tensor;
        Device target_device;
    };

    std::queue<PrefetchRequest> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    // Phase C (C5): notified by the worker after every drain so wait_for_prefetch can
    // CV-wait instead of polling on a 10 ms sleep loop. Same prefetch_mutex_ so we
    // don't double the lock count.
    std::condition_variable prefetch_drained_cv_;
    std::atomic<bool> stop_prefetch_worker_{false};
    std::thread prefetch_worker_thread_;

    // Track in-flight async prefetches the worker issued so wait_for_prefetch (and the
    // destructor) can commit the results back to the user-supplied target tensors. The
    // legacy code dropped the TransferHandle on the floor — see review item #4 — which
    // meant prefetch_to_gpu kicked off a real transfer but the user's Tensor* never saw
    // the GPU copy, defeating the whole point of the API.
    struct InFlightPrefetch {
        Tensor* target;                          ///< User-owned tensor to update on commit
        tenzor::core::TransferHandle handle;     ///< Async transfer handle; awaited on commit
    };
    std::vector<InFlightPrefetch> in_flight_prefetches_;
    std::mutex in_flight_mutex_;

    // Serializes every reassignment of a user-owned Tensor (`*tensor = ...` in
    // check_and_offload, `*f.target = ...` in wait_for_prefetch). A given
    // Tensor* may be both register_auto_offload'd (written by the monitoring
    // thread) and prefetch_to_gpu'd (written by wait_for_prefetch on another
    // thread); without a shared lock those two assignments race on the
    // intrusive_ptr refcount of the assigned-from/assigned-to storage,
    // risking use-after-free/leak and an indeterminate final device. Held only
    // across the assignment itself, never across a PCIe transfer.
    std::mutex tensor_assign_mutex_;

    // Monitoring thread for automatic offload
    std::thread monitoring_thread_;
    std::atomic<bool> stop_monitoring_{false};

    // Statistics
    std::atomic<size_t> offload_count_{0};
    std::atomic<size_t> load_count_{0};
    std::atomic<size_t> prefetch_count_{0};
    std::atomic<size_t> auto_offload_count_{0};

    // Worker thread for prefetch processing
    auto prefetch_worker() -> void;

    // Worker thread for automatic memory monitoring
    auto monitoring_worker() -> void;

    // Helper to get GPU device from tensor or use default
    auto get_gpu_device(const Tensor& tensor) const -> Device;

    // Helper to check if device is GPU type
    auto is_gpu_device(const Device& device) const -> bool;

    // Sort auto-offload registry by priority
    auto sort_auto_offload_registry() -> void;
};

} // namespace core
} // namespace tenzor
