/**
 * @file transfer_engine.hpp
 * @brief Asynchronous CPU<->GPU transfer engine for ZeRO Offload
 *
 * Provides high-performance async tensor transfers between CPU and GPU memory
 * using CUDA streams and events. Part of Phase 1 of ZeRO Offload implementation.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tenzor {
namespace core {

// Forward declarations
class TransferState;
class TransferEngine;

/**
 * @brief Handle for tracking async transfer operations
 *
 * Provides interface to check completion status and wait for async transfers.
 * Internally tracks CUDA events and transfer state.
 *
 * @code
 * TransferHandle handle = engine.cpu_to_gpu_async(cpu_tensor, Device::cuda());
 * // Do other work...
 * if (handle.is_ready()) {
 *     Tensor gpu_tensor = handle.get_tensor();
 * }
 * @endcode
 */
class TransferHandle {
public:
    /**
     * @brief Default constructor creating empty handle
     */
    TransferHandle() = default;

    /**
     * @brief Check if transfer is complete
     * @return true if transfer finished, false if still in progress
     */
    auto is_ready() const -> bool;

    /**
     * @brief Wait for transfer to complete (blocking)
     *
     * Blocks the calling thread until the transfer finishes.
     */
    auto wait() -> void;

    /**
     * @brief Get the resulting tensor after transfer completes
     * @return Transferred tensor on target device
     * @note Implicitly waits if transfer not complete
     */
    auto get_tensor() -> Tensor;

    /**
     * @brief Check if handle is valid (not empty)
     */
    auto is_valid() const -> bool { return state_ != nullptr; }

private:
    friend class TransferEngine;

    /**
     * @brief Construct handle with transfer state
     * @param state Shared state for tracking transfer
     */
    explicit TransferHandle(std::shared_ptr<TransferState> state);

    std::shared_ptr<TransferState> state_;
};

/**
 * @brief High-performance async CPU<->GPU transfer engine
 *
 * Manages asynchronous tensor transfers between CPU and GPU using:
 * - Multiple CUDA streams for parallel transfers
 * - CUDA events for completion tracking
 * - Pinned memory for fast DMA transfers
 * - Worker thread for queued transfers
 * - Transfer statistics and monitoring
 *
 * @code
 * TransferEngine::Config config;
 * config.num_streams = 4;
 * config.use_pinned_memory = true;
 *
 * TransferEngine engine(config);
 *
 * // Async transfer
 * auto handle = engine.cpu_to_gpu_async(cpu_tensor, Device::cuda());
 * // ... do other work ...
 * Tensor gpu_tensor = handle.get_tensor();
 * @endcode
 */
class TransferEngine {
public:
    /**
     * @brief Configuration for transfer engine
     */
    struct Config {
        int num_streams{4};              ///< Number of CUDA streams for parallel transfers
        size_t queue_capacity{64};       ///< Maximum pending transfers in queue
        bool use_pinned_memory{true};    ///< Use pinned memory for faster transfers
        size_t pinned_pool_size{256 * 1024 * 1024};  ///< Pinned memory pool size (256 MB default)

        Config() = default;
    };

    /**
     * @brief Transfer statistics for monitoring
     */
    struct Statistics {
        size_t total_transfers{0};       ///< Total number of transfers completed
        size_t bytes_transferred{0};     ///< Total bytes transferred
        double total_time_ms{0.0};       ///< Total transfer time in milliseconds
        size_t cpu_to_gpu_count{0};      ///< CPU->GPU transfer count
        size_t gpu_to_cpu_count{0};      ///< GPU->CPU transfer count
        double average_bandwidth_gbps{0.0};  ///< Average bandwidth in GB/s
    };

    /**
     * @brief Construct transfer engine with configuration
     * @param config Engine configuration
     * @throws std::runtime_error if CUDA initialization fails
     */
    explicit TransferEngine(const Config& config);

    /**
     * @brief Destructor - waits for pending transfers and cleanup
     */
    ~TransferEngine();

    // Disable copy/move to prevent resource management issues
    TransferEngine(const TransferEngine&) = delete;
    auto operator=(const TransferEngine&) = delete;
    TransferEngine(TransferEngine&&) = delete;
    auto operator=(TransferEngine&&) = delete;

    // ========================================================================
    // Synchronous Transfer API
    // ========================================================================

    /**
     * @brief Transfer tensor from CPU to GPU (synchronous)
     *
     * @param cpu_tensor Source tensor on CPU
     * @param gpu_device Target GPU device
     * @return New tensor on GPU with copied data
     * @throws std::runtime_error if cpu_tensor is not on CPU
     */
    auto cpu_to_gpu(const Tensor& cpu_tensor, Device gpu_device) -> Tensor;

    /**
     * @brief Transfer tensor from GPU to CPU (synchronous)
     *
     * @param gpu_tensor Source tensor on GPU
     * @return New tensor on CPU with copied data
     * @throws std::runtime_error if gpu_tensor is not on GPU
     */
    auto gpu_to_cpu(const Tensor& gpu_tensor) -> Tensor;

    // ========================================================================
    // Asynchronous Transfer API
    // ========================================================================

    /**
     * @brief Transfer tensor from CPU to GPU (asynchronous)
     *
     * Issues async transfer using CUDA streams and returns immediately.
     * Use returned handle to check completion and get result.
     *
     * @param cpu_tensor Source tensor on CPU
     * @param gpu_device Target GPU device
     * @return Handle for tracking transfer progress
     * @throws std::runtime_error if cpu_tensor is not on CPU
     */
    auto cpu_to_gpu_async(const Tensor& cpu_tensor, Device gpu_device) -> TransferHandle;

    /**
     * @brief Transfer tensor from GPU to CPU (asynchronous)
     *
     * Issues async transfer using CUDA streams and returns immediately.
     * Use returned handle to check completion and get result.
     *
     * @param gpu_tensor Source tensor on GPU
     * @return Handle for tracking transfer progress
     * @throws std::runtime_error if gpu_tensor is not on GPU
     */
    auto gpu_to_cpu_async(const Tensor& gpu_tensor) -> TransferHandle;

    // ========================================================================
    // Stream Management
    // ========================================================================

    /**
     * @brief Synchronize all transfer streams
     *
     * Blocks until all pending transfers complete on all streams.
     */
    auto synchronize() -> void;

    /**
     * @brief Synchronize specific transfer stream
     *
     * @param stream_id Stream index (0 to num_streams-1)
     * @throws std::out_of_range if stream_id invalid
     */
    auto synchronize_stream(int stream_id) -> void;

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /**
     * @brief Get total number of completed transfers
     */
    auto get_transfer_count() const -> size_t {
        return stats_.total_transfers.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get total bytes transferred
     */
    auto get_bytes_transferred() const -> size_t {
        return stats_.bytes_transferred.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get average transfer bandwidth in GB/s
     */
    auto get_average_bandwidth_gbps() const -> float;

    /**
     * @brief Get detailed transfer statistics
     */
    auto get_statistics() const -> Statistics;

    /**
     * @brief Reset transfer statistics
     */
    auto reset_statistics() -> void;

private:
    friend class TransferState;

    // Configuration
    Config config_;

    // Statistics (atomic for thread-safety)
    struct {
        std::atomic<size_t> total_transfers{0};
        std::atomic<size_t> bytes_transferred{0};
        std::atomic<size_t> cpu_to_gpu_count{0};
        std::atomic<size_t> gpu_to_cpu_count{0};
        std::atomic<double> total_time_ms{0.0};
    } stats_;

#ifdef TENZOR_USE_CUDA
    // CUDA streams for parallel transfers
    std::vector<cudaStream_t> streams_;

    // CUDA events pool for tracking completion
    std::vector<cudaEvent_t> event_pool_;
    std::mutex event_pool_mutex_;

    // Get event from pool or create new one
    auto get_event() -> cudaEvent_t;

    // Return event to pool
    auto return_event(cudaEvent_t event) -> void;
#endif

    // Pinned memory pool for fast transfers
    struct PinnedBuffer {
        void* ptr{nullptr};
        size_t size{0};
        bool in_use{false};
    };
    std::vector<PinnedBuffer> pinned_buffers_;
    std::mutex pinned_mutex_;

    // Get pinned buffer of at least 'size' bytes
    auto get_pinned_buffer(size_t size) -> void*;

    // Return pinned buffer to pool
    auto return_pinned_buffer(void* ptr) -> void;

    // Transfer queue and worker thread
    struct TransferRequest {
        enum class Type { CPU_TO_GPU, GPU_TO_CPU };

        Type type;
        Tensor source;
        Device target_device;
        std::shared_ptr<TransferState> state;
    };

    std::queue<TransferRequest> transfer_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_worker_{false};
    std::atomic<int> in_flight_transfers_{0};  ///< Transfers dequeued but not yet completed
    std::thread worker_thread_;

    // Worker thread function
    auto transfer_worker() -> void;

    // Process single transfer request
    auto process_transfer(const TransferRequest& request) -> void;

    // Allocate tensor on device
    auto allocate_tensor(const std::vector<int64_t>& shape, DType dtype, Device device) -> Tensor;

    // Record transfer statistics
    auto record_transfer(size_t bytes, double time_ms, bool cpu_to_gpu) -> void;

    // Initialize CUDA resources
    auto initialize_cuda_resources() -> void;

    // Cleanup CUDA resources
    auto cleanup_cuda_resources() -> void;
};

/**
 * @brief Internal state for tracking async transfer
 */
class TransferState {
public:
    TransferState() = default;
    ~TransferState();

    // Result tensor (set when transfer completes)
    Tensor result;

    // Completion flag
    std::atomic<bool> completed{false};

    // Synchronization
    std::mutex mutex;
    std::condition_variable cv;

#ifdef TENZOR_USE_CUDA
    // CUDA event for async completion tracking
    cudaEvent_t event{nullptr};

    // Stream used for transfer
    cudaStream_t stream{nullptr};
#endif

    // Pinned buffer (if used)
    void* pinned_buffer{nullptr};
    TransferEngine* engine{nullptr};  // For returning resources

    // Error state
    bool has_error{false};
    std::string error_message;
};

} // namespace core
} // namespace tenzor
