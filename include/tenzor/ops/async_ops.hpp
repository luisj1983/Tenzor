/**
 * @file async_ops.hpp
 * @brief Asynchronous tensor operations using Future pattern
 *
 * Provides non-blocking versions of tensor operations for improved
 * throughput via computation/communication overlap.
 */

#pragma once

#include "../core/tensor.hpp"
#include "../utils/threading/future.hpp"
#include "../utils/threading/threadpool.hpp"
#include "../backend/backend.hpp"
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace tenzor {

/**
 * @brief CUDA stream manager for async GPU operations
 *
 * Manages a pool of CUDA streams to enable concurrent kernel execution.
 * Uses round-robin scheduling to distribute work across streams.
 *
 * @note Only used when CUDA backend is available
 */
class StreamManager {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to global StreamManager
     */
    static auto instance() -> StreamManager&;

    /**
     * @brief Get or create stream for device
     *
     * Returns an available stream from the pool, creating new ones as needed.
     * Uses round-robin scheduling to balance load.
     *
     * @param device Device to get stream for
     * @return Stream handle for async operations
     */
    auto get_stream(const Device& device) -> StreamHandle;

    /**
     * @brief Synchronize specific stream
     * @param stream Stream to synchronize
     */
    auto synchronize_stream(StreamHandle stream) -> void;

    /**
     * @brief Synchronize all streams for device
     * @param device Device to synchronize
     */
    auto synchronize_device(const Device& device) -> void;

    /**
     * @brief Destroy all pooled streams and clear state.
     *
     * Must be called while the backend registry is still alive (e.g. from
     * finalize()/shutdown(), before backends are destroyed) so the cached
     * Backend* used to destroy each stream is valid and a later re-init
     * rebuilds the pool against live backends.
     */
    auto reset() -> void;

    /**
     * @brief Destructor - cleanup all streams
     */
    ~StreamManager();

private:
    StreamManager() = default;

    struct DeviceStreams {
        std::vector<StreamHandle> streams;
        size_t next_index{0};
        std::mutex mutex;
    };

    std::unordered_map<Device, DeviceStreams> device_streams_;
    // Maps each live stream back to the backend that created it, so
    // synchronize_stream()/destroy_stream() can route to the right backend
    // (streams are opaque void* handles with no device tag of their own).
    std::unordered_map<StreamHandle, Backend*> stream_owner_;
    std::mutex global_mutex_;

    // Number of streams to create per device
    static constexpr size_t STREAMS_PER_DEVICE = 4;
};

/**
 * @defgroup async_ops Asynchronous Operations
 * @brief Non-blocking tensor operations for improved throughput
 * @{
 */

/**
 * @brief Asynchronous matrix multiplication
 *
 * Performs matrix multiplication without blocking the calling thread.
 * For CPU, uses thread pool. For GPU, uses CUDA streams.
 *
 * @param a Left matrix (..., M, K)
 * @param b Right matrix (..., K, N)
 * @return Future<Tensor> for result matrix (..., M, N)
 *
 * @code
 * // Non-blocking matmul
 * auto future = async_matmul(a, b);
 *
 * // Do other work
 * process_data();
 *
 * // Wait for result
 * Tensor result = future.wait();
 * @endcode
 *
 * @see matmul for synchronous version
 */
auto async_matmul(const Tensor& a, const Tensor& b) -> Future<Tensor>;

/**
 * @brief Asynchronous element-wise addition
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Future<Tensor> for a + b
 *
 * @code
 * auto future = async_add(a, b);
 * Tensor result = future.wait();
 * @endcode
 */
auto async_add(const Tensor& a, const Tensor& b) -> Future<Tensor>;

/**
 * @brief Asynchronous element-wise multiplication
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Future<Tensor> for a * b
 */
auto async_mul(const Tensor& a, const Tensor& b) -> Future<Tensor>;

/**
 * @brief Asynchronous element-wise subtraction
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Future<Tensor> for a - b
 */
auto async_sub(const Tensor& a, const Tensor& b) -> Future<Tensor>;

/**
 * @brief Asynchronous element-wise division
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Future<Tensor> for a / b
 */
auto async_div(const Tensor& a, const Tensor& b) -> Future<Tensor>;


/**
 * @brief Asynchronous ReLU activation
 *
 * @param input Input tensor
 * @return Future<Tensor> for ReLU(input)
 */
auto async_relu(const Tensor& input) -> Future<Tensor>;

/**
 * @brief Asynchronous sigmoid activation
 *
 * @param input Input tensor
 * @return Future<Tensor> for sigmoid(input)
 */
auto async_sigmoid(const Tensor& input) -> Future<Tensor>;

/**
 * @brief Asynchronous tanh activation
 *
 * @param input Input tensor
 * @return Future<Tensor> for tanh(input)
 */
auto async_tanh(const Tensor& input) -> Future<Tensor>;

/**
 * @brief Asynchronous softmax
 *
 * @param input Input tensor
 * @param dim Dimension to compute softmax over
 * @return Future<Tensor> for softmax(input)
 */
auto async_softmax(const Tensor& input, int64_t dim = -1) -> Future<Tensor>;

/**
 * @brief Wait for multiple futures
 *
 * Convenience function to wait for multiple operations to complete.
 *
 * @param futures Vector of futures to wait for
 * @return Vector of results in same order
 *
 * @code
 * auto f1 = async_matmul(a1, b1);
 * auto f2 = async_matmul(a2, b2);
 * auto f3 = async_matmul(a3, b3);
 *
 * auto results = wait_all({f1, f2, f3});
 * @endcode
 */
auto wait_all(std::vector<Future<Tensor>>& futures) -> std::vector<Tensor>;

/**
 * @brief Check if any future is ready
 *
 * @param futures Vector of futures to check
 * @return Index of first ready future, or -1 if none ready
 */
auto wait_any(const std::vector<Future<Tensor>>& futures) -> int64_t;

/// @}

namespace detail {

/**
 * @brief Internal helper for CPU async operations
 */
template<typename Op>
auto async_cpu_op(Op&& op) -> Future<Tensor> {
    auto promise = std::make_shared<Promise<Tensor>>();
    auto future = Future<Tensor>(promise->get_state());

    thread_pool().submit([promise, op = std::forward<Op>(op)]() mutable {
        try {
            Tensor result = op();
            promise->set_value(std::move(result));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

/**
 * @brief Internal helper for GPU async operations using per-device streams.
 *
 * Backend-agnostic: StreamManager hands out a real device stream/queue for the
 * device (CUDA streams, ROCm hipStreams, OneAPI in-order queues, Vulkan
 * timeline contexts) via the backend's stream interface; for backends/devices
 * that have no stream the handle is null and the op simply runs on a worker
 * thread. The op receives the StreamHandle and the stream is synchronized
 * before the future resolves, so the result is always complete on return.
 */
template<typename Op>
auto async_gpu_op(const Device& device, Op&& op) -> Future<Tensor> {
    auto promise = std::make_shared<Promise<Tensor>>();
    auto future = Future<Tensor>(promise->get_state());

    // Get stream for this device (null if the backend has no stream support).
    StreamHandle stream = StreamManager::instance().get_stream(device);

    thread_pool().submit([promise, stream, device, op = std::forward<Op>(op)]() mutable {
        try {
            Tensor result = op(stream);
            StreamManager::instance().synchronize_stream(stream);
            promise->set_value(std::move(result));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

} // namespace detail

} // namespace tenzor
