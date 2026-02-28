/**
 * @file cuda_stream_pool.hpp
 * @brief Thread-safe CUDA stream pool with acquire/release API.
 *
 * Eliminates per-request stream creation overhead (~1-2us) by pre-allocating
 * a configurable number of streams per device and reusing them.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <cstdint>
#include <stdexcept>

namespace tenzor::cuda {

/**
 * @brief RAII guard that returns a stream to the pool on destruction.
 */
class StreamGuard {
public:
    StreamGuard() = default;
    StreamGuard(class CUDAStreamPool* pool, cudaStream_t stream, int32_t device_id)
        : pool_(pool), stream_(stream), device_id_(device_id) {}

    ~StreamGuard();

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

    StreamGuard(StreamGuard&& other) noexcept
        : pool_(other.pool_), stream_(other.stream_), device_id_(other.device_id_) {
        other.pool_ = nullptr;
        other.stream_ = nullptr;
    }

    StreamGuard& operator=(StreamGuard&& other) noexcept {
        if (this != &other) {
            release();
            pool_ = other.pool_;
            stream_ = other.stream_;
            device_id_ = other.device_id_;
            other.pool_ = nullptr;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

private:
    void release();

    class CUDAStreamPool* pool_{nullptr};
    cudaStream_t stream_{nullptr};
    int32_t device_id_{0};
};

/**
 * @brief Thread-safe pool of CUDA streams per device.
 *
 * Pre-allocates a configurable number of streams (default 8) per device.
 * acquire() returns a stream from the pool; release() returns it.
 * StreamGuard provides RAII semantics.
 */
class CUDAStreamPool {
public:
    static constexpr size_t DEFAULT_POOL_SIZE = 8;

    static auto instance() -> CUDAStreamPool& {
        static CUDAStreamPool pool;
        return pool;
    }

    /**
     * @brief Initialize pool for a device with given number of streams.
     *
     * Safe to call multiple times — subsequent calls are no-ops.
     */
    void init(int32_t device_id, size_t pool_size = DEFAULT_POOL_SIZE) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (device_id >= static_cast<int32_t>(device_pools_.size())) {
            device_pools_.resize(device_id + 1);
        }
        auto& pool = device_pools_[device_id];
        if (!pool.streams.empty()) return;  // Already initialized

        cudaSetDevice(device_id);
        pool.streams.resize(pool_size);
        pool.available.resize(pool_size, true);
        for (size_t i = 0; i < pool_size; ++i) {
            auto err = cudaStreamCreate(&pool.streams[i]);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
            }
        }
    }

    /**
     * @brief Acquire a stream from the pool.
     *
     * Returns the first available stream, or creates a new one if all are busy.
     */
    cudaStream_t acquire(int32_t device_id) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        ensure_initialized(device_id);
        auto& pool = device_pools_[device_id];

        for (size_t i = 0; i < pool.streams.size(); ++i) {
            if (pool.available[i]) {
                pool.available[i] = false;
                return pool.streams[i];
            }
        }

        // All streams busy — create a temporary one (rare path)
        cudaStream_t stream;
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream);
        pool.streams.push_back(stream);
        pool.available.push_back(false);
        return stream;
    }

    /**
     * @brief Acquire a stream with RAII guard.
     */
    StreamGuard acquire_guard(int32_t device_id) {
        return StreamGuard(this, acquire(device_id), device_id);
    }

    /**
     * @brief Release a stream back to the pool.
     */
    void release(int32_t device_id, cudaStream_t stream) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (device_id >= static_cast<int32_t>(device_pools_.size())) return;
        auto& pool = device_pools_[device_id];

        for (size_t i = 0; i < pool.streams.size(); ++i) {
            if (pool.streams[i] == stream) {
                pool.available[i] = true;
                return;
            }
        }
    }

    /**
     * @brief Destroy all pooled streams. Call during backend shutdown.
     */
    void shutdown() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        for (auto& pool : device_pools_) {
            for (auto& stream : pool.streams) {
                cudaStreamDestroy(stream);
            }
            pool.streams.clear();
            pool.available.clear();
        }
        device_pools_.clear();
    }

    ~CUDAStreamPool() {
        // Don't destroy streams here — shutdown() should be called explicitly
        // during backend shutdown before CUDA runtime is torn down.
    }

private:
    CUDAStreamPool() = default;

    void ensure_initialized(int32_t device_id) {
        if (device_id >= static_cast<int32_t>(device_pools_.size()) ||
            device_pools_[device_id].streams.empty()) {
            init(device_id);
        }
    }

    struct DevicePool {
        std::vector<cudaStream_t> streams;
        std::vector<uint8_t> available;  // NOT vector<bool> — bit-packing is not thread-safe
    };

    std::vector<DevicePool> device_pools_;
    std::recursive_mutex mutex_;
};

// StreamGuard implementation
inline StreamGuard::~StreamGuard() {
    release();
}

inline void StreamGuard::release() {
    if (pool_ && stream_) {
        pool_->release(device_id_, stream_);
        pool_ = nullptr;
        stream_ = nullptr;
    }
}

} // namespace tenzor::cuda
