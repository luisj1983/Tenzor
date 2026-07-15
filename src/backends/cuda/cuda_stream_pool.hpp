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
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include "cuda_error.hpp"

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
 *
 * Locking: each device's pool is guarded by its OWN mutex (DevicePool::mutex),
 * mirroring backend::CachingAllocator's per-device DeviceAllocator::mutex —
 * a thread acquiring/releasing a stream on GPU 0 never blocks a concurrent
 * thread doing the same on GPU 1. `pools_mutex_` guards only the
 * device_pools_ map's STRUCTURE (creating a new device's entry); it is held
 * briefly to fetch-or-create a DevicePool& and then released before that
 * pool's own mutex is locked (device_pools_ is an unordered_map, so
 * references to existing elements stay valid across insertion/rehash).
 * `stream_device_` is a separate cross-device lookup table (which device a
 * given stream belongs to) and is guarded by its own `stream_device_mutex_`
 * so that bookkeeping doesn't serialize unrelated devices either.
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
        DevicePool& pool = get_or_create_pool(device_id);
        std::lock_guard<std::mutex> lock(pool.mutex);
        init_locked(pool, device_id, pool_size);
    }

    /**
     * @brief Acquire a stream from the pool.
     *
     * Returns the first available stream, or creates a new one if all are busy.
     */
    cudaStream_t acquire(int32_t device_id) {
        DevicePool& pool = get_or_create_pool(device_id);
        std::lock_guard<std::mutex> lock(pool.mutex);
        ensure_initialized_locked(pool, device_id);

        // Make the requested device current before returning a stream from it.
        // Op wrappers launch kernels and create library handles immediately
        // after acquire(); without this, work targeting a non-default device
        // would land on whatever device happened to be current on this thread,
        // corrupting multi-GPU execution.
        CUDA_CHECK(cudaSetDevice(device_id));

        for (size_t i = 0; i < pool.streams.size(); ++i) {
            if (pool.available[i]) {
                pool.available[i] = false;
                set_stream_device(pool.streams[i], device_id);
                return pool.streams[i];
            }
        }

        // All streams busy — create a temporary one (rare path)
        cudaStream_t stream;
        CUDA_CHECK(cudaSetDevice(device_id));
        auto err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
        }
        pool.streams.push_back(stream);
        pool.available.push_back(false);
        set_stream_device(stream, device_id);
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
        DevicePool* pool = find_pool(device_id);
        if (!pool) return;

        std::lock_guard<std::mutex> lock(pool->mutex);
        for (size_t i = 0; i < pool->streams.size(); ++i) {
            if (pool->streams[i] == stream) {
                reclaim_or_free(*pool, i, stream);
                return;
            }
        }
    }

    /**
     * @brief Release a stream back to the pool without knowing its device.
     *
     * Resolves the owning device recorded at acquire() time, so the slot is
     * correctly marked available even on multi-GPU where the calling thread's
     * current device differs from the stream's creation device. Falls back to a
     * scan across all device pools if the stream was not recorded.
     */
    void release(cudaStream_t stream) {
        int32_t device_id = 0;
        bool found = false;
        {
            std::lock_guard<std::mutex> sd_lock(stream_device_mutex_);
            auto it = stream_device_.find(stream);
            if (it != stream_device_.end()) {
                device_id = it->second;
                found = true;
            }
        }
        if (found) {
            release(device_id, stream);
            return;
        }
        // Unknown stream — scan every device pool as a last resort. Holds
        // pools_mutex_ for the whole scan (rare, cold path only) so the map
        // structure can't change underneath us; per-pool mutexes are still
        // taken individually so this doesn't contend with normal
        // acquire()/release(device_id, ...) traffic any more than necessary.
        std::lock_guard<std::mutex> map_lock(pools_mutex_);
        for (auto& [dev_id, pool] : device_pools_) {
            std::lock_guard<std::mutex> lock(pool.mutex);
            for (size_t i = 0; i < pool.streams.size(); ++i) {
                if (pool.streams[i] == stream) {
                    reclaim_or_free(pool, i, stream);
                    return;
                }
            }
        }
    }

    /**
     * @brief Destroy all pooled streams. Call during backend shutdown.
     */
    void shutdown() {
        std::lock_guard<std::mutex> map_lock(pools_mutex_);
        for (auto& [dev_id, pool] : device_pools_) {
            std::lock_guard<std::mutex> lock(pool.mutex);
            for (auto& stream : pool.streams) {
                cudaStreamDestroy(stream);
            }
            pool.streams.clear();
            pool.available.clear();
        }
        device_pools_.clear();

        std::lock_guard<std::mutex> sd_lock(stream_device_mutex_);
        stream_device_.clear();
    }

    ~CUDAStreamPool() {
        // Don't destroy streams here — shutdown() should be called explicitly
        // during backend shutdown before CUDA runtime is torn down.
    }

private:
    CUDAStreamPool() = default;

    struct DevicePool {
        std::vector<cudaStream_t> streams;
        std::vector<uint8_t> available;  // NOT vector<bool> — bit-packing is not thread-safe
        size_t base_size = 0;            // count of permanent streams created in init()
        mutable std::mutex mutex;        // guards ONLY this device's pool
    };

    // Fetch (creating if absent) the DevicePool for `device_id`. Only
    // pools_mutex_ is held here, and only briefly: unordered_map guarantees
    // references to existing elements survive insertion/rehash, so it is
    // safe to return the reference after releasing the lock and have the
    // caller lock pool.mutex separately (mirrors
    // backend::CachingAllocator::allocate()'s map_mutex_ / dev_alloc->mutex
    // split).
    DevicePool& get_or_create_pool(int32_t device_id) {
        std::lock_guard<std::mutex> map_lock(pools_mutex_);
        return device_pools_[device_id];
    }

    // Like get_or_create_pool(), but does not create a new entry — used by
    // release(int32_t, cudaStream_t) where a miss just means "nothing to
    // release".
    DevicePool* find_pool(int32_t device_id) {
        std::lock_guard<std::mutex> map_lock(pools_mutex_);
        auto it = device_pools_.find(device_id);
        return it == device_pools_.end() ? nullptr : &it->second;
    }

    // Record which device a handed-out stream belongs to. Takes only
    // stream_device_mutex_ (never nested with pools_mutex_), so this
    // bookkeeping never blocks an unrelated device's acquire()/release().
    void set_stream_device(cudaStream_t stream, int32_t device_id) {
        std::lock_guard<std::mutex> sd_lock(stream_device_mutex_);
        stream_device_[stream] = device_id;
    }

    // Must be called with pool.mutex ALREADY held.
    void init_locked(DevicePool& pool, int32_t device_id, size_t pool_size) {
        if (!pool.streams.empty()) return;  // Already initialized

        // An unchecked cudaSetDevice failure would create every stream below on
        // the wrong device, silently corrupting multi-GPU execution.
        CUDA_CHECK(cudaSetDevice(device_id));
        pool.base_size = pool_size;  // permanent streams [0, base_size) are never reclaimed
        pool.streams.resize(pool_size);
        pool.available.resize(pool_size, true);
        for (size_t i = 0; i < pool_size; ++i) {
            auto err = cudaStreamCreate(&pool.streams[i]);
            if (err != cudaSuccess) {
                // Exception safety: destroy every stream successfully created
                // before this failure and leave pool.streams EMPTY. The only
                // re-init trigger is ensure_initialized_locked()'s
                // `pool.streams.empty()` check — if we left the partially
                // filled vector in place (streams[0..i) real handles,
                // streams[i..pool_size) default-constructed nullptr, all
                // marked available), every later acquire() would skip
                // re-init and could hand out nullptr as if it were a real
                // pooled stream (nullptr == the legacy default stream).
                // Resetting to empty here makes the failure unambiguously
                // detectable so the next ensure_initialized_locked() call
                // retries a full init() instead of silently continuing in a
                // corrupted state.
                for (size_t j = 0; j < i; ++j) {
                    cudaStreamDestroy(pool.streams[j]);
                }
                pool.streams.clear();
                pool.available.clear();
                pool.base_size = 0;
                throw std::runtime_error(
                    "Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
            }
        }
    }

    // Must be called with pool.mutex ALREADY held.
    void ensure_initialized_locked(DevicePool& pool, int32_t device_id) {
        if (pool.streams.empty()) {
            size_t pool_size = DEFAULT_POOL_SIZE;
            const char* env = std::getenv("TENZOR_CUDA_STREAM_POOL_SIZE");
            if (env) {
                int val = std::atoi(env);
                if (val > 0 && val <= 256) {
                    pool_size = static_cast<size_t>(val);
                }
            }
            init_locked(pool, device_id, pool_size);
        }
    }

    // Reclaim an overflow stream (index >= base_size) on release: destroy it,
    // remove its pool slot, and drop its stream_device_ entry so the pool and
    // the map cannot grow without bound under bursts of concurrent acquires.
    // Permanent streams [0, base_size) are only flagged available. `idx` must be
    // a valid index into `pool.streams`. Caller holds pool.mutex.
    void reclaim_or_free(DevicePool& pool, size_t idx, cudaStream_t stream) {
        if (idx >= pool.base_size) {
            cudaStreamDestroy(stream);
            {
                std::lock_guard<std::mutex> sd_lock(stream_device_mutex_);
                stream_device_.erase(stream);
            }
            pool.streams.erase(pool.streams.begin() + idx);
            pool.available.erase(pool.available.begin() + idx);
        } else {
            pool.available[idx] = true;
        }
    }

    // Guards device_pools_' structure only (see class doc comment above).
    std::mutex pools_mutex_;
    std::unordered_map<int32_t, DevicePool> device_pools_;

    // Maps each handed-out stream to the device it was acquired/created on, so
    // release(stream) can find the right pool regardless of the caller's
    // current device. Cross-device, so guarded by its own mutex rather than
    // pools_mutex_ or any single DevicePool::mutex.
    std::mutex stream_device_mutex_;
    std::unordered_map<cudaStream_t, int32_t> stream_device_;
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
