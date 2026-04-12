/**
 * @file buffer_pool.hpp
 * @brief Thread-local buffer pool for temporary allocations
 *
 * Eliminates repeated heap allocations in hot paths like:
 * - Conv2d im2col buffers
 * - GEMM packing buffers
 * - Reduction temporary arrays
 *
 * Features:
 * - Thread-local storage for zero-contention
 * - Size-based bucketing for efficient reuse
 * - 64-byte aligned allocations for SIMD
 * - Automatic growth with configurable max size
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <array>
#include <algorithm>
#include <memory>

#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
    inline void* aligned_alloc_wrapper(size_t alignment, size_t size) {
        return _aligned_malloc(size, alignment);
    }
#else
    #include <cstdlib>
    #define ALIGNED_FREE(ptr) std::free(ptr)
    // Use posix_memalign for better compatibility and error handling
    inline void* aligned_alloc_wrapper(size_t alignment, size_t size) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
        return ptr;
    }
#endif
#define ALIGNED_ALLOC(alignment, size) aligned_alloc_wrapper(alignment, size)

namespace tenzor {
namespace cpu {

// ============================================================================
// Buffer Pool Configuration
// ============================================================================

constexpr size_t BUFFER_ALIGNMENT = 64;  // AVX-512 cache line
constexpr size_t NUM_SIZE_CLASSES = 16;  // Log2-based size buckets
constexpr size_t MIN_BUFFER_SIZE = 1024; // 1KB minimum
constexpr size_t MAX_BUFFER_SIZE = 256 * 1024 * 1024; // 256MB maximum
constexpr size_t MAX_CACHED_BUFFERS = 4; // Max buffers per size class
constexpr size_t MAX_CACHED_PER_CLASS = 32; // Max raw pointers cached per bucket

// ============================================================================
// Aligned Buffer Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for aligned memory
 */
class AlignedBuffer {
public:
    AlignedBuffer() : data_(nullptr), size_(0) {}

    explicit AlignedBuffer(size_t size) : size_(size) {
        if (size > 0) {
            // Round up to alignment
            size_t aligned_size = (size + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
            data_ = ALIGNED_ALLOC(BUFFER_ALIGNMENT, aligned_size);
        } else {
            data_ = nullptr;
        }
    }

    ~AlignedBuffer() {
        if (data_) {
            ALIGNED_FREE(data_);
        }
    }

    // Move-only
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                ALIGNED_FREE(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }

    template<typename T>
    T* as() { return static_cast<T*>(data_); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(data_); }

private:
    void* data_;
    size_t size_;
};

// ============================================================================
// Thread-Local Buffer Pool
// ============================================================================

/**
 * @brief Get size class for a given size (log2 based)
 */
inline size_t get_size_class(size_t size) {
    if (size <= MIN_BUFFER_SIZE) return 0;

    // Find the log2 bucket
    size_t normalized = (size - 1) / MIN_BUFFER_SIZE;
    size_t class_idx = 0;
    while (normalized > 0) {
        normalized >>= 1;
        class_idx++;
    }
    return std::min(class_idx, NUM_SIZE_CLASSES - 1);
}

/**
 * @brief Get actual size for a size class
 */
inline size_t get_class_size(size_t class_idx) {
    return MIN_BUFFER_SIZE << class_idx;
}

/**
 * @brief Thread-local buffer pool
 */
class BufferPool {
public:
    BufferPool() = default;
    ~BufferPool() { clear(); }

    // Non-copyable, non-movable (thread-local singleton)
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /**
     * @brief Get a buffer of at least the requested size
     */
    void* acquire(size_t size) {
        if (size == 0) return nullptr;

        // std::aligned_alloc requires size to be a multiple of alignment
        size_t aligned_size = (size + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

        if (aligned_size > MAX_BUFFER_SIZE) {
            // Too large for pool, allocate directly
            return ALIGNED_ALLOC(BUFFER_ALIGNMENT, aligned_size);
        }

        size_t class_idx = get_size_class(size);
        auto& bucket = buckets_[class_idx];

        // Try to reuse a cached buffer
        if (!bucket.empty()) {
            void* ptr = bucket.back();
            bucket.pop_back();
            // buffer_sizes_ already has this pointer's allocated size from
            // the original acquire or from release caching it back
            return ptr;
        }

        // Allocate new buffer - must be at least the requested size
        // get_class_size may return less than requested for sizes between
        // class boundaries, so we take the max to ensure sufficient allocation
        size_t alloc_size = std::max(size, get_class_size(class_idx));
        void* ptr = ALIGNED_ALLOC(BUFFER_ALIGNMENT, alloc_size);
        if (ptr) {
            buffer_sizes_[ptr] = alloc_size;
        }
        return ptr;
    }

    /**
     * @brief Return a buffer to the pool for reuse
     */
    void release(void* ptr) {
        if (!ptr) return;

        auto it = buffer_sizes_.find(ptr);
        if (it == buffer_sizes_.end()) {
            // Not from our pool (was too large), free directly
            ALIGNED_FREE(ptr);
            return;
        }

        size_t alloc_size = it->second;
        size_t class_idx = get_size_class(alloc_size);
        auto& bucket = buckets_[class_idx];

        if (bucket.size() < MAX_CACHED_PER_CLASS) {
            // Cache the buffer for reuse — keep the buffer_sizes_ entry
            bucket.push_back(ptr);
        } else {
            // Bucket full, actually free
            buffer_sizes_.erase(it);
            ALIGNED_FREE(ptr);
        }
    }

    /**
     * @brief Clear all cached buffers
     */
    void clear() {
        for (auto& bucket : buckets_) {
            for (void* ptr : bucket) {
                ALIGNED_FREE(ptr);
                buffer_sizes_.erase(ptr);
            }
            bucket.clear();
        }
        // Note: Outstanding buffers (not yet released) will leak if not returned
        buffer_sizes_.clear();
    }

    /**
     * @brief Get total cached memory
     */
    size_t cached_bytes() const {
        size_t total = 0;
        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            total += buckets_[i].size() * get_class_size(i);
        }
        return total;
    }

private:
    std::array<std::vector<void*>, NUM_SIZE_CLASSES> buckets_;
    std::unordered_map<void*, size_t> buffer_sizes_;
};

/**
 * @brief Get the thread-local buffer pool
 */
inline BufferPool& get_buffer_pool() {
    thread_local BufferPool pool;
    return pool;
}

// ============================================================================
// RAII Buffer Handle
// ============================================================================

/**
 * @brief RAII handle for pooled buffer
 */
template<typename T>
class PooledBuffer {
public:
    PooledBuffer() : data_(nullptr), size_(0) {}

    explicit PooledBuffer(size_t count)
        : size_(count * sizeof(T)) {
        data_ = static_cast<T*>(get_buffer_pool().acquire(size_));
    }

    ~PooledBuffer() {
        if (data_) {
            get_buffer_pool().release(data_);
        }
    }

    // Move-only
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;

    PooledBuffer(PooledBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    PooledBuffer& operator=(PooledBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                get_buffer_pool().release(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_ / sizeof(T); }
    size_t size_bytes() const { return size_; }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T* begin() { return data_; }
    T* end() { return data_ + size(); }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size(); }

private:
    T* data_;
    size_t size_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Acquire a typed buffer from the pool
 */
template<typename T>
inline PooledBuffer<T> acquire_buffer(size_t count) {
    return PooledBuffer<T>(count);
}

/**
 * @brief Acquire workspace for im2col
 *
 * @param batch Batch size
 * @param out_h Output height
 * @param out_w Output width
 * @param in_channels Input channels
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 */
template<typename T>
inline PooledBuffer<T> acquire_im2col_buffer(
    int64_t batch, int64_t out_h, int64_t out_w,
    int64_t in_channels, int64_t kernel_h, int64_t kernel_w
) {
    size_t col_rows = batch * out_h * out_w;
    size_t col_cols = in_channels * kernel_h * kernel_w;
    return PooledBuffer<T>(col_rows * col_cols);
}

/**
 * @brief Acquire workspace for GEMM packing
 */
template<typename T>
inline PooledBuffer<T> acquire_pack_buffer(int64_t rows, int64_t cols) {
    return PooledBuffer<T>(rows * cols);
}

} // namespace cpu
} // namespace tenzor

// Cleanup macros
#undef ALIGNED_ALLOC
#undef ALIGNED_FREE
