#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>
#include <functional>

// Note: We use void* for SYCL queue to avoid including sycl.hpp in headers.
// The actual sycl::queue* is cast in the implementation file.

namespace tenzor {
namespace backend {

/**
 * @brief Memory block representation for the OneAPI caching allocator
 */
struct OneAPIBlock {
    void* ptr;                  // USM pointer
    size_t size;                // Block size in bytes
    bool allocated;             // Whether block is currently allocated
    int device;                 // Device index
    bool is_shared;             // Whether this is shared memory (vs device-only)

    OneAPIBlock(void* p, size_t s, int dev, bool shared = true)
        : ptr(p), size(s), allocated(false), device(dev), is_shared(shared) {}

    // Comparison for ordered containers (by size, then by pointer)
    bool operator<(const OneAPIBlock& other) const {
        if (size != other.size) return size < other.size;
        return ptr < other.ptr;
    }
};

/**
 * @brief Comparator for ordering blocks by size (for best-fit allocation)
 */
struct OneAPIBlockComparator {
    bool operator()(const OneAPIBlock* a, const OneAPIBlock* b) const {
        if (a->size != b->size) return a->size < b->size;
        return a->ptr < b->ptr;
    }
};

/**
 * @brief Statistics for OneAPI/SYCL memory usage tracking
 */
struct OneAPIMemoryStats {
    size_t allocated_bytes;     // Currently allocated to user
    size_t reserved_bytes;      // Total reserved (allocated + cached)
    size_t cached_bytes;        // Cached but not allocated
    size_t num_allocations;     // Total allocation calls
    size_t num_frees;           // Total free calls
    size_t num_cache_hits;      // Allocations satisfied from cache
    size_t num_splits;          // Number of block splits
    size_t num_merges;          // Number of block merges
    size_t peak_allocated;      // Peak allocated bytes
    size_t shared_memory_bytes; // Bytes in shared (host+device) allocations
    size_t device_memory_bytes; // Bytes in device-only allocations

    OneAPIMemoryStats()
        : allocated_bytes(0), reserved_bytes(0), cached_bytes(0),
          num_allocations(0), num_frees(0), num_cache_hits(0),
          num_splits(0), num_merges(0), peak_allocated(0),
          shared_memory_bytes(0), device_memory_bytes(0) {}
};

/**
 * @brief OneAPICachingAllocator for efficient Intel GPU/CPU memory management
 *
 * Implements a memory pooling system that reuses freed USM (Unified Shared Memory)
 * allocations to reduce sycl::malloc_shared/sycl::free overhead. Uses best-fit
 * allocation strategy with block splitting to reduce fragmentation.
 *
 * Supports both shared memory (accessible from host and device) and device-only
 * memory for optimal performance.
 *
 * Thread-safe and supports multiple SYCL devices.
 */
class OneAPICachingAllocator {
public:
    /**
     * @brief Get the singleton instance
     */
    static OneAPICachingAllocator& get();

    /**
     * @brief Initialize the allocator with SYCL queue
     *
     * @param queue SYCL queue for the device (pass as void* to avoid header dependency)
     * @param device_index Device index for multi-device support
     */
    void initialize(void* queue, int device_index);

    /**
     * @brief Check if allocator is initialized for a device
     */
    bool is_initialized(int device_index) const;

    /**
     * @brief Allocate shared memory from the cache or device
     *
     * Shared memory is accessible from both host and device, suitable for
     * most tensor operations.
     *
     * @param size Size in bytes to allocate
     * @param device Device index
     * @return Pointer to allocated memory
     * @throws std::runtime_error if allocation fails
     */
    void* allocate_shared(size_t size, int device = 0);

    /**
     * @brief Allocate device-only memory from the cache or device
     *
     * Device-only memory may have better performance for GPU-only operations
     * but cannot be directly accessed from the host.
     *
     * @param size Size in bytes to allocate
     * @param device Device index
     * @return Pointer to allocated memory
     * @throws std::runtime_error if allocation fails
     */
    void* allocate_device(size_t size, int device = 0);

    /**
     * @brief Allocate memory (defaults to shared memory)
     *
     * @param size Size in bytes to allocate
     * @param device Device index
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size, int device = 0) {
        return allocate_shared(size, device);
    }

    /**
     * @brief Free memory back to the cache
     *
     * @param ptr Pointer to free
     * @param device Device index
     */
    void free(void* ptr, int device = 0);

    /**
     * @brief Empty the cache and release all cached blocks
     *
     * @param device Device ID (-1 for all devices)
     */
    void empty_cache(int device = -1);

    /**
     * @brief Garbage collection - release unused cached blocks
     *
     * @param device Device ID (-1 for all devices)
     * @param aggressive If true, releases all cached blocks; otherwise only large ones
     */
    void garbage_collect(int device = -1, bool aggressive = false);

    /**
     * @brief Get currently allocated memory in bytes
     *
     * @param device Device ID (-1 for all devices)
     * @return Size in bytes
     */
    size_t memory_allocated(int device = -1) const;

    /**
     * @brief Get total reserved memory (allocated + cached) in bytes
     *
     * @param device Device ID (-1 for all devices)
     * @return Size in bytes
     */
    size_t memory_reserved(int device = -1) const;

    /**
     * @brief Get cached memory in bytes
     *
     * @param device Device ID (-1 for all devices)
     * @return Size in bytes
     */
    size_t memory_cached(int device = -1) const;

    /**
     * @brief Get memory statistics
     *
     * @param device Device ID (-1 for all devices)
     * @return Memory statistics
     */
    OneAPIMemoryStats get_stats(int device = -1) const;

    /**
     * @brief Reset statistics counters
     */
    void reset_stats();

    /**
     * @brief Set memory allocation alignment (default: 64 bytes)
     *
     * @param alignment Alignment in bytes (must be power of 2)
     */
    void set_alignment(size_t alignment);

    /**
     * @brief Set maximum cached memory per device (default: unlimited)
     *
     * @param max_bytes Maximum cached memory in bytes (0 = unlimited)
     */
    void set_max_cached_memory(size_t max_bytes);

    /**
     * @brief Set minimum block size for splitting (default: 512 bytes)
     *
     * @param min_size Minimum size in bytes
     */
    void set_min_split_size(size_t min_size);

    /**
     * @brief Set large allocation threshold
     *
     * Allocations larger than this threshold are released immediately
     * on free rather than being cached.
     *
     * @param threshold Threshold in bytes (default: 2GB)
     */
    void set_large_allocation_threshold(size_t threshold);

private:
    OneAPICachingAllocator();
    ~OneAPICachingAllocator();

    // Prevent copying
    OneAPICachingAllocator(const OneAPICachingAllocator&) = delete;
    OneAPICachingAllocator& operator=(const OneAPICachingAllocator&) = delete;

    /**
     * @brief Internal allocation implementation
     */
    void* allocate_impl(size_t size, int device, bool shared);

    /**
     * @brief Try to allocate from cache
     */
    OneAPIBlock* try_allocate_from_cache(size_t size, int device, bool shared);

    /**
     * @brief Allocate new block from device
     */
    OneAPIBlock* allocate_new_block(size_t size, int device, bool shared);

    /**
     * @brief Split a block if it's larger than needed
     */
    bool split_block(OneAPIBlock* block, size_t size);

    /**
     * @brief Try to merge block with adjacent free blocks
     */
    bool try_merge_blocks(OneAPIBlock* block);

    /**
     * @brief Round size up to alignment
     */
    size_t round_size(size_t size) const;

    /**
     * @brief Free blocks exceeding cache limit
     */
    void enforce_cache_limit(int device);

    /**
     * @brief Release a block back to device
     */
    void release_block(OneAPIBlock* block);

    // Per-device data structures
    struct DeviceAllocator {
        void* queue = nullptr;  // Actually sycl::queue*, cast in implementation
        bool initialized = false;

        // Free blocks ordered by size (for best-fit) - separate pools for shared vs device
        std::set<OneAPIBlock*, OneAPIBlockComparator> free_shared_blocks;
        std::set<OneAPIBlock*, OneAPIBlockComparator> free_device_blocks;

        // All blocks (free and allocated) by pointer
        std::unordered_map<void*, std::unique_ptr<OneAPIBlock>> all_blocks;

        // Statistics
        OneAPIMemoryStats stats;

        DeviceAllocator() = default;
    };

    // Mutex for thread safety
    mutable std::mutex mutex_;

    // Per-device allocators
    std::unordered_map<int, DeviceAllocator> device_allocators_;

    // Configuration
    size_t alignment_;
    size_t max_cached_memory_;
    size_t min_split_size_;
    size_t large_allocation_threshold_;
    bool merge_enabled_;
};

/**
 * @brief RAII wrapper for OneAPI cached memory allocation
 */
class OneAPICachedMemoryGuard {
public:
    OneAPICachedMemoryGuard(size_t size, int device = 0, bool shared = true)
        : ptr_(nullptr), device_(device), size_(size) {
        if (shared) {
            ptr_ = OneAPICachingAllocator::get().allocate_shared(size, device);
        } else {
            ptr_ = OneAPICachingAllocator::get().allocate_device(size, device);
        }
    }

    ~OneAPICachedMemoryGuard() {
        if (ptr_) {
            OneAPICachingAllocator::get().free(ptr_, device_);
        }
    }

    // Move semantics
    OneAPICachedMemoryGuard(OneAPICachedMemoryGuard&& other) noexcept
        : ptr_(other.ptr_), device_(other.device_), size_(other.size_) {
        other.ptr_ = nullptr;
    }

    OneAPICachedMemoryGuard& operator=(OneAPICachedMemoryGuard&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                OneAPICachingAllocator::get().free(ptr_, device_);
            }
            ptr_ = other.ptr_;
            device_ = other.device_;
            size_ = other.size_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // Prevent copying
    OneAPICachedMemoryGuard(const OneAPICachedMemoryGuard&) = delete;
    OneAPICachedMemoryGuard& operator=(const OneAPICachedMemoryGuard&) = delete;

    void* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void* ptr_;
    int device_;
    size_t size_;
};

} // namespace backend
} // namespace tenzor
