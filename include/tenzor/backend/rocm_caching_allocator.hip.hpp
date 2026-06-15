#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

// Forward declare HIP types to avoid requiring hip_runtime.h
#ifndef __HIP_PLATFORM_AMD__
struct ihipStream_t;
typedef struct ihipStream_t* hipStream_t;
#else
#include <hip/hip_runtime.h>
#endif

namespace tenzor {
namespace backend {
namespace rocm {

/**
 * @brief Memory block representation for the ROCm caching allocator
 */
struct Block {
    void* ptr;              // Device pointer
    size_t size;            // Block size in bytes
    bool allocated;         // Whether block is currently allocated
    int device;             // HIP device ID
    hipStream_t stream;     // Associated stream (for async operations)
    size_t alignment;       // Block alignment for HBM optimization
    void* original_ptr;     // Original hipMalloc pointer (for merge tracking)

    Block(void* p, size_t s, int dev, hipStream_t str = nullptr, size_t align = 256)
        : ptr(p), size(s), allocated(false), device(dev), stream(str),
          alignment(align), original_ptr(p) {}

    // Comparison for ordered containers (by size, then by pointer)
    bool operator<(const Block& other) const {
        if (size != other.size) return size < other.size;
        return ptr < other.ptr;
    }
};

/**
 * @brief Comparator for ordering blocks by size (for best-fit allocation)
 */
struct BlockComparator {
    bool operator()(const Block* a, const Block* b) const {
        if (a->size != b->size) return a->size < b->size;
        return a->ptr < b->ptr;
    }
};

/**
 * @brief Statistics for memory usage tracking
 */
struct MemoryStats {
    size_t allocated_bytes;     // Currently allocated to user
    size_t reserved_bytes;      // Total reserved (allocated + cached)
    size_t cached_bytes;        // Cached but not allocated
    size_t num_allocations;     // Total allocation calls
    size_t num_frees;           // Total free calls
    size_t num_cache_hits;      // Allocations satisfied from cache
    size_t num_splits;          // Number of block splits
    size_t num_merges;          // Number of block merges
    size_t peak_allocated;      // Peak allocated memory
    size_t peak_reserved;       // Peak reserved memory
    size_t num_oom_errors;      // Out of memory errors
    size_t hbm_bytes;           // HBM-optimized allocations

    MemoryStats()
        : allocated_bytes(0), reserved_bytes(0), cached_bytes(0),
          num_allocations(0), num_frees(0), num_cache_hits(0),
          num_splits(0), num_merges(0), peak_allocated(0),
          peak_reserved(0), num_oom_errors(0), hbm_bytes(0) {}
};

/**
 * @brief Device properties for AMD GPUs
 */
struct DeviceProperties {
    size_t total_memory;        // Total device memory
    size_t available_memory;    // Available memory
    bool has_hbm;               // Has HBM (MI series)
    int compute_units;          // Number of compute units
    size_t max_shared_memory;   // Max shared memory per block
    int warp_size;              // Warp size (typically 64 for AMD)
    std::string device_name;    // Device name

    DeviceProperties()
        : total_memory(0), available_memory(0), has_hbm(false),
          compute_units(0), max_shared_memory(0), warp_size(64) {}
};

/**
 * @brief ROCm CachingAllocator for efficient AMD GPU memory management
 *
 * Implements a memory pooling system that reuses freed memory blocks
 * to reduce hipMalloc/hipFree overhead. Uses best-fit allocation
 * strategy with block splitting and merging to reduce fragmentation.
 *
 * Optimized for AMD GPU characteristics:
 * - HBM support for MI series GPUs (MI50, MI100, MI200, MI300)
 * - 256-byte alignment for optimal memory coalescing
 * - Efficient handling of large allocations (>2GB)
 * - Per-stream memory pools for concurrent kernel execution
 * - Garbage collection to prevent memory fragmentation
 *
 * Thread-safe and supports multiple HIP devices.
 */
class RocmCachingAllocator {
public:
    /**
     * @brief Get the singleton instance
     */
    static RocmCachingAllocator& get();

    /**
     * @brief Allocate memory from the cache or device
     *
     * @param size Size in bytes to allocate
     * @param device HIP device ID
     * @param stream HIP stream for async operations
     * @return Device pointer to allocated memory
     * @throws std::runtime_error if allocation fails
     */
    void* allocate(size_t size, int device = 0, hipStream_t stream = nullptr);

    /**
     * @brief Free memory back to the cache
     *
     * @param ptr Device pointer to free
     * @param device HIP device ID
     */
    void free(void* ptr, int device = 0);

    /**
     * @brief Empty the cache and release all cached blocks
     *
     * @param device Device ID (-1 for all devices)
     */
    void empty_cache(int device = -1);

    /**
     * @brief Perform garbage collection to reduce fragmentation
     *
     * @param device Device ID (-1 for all devices)
     * @param aggressive If true, performs more aggressive cleanup
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
    MemoryStats get_stats(int device = -1) const;

    /**
     * @brief Get device properties
     *
     * @param device Device ID
     * @return Device properties
     */
    DeviceProperties get_device_properties(int device) const;

    /**
     * @brief Reset statistics counters
     */
    void reset_stats();

    /**
     * @brief Set memory allocation alignment (default: 256 bytes for HBM)
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
     * @brief Enable or disable block merging (default: enabled)
     *
     * @param enable True to enable merging
     */
    void set_merge_enabled(bool enable);

    /**
     * @brief Set minimum block size for splitting (default: 512 bytes)
     *
     * @param min_size Minimum size in bytes
     */
    void set_min_split_size(size_t min_size);

    /**
     * @brief Enable or disable logging (default: disabled)
     *
     * @param enable True to enable logging
     */
    void set_logging_enabled(bool enable);

    /**
     * @brief Set large allocation threshold (default: 2GB)
     *
     * Allocations larger than this will use special handling
     *
     * @param threshold Threshold in bytes
     */
    void set_large_alloc_threshold(size_t threshold);

    /**
     * @brief Synchronize all streams for a device
     *
     * @param device Device ID
     */
    void synchronize_device(int device);

private:
    RocmCachingAllocator();
    ~RocmCachingAllocator();

    // Prevent copying
    RocmCachingAllocator(const RocmCachingAllocator&) = delete;
    RocmCachingAllocator& operator=(const RocmCachingAllocator&) = delete;

    /**
     * @brief Try to allocate from cache
     *
     * @param size Size in bytes
     * @param device Device ID
     * @param stream HIP stream
     * @return Block pointer if found, nullptr otherwise
     */
    Block* try_allocate_from_cache(size_t size, int device, hipStream_t stream);

    /**
     * @brief Allocate new block from device
     *
     * @param size Size in bytes
     * @param device Device ID
     * @param stream HIP stream
     * @return New block pointer
     */
    Block* allocate_new_block(size_t size, int device, hipStream_t stream);

    /**
     * @brief Split a block if it's larger than needed
     *
     * @param block Block to split
     * @param size Desired size
     * @return True if block was split
     */
    bool split_block(Block* block, size_t size);

    /**
     * @brief Try to merge block with adjacent free blocks
     *
     * @param block Block to merge
     * @return True if any merges occurred
     */
    bool try_merge_blocks(Block* block);

    /**
     * @brief Round size up to alignment
     *
     * @param size Original size
     * @return Aligned size
     */
    size_t round_size(size_t size) const;

    /**
     * @brief Free blocks exceeding cache limit
     *
     * @param device Device ID
     */
    void enforce_cache_limit(int device);

    /**
     * @brief Release a block back to device
     *
     * Only a block that still owns its original hipMalloc pointer
     * (`ptr == original_ptr`) may be device-freed; interior split-remainder
     * blocks are skipped (returns false) and reclaimed only once merging
     * reassembles the full allocation.
     *
     * @param block Block to release
     * @return True if the block was device-freed, false if it was an interior
     *         sub-block that cannot be freed standalone.
     */
    bool release_block(Block* block);

    /**
     * @brief Initialize device properties
     *
     * @param device Device ID
     */
    void initialize_device_properties(int device);

    /**
     * @brief Check if device has HBM memory
     *
     * @param device Device ID
     * @return True if device has HBM
     */
    bool is_hbm_device(int device);

    /**
     * @brief Log message (if logging enabled)
     *
     * @param message Message to log
     */
    void log_message(const std::string& message);

    /**
     * @brief Handle allocation failure with retry and GC
     *
     * @param size Size to allocate
     * @param device Device ID
     * @return True if retry should be attempted
     */
    bool handle_allocation_failure(size_t size, int device);

    // Per-device data structures
    struct DeviceAllocator {
        // Free blocks ordered by size (for best-fit)
        std::set<Block*, BlockComparator> free_blocks;

        // All blocks (free and allocated) by pointer
        std::unordered_map<void*, std::unique_ptr<Block>> all_blocks;

        // Address-ordered index (non-owning) into all_blocks, enabling O(log n)
        // predecessor lookup for backward coalescing on free(). Mirrors the
        // CUDA reference allocator.
        std::map<void*, Block*> blocks_by_addr;

        // Statistics
        MemoryStats stats;

        // Device properties
        DeviceProperties properties;

        // Per-stream blocks for better locality
        std::unordered_map<hipStream_t, std::vector<Block*>> stream_blocks;

        DeviceAllocator() = default;
    };

    // Mutex for thread safety
    mutable std::mutex mutex_;

    // Per-device allocators
    std::unordered_map<int, DeviceAllocator> device_allocators_;

    // Configuration
    size_t alignment_;
    size_t max_cached_memory_;
    bool merge_enabled_;
    size_t min_split_size_;
    bool logging_enabled_;
    size_t large_alloc_threshold_;
};

/**
 * @brief RAII wrapper for ROCm cached memory allocation
 */
class RocmCachedMemoryGuard {
public:
    RocmCachedMemoryGuard(size_t size, int device = 0, hipStream_t stream = nullptr)
        : ptr_(nullptr), device_(device), size_(size) {
        ptr_ = RocmCachingAllocator::get().allocate(size, device, stream);
    }

    ~RocmCachedMemoryGuard() {
        if (ptr_) {
            RocmCachingAllocator::get().free(ptr_, device_);
        }
    }

    // Move semantics
    RocmCachedMemoryGuard(RocmCachedMemoryGuard&& other) noexcept
        : ptr_(other.ptr_), device_(other.device_), size_(other.size_) {
        other.ptr_ = nullptr;
    }

    RocmCachedMemoryGuard& operator=(RocmCachedMemoryGuard&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                RocmCachingAllocator::get().free(ptr_, device_);
            }
            ptr_ = other.ptr_;
            device_ = other.device_;
            size_ = other.size_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // Prevent copying
    RocmCachedMemoryGuard(const RocmCachedMemoryGuard&) = delete;
    RocmCachedMemoryGuard& operator=(const RocmCachedMemoryGuard&) = delete;

    void* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void* ptr_;
    int device_;
    size_t size_;
};

} // namespace rocm
} // namespace backend
} // namespace tenzor
