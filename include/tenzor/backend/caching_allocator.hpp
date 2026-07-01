#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declare CUDA types to avoid requiring cuda_runtime.h
#ifndef __CUDACC__
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
struct CUevent_st;
typedef struct CUevent_st* cudaEvent_t;
#else
#include <cuda_runtime.h>
#endif

namespace tenzor {
namespace backend {

/**
 * @brief Memory block representation for the caching allocator
 */
struct Block {
    void* ptr;              // Device pointer
    size_t size;            // Block size in bytes
    bool allocated;         // Whether block is currently allocated
    int device;             // CUDA device ID
    cudaStream_t stream;    // Associated stream (for async operations)
    void* original_ptr;     // Original cudaMalloc pointer (for merge tracking)

    // Pending CUDA events that must complete before this block's memory may be
    // safely reused on a *different* stream. One is recorded on free() against
    // the block's last-use stream; block merges splice neighbors' events in,
    // and a split hands still-pending events to the remainder. Empty whenever
    // reuse needs no cross-stream synchronization. The second element of each
    // pair is the stream the event was recorded on (used to take the
    // same-stream fast path on reuse).
    std::vector<std::pair<cudaEvent_t, cudaStream_t>> free_events;

    Block(void* p, size_t s, int dev, cudaStream_t str = nullptr)
        : ptr(p), size(s), allocated(false), device(dev), stream(str), original_ptr(p) {}

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

    MemoryStats()
        : allocated_bytes(0), reserved_bytes(0), cached_bytes(0),
          num_allocations(0), num_frees(0), num_cache_hits(0),
          num_splits(0), num_merges(0) {}
};

/**
 * @brief CachingAllocator for efficient GPU memory management
 *
 * Implements a memory pooling system that reuses freed memory blocks
 * to reduce cudaMalloc/cudaFree overhead. Uses best-fit allocation
 * strategy with block splitting and merging to reduce fragmentation.
 *
 * Thread-safe and supports multiple CUDA devices.
 */
class CachingAllocator {
public:
    /**
     * @brief Get the singleton instance
     */
    static CachingAllocator& get();

    /**
     * @brief Allocate memory from the cache or device
     *
     * @param size Size in bytes to allocate
     * @param device CUDA device ID
     * @param stream CUDA stream for async operations
     * @return Device pointer to allocated memory
     * @throws std::runtime_error if allocation fails
     */
    void* allocate(size_t size, int device = 0, cudaStream_t stream = nullptr);

    /**
     * @brief Free memory back to the cache
     *
     * @param ptr Device pointer to free
     * @param device CUDA device ID
     */
    void free(void* ptr, int device = 0);

    /**
     * @brief Empty the cache and release all cached blocks
     *
     * @param device Device ID (-1 for all devices)
     */
    void empty_cache(int device = -1);

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
     * @brief Reset statistics counters
     */
    void reset_stats();

    /**
     * @brief Set memory allocation alignment (default: 512 bytes)
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

private:
    CachingAllocator();
    ~CachingAllocator();

    // Prevent copying
    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;

    /**
     * @brief Try to allocate from cache
     *
     * @param size Size in bytes
     * @param device Device ID
     * @param stream CUDA stream
     * @return Block pointer if found, nullptr otherwise
     */
    Block* try_allocate_from_cache(size_t size, int device, cudaStream_t stream);

    /**
     * @brief Allocate new block from device
     *
     * @param size Size in bytes
     * @param device Device ID
     * @param stream CUDA stream
     * @return New block pointer
     */
    Block* allocate_new_block(size_t size, int device, cudaStream_t stream);

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
     * @brief Release a block back to the device.
     *
     * Only frees the underlying allocation when @p block still owns the
     * original cudaMalloc pointer (ptr == original_ptr). Interior split
     * remainder blocks are left tracked and not freed.
     *
     * @param block Block to release
     * @return true if the device allocation was actually freed, false if the
     *         block was an interior sub-block and skipped.
     */
    bool release_block(Block* block);

    // --- Stream-ordered reuse event helpers (called with device mutex held) ---

    /// Take an event from the pool, creating a new timing-disabled event if the
    /// pool is empty. Returns nullptr if event creation fails.
    cudaEvent_t acquire_event(std::vector<cudaEvent_t>& pool);

    /// Return an event to the pool for later reuse (no-op for nullptr).
    void recycle_event(std::vector<cudaEvent_t>& pool, cudaEvent_t event);

    /// Record an event on @p block 's last-use stream and stash it on the block
    /// so a later cross-stream reuse can wait for in-flight work to finish.
    void record_free_event(std::vector<cudaEvent_t>& pool, Block* block);

    /// Destroy every event currently held in @p pool (cache teardown).
    void destroy_event_pool(std::vector<cudaEvent_t>& pool);

    // Per-device data structures
    struct DeviceAllocator {
        // Per-device mutex for concurrent multi-GPU allocation
        mutable std::mutex mutex;

        // Free blocks ordered by size (for best-fit)
        std::set<Block*, BlockComparator> free_blocks;

        // All blocks (free and allocated) by pointer
        std::unordered_map<void*, std::unique_ptr<Block>> all_blocks;

        // Address-ordered index (non-owning) into all_blocks, enabling O(log n)
        // predecessor lookup for backward coalescing on free().
        std::map<void*, Block*> blocks_by_addr;

        // Pool of reusable CUDA events (recorded on free, consumed on reuse) to
        // avoid cudaEventCreate/Destroy churn on the allocation hot path.
        std::vector<cudaEvent_t> event_pool;

        // Statistics
        MemoryStats stats;

        /// Preferred NUMA node for CPU allocations (-1 = no preference)
        int preferred_numa_node{-1};

        DeviceAllocator() = default;
    };

    // Mutex for protecting device_allocators_ map structure
    mutable std::mutex map_mutex_;

    // Per-device allocators
    std::unordered_map<int, DeviceAllocator> device_allocators_;

    // Configuration
    size_t alignment_;
    size_t max_cached_memory_;
    bool merge_enabled_;
    size_t min_split_size_;
};

/**
 * @brief RAII wrapper for cached memory allocation
 */
class CachedMemoryGuard {
public:
    CachedMemoryGuard(size_t size, int device = 0)
        : ptr_(nullptr), device_(device), size_(size) {
        ptr_ = CachingAllocator::get().allocate(size, device);
    }

    ~CachedMemoryGuard() {
        if (ptr_) {
            CachingAllocator::get().free(ptr_, device_);
        }
    }

    // Move semantics
    CachedMemoryGuard(CachedMemoryGuard&& other) noexcept
        : ptr_(other.ptr_), device_(other.device_), size_(other.size_) {
        other.ptr_ = nullptr;
    }

    CachedMemoryGuard& operator=(CachedMemoryGuard&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                CachingAllocator::get().free(ptr_, device_);
            }
            ptr_ = other.ptr_;
            device_ = other.device_;
            size_ = other.size_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // Prevent copying
    CachedMemoryGuard(const CachedMemoryGuard&) = delete;
    CachedMemoryGuard& operator=(const CachedMemoryGuard&) = delete;

    void* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void* ptr_;
    int device_;
    size_t size_;
};

} // namespace backend
} // namespace tenzor
