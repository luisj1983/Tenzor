#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

// Forward declare CUDA types to avoid requiring cuda_runtime.h
#ifndef __CUDACC__
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
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
     * @brief Release a block back to device
     *
     * @param block Block to release
     */
    void release_block(Block* block);

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

        // Statistics
        MemoryStats stats;

        /// Preferred NUMA node for CPU allocations (-1 = no preference)
        int preferred_numa_node{-1};

        DeviceAllocator() = default;
    };

    /// Pending async free for stream-ordered memory management
    struct PendingFree {
        void* ptr;
        size_t size;
        void* event;  ///< Backend-specific event handle (e.g., cudaEvent_t)
        int device;
    };

    /// Pending frees awaiting stream completion
    std::vector<PendingFree> pending_frees_;

    /// CUDA graph capture mode
    bool capture_mode_{false};
    std::vector<std::pair<size_t, void*>> captured_allocations_;

public:
    /// Set NUMA preference for a device allocator
    auto set_preferred_numa_node(int device, int numa_node) -> void;

    /// Get NUMA preference for a device
    auto get_preferred_numa_node(int device) const -> int;

    /// Record an async free (deferred until event completes)
    auto free_async(void* ptr, size_t size, void* event, int device = 0) -> void;

    /// Process completed pending frees
    auto process_pending_frees() -> void;

    /// Enter CUDA graph capture mode (allocations are recorded, not executed)
    auto begin_capture_mode() -> void { capture_mode_ = true; captured_allocations_.clear(); }

    /// Exit capture mode and return the recorded allocations
    auto end_capture_mode() -> std::vector<std::pair<size_t, void*>> {
        capture_mode_ = false;
        return std::move(captured_allocations_);
    }

    /// Check if in capture mode
    auto in_capture_mode() const -> bool { return capture_mode_; }

private:

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
