/**
 * @file pinned_allocator.hpp
 * @brief Pinned (page-locked) memory allocator for fast CPU<->GPU transfers
 *
 * Part of ZeRO Offload Phase 1 - provides efficient pinned memory management
 * with O(1) allocation using a free-list based memory pool.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>
#include <map>
#include <atomic>

namespace tenzor::core {

/**
 * @brief Memory block metadata for free-list management
 */
struct MemoryBlock {
    void* ptr;              // Pointer to memory block
    size_t size;            // Size of this block in bytes
    bool is_free;           // Whether this block is available
    MemoryBlock* next;      // Next block in list
    MemoryBlock* prev;      // Previous block in list

    MemoryBlock(void* p, size_t s, bool free = true)
        : ptr(p), size(s), is_free(free), next(nullptr), prev(nullptr) {}
};

/**
 * @brief Statistics for pinned memory allocator
 */
struct PinnedMemoryStats {
    size_t total_size;          // Total pool size in bytes
    size_t allocated_size;      // Currently allocated bytes
    size_t free_size;           // Available bytes
    size_t num_allocations;     // Number of active allocations
    size_t num_blocks;          // Total number of blocks
    size_t num_free_blocks;     // Number of free blocks
    float fragmentation_ratio;  // Fragmentation ratio (0.0 to 1.0)
    size_t peak_allocated;      // Peak allocated size
    size_t num_defragmentations; // Number of defragmentation operations
};

/**
 * @brief Fast pinned memory allocator for CPU<->GPU transfers
 *
 * Uses cudaHostAlloc to pre-allocate a large pool of pinned (page-locked)
 * memory. Provides O(1) allocation using a best-fit free-list algorithm.
 * Thread-safe with automatic coalescing of adjacent free blocks.
 *
 * Key features:
 * - Fast O(1) allocation from pre-allocated pool
 * - Automatic coalescing of adjacent free blocks
 * - Thread-safe operations
 * - Defragmentation support
 * - Detailed statistics tracking
 *
 * Example:
 * @code
 * PinnedMemoryAllocator::Config config;
 * config.pool_size = 1024 * 1024 * 1024;  // 1 GB
 * config.min_block_size = 4096;           // 4 KB minimum
 * config.allow_growth = false;
 *
 * PinnedMemoryAllocator allocator(config);
 * void* ptr = allocator.allocate(1024 * 1024);  // Allocate 1 MB
 * // ... use memory for fast GPU transfers ...
 * allocator.deallocate(ptr);
 * @endcode
 */
class PinnedMemoryAllocator {
public:
    /**
     * @brief Configuration for pinned memory allocator
     */
    struct Config {
        size_t pool_size = 1024 * 1024 * 1024;  // Default 1 GB
        size_t min_block_size = 4096;           // Default 4 KB
        bool allow_growth = false;              // Don't allow pool growth by default
        size_t growth_increment = 256 * 1024 * 1024;  // 256 MB growth chunks
        size_t max_pool_size = 4ULL * 1024 * 1024 * 1024;  // 4 GB max
        bool enable_defragmentation = true;     // Enable automatic defragmentation
        bool throw_on_oom = false;              // Throw instead of returning nullptr on exhaustion
    };

    /**
     * @brief Construct pinned memory allocator
     * @param config Configuration for allocator
     * @throws std::runtime_error if CUDA allocation fails
     */
    explicit PinnedMemoryAllocator(const Config& config);

    /**
     * @brief Destructor - frees all pinned memory
     */
    ~PinnedMemoryAllocator();

    // Non-copyable but movable
    PinnedMemoryAllocator(const PinnedMemoryAllocator&) = delete;
    PinnedMemoryAllocator& operator=(const PinnedMemoryAllocator&) = delete;
    PinnedMemoryAllocator(PinnedMemoryAllocator&&) noexcept;
    PinnedMemoryAllocator& operator=(PinnedMemoryAllocator&&) noexcept;

    // =========================================================================
    // Allocation Interface
    // =========================================================================

    /**
     * @brief Allocate pinned memory from pool
     * @param bytes Number of bytes to allocate
     * @return Pointer to allocated memory, or nullptr if allocation fails
     * @note This is thread-safe
     */
    auto allocate(size_t bytes) -> void*;

    /**
     * @brief Deallocate previously allocated memory
     * @param ptr Pointer to memory to deallocate
     * @note This is thread-safe and automatically coalesces adjacent free blocks
     */
    auto deallocate(void* ptr) -> void;

    // =========================================================================
    // Pool Management
    // =========================================================================

    /**
     * @brief Defragment memory pool by coalescing all adjacent free blocks
     * @return Number of blocks coalesced
     */
    auto defragment() -> size_t;

    /**
     * @brief Reset allocator - deallocate everything
     * @note This does not free the pool, just marks all blocks as free
     */
    auto reset() -> void;

    /**
     * @brief Grow pool size (if allow_growth is enabled)
     * @param additional_bytes Number of bytes to add to pool
     * @return true if growth succeeded, false otherwise
     */
    auto grow_pool(size_t additional_bytes) -> bool;

    // =========================================================================
    // Statistics and Monitoring
    // =========================================================================

    /**
     * @brief Get total pool size in bytes
     */
    auto get_total_size() const -> size_t;

    /**
     * @brief Get currently allocated size in bytes
     */
    auto get_allocated_size() const -> size_t;

    /**
     * @brief Get free (available) size in bytes
     */
    auto get_free_size() const -> size_t;

    /**
     * @brief Get fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
     *
     * Calculated as: (num_free_blocks - 1) / max_possible_fragments
     * A high fragmentation ratio indicates memory is split into many small blocks.
     */
    auto get_fragmentation_ratio() const -> float;

    /**
     * @brief Get number of active allocations
     */
    auto get_allocation_count() const -> size_t;

    /**
     * @brief Get detailed statistics
     */
    auto get_stats() const -> PinnedMemoryStats;

    /**
     * @brief Check if allocator is valid (has allocated pool)
     */
    auto is_valid() const -> bool;

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Find best-fit free block for allocation
     * @param size Requested size
     * @return Block that fits, or nullptr if none found
     */
    auto find_best_fit(size_t size) -> MemoryBlock*;

    /**
     * @brief Split block into allocated and free portions
     * @param block Block to split
     * @param size Size to allocate from block
     * @return New free block (remainder), or nullptr if no split needed
     */
    auto split_block(MemoryBlock* block, size_t size) -> MemoryBlock*;

    /**
     * @brief Coalesce adjacent free blocks
     * @param block Block to coalesce with neighbors
     */
    auto coalesce_block(MemoryBlock* block) -> void;

    /**
     * @brief Find block by pointer
     * @param ptr Pointer to find
     * @return Block containing ptr, or nullptr if not found
     */
    auto find_block(void* ptr) -> MemoryBlock*;

    /**
     * @brief Allocate CUDA pinned memory
     * @param size Size in bytes
     * @return Pointer to allocated memory
     * @throws std::runtime_error on CUDA error
     */
    auto allocate_cuda_pinned(size_t size) -> void*;

    /**
     * @brief Free CUDA pinned memory
     * @param ptr Pointer to free
     */
    auto free_cuda_pinned(void* ptr) -> void;

    /**
     * @brief Calculate fragmentation ratio
     */
    auto calculate_fragmentation() const -> float;

    // =========================================================================
    // Member Variables
    // =========================================================================

    Config config_;                          // Allocator configuration

    // Memory pools (support multiple pools if growth is enabled)
    struct PoolRegion {
        void* base_ptr;                      // Base pointer to pool
        size_t size;                         // Size of this pool
        MemoryBlock* head;                   // Head of free list for this pool
    };
    std::vector<PoolRegion> pools_;          // All allocated pools

    // Free list management
    MemoryBlock* free_list_head_;            // Head of global free list
    std::map<void*, std::unique_ptr<MemoryBlock>> block_map_; // Map pointer -> block (owns lifetime)

    // Statistics tracking
    std::atomic<size_t> total_allocated_;    // Total bytes allocated
    std::atomic<size_t> num_allocations_;    // Number of active allocations
    std::atomic<size_t> peak_allocated_;     // Peak allocated bytes
    std::atomic<size_t> num_defragmentations_; // Defragmentation count

    // Thread safety
    mutable std::mutex mutex_;               // Mutex for thread-safe operations

    // Validation
    bool is_initialized_;                    // Whether allocator is initialized
};

} // namespace tenzor::core
