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

struct MemoryBlock;

/// Size-ordered index of free blocks: size -> block, enabling O(log N)
/// best-fit lookup instead of an O(N) scan over every (free + allocated) block.
using FreeBlockIndex = std::multimap<size_t, MemoryBlock*>;

/**
 * @brief Memory block metadata for free-list management
 */
struct MemoryBlock {
    void* ptr;              // Pointer to memory block
    size_t size;            // Size of this block in bytes
    bool is_free;           // Whether this block is available
    MemoryBlock* next;      // Next block in list
    MemoryBlock* prev;      // Previous block in list

    // When is_free is true, this iterator points at this block's entry in the
    // allocator's free_blocks_ index; otherwise it is unset (== index end()).
    // Stored per-block so the index entry can be erased/updated in O(log N).
    FreeBlockIndex::iterator free_it{};
    bool indexed{false};    // Whether free_it currently references a live entry

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
 * memory. Provides allocation using a best-fit free-list algorithm backed by a
 * size-ordered index of free blocks (free_blocks_), giving O(log N) best-fit
 * lookup instead of an O(N) scan over every block.
 * Thread-safe with automatic coalescing of adjacent free blocks.
 *
 * Key features:
 * - Best-fit allocation from a pre-allocated pool (O(log N) via free_blocks_)
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
     * @brief Register a block in the size-ordered free index.
     *
     * Marks the block as free and inserts it into free_blocks_ so subsequent
     * best-fit lookups can find it in O(log N). Safe to call when already
     * indexed (no-op re-insert is avoided via the indexed flag).
     */
    auto index_free_block(MemoryBlock* block) -> void;

    /**
     * @brief Remove a block from the size-ordered free index.
     *
     * Erases the block's entry from free_blocks_ (O(log N)) if it is present.
     * Does not change block->is_free; callers update that as appropriate.
     */
    auto unindex_free_block(MemoryBlock* block) -> void;

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
     * @brief Allocate GPU-DMA-capable pinned memory (CUDA cudaHostAlloc, ROCm
     *        hipHostMalloc, or an OS-level mlock/VirtualLock fallback).
     * @param size Size in bytes
     * @param out_via_rocm Set to true if the allocation used hipHostMalloc
     *        (ROCm), so free_cuda_pinned() can free it the same way.
     * @return Pointer to allocated memory
     * @throws std::runtime_error on CUDA error
     */
    auto allocate_cuda_pinned(size_t size, bool& out_via_rocm) -> void*;

    /**
     * @brief Free memory allocated by allocate_cuda_pinned()
     * @param ptr Pointer to free
     * @param size Requested size in bytes used at allocation time (so the full
     *             locked region can be unlocked on the mlock/VirtualLock path)
     * @param via_rocm Must match the out_via_rocm value returned by the
     *        matching allocate_cuda_pinned() call.
     */
    auto free_cuda_pinned(void* ptr, size_t size, bool via_rocm) -> void;

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
        bool via_rocm{false};                // true if allocated via hipHostMalloc
    };
    std::vector<PoolRegion> pools_;          // All allocated pools

    // Free list management
    MemoryBlock* free_list_head_;            // Head of global free list
    std::map<void*, std::unique_ptr<MemoryBlock>> block_map_; // Map pointer -> block (owns lifetime)

    // Size-ordered index of free blocks for O(log N) best-fit allocation.
    // Mirrors the free subset of block_map_; kept in sync via
    // index_free_block() / unindex_free_block().
    FreeBlockIndex free_blocks_;

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
