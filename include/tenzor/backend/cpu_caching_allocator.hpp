/**
 * @file cpu_caching_allocator.hpp
 * @brief Thread-local caching memory allocator for CPU tensors
 *
 * Provides high-performance memory allocation by caching freed blocks
 * for reuse, eliminating repeated system calls to posix_memalign/free.
 *
 * Features:
 * - Thread-local pools for zero-contention fast path
 * - Global pool for cross-thread block sharing
 * - Best-fit allocation strategy
 * - Block splitting for large cached blocks
 * - Proactive memory release under pressure
 * - 64-byte aligned allocations for SIMD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>

namespace tenzor {
namespace cpu {

/**
 * @brief Thread-local caching allocator for CPU memory
 *
 * Singleton allocator that maintains per-thread free block pools
 * for fast allocation/deallocation without lock contention.
 * Overflow blocks migrate to a global pool for cross-thread reuse.
 */
class CPUCachingAllocator {
public:
    /**
     * @brief Memory statistics
     */
    struct Stats {
        size_t total_allocations{0};    ///< Total allocation requests
        size_t cache_hits{0};           ///< Allocations served from cache
        size_t allocated_bytes{0};      ///< Currently allocated bytes
        size_t cached_bytes{0};         ///< Bytes in free pools
        size_t peak_allocated_bytes{0}; ///< Peak allocated bytes
        size_t num_splits{0};           ///< Number of block splits
        size_t num_backend_allocs{0};   ///< Calls to posix_memalign
        size_t num_backend_frees{0};    ///< Calls to free()
    };

    /**
     * @brief Get singleton instance
     */
    static auto instance() -> CPUCachingAllocator&;

    // Non-copyable, non-movable
    CPUCachingAllocator(const CPUCachingAllocator&) = delete;
    CPUCachingAllocator& operator=(const CPUCachingAllocator&) = delete;
    CPUCachingAllocator(CPUCachingAllocator&&) = delete;
    CPUCachingAllocator& operator=(CPUCachingAllocator&&) = delete;

    /**
     * @brief Allocate aligned memory
     *
     * Attempts to reuse a cached block from thread-local pool first,
     * then global pool, falling back to posix_memalign if needed.
     *
     * @param bytes Number of bytes to allocate
     * @return Pointer to allocated memory (64-byte aligned)
     * @throws std::bad_alloc if allocation fails
     */
    auto allocate(size_t bytes) -> void*;

    /**
     * @brief Return memory to cache
     *
     * Does not immediately free memory - caches it for reuse.
     * Memory is freed to OS when cache exceeds limits.
     *
     * @param ptr Pointer to deallocate (nullptr is no-op)
     */
    void deallocate(void* ptr);

    /**
     * @brief Release all cached memory to OS
     *
     * Frees all blocks in thread-local and global free pools.
     * Does not affect currently allocated blocks.
     */
    void release_cached_memory();

    /**
     * @brief Set maximum cached memory limit
     *
     * When cached memory exceeds this limit, oldest blocks are freed.
     *
     * @param bytes Maximum bytes to cache (default 1GB)
     */
    void set_max_cached_bytes(size_t bytes);

    /**
     * @brief Set maximum per-thread cached memory
     *
     * When a thread's cache exceeds this, blocks migrate to global pool.
     *
     * @param bytes Maximum bytes per thread (default 256MB)
     */
    void set_max_local_cached_bytes(size_t bytes);

    /**
     * @brief Set minimum block size for splitting
     *
     * Cached blocks larger than 2x request and >= this size are split.
     *
     * @param bytes Minimum split size (default 1MB)
     */
    void set_min_split_size(size_t bytes);

    /**
     * @brief Get current memory statistics
     */
    auto get_stats() const -> Stats;

    /**
     * @brief Erase the calling thread's entry from per_thread_pending_frees_.
     *
     * Called from ThreadLocalPoolWrapper's destructor when a thread exits.
     * Any pending-decrement entries for a dead thread's tid are stale (the
     * memory was already returned globally); discarding them is safe and
     * prevents the map from growing without bound under short-lived producer
     * thread churn.
     */
    void erase_pending_for_current_thread();

#ifdef TENZOR_TESTING
    /**
     * @brief Return the number of entries in per_thread_pending_frees_.
     *
     * Diagnostic accessor for unit tests only.  Allows tests to assert that
     * the map does not grow unboundedly when short-lived producer threads exit.
     */
    auto get_pending_map_size() const -> std::size_t {
        std::lock_guard<std::mutex> lock(global_mutex_);
        return per_thread_pending_frees_.size();
    }
#endif

    /**
     * @brief Reset statistics counters
     */
    void reset_stats();

    /**
     * @brief Per-thread local pool statistics
     */
    struct LocalStats {
        size_t allocated_bytes{0};  ///< Bytes currently allocated by this thread
        size_t cached_bytes{0};     ///< Bytes cached in this thread's free pool
    };

    /**
     * @brief Get statistics for the calling thread's local pool.
     *
     * Drains any pending cross-thread decrements before returning so the
     * value is accurate immediately after a cross-thread deallocate.
     */
    auto get_local_stats() -> LocalStats;

    /**
     * @brief Block metadata (public for thread_local storage)
     */
    struct Block {
        void* ptr{nullptr};                         ///< Memory pointer
        size_t size{0};                             ///< Usable size
        size_t allocated_size{0};                   ///< Original allocation size (before split)
        void* root_ptr{nullptr};                    ///< Original allocation pointer (for free)
        bool is_split{false};                       ///< True if this block was created by splitting
        std::thread::id originating_tid{};          ///< Thread that allocated this block
    };

    /**
     * @brief Root allocation tracking for coalescing
     */
    struct RootAllocation {
        void* ptr{nullptr};         ///< Original pointer from posix_memalign
        size_t size{0};             ///< Original allocation size
        int fragment_count{1};      ///< Number of fragments (1 = not split)
        size_t freed_size{0};       ///< Size of fragments returned to free pool
    };

    /**
     * @brief Thread-local memory pool (public for thread_local storage)
     *
     * Note: Root allocations are tracked centrally in global_root_allocations_
     * to correctly handle fragments that may be split across local and global pools.
     */
    struct ThreadLocalPool {
        std::multimap<size_t, Block> free_blocks;           ///< Free blocks by size
        std::unordered_map<void*, Block> allocated_blocks;  ///< Tracked allocations
        size_t cached_bytes{0};                             ///< Total cached size
        size_t allocated_bytes{0};                          ///< Total allocated size
    };

private:
    CPUCachingAllocator();
    ~CPUCachingAllocator();

    /**
     * @brief Get thread-local pool (creates if needed)
     */
    static auto get_local_pool() -> ThreadLocalPool&;

    /**
     * @brief Try to allocate from thread-local pool
     */
    auto try_allocate_local(size_t bytes) -> void*;

    /**
     * @brief Try to allocate from global pool
     */
    auto try_allocate_global(size_t bytes) -> void*;

    /**
     * @brief Allocate new block from system
     */
    auto allocate_from_system(size_t bytes) -> void*;

    /**
     * @brief Find best-fit block in a free block map
     */
    auto find_best_fit(std::multimap<size_t, Block>& blocks, size_t bytes)
        -> std::multimap<size_t, Block>::iterator;

    /**
     * @brief Split a block if beneficial (thread-local version)
     *
     * @param block Block to potentially split
     * @param requested_size Size actually needed
     * @param free_blocks Pool to return remainder to
     * @param local Thread-local pool for root allocation tracking
     * @return true if block was split
     */
    auto maybe_split_block(Block& block, size_t requested_size,
                           std::multimap<size_t, Block>& free_blocks,
                           ThreadLocalPool& local) -> bool;

    /**
     * @brief Split a block if beneficial (global version)
     */
    auto maybe_split_block_global(Block& block, size_t requested_size,
                                   std::multimap<size_t, Block>& free_blocks) -> bool;

    /**
     * @brief Migrate blocks from local to global pool
     */
    void migrate_to_global(ThreadLocalPool& local);

    /**
     * @brief Check and handle memory pressure
     */
    void check_memory_pressure();

    /**
     * @brief Free a block back to the system
     */
    void free_to_system(void* ptr);

    /**
     * @brief Try to coalesce a freed block with its siblings
     * @return true if the root allocation can now be freed
     */
    bool try_coalesce_and_free(Block& block, ThreadLocalPool& local);

    /**
     * @brief Check if a root allocation is fully coalesced
     */
    bool is_fully_coalesced(const RootAllocation& root) const;

    /**
     * @brief Drain cross-thread pending decrements for the calling thread.
     *
     * When a block allocated by thread A is freed by thread B, the
     * cross-thread free path cannot safely touch thread A's thread_local
     * state.  Instead it records the freed pointer into
     * per_thread_pending_frees_ (keyed by originating tid) under
     * global_mutex_.  The next time thread A enters the allocator it calls
     * this function to remove those stale entries from its local pool and
     * reconcile local.allocated_bytes.
     */
    void drain_pending_decrements(ThreadLocalPool& local);

    // Global pool (accessed with mutex)
    std::multimap<size_t, Block> global_free_blocks_;
    std::unordered_map<void*, Block> global_allocated_blocks_;
    std::unordered_map<void*, RootAllocation> global_root_allocations_;
    mutable std::mutex global_mutex_;

    // Cross-thread dealloc reconciliation: maps originating tid → list of
    // pointers freed by a different thread.  Entries are drained by the
    // originating thread on its next allocator call.  Protected by global_mutex_.
    std::unordered_map<std::thread::id, std::vector<void*>> per_thread_pending_frees_;

    // Configuration
    std::atomic<size_t> max_cached_bytes_{1ULL << 30};        // 1GB
    std::atomic<size_t> max_local_cached_bytes_{256ULL << 20}; // 256MB
    std::atomic<size_t> min_split_size_{1ULL << 20};          // 1MB

    // Global statistics (atomic for thread-safety)
    mutable std::mutex stats_mutex_;
    Stats global_stats_;

    // Constants
    static constexpr size_t ALIGNMENT = 64;  // Cache line alignment
};

} // namespace cpu
} // namespace tenzor
