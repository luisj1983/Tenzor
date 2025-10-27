/**
 * @file caching_allocator.hpp
 * @brief Memory pooling allocator with block caching and reuse
 *
 * Provides a caching allocator that reduces backend allocation calls by
 * maintaining a pool of freed memory blocks for reuse. Implements delayed
 * deallocation, size-based block allocation, and defragmentation support.
 */

#pragma once

#include <map>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstddef>
#include <stdexcept>
#include "device.hpp"

namespace tenzor {

// Forward declaration
class Backend;

/**
 * @brief Memory pooling allocator with caching and reuse.
 *
 * The CachingAllocator reduces memory allocation overhead by maintaining
 * a pool of freed memory blocks that can be reused for future allocations.
 * This is particularly beneficial for GPU memory where allocation/deallocation
 * operations are expensive.
 *
 * Features:
 * - Size-based block allocation with best-fit strategy
 * - Delayed deallocation (caching) for improved performance
 * - Defragmentation support to reduce fragmentation
 * - Thread-safe operations with mutex protection
 * - Per-device memory tracking
 * - Statistics tracking for monitoring and optimization
 *
 * @code
 * Backend* backend = get_cuda_backend();
 * Device device = Device::cuda(0);
 * auto allocator = std::make_unique<CachingAllocator>(backend, device);
 *
 * // Allocate memory (may reuse cached block)
 * void* ptr1 = allocator->allocate(1024);
 *
 * // Deallocate (cached for reuse)
 * allocator->deallocate(ptr1);
 *
 * // Reuses cached block if size matches
 * void* ptr2 = allocator->allocate(1024);
 * @endcode
 *
 * @note Thread-safe for concurrent allocate/deallocate calls
 * @note Memory is not automatically freed until defragment() or destructor
 */
class CachingAllocator {
public:
    /**
     * @brief Construct caching allocator for specific backend and device.
     *
     * @param backend Backend managing memory allocation (must not be null)
     * @param device Device where memory will be allocated
     * @throws std::invalid_argument if backend is null
     *
     * @code
     * Backend* cuda_backend = get_backend("cuda");
     * Device device = Device::cuda(0);
     * CachingAllocator allocator(cuda_backend, device);
     * @endcode
     */
    explicit CachingAllocator(Backend* backend, Device device);

    /**
     * @brief Destructor frees all cached and allocated blocks.
     *
     * Ensures all memory is properly returned to the backend,
     * preventing leaks on allocator destruction.
     */
    ~CachingAllocator();

    // Non-copyable, movable
    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;
    CachingAllocator(CachingAllocator&&) noexcept;
    CachingAllocator& operator=(CachingAllocator&&) noexcept;

    /**
     * @brief Allocate memory block of specified size.
     *
     * Attempts to reuse a cached block of sufficient size. If no suitable
     * cached block exists, allocates new memory from the backend.
     *
     * Uses best-fit strategy: finds the smallest cached block that can
     * accommodate the requested size to minimize fragmentation.
     *
     * @param bytes Number of bytes to allocate (must be > 0)
     * @return Pointer to allocated memory
     * @throws std::invalid_argument if bytes is 0
     * @throws std::runtime_error if backend allocation fails
     *
     * @note Thread-safe
     * @note Returned pointer is aligned according to backend requirements
     *
     * @code
     * void* ptr = allocator->allocate(4096);
     * // Use memory...
     * allocator->deallocate(ptr);
     * @endcode
     */
    auto allocate(size_t bytes) -> void*;

    /**
     * @brief Deallocate memory block, caching for reuse.
     *
     * Instead of immediately freeing memory to the backend, stores the
     * block in the free pool for potential reuse. This reduces expensive
     * backend allocation calls.
     *
     * @param ptr Pointer returned by allocate() (null is a no-op)
     * @throws std::runtime_error if ptr is unknown (not allocated by this allocator)
     *
     * @note Thread-safe
     * @note Memory is not returned to backend until defragment() or destruction
     *
     * @code
     * void* ptr = allocator->allocate(1024);
     * allocator->deallocate(ptr);  // Cached, not freed
     * @endcode
     */
    auto deallocate(void* ptr) -> void;

    /**
     * @brief Free all cached blocks back to backend.
     *
     * Returns all free (cached) blocks to the backend, reducing overall
     * memory usage. Useful for periodic cleanup or memory pressure situations.
     *
     * Does not affect currently allocated blocks.
     *
     * @note Thread-safe
     * @note Expensive operation - use sparingly
     *
     * @code
     * // Periodic cleanup
     * allocator->defragment();
     * @endcode
     */
    auto defragment() -> void;

    /**
     * @brief Get total allocated memory in bytes.
     *
     * Returns the sum of all allocated blocks, including both
     * in-use and cached (free) blocks.
     *
     * @return Total allocated bytes
     *
     * @note Thread-safe
     */
    auto total_allocated_bytes() const -> size_t;

    /**
     * @brief Get total cached (free) memory in bytes.
     *
     * Returns the sum of all cached blocks available for reuse.
     *
     * @return Total cached bytes
     *
     * @note Thread-safe
     */
    auto total_cached_bytes() const -> size_t;

    /**
     * @brief Get number of allocated blocks.
     *
     * @return Count of allocated blocks (in-use + cached)
     *
     * @note Thread-safe
     */
    auto allocated_block_count() const -> size_t;

    /**
     * @brief Get number of cached (free) blocks.
     *
     * @return Count of cached blocks available for reuse
     *
     * @note Thread-safe
     */
    auto cached_block_count() const -> size_t;

    /**
     * @brief Get cache hit rate as percentage.
     *
     * Returns the percentage of allocations that were satisfied by
     * cached blocks (0.0 - 100.0).
     *
     * @return Cache hit rate percentage
     *
     * @note Thread-safe
     * @note Returns 0.0 if no allocations have been performed
     */
    auto cache_hit_rate() const -> double;

    /**
     * @brief Get device associated with this allocator.
     *
     * @return Device specification
     */
    auto device() const -> Device { return device_; }

private:
    /**
     * @brief Find suitable free block for allocation.
     *
     * Searches free_blocks_ for a block >= requested size using
     * lower_bound for efficient O(log n) lookup.
     *
     * @param bytes Minimum required size
     * @return Pointer to cached block or nullptr if none found
     *
     * @note Must be called with mutex_ locked
     */
    auto find_free_block(size_t bytes) -> void*;

    /**
     * @brief Get size of allocated block.
     *
     * @param ptr Pointer to allocated block
     * @return Size in bytes
     * @throws std::runtime_error if ptr is unknown
     *
     * @note Must be called with mutex_ locked
     */
    auto size_of(void* ptr) const -> size_t;

    /**
     * @brief Free all blocks in free list.
     *
     * Internal helper for defragment() and destructor.
     *
     * @note Must be called with mutex_ locked
     */
    auto free_cached_blocks() -> void;

    Backend* backend_;                                      ///< Backend for memory allocation
    Device device_;                                         ///< Target device for allocations
    mutable std::mutex mutex_;                              ///< Mutex for thread safety

    // Memory tracking
    std::multimap<size_t, void*> free_blocks_;             ///< Free blocks (size -> ptr)
    std::unordered_map<void*, size_t> allocated_blocks_;   ///< All allocated blocks (ptr -> size)

    // Statistics
    size_t total_allocations_{0};      ///< Total allocation requests
    size_t cache_hits_{0};             ///< Allocations satisfied by cache
    size_t backend_allocations_{0};    ///< Allocations from backend
    size_t total_allocated_bytes_{0};  ///< Total memory from backend
    size_t total_cached_bytes_{0};     ///< Total memory in free pool
};

} // namespace tenzor
