/**
 * @file cpu_caching_allocator.cpp
 * @brief Implementation of thread-local caching allocator for CPU tensors
 */

#include "tenzor/backend/cpu_caching_allocator.hpp"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace tenzor {
namespace cpu {

// Thread-local pool wrapper for tracking validity during shutdown
struct ThreadLocalPoolWrapper {
    CPUCachingAllocator::ThreadLocalPool pool;
    bool valid{true};

    ~ThreadLocalPoolWrapper() {
        // Mark as invalid - do NOT free memory here.
        // Reasons:
        // 1. Root allocations may have been migrated to global pool and freed by singleton
        // 2. At process exit, OS reclaims all memory anyway
        // 3. For thread exit during runtime, allocated memory should still be in use
        //    or have been properly deallocated through the allocator API
        valid = false;
    }
};

static thread_local ThreadLocalPoolWrapper tl_pool_wrapper_;

auto CPUCachingAllocator::instance() -> CPUCachingAllocator& {
    static CPUCachingAllocator instance;
    return instance;
}

CPUCachingAllocator::CPUCachingAllocator() = default;

CPUCachingAllocator::~CPUCachingAllocator() {
    // During static destruction, thread-local storage may already be destroyed.
    // Free all roots - at shutdown we release everything regardless of coalescing state.
    // Any "leaked" blocks would have their root freed anyway when we exit.
    std::lock_guard<std::mutex> lock(global_mutex_);
    for (auto& [root_ptr, root] : global_root_allocations_) {
        free_to_system(root.ptr);
    }
    global_root_allocations_.clear();
    global_free_blocks_.clear();
    global_allocated_blocks_.clear();
}

auto CPUCachingAllocator::get_local_pool() -> ThreadLocalPool& {
    return tl_pool_wrapper_.pool;
}

auto CPUCachingAllocator::allocate(size_t bytes) -> void* {
    if (bytes == 0) {
        return nullptr;
    }

    // Round up to alignment
    bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    void* ptr = nullptr;

    // Try thread-local pool first (no lock)
    ptr = try_allocate_local(bytes);
    if (ptr) {
        return ptr;
    }

    // Try global pool (with lock)
    ptr = try_allocate_global(bytes);
    if (ptr) {
        return ptr;
    }

    // Allocate from system
    ptr = allocate_from_system(bytes);
    if (!ptr) {
        // Try releasing cache and retrying
        release_cached_memory();
        ptr = allocate_from_system(bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
    }

    // Track allocation GLOBALLY so any thread can deallocate
    Block block;
    block.ptr = ptr;
    block.size = bytes;
    block.allocated_size = bytes;
    block.root_ptr = ptr;
    block.is_split = false;

    {
        std::lock_guard<std::mutex> lock(global_mutex_);
        global_allocated_blocks_[ptr] = block;

        // Track root allocation
        RootAllocation root;
        root.ptr = ptr;
        root.size = bytes;
        root.fragment_count = 1;
        root.freed_size = 0;
        global_root_allocations_[ptr] = root;
    }

    // Also track locally for fast deallocation path
    auto& local = get_local_pool();
    local.allocated_blocks[ptr] = block;
    local.allocated_bytes += bytes;

    // Update stats
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        global_stats_.total_allocations++;
        global_stats_.allocated_bytes += bytes;
        global_stats_.peak_allocated_bytes = std::max(
            global_stats_.peak_allocated_bytes,
            global_stats_.allocated_bytes
        );
    }

    return ptr;
}

void CPUCachingAllocator::deallocate(void* ptr) {
    if (!ptr) {
        return;
    }

    auto& local = get_local_pool();

    // Check thread-local allocated blocks first
    auto it = local.allocated_blocks.find(ptr);
    if (it != local.allocated_blocks.end()) {
        Block block = it->second;
        local.allocated_blocks.erase(it);
        local.allocated_bytes -= block.size;

        // Update CENTRAL tracking: remove from global_allocated_blocks_ and update root
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_allocated_blocks_.erase(ptr);

            auto root_it = global_root_allocations_.find(block.root_ptr);
            if (root_it != global_root_allocations_.end()) {
                root_it->second.freed_size += block.size;
            }
        }

        // Add to thread-local free pool
        local.free_blocks.insert({block.size, block});
        local.cached_bytes += block.size;

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            global_stats_.allocated_bytes -= block.size;
            global_stats_.cached_bytes += block.size;
        }

        // Check if local pool is too large
        if (local.cached_bytes > max_local_cached_bytes_.load()) {
            migrate_to_global(local);
        }

        // Check global memory pressure
        check_memory_pressure();
        return;
    }

    // Check global allocated blocks
    {
        std::lock_guard<std::mutex> lock(global_mutex_);
        auto git = global_allocated_blocks_.find(ptr);
        if (git != global_allocated_blocks_.end()) {
            Block block = git->second;
            global_allocated_blocks_.erase(git);

            // Update global root allocation tracking
            auto root_it = global_root_allocations_.find(block.root_ptr);
            if (root_it != global_root_allocations_.end()) {
                root_it->second.freed_size += block.size;
            }

            // Add to global free pool
            global_free_blocks_.insert({block.size, block});

            // Update stats
            {
                std::lock_guard<std::mutex> slock(stats_mutex_);
                global_stats_.allocated_bytes -= block.size;
                global_stats_.cached_bytes += block.size;
            }

            return;
        }
    }

    // Unknown pointer - might be from before caching was enabled
    // or allocated directly. Just free it.
    free_to_system(ptr);
}

bool CPUCachingAllocator::is_fully_coalesced(const RootAllocation& root) const {
    return root.freed_size == root.size;
}

void CPUCachingAllocator::release_cached_memory() {
    std::lock_guard<std::mutex> lock(global_mutex_);

    // Find all fully coalesced root allocations that can be freed
    std::vector<void*> roots_to_free;
    for (auto& [root_ptr, root] : global_root_allocations_) {
        if (is_fully_coalesced(root)) {
            roots_to_free.push_back(root_ptr);
        }
    }

    // Free fully coalesced allocations and remove their blocks from all pools
    for (void* root_ptr : roots_to_free) {
        size_t freed_bytes = 0;

        // Remove blocks from thread-local free pool
        if (tl_pool_wrapper_.valid) {
            auto& local = get_local_pool();
            for (auto it = local.free_blocks.begin(); it != local.free_blocks.end(); ) {
                if (it->second.root_ptr == root_ptr) {
                    freed_bytes += it->second.size;
                    local.cached_bytes -= it->second.size;
                    it = local.free_blocks.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // Remove blocks from global free pool
        for (auto it = global_free_blocks_.begin(); it != global_free_blocks_.end(); ) {
            if (it->second.root_ptr == root_ptr) {
                freed_bytes += it->second.size;
                it = global_free_blocks_.erase(it);
            } else {
                ++it;
            }
        }

        // Free the root allocation
        free_to_system(root_ptr);
        global_root_allocations_.erase(root_ptr);

        // Update stats
        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            global_stats_.cached_bytes -= freed_bytes;
        }
    }
}

void CPUCachingAllocator::set_max_cached_bytes(size_t bytes) {
    max_cached_bytes_.store(bytes);
    check_memory_pressure();
}

void CPUCachingAllocator::set_max_local_cached_bytes(size_t bytes) {
    max_local_cached_bytes_.store(bytes);
}

void CPUCachingAllocator::set_min_split_size(size_t bytes) {
    min_split_size_.store(bytes);
}

auto CPUCachingAllocator::get_stats() const -> Stats {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return global_stats_;
}

void CPUCachingAllocator::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    // Preserve current allocated/cached counts
    size_t allocated = global_stats_.allocated_bytes;
    size_t cached = global_stats_.cached_bytes;
    global_stats_ = Stats{};
    global_stats_.allocated_bytes = allocated;
    global_stats_.cached_bytes = cached;
}

auto CPUCachingAllocator::try_allocate_local(size_t bytes) -> void* {
    auto& local = get_local_pool();

    auto it = find_best_fit(local.free_blocks, bytes);
    if (it == local.free_blocks.end()) {
        return nullptr;
    }

    Block block = it->second;
    local.free_blocks.erase(it);
    local.cached_bytes -= block.size;

    // Check if we should split
    size_t min_split = min_split_size_.load();
    size_t remainder = block.size - bytes;
    bool should_split = (block.size >= 2 * bytes && remainder >= min_split);

    // Update CENTRAL root allocation - block is being taken from free pool
    {
        std::lock_guard<std::mutex> lock(global_mutex_);
        auto root_it = global_root_allocations_.find(block.root_ptr);
        if (root_it != global_root_allocations_.end()) {
            root_it->second.freed_size -= block.size;

            // If splitting, remainder goes back to free pool
            if (should_split) {
                root_it->second.fragment_count++;
                root_it->second.freed_size += remainder;
            }
        }
    }

    // Perform the actual split
    if (should_split) {
        Block remainder_block;
        remainder_block.ptr = static_cast<char*>(block.ptr) + bytes;
        remainder_block.size = remainder;
        remainder_block.allocated_size = block.allocated_size;
        remainder_block.root_ptr = block.root_ptr;
        remainder_block.is_split = true;

        local.free_blocks.insert({remainder_block.size, remainder_block});
        local.cached_bytes += remainder;

        block.size = bytes;
        block.is_split = true;

        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            global_stats_.num_splits++;
            global_stats_.cached_bytes += remainder;
        }
    }

    // Track as allocated - BOTH globally (for cross-thread dealloc) and locally (fast path)
    {
        std::lock_guard<std::mutex> glock(global_mutex_);
        global_allocated_blocks_[block.ptr] = block;
    }
    local.allocated_blocks[block.ptr] = block;
    local.allocated_bytes += block.size;

    // Update stats
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        global_stats_.total_allocations++;
        global_stats_.cache_hits++;
        global_stats_.cached_bytes -= block.size;
        global_stats_.allocated_bytes += block.size;
        global_stats_.peak_allocated_bytes = std::max(
            global_stats_.peak_allocated_bytes,
            global_stats_.allocated_bytes
        );
    }

    return block.ptr;
}

auto CPUCachingAllocator::try_allocate_global(size_t bytes) -> void* {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = find_best_fit(global_free_blocks_, bytes);
    if (it == global_free_blocks_.end()) {
        return nullptr;
    }

    Block block = it->second;
    global_free_blocks_.erase(it);

    // Check if we should split
    size_t min_split = min_split_size_.load();
    size_t remainder = block.size - bytes;
    bool should_split = (block.size >= 2 * bytes && remainder >= min_split);

    // Update CENTRAL root allocation tracking (already have the lock)
    auto root_it = global_root_allocations_.find(block.root_ptr);
    if (root_it != global_root_allocations_.end()) {
        root_it->second.freed_size -= block.size;

        if (should_split) {
            root_it->second.fragment_count++;
            root_it->second.freed_size += remainder;
        }
    }

    // Perform split if needed
    if (should_split) {
        Block remainder_block;
        remainder_block.ptr = static_cast<char*>(block.ptr) + bytes;
        remainder_block.size = remainder;
        remainder_block.allocated_size = block.allocated_size;
        remainder_block.root_ptr = block.root_ptr;
        remainder_block.is_split = true;

        global_free_blocks_.insert({remainder_block.size, remainder_block});

        block.size = bytes;
        block.is_split = true;

        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            global_stats_.num_splits++;
            global_stats_.cached_bytes += remainder;
        }
    }

    // Track as allocated - BOTH globally (for cross-thread dealloc) and locally (fast path)
    // Note: already holding global_mutex_
    global_allocated_blocks_[block.ptr] = block;

    auto& local = get_local_pool();
    local.allocated_blocks[block.ptr] = block;
    local.allocated_bytes += block.size;

    // Update stats
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        global_stats_.total_allocations++;
        global_stats_.cache_hits++;
        global_stats_.cached_bytes -= block.size;
        global_stats_.allocated_bytes += block.size;
        global_stats_.peak_allocated_bytes = std::max(
            global_stats_.peak_allocated_bytes,
            global_stats_.allocated_bytes
        );
    }

    return block.ptr;
}

auto CPUCachingAllocator::allocate_from_system(size_t bytes) -> void* {
    void* ptr = nullptr;

#ifdef _WIN32
    ptr = _aligned_malloc(bytes, ALIGNMENT);
#else
    if (posix_memalign(&ptr, ALIGNMENT, bytes) != 0) {
        ptr = nullptr;
    }
#endif

    if (ptr) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        global_stats_.num_backend_allocs++;
    }

    return ptr;
}

auto CPUCachingAllocator::find_best_fit(std::multimap<size_t, Block>& blocks,
                                         size_t bytes)
    -> std::multimap<size_t, Block>::iterator {
    // Find first block with size >= bytes (best-fit via lower_bound)
    return blocks.lower_bound(bytes);
}

auto CPUCachingAllocator::maybe_split_block(Block& block, size_t requested_size,
                                             std::multimap<size_t, Block>& free_blocks,
                                             ThreadLocalPool& local)
    -> bool {
    size_t min_split = min_split_size_.load();

    // Only split if:
    // 1. Block is at least 2x the requested size
    // 2. Remainder would be >= min_split_size
    size_t remainder = block.size - requested_size;
    if (block.size >= 2 * requested_size && remainder >= min_split) {
        // Create remainder block
        Block remainder_block;
        remainder_block.ptr = static_cast<char*>(block.ptr) + requested_size;
        remainder_block.size = remainder;
        remainder_block.allocated_size = block.allocated_size;
        remainder_block.root_ptr = block.root_ptr;
        remainder_block.is_split = true;

        // Add remainder to free pool
        free_blocks.insert({remainder_block.size, remainder_block});
        local.cached_bytes += remainder;

        // Update CENTRAL root allocation - increment fragment count, add freed_size for remainder
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            auto root_it = global_root_allocations_.find(block.root_ptr);
            if (root_it != global_root_allocations_.end()) {
                root_it->second.fragment_count++;
                root_it->second.freed_size += remainder;
            }
        }

        // Update original block
        block.size = requested_size;
        block.is_split = true;

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            global_stats_.num_splits++;
            global_stats_.cached_bytes += remainder;
        }

        return true;
    }

    return false;
}

auto CPUCachingAllocator::maybe_split_block_global(Block& block, size_t requested_size,
                                                    std::multimap<size_t, Block>& free_blocks)
    -> bool {
    size_t min_split = min_split_size_.load();

    size_t remainder = block.size - requested_size;
    if (block.size >= 2 * requested_size && remainder >= min_split) {
        Block remainder_block;
        remainder_block.ptr = static_cast<char*>(block.ptr) + requested_size;
        remainder_block.size = remainder;
        remainder_block.allocated_size = block.allocated_size;
        remainder_block.root_ptr = block.root_ptr;
        remainder_block.is_split = true;

        free_blocks.insert({remainder_block.size, remainder_block});

        // Update global root allocation
        auto root_it = global_root_allocations_.find(block.root_ptr);
        if (root_it != global_root_allocations_.end()) {
            root_it->second.fragment_count++;
            root_it->second.freed_size += remainder;
        }

        block.size = requested_size;
        block.is_split = true;

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            global_stats_.num_splits++;
            global_stats_.cached_bytes += remainder;
        }

        return true;
    }

    return false;
}

void CPUCachingAllocator::migrate_to_global(ThreadLocalPool& local) {
    std::lock_guard<std::mutex> lock(global_mutex_);

    // Move half of the cached blocks to global pool
    size_t target = local.cached_bytes / 2;
    size_t moved = 0;

    auto it = local.free_blocks.begin();
    while (it != local.free_blocks.end() && moved < target) {
        Block block = it->second;
        global_free_blocks_.insert({block.size, block});

        // With centralized root tracking, freed_size doesn't change when moving
        // between pools - the block is still in a free pool, just a different one.

        moved += block.size;
        it = local.free_blocks.erase(it);
    }

    local.cached_bytes -= moved;
}

void CPUCachingAllocator::check_memory_pressure() {
    // Check if total cached bytes exceeds limit
    size_t total_cached = 0;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        total_cached = global_stats_.cached_bytes;
    }

    if (total_cached <= max_cached_bytes_.load()) {
        return;  // Under limit, nothing to do
    }

    // We can only safely free roots if ALL their blocks are in the global pool.
    // If any blocks are in thread-local pools, we cannot free the root here
    // because other threads might still reference those blocks.
    //
    // To determine this, we check if the blocks in global_free_blocks_ for a root
    // account for its entire freed_size. If so, all fragments are in global pool.

    std::vector<void*> roots_to_free;
    std::lock_guard<std::mutex> lock(global_mutex_);

    // First, compute how much of each root's freed_size is in global pool
    std::unordered_map<void*, size_t> global_freed_per_root;
    for (auto& [size, block] : global_free_blocks_) {
        global_freed_per_root[block.root_ptr] += block.size;
    }

    // Find roots that are fully coalesced AND have all fragments in global pool
    for (auto& [root_ptr, root] : global_root_allocations_) {
        if (is_fully_coalesced(root)) {
            // Check if all freed blocks are in global pool
            auto git = global_freed_per_root.find(root_ptr);
            size_t global_freed = (git != global_freed_per_root.end()) ? git->second : 0;
            if (global_freed == root.freed_size) {
                // All blocks are in global pool, safe to free
                roots_to_free.push_back(root_ptr);
            }
        }
    }

    if (roots_to_free.empty()) {
        return;  // No roots that can be safely freed
    }

    // Free roots and remove their blocks from global free pool
    for (void* root_ptr : roots_to_free) {
        size_t freed_bytes = 0;

        // Remove blocks from global free pool
        for (auto it = global_free_blocks_.begin(); it != global_free_blocks_.end(); ) {
            if (it->second.root_ptr == root_ptr) {
                freed_bytes += it->second.size;
                it = global_free_blocks_.erase(it);
            } else {
                ++it;
            }
        }

        // Free the root allocation
        free_to_system(root_ptr);
        global_root_allocations_.erase(root_ptr);

        // Update stats
        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            global_stats_.cached_bytes -= freed_bytes;
        }

        // Check if we've freed enough
        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            if (global_stats_.cached_bytes <= max_cached_bytes_.load()) {
                break;
            }
        }
    }
}

void CPUCachingAllocator::free_to_system(void* ptr) {
    if (!ptr) {
        return;
    }

#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        global_stats_.num_backend_frees++;
    }
}

bool CPUCachingAllocator::try_coalesce_and_free(Block& block, ThreadLocalPool& /* local */) {
    // Use centralized root tracking - caller should already hold global_mutex_
    auto root_it = global_root_allocations_.find(block.root_ptr);
    if (root_it == global_root_allocations_.end()) {
        return false;
    }

    return is_fully_coalesced(root_it->second);
}

} // namespace cpu
} // namespace tenzor
