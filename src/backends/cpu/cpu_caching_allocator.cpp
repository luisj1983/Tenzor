/**
 * @file cpu_caching_allocator.cpp
 * @brief Implementation of thread-local caching allocator for CPU tensors
 */

#include "tenzor/backend/cpu_caching_allocator.hpp"
#include "tenzor/backend/loader.hpp"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <thread>
#include <sstream>
#include <iomanip>

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
        // Mark as invalid so subsequent deallocate() calls on this thread skip
        // touching the now-torn-down thread_local storage.
        valid = false;

        // Erase any pending-decrement entries queued under this thread's tid by
        // other threads' cross-thread frees.  Once this thread exits its local
        // counters are never consulted again, so the entries are stale bookkeeping.
        // The memory itself was already returned to the global pool via the
        // cross-thread free path, so discarding the pending list is safe.
        //
        // Without this erase, per_thread_pending_frees_ accumulates one dead
        // entry per exited producer thread and grows without bound.
        //
        // Guard against process-exit static-destruction order: if the backend
        // registry is already gone the singleton may be partially destroyed, so
        // skip the erase (the OS reclaims everything at process exit anyway).
        if (is_backend_registry_alive()) {
            CPUCachingAllocator::instance().erase_pending_for_current_thread();
        }
    }
};

static thread_local ThreadLocalPoolWrapper tl_pool_wrapper_;

// ---------------------------------------------------------------------------
// Split-sibling coalescing helpers (Task 7.1)
// Defined before deallocate() which calls them.
// ---------------------------------------------------------------------------

static void merge_adjacent_in_map(
    std::multimap<size_t, CPUCachingAllocator::Block>& free_map,
    CPUCachingAllocator::Block& block,
    CPUCachingAllocator::RootAllocation& root,
    size_t& /*stats_cached_delta*/  // no net change in cached bytes during merging
) {
    bool merged = true;
    while (merged) {
        merged = false;
        for (auto it = free_map.begin(); it != free_map.end(); ++it) {
            auto& candidate = it->second;
            if (candidate.root_ptr != block.root_ptr) {
                continue;
            }
            char* cand_start = static_cast<char*>(candidate.ptr);
            char* cand_end   = cand_start + candidate.size;
            char* blk_start  = static_cast<char*>(block.ptr);
            char* blk_end    = blk_start  + block.size;

            bool left_adjacent  = (cand_end  == blk_start);
            bool right_adjacent = (blk_end   == cand_start);
            if (!left_adjacent && !right_adjacent) {
                continue;
            }

            size_t new_size = block.size + candidate.size;
            if (left_adjacent) {
                block.ptr = candidate.ptr;
            }
            block.size        = new_size;
            block.allocated_size = root.size;
            block.is_split    = (new_size < root.size);

            free_map.erase(it);
            root.fragment_count = std::max(1, root.fragment_count - 1);
            merged = true;
            break;
        }
    }
}

// Called from deallocate() after the block has been placed in the local free
// pool.  Merges adjacent siblings; if the root is fully coalesced, moves the
// merged block to the global pool so check_memory_pressure() can return it.
// Must NOT hold global_mutex_ on entry.
static void coalesce_local_block(
    CPUCachingAllocator::Block& block,
    CPUCachingAllocator::ThreadLocalPool& local,
    std::multimap<size_t, CPUCachingAllocator::Block>& global_free,
    std::unordered_map<void*, CPUCachingAllocator::RootAllocation>& root_map,
    std::mutex& global_mutex
) {
    // Remove freshly-freed block from local pool so we can work on it.
    bool found = false;
    for (auto it = local.free_blocks.begin(); it != local.free_blocks.end(); ++it) {
        if (it->second.ptr == block.ptr) {
            local.free_blocks.erase(it);
            found = true;
            break;
        }
    }
    if (!found) {
        return;
    }

    // Merge adjacent siblings in the local pool.
    {
        std::lock_guard<std::mutex> lock(global_mutex);
        auto root_it = root_map.find(block.root_ptr);
        if (root_it == root_map.end()) {
            local.free_blocks.insert({block.size, block});
            return;
        }
        size_t delta = 0;
        merge_adjacent_in_map(local.free_blocks, block, root_it->second, delta);

        // If fully coalesced, migrate to global pool for OS-return under pressure.
        if (block.size == root_it->second.size &&
            root_it->second.freed_size == root_it->second.size) {
            local.cached_bytes -= block.size;
            global_free.insert({block.size, block});
            return;
        }
    }

    // Not fully coalesced: put the (possibly enlarged) block back.
    local.free_blocks.insert({block.size, block});
}

// ---------------------------------------------------------------------------
// fmt_bytes helper for memory_summary()
// ---------------------------------------------------------------------------
static std::string fmt_bytes(size_t bytes) {
    std::ostringstream oss;
    if (bytes >= (1ULL << 30)) {
        oss << std::fixed << std::setprecision(2) << (double)bytes / (1ULL << 30) << " GiB";
    } else if (bytes >= (1ULL << 20)) {
        oss << std::fixed << std::setprecision(2) << (double)bytes / (1ULL << 20) << " MiB";
    } else if (bytes >= (1ULL << 10)) {
        oss << std::fixed << std::setprecision(2) << (double)bytes / (1ULL << 10) << " KiB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

auto CPUCachingAllocator::instance() -> CPUCachingAllocator& {
    static CPUCachingAllocator instance;
    return instance;
}

CPUCachingAllocator::CPUCachingAllocator() = default;

CPUCachingAllocator::~CPUCachingAllocator() {
    // If the backend registry has already been shut down (finalize() ran),
    // skip freeing — the OS reclaims all memory at process exit anyway.
    // This prevents use-after-free if this static outlives other statics
    // during unordered static destruction.
    if (!is_backend_registry_alive()) {
        return;
    }

    // During normal shutdown, free all roots.
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

    // Drain any cross-thread pending decrements before touching local state.
    if (tl_pool_wrapper_.valid) {
        drain_pending_decrements(get_local_pool());
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
    block.originating_tid = std::this_thread::get_id();

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

    // Shutdown-safety: if the thread-local pool wrapper has already been
    // destroyed (its dtor sets `valid=false`), avoid touching its
    // unordered_map — the storage has been torn down and dereferencing it
    // is UB. The OS reclaims any leaked bytes at process exit, and tensors
    // that outlive the pool are only hit during static-destructor teardown
    // (e.g. parametrization_registry dropping saved tensors on exit).
    if (!tl_pool_wrapper_.valid) {
        return;
    }

    auto& local = get_local_pool();

    // Drain any cross-thread pending decrements before touching local state.
    drain_pending_decrements(local);

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

        // Attempt split-sibling coalescing (Task 7.1).
        // Only bother when the block originated from a split — unsplit blocks
        // have no siblings to merge with.
        if (block.is_split) {
            coalesce_local_block(block, local,
                                 global_free_blocks_,
                                 global_root_allocations_,
                                 global_mutex_);
        }

        // Check if local pool is too large
        if (local.cached_bytes > max_local_cached_bytes_.load()) {
            migrate_to_global(local);
        }

        // Check global memory pressure
        check_memory_pressure();
        return;
    }

    // Check global allocated blocks (cross-thread dealloc path)
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

            // Add to global free pool, then attempt sibling coalescing.
            if (block.is_split) {
                auto root_it2 = global_root_allocations_.find(block.root_ptr);
                if (root_it2 != global_root_allocations_.end()) {
                    size_t delta = 0;
                    merge_adjacent_in_map(global_free_blocks_, block, root_it2->second, delta);
                }
            }
            global_free_blocks_.insert({block.size, block});

            // Update stats
            {
                std::lock_guard<std::mutex> slock(stats_mutex_);
                global_stats_.allocated_bytes -= block.size;
                global_stats_.cached_bytes += block.size;
            }

            // Audit P0 #8: queue a pending decrement for the originating
            // thread so its local.allocated_bytes and local.allocated_blocks
            // stay in sync.  The originating thread drains this on its next
            // allocator call via drain_pending_decrements().
            if (block.originating_tid != std::thread::id{}) {
                per_thread_pending_frees_[block.originating_tid].push_back(ptr);
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
    block.originating_tid = std::this_thread::get_id();
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
    block.originating_tid = std::this_thread::get_id();
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
    // IMPORTANT: We also verify that no blocks are currently allocated from the root.
    // A root can only be freed if freed_size == root.size (all blocks returned to pools).

    std::lock_guard<std::mutex> lock(global_mutex_);

    // First, compute how much of each root's size is in global FREE pool
    std::unordered_map<void*, size_t> global_freed_per_root;
    for (auto& [size, block] : global_free_blocks_) {
        global_freed_per_root[block.root_ptr] += block.size;
    }

    // Also check if any blocks from each root are currently ALLOCATED
    std::unordered_set<void*> roots_with_allocated_blocks;
    for (auto& [ptr, block] : global_allocated_blocks_) {
        roots_with_allocated_blocks.insert(block.root_ptr);
    }

    // Find roots that:
    // 1. Are fully coalesced (freed_size == root.size)
    // 2. Have ALL freed blocks in global pool (global_freed == freed_size)
    // 3. Have NO currently allocated blocks
    std::vector<void*> roots_to_free;
    for (auto& [root_ptr, root] : global_root_allocations_) {
        // Skip if any blocks are currently allocated
        if (roots_with_allocated_blocks.count(root_ptr) > 0) {
            continue;
        }

        // Check if fully coalesced
        if (!is_fully_coalesced(root)) {
            continue;
        }

        // Check if all freed blocks are in global pool
        auto git = global_freed_per_root.find(root_ptr);
        size_t global_freed = (git != global_freed_per_root.end()) ? git->second : 0;

        // CRITICAL: Only free if global_freed == root.size (not just freed_size)
        // This ensures we account for the full allocation, not just tracked freed bytes
        if (global_freed == root.size) {
            roots_to_free.push_back(root_ptr);
        }
    }

    if (roots_to_free.empty()) {
        return;  // No roots that can be safely freed
    }

    // Free roots and remove their blocks from global free pool
    for (void* root_ptr : roots_to_free) {
        // Recheck: ensure root still exists and has no allocated blocks
        auto root_it = global_root_allocations_.find(root_ptr);
        if (root_it == global_root_allocations_.end()) {
            continue;  // Root was removed by another thread
        }

        // Double-check no allocated blocks reference this root
        bool has_allocated = false;
        for (auto& [ptr, block] : global_allocated_blocks_) {
            if (block.root_ptr == root_ptr) {
                has_allocated = true;
                break;
            }
        }
        if (has_allocated) {
            continue;  // Safety: skip if any blocks are now allocated
        }

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

        // CRITICAL: Verify freed_bytes matches root size before freeing
        // If they don't match, there's a tracking bug - skip this root
        if (freed_bytes != root_it->second.size) {
            // Tracking inconsistency detected - don't free, just continue
            // The blocks have been removed from free pool but we won't free the root
            // This leaks memory but avoids corruption
            continue;
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

void CPUCachingAllocator::drain_pending_decrements(ThreadLocalPool& local) {
    auto tid = std::this_thread::get_id();

    // Steal the pending-frees vector under the lock, then process it without
    // holding the lock (thread-local state must not be modified under
    // global_mutex_ because allocate/deallocate acquire global_mutex_ while
    // local state is active, and taking the lock a second time would deadlock
    // on a non-recursive mutex).
    std::vector<void*> pending;
    {
        std::lock_guard<std::mutex> lock(global_mutex_);
        auto it = per_thread_pending_frees_.find(tid);
        if (it == per_thread_pending_frees_.end() || it->second.empty()) {
            return;
        }
        pending = std::move(it->second);
        per_thread_pending_frees_.erase(it);
    } // global_mutex_ released here

    // Reconcile local state: remove stale entries left by cross-thread frees.
    for (void* p : pending) {
        auto local_it = local.allocated_blocks.find(p);
        if (local_it != local.allocated_blocks.end()) {
            local.allocated_bytes -= local_it->second.size;
            local.allocated_blocks.erase(local_it);
        }
    }
}

void CPUCachingAllocator::erase_pending_for_current_thread() {
    std::lock_guard<std::mutex> lock(global_mutex_);
    per_thread_pending_frees_.erase(std::this_thread::get_id());
}

auto CPUCachingAllocator::get_local_stats() -> LocalStats {
    if (!tl_pool_wrapper_.valid) {
        return {};
    }
    auto& local = get_local_pool();
    drain_pending_decrements(local);
    return {local.allocated_bytes, local.cached_bytes};
}

bool CPUCachingAllocator::try_coalesce_and_free(Block& block, ThreadLocalPool& /* local */) {
    // Use centralized root tracking - caller should already hold global_mutex_
    auto root_it = global_root_allocations_.find(block.root_ptr);
    if (root_it == global_root_allocations_.end()) {
        return false;
    }

    return is_fully_coalesced(root_it->second);
}

// ---------------------------------------------------------------------------
// memory_summary() — Task 7.2
// ---------------------------------------------------------------------------

auto CPUCachingAllocator::memory_summary() -> std::string {
    Stats s = get_stats();
    LocalStats ls = get_local_stats();

    std::ostringstream out;
    out << "=== CPUCachingAllocator memory summary ===\n";
    out << "  allocated:        " << fmt_bytes(s.allocated_bytes)
        << "  (peak: " << fmt_bytes(s.peak_allocated_bytes) << ")\n";
    out << "  cached:           " << fmt_bytes(s.cached_bytes) << "\n";
    out << "  total allocations:" << std::setw(10) << s.total_allocations << "\n";
    out << "  cache hits:       " << std::setw(10) << s.cache_hits << "\n";
    out << "  backend allocs:   " << std::setw(10) << s.num_backend_allocs << "\n";
    out << "  backend frees:    " << std::setw(10) << s.num_backend_frees << "\n";
    out << "  splits:           " << std::setw(10) << s.num_splits << "\n";
    out << "--- calling thread (local pool) ---\n";
    out << "  thread allocated: " << fmt_bytes(ls.allocated_bytes) << "\n";
    out << "  thread cached:    " << fmt_bytes(ls.cached_bytes) << "\n";
    out << "==========================================\n";
    return out.str();
}

} // namespace cpu
} // namespace tenzor
