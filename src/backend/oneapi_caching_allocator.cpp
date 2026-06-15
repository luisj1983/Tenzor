#include "tenzor/backend/oneapi_caching_allocator.hpp"
#include "tenzor/backend/loader.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace tenzor {
namespace backend {

// Default configuration
constexpr size_t DEFAULT_ALIGNMENT = 64;  // Optimal for Intel GPUs
constexpr size_t DEFAULT_MIN_SPLIT_SIZE = 512;
constexpr size_t DEFAULT_LARGE_ALLOCATION_THRESHOLD = 2ULL * 1024 * 1024 * 1024;  // 2GB

OneAPICachingAllocator::OneAPICachingAllocator()
    : alignment_(DEFAULT_ALIGNMENT),
      max_cached_memory_(0),  // Unlimited by default
      min_split_size_(DEFAULT_MIN_SPLIT_SIZE),
      large_allocation_threshold_(DEFAULT_LARGE_ALLOCATION_THRESHOLD),
      merge_enabled_(true) {
}

OneAPICachingAllocator::~OneAPICachingAllocator() {
    // During static destruction the SYCL runtime may already be torn down,
    // making USM free calls crash. Skip cleanup if the backend registry
    // is already shut down — the OS reclaims all memory at exit anyway.
    if (!is_backend_registry_alive()) {
        return;
    }
    // Release all cached memory
    empty_cache(-1);
}

OneAPICachingAllocator& OneAPICachingAllocator::get() {
    static OneAPICachingAllocator instance;
    return instance;
}

void OneAPICachingAllocator::initialize(void* queue, int device_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& device_alloc = device_allocators_[device_index];
    if (device_alloc.initialized) {
        return;  // Already initialized
    }

    device_alloc.queue = queue;  // Store as void*, cast when using
    device_alloc.initialized = true;
}

bool OneAPICachingAllocator::is_initialized(int device_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_allocators_.find(device_index);
    return it != device_allocators_.end() && it->second.initialized;
}

void* OneAPICachingAllocator::allocate_shared(size_t size, int device) {
    return allocate_impl(size, device, true);
}

void* OneAPICachingAllocator::allocate_device(size_t size, int device) {
    return allocate_impl(size, device, false);
}

void* OneAPICachingAllocator::allocate_impl(size_t size, int device, bool shared) {
    if (size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_allocators_.find(device);
    if (it == device_allocators_.end() || !it->second.initialized) {
        throw std::runtime_error("OneAPICachingAllocator: Device " + std::to_string(device) +
                                 " not initialized. Call initialize() first.");
    }

    // Round size to alignment
    size = round_size(size);

    auto& device_alloc = device_allocators_[device];
    device_alloc.stats.num_allocations++;

    // Try to find a suitable block in cache
    OneAPIBlock* block = try_allocate_from_cache(size, device, shared);

    bool from_cache = (block != nullptr);

    if (!block) {
        // No suitable cached block, allocate new one
        block = allocate_new_block(size, device, shared);
    } else {
        device_alloc.stats.num_cache_hits++;
    }

    // Mark block as allocated
    block->allocated = true;

    // Update statistics
    device_alloc.stats.allocated_bytes += block->size;
    if (device_alloc.stats.allocated_bytes > device_alloc.stats.peak_allocated) {
        device_alloc.stats.peak_allocated = device_alloc.stats.allocated_bytes;
    }

    if (shared) {
        device_alloc.stats.shared_memory_bytes += block->size;
    } else {
        device_alloc.stats.device_memory_bytes += block->size;
    }

    // Only subtract from cached_bytes if we got the block from cache
    if (from_cache && device_alloc.stats.cached_bytes >= block->size) {
        device_alloc.stats.cached_bytes -= block->size;
    }

    return block->ptr;
}

// Wait for (and discard) a block's release fence. Must be called before the
// block's memory is recycled to a new owner or returned to the runtime.
static void wait_and_clear_release_fence(OneAPIBlock* block) {
    if (block->release_fence) {
        auto* ev = static_cast<sycl::event*>(block->release_fence);
        try { ev->wait(); } catch (...) { /* device teardown */ }
        delete ev;
        block->release_fence = nullptr;
    }
}

void OneAPICachingAllocator::free(void* ptr, int device) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto& device_alloc = device_allocators_[device];
    device_alloc.stats.num_frees++;

    // Find the block
    auto it = device_alloc.all_blocks.find(ptr);
    if (it == device_alloc.all_blocks.end()) {
        throw std::runtime_error("OneAPICachingAllocator: Attempted to free pointer not allocated by this allocator");
    }

    OneAPIBlock* block = it->second.get();
    if (!block->allocated) {
        throw std::runtime_error("OneAPICachingAllocator: Attempted to free already freed pointer");
    }

    // Mark as free
    block->allocated = false;

    // Fence the free: operations already enqueued on the (in-order) device
    // queue may still touch this block. Attach a barrier event; reuse and
    // sycl::free both wait on it (see OneAPIBlock::release_fence).
    if (sycl::queue* q = static_cast<sycl::queue*>(device_alloc.queue)) {
        try {
            wait_and_clear_release_fence(block);  // none expected, be safe
            block->release_fence = new sycl::event(q->ext_oneapi_submit_barrier());
        } catch (...) {
            // Queue unusable (teardown) — no pending work to fence.
        }
    }

    // Update statistics
    if (device_alloc.stats.allocated_bytes >= block->size) {
        device_alloc.stats.allocated_bytes -= block->size;
    }

    if (block->is_shared) {
        if (device_alloc.stats.shared_memory_bytes >= block->size) {
            device_alloc.stats.shared_memory_bytes -= block->size;
        }
    } else {
        if (device_alloc.stats.device_memory_bytes >= block->size) {
            device_alloc.stats.device_memory_bytes -= block->size;
        }
    }

    // For very large allocations, release immediately rather than caching
    if (block->size >= large_allocation_threshold_) {
        release_block(block);
        return;
    }

    device_alloc.stats.cached_bytes += block->size;

    // Try to merge with adjacent blocks
    if (merge_enabled_) {
        try_merge_blocks(block);
    }

    // Add to appropriate free blocks set
    if (block->is_shared) {
        device_alloc.free_shared_blocks.insert(block);
    } else {
        device_alloc.free_device_blocks.insert(block);
    }

    // Enforce cache limit if set
    if (max_cached_memory_ > 0) {
        enforce_cache_limit(device);
    }
}

void OneAPICachingAllocator::empty_cache(int device) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        // Empty all devices
        for (auto& pair : device_allocators_) {
            auto& device_alloc = pair.second;
            if (!device_alloc.initialized) continue;

            // Release all free blocks
            std::vector<OneAPIBlock*> blocks_to_release;
            for (OneAPIBlock* block : device_alloc.free_shared_blocks) {
                blocks_to_release.push_back(block);
            }
            for (OneAPIBlock* block : device_alloc.free_device_blocks) {
                blocks_to_release.push_back(block);
            }

            for (OneAPIBlock* block : blocks_to_release) {
                release_block(block);
            }
        }
    } else {
        // Empty specific device
        auto it = device_allocators_.find(device);
        if (it == device_allocators_.end() || !it->second.initialized) {
            return;
        }

        auto& device_alloc = it->second;

        std::vector<OneAPIBlock*> blocks_to_release;
        for (OneAPIBlock* block : device_alloc.free_shared_blocks) {
            blocks_to_release.push_back(block);
        }
        for (OneAPIBlock* block : device_alloc.free_device_blocks) {
            blocks_to_release.push_back(block);
        }

        for (OneAPIBlock* block : blocks_to_release) {
            release_block(block);
        }
    }
}

void OneAPICachingAllocator::release_all() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [dev_id, device_alloc] : device_allocators_) {
        if (!device_alloc.initialized) continue;
        sycl::queue* queue = static_cast<sycl::queue*>(device_alloc.queue);

        // Free every tracked USM pointer (cached and still-allocated).
        for (auto& [ptr, block] : device_alloc.all_blocks) {
            if (ptr) {
                // Wait on and reclaim the heap-allocated release_fence event so
                // pending work is fenced before sycl::free and the event isn't
                // leaked (release_block() does this; release_all() must too).
                wait_and_clear_release_fence(block.get());
                try {
                    sycl::free(ptr, *queue);
                } catch (const sycl::exception&) {
                    // Swallow — cleanup context
                }
            }
        }

        device_alloc.all_blocks.clear();
        device_alloc.free_shared_blocks.clear();
        device_alloc.free_device_blocks.clear();
        device_alloc.stats = OneAPIMemoryStats{};
    }
}

void OneAPICachingAllocator::garbage_collect(int device, bool aggressive) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto do_gc = [&](DeviceAllocator& device_alloc) {
        if (!device_alloc.initialized) return;

        if (aggressive) {
            // Release all cached blocks
            std::vector<OneAPIBlock*> blocks_to_release;
            for (OneAPIBlock* block : device_alloc.free_shared_blocks) {
                blocks_to_release.push_back(block);
            }
            for (OneAPIBlock* block : device_alloc.free_device_blocks) {
                blocks_to_release.push_back(block);
            }
            for (OneAPIBlock* block : blocks_to_release) {
                release_block(block);
            }
        } else {
            // Only release large blocks (> 1MB)
            constexpr size_t LARGE_BLOCK_THRESHOLD = 1024 * 1024;

            std::vector<OneAPIBlock*> blocks_to_release;
            for (OneAPIBlock* block : device_alloc.free_shared_blocks) {
                if (block->size >= LARGE_BLOCK_THRESHOLD) {
                    blocks_to_release.push_back(block);
                }
            }
            for (OneAPIBlock* block : device_alloc.free_device_blocks) {
                if (block->size >= LARGE_BLOCK_THRESHOLD) {
                    blocks_to_release.push_back(block);
                }
            }
            for (OneAPIBlock* block : blocks_to_release) {
                release_block(block);
            }
        }
    };

    if (device == -1) {
        for (auto& pair : device_allocators_) {
            do_gc(pair.second);
        }
    } else {
        auto it = device_allocators_.find(device);
        if (it != device_allocators_.end()) {
            do_gc(it->second);
        }
    }
}

size_t OneAPICachingAllocator::memory_allocated(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            total += pair.second.stats.allocated_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats.allocated_bytes;
    }
    return 0;
}

size_t OneAPICachingAllocator::memory_reserved(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            total += pair.second.stats.reserved_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats.reserved_bytes;
    }
    return 0;
}

size_t OneAPICachingAllocator::memory_cached(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            total += pair.second.stats.cached_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats.cached_bytes;
    }
    return 0;
}

OneAPIMemoryStats OneAPICachingAllocator::get_stats(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        OneAPIMemoryStats total;
        for (const auto& pair : device_allocators_) {
            const auto& stats = pair.second.stats;
            total.allocated_bytes += stats.allocated_bytes;
            total.reserved_bytes += stats.reserved_bytes;
            total.cached_bytes += stats.cached_bytes;
            total.num_allocations += stats.num_allocations;
            total.num_frees += stats.num_frees;
            total.num_cache_hits += stats.num_cache_hits;
            total.num_splits += stats.num_splits;
            total.num_merges += stats.num_merges;
            total.shared_memory_bytes += stats.shared_memory_bytes;
            total.device_memory_bytes += stats.device_memory_bytes;
            if (stats.peak_allocated > total.peak_allocated) {
                total.peak_allocated = stats.peak_allocated;
            }
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats;
    }
    return OneAPIMemoryStats();
}

void OneAPICachingAllocator::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : device_allocators_) {
        auto& stats = pair.second.stats;
        stats.num_allocations = 0;
        stats.num_frees = 0;
        stats.num_cache_hits = 0;
        stats.num_splits = 0;
        stats.num_merges = 0;
    }
}

void OneAPICachingAllocator::set_alignment(size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    alignment_ = alignment;
}

void OneAPICachingAllocator::set_max_cached_memory(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_cached_memory_ = max_bytes;
}

void OneAPICachingAllocator::set_min_split_size(size_t min_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_split_size_ = min_size;
}

void OneAPICachingAllocator::set_large_allocation_threshold(size_t threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    large_allocation_threshold_ = threshold;
}

OneAPIBlock* OneAPICachingAllocator::try_allocate_from_cache(size_t size, int device, bool shared) {
    auto& device_alloc = device_allocators_[device];

    // Select appropriate free blocks set
    auto& free_blocks = shared ? device_alloc.free_shared_blocks : device_alloc.free_device_blocks;

    // Find smallest free block that fits (best-fit)
    OneAPIBlock search_block(nullptr, size, device, shared);
    auto it = free_blocks.lower_bound(&search_block);

    if (it != free_blocks.end()) {
        OneAPIBlock* block = *it;

        // The previous owner's enqueued work must be complete before this
        // memory gets a new owner (see OneAPIBlock::release_fence). Use a
        // NON-blocking completeness check: waiting here would stall every
        // cached-block reuse on the whole in-order queue (measured: DeepLab
        // GradientFlow went from seconds to a 1200 s timeout with a blocking
        // wait). If the fence hasn't signalled, leave the block cached and
        // let the caller allocate fresh memory instead.
        if (block->release_fence) {
            auto* ev = static_cast<sycl::event*>(block->release_fence);
            bool complete = false;
            try {
                complete = ev->get_info<sycl::info::event::command_execution_status>() ==
                           sycl::info::event_command_status::complete;
            } catch (...) {
                complete = true;  // device teardown — nothing left in flight
            }
            if (!complete) {
                return nullptr;  // still in flight: skip reuse, keep cached
            }
            delete ev;
            block->release_fence = nullptr;
        }

        // Remove from free blocks
        free_blocks.erase(it);

        // Try to split if block is too large
        if (block->size >= size + min_split_size_) {
            split_block(block, size);
        }

        return block;
    }

    return nullptr;
}

OneAPIBlock* OneAPICachingAllocator::allocate_new_block(size_t size, int device, bool shared) {
    auto& device_alloc = device_allocators_[device];
    sycl::queue* queue = static_cast<sycl::queue*>(device_alloc.queue);

    void* ptr = nullptr;
    try {
        if (shared) {
            ptr = sycl::malloc_shared(size, *queue);
        } else {
            ptr = sycl::malloc_device(size, *queue);
        }
    } catch (const sycl::exception& e) {
        throw std::runtime_error("OneAPICachingAllocator: SYCL allocation failed: " +
                                 std::string(e.what()));
    }

    if (ptr == nullptr) {
        throw std::runtime_error("OneAPICachingAllocator: SYCL allocation returned nullptr");
    }

    // Create block
    auto block = std::make_unique<OneAPIBlock>(ptr, size, device, shared);
    OneAPIBlock* block_ptr = block.get();

    // Add to all_blocks
    device_alloc.all_blocks[ptr] = std::move(block);

    // Update statistics
    device_alloc.stats.reserved_bytes += size;

    return block_ptr;
}

bool OneAPICachingAllocator::split_block(OneAPIBlock* block, size_t size) {
    // USM allocations cannot be sub-divided — each sycl::malloc_* call returns
    // an independent allocation. The previous implementation allocated NEW memory
    // for the remainder, which defeats the purpose of caching. Instead, we simply
    // return the oversized block. ML workloads have stable allocation patterns,
    // so fragmentation from oversized blocks is minimal.
    (void)block;
    (void)size;
    return false;
}

bool OneAPICachingAllocator::try_merge_blocks(OneAPIBlock* block) {
    // Note: USM doesn't support merging like contiguous device memory,
    // so this is a no-op for now. Each USM allocation is independent.
    // We keep this function for API compatibility and potential future optimization.
    (void)block;  // Suppress unused parameter warning
    return false;
}

size_t OneAPICachingAllocator::round_size(size_t size) const {
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void OneAPICachingAllocator::enforce_cache_limit(int device) {
    auto& device_alloc = device_allocators_[device];

    // Helper to release from a free block set
    auto release_from_set = [&](std::set<OneAPIBlock*, OneAPIBlockComparator>& free_set) {
        while (device_alloc.stats.cached_bytes > max_cached_memory_ && !free_set.empty()) {
            // Release largest block first
            auto it = free_set.rbegin();
            OneAPIBlock* block = *it;

            // Convert reverse iterator to forward iterator for erase
            auto forward_it = std::next(it).base();
            free_set.erase(forward_it);

            release_block(block);
        }
    };

    // Release from both pools
    release_from_set(device_alloc.free_shared_blocks);
    release_from_set(device_alloc.free_device_blocks);
}

void OneAPICachingAllocator::release_block(OneAPIBlock* block) {
    auto& device_alloc = device_allocators_[block->device];
    sycl::queue* queue = static_cast<sycl::queue*>(device_alloc.queue);

    // Remove from free blocks if present
    if (block->is_shared) {
        device_alloc.free_shared_blocks.erase(block);
    } else {
        device_alloc.free_device_blocks.erase(block);
    }

    // Free SYCL memory — only after any work enqueued before the free has
    // completed (sycl::free with in-flight operations corrupts the Intel
    // runtime's internal allocator).
    if (block->ptr) {
        wait_and_clear_release_fence(block);
        try {
            sycl::free(block->ptr, *queue);
        } catch (const sycl::exception&) {
            // Log error but don't throw in cleanup context
        }
    }

    // Update statistics
    if (device_alloc.stats.reserved_bytes >= block->size) {
        device_alloc.stats.reserved_bytes -= block->size;
    }
    if (device_alloc.stats.cached_bytes >= block->size) {
        device_alloc.stats.cached_bytes -= block->size;
    }

    // Remove from all_blocks
    device_alloc.all_blocks.erase(block->ptr);
}

} // namespace backend
} // namespace tenzor
