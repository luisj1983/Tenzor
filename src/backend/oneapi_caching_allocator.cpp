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

    // Re-initialization after a release_all() must clear the released flag,
    // otherwise free()/allocate() stay permanently disabled. Reset before the
    // already-initialized early-return so it runs on every initialize() call.
    released_ = false;

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

    // First undo the cached_bytes contribution recorded when this block was
    // freed — that was the PREVIOUS owner's requested_size, still on the block.
    if (from_cache && device_alloc.stats.cached_bytes >= block->requested_size) {
        device_alloc.stats.cached_bytes -= block->requested_size;
    }

    // Account statistics against the rounded user request, NOT the physical
    // block size: an oversized cached block (split_block is a no-op for USM)
    // would otherwise inflate allocated/peak/cached bytes (e.g. a 64-byte
    // request served from a 1 GB cached block counting as 1 GB).
    block->requested_size = size;

    // Update statistics
    device_alloc.stats.allocated_bytes += block->requested_size;
    if (device_alloc.stats.allocated_bytes > device_alloc.stats.peak_allocated) {
        device_alloc.stats.peak_allocated = device_alloc.stats.allocated_bytes;
    }

    if (shared) {
        device_alloc.stats.shared_memory_bytes += block->requested_size;
    } else {
        device_alloc.stats.device_memory_bytes += block->requested_size;
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
        // After release_all() (backend shutdown) every block has already been
        // sycl::free'd and forgotten. A Storage that outlives the backend will
        // still call free() on its now-stale pointer; throwing here would
        // escape a noexcept Storage destructor and call std::terminate(). The
        // memory is already gone, so treat the late free as a no-op.
        if (released_) {
            return;
        }
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

    // Update statistics against the owner's requested_size (what allocate_impl
    // added), not the physical block size — keeps the counters balanced for
    // oversized cached blocks.
    if (device_alloc.stats.allocated_bytes >= block->requested_size) {
        device_alloc.stats.allocated_bytes -= block->requested_size;
    }

    if (block->is_shared) {
        if (device_alloc.stats.shared_memory_bytes >= block->requested_size) {
            device_alloc.stats.shared_memory_bytes -= block->requested_size;
        }
    } else {
        if (device_alloc.stats.device_memory_bytes >= block->requested_size) {
            device_alloc.stats.device_memory_bytes -= block->requested_size;
        }
    }

    // For very large allocations, release immediately rather than caching.
    // (Threshold compares the PHYSICAL block size — this is about reclaiming
    // physical USM, independent of how much the user requested.) A block this
    // large is virtually never an interior split remainder, but if it is (an
    // interior sub-range of an even bigger original allocation still sharing
    // live siblings), release_block declines rather than freeing it — fall
    // through to the normal caching path below instead of stranding it
    // untracked by any free set.
    if (block->size >= large_allocation_threshold_ && release_block(block)) {
        return;
    }

    // Cache the block. Record its requested_size as the cached contribution so
    // allocate_impl can undo exactly this amount on reuse.
    device_alloc.stats.cached_bytes += block->requested_size;

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

        // Drain every block's release fence first — including interior split
        // remainders, which can carry their own outstanding fence from a prior
        // free() even though only the root block gets sycl::free'd below. An
        // allocation must not be freed while ANY of its sub-blocks still has
        // enqueued work in flight against its sub-range.
        for (auto& [ptr, block] : device_alloc.all_blocks) {
            wait_and_clear_release_fence(block.get());
        }

        // Free every ROOT USM allocation (cached and still-allocated). A split
        // remainder block has ptr == original_ptr + offset — sycl::free requires
        // the exact pointer sycl::malloc_* returned, so only blocks with
        // ptr == original_ptr may be freed here; each such block covers the
        // entire underlying allocation (including any interior sibling
        // sub-blocks), so freeing just the root is correct and complete.
        for (auto& [ptr, block] : device_alloc.all_blocks) {
            if (ptr && block->ptr == block->original_ptr) {
                try {
                    sycl::free(ptr, *queue);
                } catch (const sycl::exception&) {
                    // Swallow — cleanup context
                }
            }
        }

        device_alloc.all_blocks.clear();
        device_alloc.blocks_by_addr.clear();
        device_alloc.free_shared_blocks.clear();
        device_alloc.free_device_blocks.clear();
        device_alloc.stats = OneAPIMemoryStats{};
    }

    // Mark released: subsequent free() calls for now-forgotten pointers become
    // no-ops instead of throwing out of (noexcept) Storage destructors.
    released_ = true;
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
    // Trim already-cached blocks down to the new limit immediately; otherwise
    // the limit only takes effect on the next free() and memory_cached() can
    // report more than the configured maximum. enforce_cache_limit assumes the
    // mutex is held (as on the free() path), which it is here.
    if (max_cached_memory_ > 0) {
        for (auto& entry : device_allocators_) {
            enforce_cache_limit(entry.first);
        }
    }
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

    // Find smallest free block that fits (best-fit), then scan forward over the
    // larger ready blocks.
    OneAPIBlock search_block(nullptr, size, device, shared);

    // The previous owner's enqueued work must be complete before this memory
    // gets a new owner (see OneAPIBlock::release_fence). Use a NON-blocking
    // completeness check: a blocking wait would stall every cached-block reuse
    // on the whole in-order queue (measured: DeepLab GradientFlow went from
    // seconds to a 1200 s timeout with a blocking wait).
    //
    // Critically, do NOT give up after the first best-fit block: in a tight
    // train loop the just-freed best-fit block's release fence is almost always
    // still in flight at the next same-size request, so returning nullptr here
    // would bypass the cache entirely and accumulate fresh USM blocks forever.
    // Instead, scan forward (towards larger blocks) for the first one whose
    // fence has already signalled and reuse that, only falling back to a fresh
    // allocation when every fitting cached block is still busy.
    for (auto it = free_blocks.lower_bound(&search_block); it != free_blocks.end(); ) {
        OneAPIBlock* block = *it;

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
                ++it;            // still in flight: try the next (larger) ready block
                continue;
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

    // Add to all_blocks (and the address-ordered index)
    device_alloc.all_blocks[ptr] = std::move(block);
    device_alloc.blocks_by_addr[ptr] = block_ptr;

    // Update statistics
    device_alloc.stats.reserved_bytes += size;

    return block_ptr;
}

bool OneAPICachingAllocator::split_block(OneAPIBlock* block, size_t size) {
    // A USM allocation must be sycl::free'd with the exact pointer returned by
    // sycl::malloc_*, so a split-off remainder can never be independently freed
    // — it stays tracked as an interior sub-range of `block`'s original
    // allocation (original_ptr), exactly like the CUDA/ROCm caching allocators'
    // hipMalloc/cudaMalloc sub-blocks. release_block() refuses to sycl::free an
    // interior block, and try_merge_blocks() reassembles siblings sharing the
    // same original_ptr back into one block once they are all free again — so
    // the underlying allocation is correctly freed exactly once, from the block
    // that still owns original_ptr.
    if (block->size < size + min_split_size_) {
        return false;
    }

    auto& device_alloc = device_allocators_[block->device];

    size_t remaining_size = block->size - size;
    void* new_ptr = static_cast<char*>(block->ptr) + size;

    auto new_block = std::make_unique<OneAPIBlock>(new_ptr, remaining_size, block->device, block->is_shared);
    new_block->allocated = false;
    new_block->original_ptr = block->original_ptr;  // Inherit for merge/release tracking
    // A freshly split-off remainder hasn't been requested by anyone yet; its
    // full physical size is genuinely free, so requested_size == size keeps the
    // cached_bytes invariant (cached_bytes == sum of free blocks' requested_size)
    // intact when it is later reused or re-merged.
    new_block->requested_size = remaining_size;
    OneAPIBlock* new_block_ptr = new_block.get();

    // Add to all_blocks (and the address-ordered index)
    device_alloc.all_blocks[new_ptr] = std::move(new_block);
    device_alloc.blocks_by_addr[new_ptr] = new_block_ptr;

    // Add to the free set matching the parent's memory kind
    if (block->is_shared) {
        device_alloc.free_shared_blocks.insert(new_block_ptr);
    } else {
        device_alloc.free_device_blocks.insert(new_block_ptr);
    }

    // Shrink the original block to exactly what was requested
    block->size = size;

    device_alloc.stats.num_splits++;
    device_alloc.stats.cached_bytes += remaining_size;

    return true;
}

bool OneAPICachingAllocator::try_merge_blocks(OneAPIBlock* block) {
    auto& device_alloc = device_allocators_[block->device];
    auto& free_blocks = block->is_shared ? device_alloc.free_shared_blocks
                                          : device_alloc.free_device_blocks;
    bool merged = false;

    // Forward merge: is there a free block immediately after this one, split
    // from the same original allocation?
    {
        void* next_ptr = static_cast<char*>(block->ptr) + block->size;
        auto next_it = device_alloc.all_blocks.find(next_ptr);
        if (next_it != device_alloc.all_blocks.end()) {
            OneAPIBlock* next_block = next_it->second.get();
            // Only fuse blocks split from the same sycl::malloc_* call. Two
            // address-adjacent blocks from distinct allocations must never be
            // merged: release_block() would only ever sycl::free the first
            // underlying allocation (leaking the second), and the merged block
            // could be handed out for OOB access past the first allocation.
            if (!next_block->allocated && next_block->original_ptr == block->original_ptr) {
                // Drain any outstanding release fence on the neighbor before
                // absorbing its range — the surviving block keeps only ITS OWN
                // fence, so any work still in flight against the neighbor's
                // memory must be proven complete now, not silently dropped.
                wait_and_clear_release_fence(next_block);

                free_blocks.erase(next_block);
                block->size += next_block->size;
                // Both blocks' bytes are already counted in cached_bytes (each
                // was credited when it was freed/split off); fold the
                // neighbor's requested_size into the survivor so a later reuse
                // decrements cached_bytes by the full merged amount.
                block->requested_size += next_block->requested_size;

                device_alloc.blocks_by_addr.erase(next_ptr);
                device_alloc.all_blocks.erase(next_it);

                device_alloc.stats.num_merges++;
                merged = true;
            }
        }
    }

    // Backward merge: is there a free predecessor immediately before this
    // block, split from the same original allocation? `block` must remain the
    // surviving object (the caller still references it), so absorb the
    // predecessor INTO `block` and re-key it to the predecessor's lower
    // address.
    {
        auto addr_it = device_alloc.blocks_by_addr.find(block->ptr);
        if (addr_it != device_alloc.blocks_by_addr.end() &&
            addr_it != device_alloc.blocks_by_addr.begin()) {
            auto prev_it = std::prev(addr_it);
            OneAPIBlock* prev_block = prev_it->second;
            void* prev_end = static_cast<char*>(prev_block->ptr) + prev_block->size;

            if (prev_end == block->ptr && !prev_block->allocated &&
                prev_block->original_ptr == block->original_ptr) {
                wait_and_clear_release_fence(prev_block);

                void* old_ptr = block->ptr;
                void* new_ptr = prev_block->ptr;
                size_t prev_requested = prev_block->requested_size;

                free_blocks.erase(prev_block);

                block->ptr = new_ptr;
                block->size += prev_block->size;
                block->requested_size += prev_requested;

                // Move block's owning entry from old_ptr -> new_ptr and drop
                // the predecessor's entry (destroys prev_block).
                auto node = device_alloc.all_blocks.extract(old_ptr);
                device_alloc.all_blocks.erase(new_ptr);
                node.key() = new_ptr;
                device_alloc.all_blocks.insert(std::move(node));

                device_alloc.blocks_by_addr.erase(old_ptr);
                device_alloc.blocks_by_addr[new_ptr] = block;

                device_alloc.stats.num_merges++;
                merged = true;
            }
        }
    }

    return merged;
}

size_t OneAPICachingAllocator::round_size(size_t size) const {
    // Guard the round-up against size_t overflow (see CachingAllocator::round_size).
    if (size > SIZE_MAX - (alignment_ - 1)) {
        throw std::bad_alloc();
    }
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void OneAPICachingAllocator::enforce_cache_limit(int device) {
    auto& device_alloc = device_allocators_[device];

    // Helper to release from a free block set
    auto release_from_set = [&](std::set<OneAPIBlock*, OneAPIBlockComparator>& free_set) {
        // Snapshot largest-first: release_block only actually frees (and
        // erases from free_set) a block that solely owns its original
        // allocation — it returns false, WITHOUT touching free_set, for an
        // interior split remainder still sharing that allocation with a live
        // sibling. Iterating a snapshot (rather than re-reading free_set.rbegin()
        // each pass) means a declined block is simply skipped instead of being
        // re-selected forever.
        std::vector<OneAPIBlock*> candidates(free_set.rbegin(), free_set.rend());
        for (OneAPIBlock* block : candidates) {
            if (device_alloc.stats.cached_bytes <= max_cached_memory_) {
                break;
            }
            release_block(block);
        }
    };

    // Release from both pools
    release_from_set(device_alloc.free_shared_blocks);
    release_from_set(device_alloc.free_device_blocks);
}

bool OneAPICachingAllocator::release_block(OneAPIBlock* block) {
    auto& device_alloc = device_allocators_[block->device];

    // Only the block that still owns the original sycl::malloc_* pointer may be
    // sycl::free'd. A split remainder has ptr == original_ptr + offset (an
    // interior pointer): sycl::free on it is undefined behavior, and freeing it
    // would orphan the rest of the underlying allocation that sibling blocks
    // still reference. Interior sub-blocks are only ever reclaimed by
    // try_merge_blocks reassembling the full allocation back into one block
    // whose ptr == original_ptr (backward merges re-key to the lowest address).
    if (block->ptr != block->original_ptr) {
        return false;
    }

    // Even the root block (ptr == original_ptr) must not be freed while a
    // split-off sibling from the same original allocation is still tracked:
    // sycl::free(original_ptr) releases the ENTIRE underlying allocation, so
    // freeing the root while a remainder sibling is still live would dangle
    // that sibling (use-after-free if it's later handed out). This only
    // matters with merging disabled or mid-cascade; try_merge_blocks normally
    // reassembles the full allocation before this is reached.
    for (const auto& [addr, other] : device_alloc.all_blocks) {
        if (other.get() != block && other->original_ptr == block->original_ptr) {
            return false;
        }
    }

    sycl::queue* queue = static_cast<sycl::queue*>(device_alloc.queue);

    // Remove from free blocks if present. The erase count tells us whether this
    // block was actually cached (and thus contributing to cached_bytes): the
    // free() large-allocation fast path releases a block that was never added
    // to a free set, so we must not subtract its size from cached_bytes.
    size_t was_cached = 0;
    if (block->is_shared) {
        was_cached = device_alloc.free_shared_blocks.erase(block);
    } else {
        was_cached = device_alloc.free_device_blocks.erase(block);
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

    // reserved_bytes tracks PHYSICAL USM, so subtract the physical block size.
    if (device_alloc.stats.reserved_bytes >= block->size) {
        device_alloc.stats.reserved_bytes -= block->size;
    }
    // cached_bytes was incremented by requested_size when the block was cached;
    // only undo it if the block was in fact cached.
    if (was_cached && device_alloc.stats.cached_bytes >= block->requested_size) {
        device_alloc.stats.cached_bytes -= block->requested_size;
    }

    // Remove from all_blocks (and the address-ordered index)
    device_alloc.blocks_by_addr.erase(block->ptr);
    device_alloc.all_blocks.erase(block->ptr);
    return true;
}

} // namespace backend
} // namespace tenzor
