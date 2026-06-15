#include "tenzor/backend/caching_allocator.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstdlib>

namespace tenzor {
namespace backend {

// Debug logging (enable with TENZOR_DEBUG_ALLOCATOR=1)
static bool debug_allocator() {
    static bool enabled = (std::getenv("TENZOR_DEBUG_ALLOCATOR") != nullptr);
    return enabled;
}

#define ALLOC_DEBUG(msg) if (debug_allocator()) { std::cerr << "[ALLOC] " << msg << std::endl; }

// Default configuration
constexpr size_t DEFAULT_ALIGNMENT = 512;
constexpr size_t DEFAULT_MIN_SPLIT_SIZE = 512;

CachingAllocator::CachingAllocator()
    : alignment_(DEFAULT_ALIGNMENT),
      max_cached_memory_(0),  // Unlimited by default
      merge_enabled_(true),
      min_split_size_(DEFAULT_MIN_SPLIT_SIZE) {
}

CachingAllocator::~CachingAllocator() {
    // Release all cached memory
    empty_cache(-1);
}

CachingAllocator& CachingAllocator::get() {
    static CachingAllocator instance;
    return instance;
}

void* CachingAllocator::allocate(size_t size, int device, cudaStream_t stream) {
    if (size == 0) {
        return nullptr;
    }

    // Get or create device allocator (map_mutex_ protects map structure)
    DeviceAllocator* dev_alloc;
    {
        std::lock_guard<std::mutex> map_lock(map_mutex_);
        dev_alloc = &device_allocators_[device];
    }

    // Lock only this device's mutex for the actual allocation
    std::lock_guard<std::mutex> lock(dev_alloc->mutex);

    // Round size to alignment
    size_t original_size = size;
    size = round_size(size);

    dev_alloc->stats.num_allocations++;

    // Try to find a suitable block in cache
    Block* block = try_allocate_from_cache(size, device, stream);

    bool from_cache = (block != nullptr);

    if (!block) {
        // No suitable cached block, allocate new one
        block = allocate_new_block(size, device, stream);
        ALLOC_DEBUG("NEW alloc: ptr=" << block->ptr << " size=" << size
                    << " (requested=" << original_size << ") device=" << device);
    } else {
        dev_alloc->stats.num_cache_hits++;
        ALLOC_DEBUG("CACHE HIT: ptr=" << block->ptr << " block_size=" << block->size
                    << " requested=" << size << " device=" << device);
    }

    // Mark block as allocated
    block->allocated = true;

    // Update statistics
    dev_alloc->stats.allocated_bytes += block->size;

    // Only subtract from cached_bytes if we got the block from cache
    if (from_cache && dev_alloc->stats.cached_bytes >= block->size) {
        dev_alloc->stats.cached_bytes -= block->size;
    }

    return block->ptr;
}

void CachingAllocator::free(void* ptr, int device) {
    if (!ptr) {
        return;
    }

    // Get or create device allocator
    DeviceAllocator* dev_alloc;
    {
        std::lock_guard<std::mutex> map_lock(map_mutex_);
        dev_alloc = &device_allocators_[device];
    }

    bool not_found = false;
    {
        std::lock_guard<std::mutex> lock(dev_alloc->mutex);

        dev_alloc->stats.num_frees++;

        // Find the block
        auto it = dev_alloc->all_blocks.find(ptr);
        if (it == dev_alloc->all_blocks.end()) {
            ALLOC_DEBUG("FREE ERROR: ptr=" << ptr << " device=" << device << " NOT FOUND in all_blocks!");
            not_found = true;
        } else {
            Block* block = it->second.get();
            if (!block->allocated) {
                ALLOC_DEBUG("FREE ERROR: ptr=" << ptr << " DOUBLE FREE!");
                throw std::runtime_error("Attempted to free already freed pointer");
            }

            ALLOC_DEBUG("FREE: ptr=" << ptr << " size=" << block->size << " device=" << device);

            // Mark as free
            block->allocated = false;

            // Update statistics
            if (dev_alloc->stats.allocated_bytes >= block->size) {
                dev_alloc->stats.allocated_bytes -= block->size;
            }
            dev_alloc->stats.cached_bytes += block->size;

            // Try to merge with adjacent blocks
            if (merge_enabled_) {
                try_merge_blocks(block);
            }

            // Add to free blocks set
            dev_alloc->free_blocks.insert(block);

            // Enforce cache limit if set
            if (max_cached_memory_ > 0) {
                enforce_cache_limit(device);
            }
        }
    }  // device mutex released

    if (not_found) {
        // Diagnostic cross-device scan. Performed AFTER releasing this device's
        // mutex so it obeys the global lock order (map_mutex_ before any device
        // mutex). Acquiring map_mutex_ while still holding dev_alloc->mutex was
        // a device->map inversion that could deadlock against empty_cache() /
        // stats (which take map->device).
        {
            std::lock_guard<std::mutex> map_lock(map_mutex_);
            for (const auto& [dev_id, da] : device_allocators_) {
                if (dev_id == device) continue;
                std::lock_guard<std::mutex> other_lock(da.mutex);
                auto found = da.all_blocks.find(ptr);
                if (found != da.all_blocks.end()) {
                    ALLOC_DEBUG("  -> Found in device " << dev_id << " instead!");
                }
            }
        }
        throw std::runtime_error("Attempted to free pointer not allocated by CachingAllocator");
    }
}

void CachingAllocator::empty_cache(int device) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);

    if (device == -1) {
        // Empty all devices
        for (auto& pair : device_allocators_) {
            std::lock_guard<std::mutex> dev_lock(pair.second.mutex);

            // Release all free blocks
            std::vector<Block*> blocks_to_release;
            for (Block* block : pair.second.free_blocks) {
                blocks_to_release.push_back(block);
            }

            for (Block* block : blocks_to_release) {
                release_block(block);
            }
        }
    } else {
        // Empty specific device
        auto it = device_allocators_.find(device);
        if (it == device_allocators_.end()) {
            return;  // Device doesn't exist, nothing to empty
        }

        std::lock_guard<std::mutex> dev_lock(it->second.mutex);

        std::vector<Block*> blocks_to_release;
        for (Block* block : it->second.free_blocks) {
            blocks_to_release.push_back(block);
        }

        for (Block* block : blocks_to_release) {
            release_block(block);
        }
    }
}

size_t CachingAllocator::memory_allocated(int device) const {
    std::lock_guard<std::mutex> map_lock(map_mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            std::lock_guard<std::mutex> dev_lock(pair.second.mutex);
            total += pair.second.stats.allocated_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        std::lock_guard<std::mutex> dev_lock(it->second.mutex);
        return it->second.stats.allocated_bytes;
    }
    return 0;
}

size_t CachingAllocator::memory_reserved(int device) const {
    std::lock_guard<std::mutex> map_lock(map_mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            std::lock_guard<std::mutex> dev_lock(pair.second.mutex);
            total += pair.second.stats.reserved_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        std::lock_guard<std::mutex> dev_lock(it->second.mutex);
        return it->second.stats.reserved_bytes;
    }
    return 0;
}

size_t CachingAllocator::memory_cached(int device) const {
    std::lock_guard<std::mutex> map_lock(map_mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            std::lock_guard<std::mutex> dev_lock(pair.second.mutex);
            total += pair.second.stats.cached_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        std::lock_guard<std::mutex> dev_lock(it->second.mutex);
        return it->second.stats.cached_bytes;
    }
    return 0;
}

MemoryStats CachingAllocator::get_stats(int device) const {
    std::lock_guard<std::mutex> map_lock(map_mutex_);

    if (device == -1) {
        MemoryStats total;
        for (const auto& pair : device_allocators_) {
            std::lock_guard<std::mutex> dev_lock(pair.second.mutex);
            const auto& stats = pair.second.stats;
            total.allocated_bytes += stats.allocated_bytes;
            total.reserved_bytes += stats.reserved_bytes;
            total.cached_bytes += stats.cached_bytes;
            total.num_allocations += stats.num_allocations;
            total.num_frees += stats.num_frees;
            total.num_cache_hits += stats.num_cache_hits;
            total.num_splits += stats.num_splits;
            total.num_merges += stats.num_merges;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        std::lock_guard<std::mutex> dev_lock(it->second.mutex);
        return it->second.stats;
    }
    return MemoryStats();
}

void CachingAllocator::reset_stats() {
    std::lock_guard<std::mutex> map_lock(map_mutex_);

    for (auto& pair : device_allocators_) {
        std::lock_guard<std::mutex> dev_lock(pair.second.mutex);
        auto& stats = pair.second.stats;
        stats.num_allocations = 0;
        stats.num_frees = 0;
        stats.num_cache_hits = 0;
        stats.num_splits = 0;
        stats.num_merges = 0;
    }
}

void CachingAllocator::set_alignment(size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    alignment_ = alignment;
}

void CachingAllocator::set_max_cached_memory(size_t max_bytes) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    max_cached_memory_ = max_bytes;
}

void CachingAllocator::set_merge_enabled(bool enable) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    merge_enabled_ = enable;
}

void CachingAllocator::set_min_split_size(size_t min_size) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    min_split_size_ = min_size;
}

// NOTE: All functions below are called with the per-device mutex already held

Block* CachingAllocator::try_allocate_from_cache(size_t size, int device, cudaStream_t stream) {
    auto& device_alloc = device_allocators_[device];

    // Find smallest free block that fits (best-fit)
    Block search_block(nullptr, size, device, stream);
    auto it = device_alloc.free_blocks.lower_bound(&search_block);

    if (it != device_alloc.free_blocks.end()) {
        Block* block = *it;

        // Remove from free blocks
        device_alloc.free_blocks.erase(it);

        // Try to split if block is too large
        if (block->size >= size + min_split_size_) {
            split_block(block, size);
        }

        return block;
    }

    return nullptr;
}

Block* CachingAllocator::allocate_new_block(size_t size, int device, cudaStream_t stream) {
    auto& device_alloc = device_allocators_[device];

    // Set device
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Allocate from device
    void* ptr = nullptr;
    err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Create block
    auto block = std::make_unique<Block>(ptr, size, device, stream);
    Block* block_ptr = block.get();

    // Add to all_blocks (and the address-ordered index)
    device_alloc.all_blocks[ptr] = std::move(block);
    device_alloc.blocks_by_addr[ptr] = block_ptr;

    // Update statistics
    device_alloc.stats.reserved_bytes += size;

    return block_ptr;
}

bool CachingAllocator::split_block(Block* block, size_t size) {
    if (block->size < size + min_split_size_) {
        return false;
    }

    auto& device_alloc = device_allocators_[block->device];

    // Calculate split size
    size_t remaining_size = block->size - size;

    // Create new block for remaining memory
    void* new_ptr = static_cast<char*>(block->ptr) + size;

    ALLOC_DEBUG("SPLIT: original_ptr=" << block->ptr << " original_size=" << block->size
                << " -> keep_size=" << size << " split_ptr=" << new_ptr << " split_size=" << remaining_size);

    auto new_block = std::make_unique<Block>(new_ptr, remaining_size, block->device, block->stream);
    new_block->allocated = false;
    new_block->original_ptr = block->original_ptr;  // Inherit from parent for merge tracking
    Block* new_block_ptr = new_block.get();

    // Add to all_blocks (and the address-ordered index)
    device_alloc.all_blocks[new_ptr] = std::move(new_block);
    device_alloc.blocks_by_addr[new_ptr] = new_block_ptr;

    // Add to free blocks
    device_alloc.free_blocks.insert(new_block_ptr);

    // Update original block size
    block->size = size;

    // Update statistics
    device_alloc.stats.num_splits++;
    device_alloc.stats.cached_bytes += remaining_size;

    return true;
}

bool CachingAllocator::try_merge_blocks(Block* block) {
    auto& device_alloc = device_allocators_[block->device];
    bool merged = false;

    // Repeat until no further coalescing is possible so a freed block fully
    // absorbs ALL contiguous free neighbors on both sides in one call (not just
    // the single immediate predecessor/successor). Each successful merge can
    // expose a new neighbor that the next pass picks up.
    bool merged_this_pass = true;
    while (merged_this_pass) {
        merged_this_pass = false;

        // Forward merge: absorb the immediately-following contiguous free block.
        void* next_ptr = static_cast<char*>(block->ptr) + block->size;
        auto next_it = device_alloc.all_blocks.find(next_ptr);
        if (next_it != device_alloc.all_blocks.end()) {
            Block* next_block = next_it->second.get();
            // Only merge blocks from the same original cudaMalloc allocation
            if (!next_block->allocated && next_block->original_ptr == block->original_ptr) {
                ALLOC_DEBUG("MERGE: block_ptr=" << block->ptr << " block_size=" << block->size
                            << " + next_ptr=" << next_block->ptr << " next_size=" << next_block->size
                            << " -> merged_size=" << (block->size + next_block->size));

                // Merge with next block
                device_alloc.free_blocks.erase(next_block);

                // When merging, next_block's size was already in cached_bytes
                // Subtract it before expanding the current block to avoid double-counting
                if (device_alloc.stats.cached_bytes >= next_block->size) {
                    device_alloc.stats.cached_bytes -= next_block->size;
                }

                // Expand current block
                block->size += next_block->size;

                // Remove next block (and its address-index entry)
                device_alloc.blocks_by_addr.erase(next_ptr);
                device_alloc.all_blocks.erase(next_it);

                device_alloc.stats.num_merges++;
                merged = true;
                merged_this_pass = true;
            }
        }

        // Backward merge: find the immediately-preceding block via the address
        // index (O(log n)). `block` must remain the surviving object because the
        // caller still references it, so we absorb the predecessor INTO `block`
        // and re-key it to the predecessor's (lower) address.
        auto addr_it = device_alloc.blocks_by_addr.find(block->ptr);
        if (addr_it != device_alloc.blocks_by_addr.end() &&
            addr_it != device_alloc.blocks_by_addr.begin()) {
            auto prev_it = std::prev(addr_it);
            Block* prev_block = prev_it->second;
            void* prev_end = static_cast<char*>(prev_block->ptr) + prev_block->size;
            if (prev_end == block->ptr && !prev_block->allocated &&
                prev_block->original_ptr == block->original_ptr) {
                ALLOC_DEBUG("MERGE(prev): prev_ptr=" << prev_block->ptr
                            << " prev_size=" << prev_block->size
                            << " + block_ptr=" << block->ptr
                            << " block_size=" << block->size);

                void* old_ptr = block->ptr;
                void* new_ptr = prev_block->ptr;
                size_t prev_size = prev_block->size;

                // prev_block is currently a free block; mirror the forward-merge
                // accounting (subtract the absorbed block's cached bytes).
                device_alloc.free_blocks.erase(prev_block);
                if (device_alloc.stats.cached_bytes >= prev_size) {
                    device_alloc.stats.cached_bytes -= prev_size;
                }

                // Grow `block` downward to cover the predecessor's range.
                block->ptr = new_ptr;
                block->size += prev_size;

                // Move `block`'s owning entry from old_ptr -> new_ptr and drop
                // the predecessor's entry (destroying the prev Block object).
                auto node = device_alloc.all_blocks.extract(old_ptr);
                device_alloc.all_blocks.erase(new_ptr);  // destroys prev_block
                node.key() = new_ptr;
                device_alloc.all_blocks.insert(std::move(node));

                // Update the address index to match.
                device_alloc.blocks_by_addr.erase(old_ptr);
                device_alloc.blocks_by_addr[new_ptr] = block;

                device_alloc.stats.num_merges++;
                merged = true;
                merged_this_pass = true;
            }
        }
    }

    return merged;
}

size_t CachingAllocator::round_size(size_t size) const {
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void CachingAllocator::enforce_cache_limit(int device) {
    auto& device_alloc = device_allocators_[device];

    // Release blocks until we're under the limit. Only blocks that still own
    // their original cudaMalloc pointer can actually be freed; interior split
    // remainder blocks are skipped (release_block returns false) — collect them
    // and re-insert afterwards so the largest-first scan makes progress instead
    // of repeatedly picking the same un-freeable interior block.
    std::vector<Block*> skipped;
    while (device_alloc.stats.cached_bytes > max_cached_memory_ &&
           !device_alloc.free_blocks.empty()) {
        // Release largest block first
        auto it = device_alloc.free_blocks.rbegin();
        Block* block = *it;

        // Convert reverse iterator to forward iterator for erase
        auto forward_it = std::next(it).base();
        device_alloc.free_blocks.erase(forward_it);

        if (!release_block(block)) {
            // Interior sub-block: cannot be freed standalone. Hold it aside so
            // we don't re-select it on the next iteration.
            skipped.push_back(block);
        }
    }
    // Restore the interior blocks we set aside so they remain available for
    // future coalescing.
    for (Block* block : skipped) {
        device_alloc.free_blocks.insert(block);
    }
}

bool CachingAllocator::release_block(Block* block) {
    auto& device_alloc = device_allocators_[block->device];

    // Only the block that still owns the original cudaMalloc pointer may be
    // device-freed. A split remainder block has `ptr = original_ptr + offset`
    // (an interior pointer): calling cudaFree on it is an invalid free, and
    // freeing it would orphan/leak the rest of the underlying allocation that
    // sibling blocks still reference. Such interior sub-blocks are only ever
    // reclaimed by try_merge_blocks reassembling the full allocation back into
    // a single block whose ptr == original_ptr (backward merges re-key to the
    // lowest address). Skip interior blocks here, leaving them tracked.
    if (block->ptr != block->original_ptr) {
        return false;
    }

    // Even the lower split block keeps ptr == original_ptr (split_block only
    // shrinks its size and creates a remainder at original_ptr + size that
    // inherits the same original_ptr). cudaFree(original_ptr) frees the ENTIRE
    // underlying allocation, so device-freeing the lower block while a remainder
    // sibling is still tracked would leave that sibling dangling (use-after-free
    // if handed back out) and leak the remainder's reserved accounting. This
    // normally cannot happen because try_merge_blocks reassembles the full block
    // first, but with merging disabled (set_merge_enabled(false)) the lower block
    // is still selectable by enforce_cache_limit/empty_cache. Refuse to free
    // unless this block is the SOLE block for its original allocation.
    for (const auto& [addr, other] : device_alloc.all_blocks) {
        if (other.get() != block && other->original_ptr == block->original_ptr) {
            // A sibling (remainder or another split fragment) still references
            // this allocation. Leave the block cached so it can be coalesced
            // later rather than freeing memory still in use.
            return false;
        }
    }

    // Remove from free blocks if present
    device_alloc.free_blocks.erase(block);

    // Free device memory. Surface the error rather than silently swallowing it
    // (a failed cudaFree means the underlying allocation leaks).
    cudaError_t err = cudaFree(block->ptr);
    if (err != cudaSuccess) {
        ALLOC_DEBUG("cudaFree failed for ptr=" << block->ptr
                    << " err=" << static_cast<int>(err));
    }

    // Update statistics
    if (device_alloc.stats.reserved_bytes >= block->size) {
        device_alloc.stats.reserved_bytes -= block->size;
    }
    if (device_alloc.stats.cached_bytes >= block->size) {
        device_alloc.stats.cached_bytes -= block->size;
    }

    // Remove from all_blocks (and the address-ordered index)
    device_alloc.blocks_by_addr.erase(block->ptr);
    device_alloc.all_blocks.erase(block->ptr);
    return true;
}

} // namespace backend
} // namespace tenzor
