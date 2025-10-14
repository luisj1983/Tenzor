#include "tenzor/backend/caching_allocator.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// CUDA runtime stubs for when CUDA is not available
typedef enum { cudaSuccess = 0 } cudaError_t;
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** ptr, size_t) { *ptr = nullptr; return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA not available"; }

struct cudaPointerAttributes {
    int device;
};
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attr, const void*) {
    attr->device = 0;
    return cudaSuccess;
}
#endif

#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace tenzor {
namespace backend {

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

    std::lock_guard<std::mutex> lock(mutex_);

    // Round size to alignment
    size = round_size(size);

    auto& device_alloc = device_allocators_[device];
    device_alloc.stats.num_allocations++;

    // Try to find a suitable block in cache
    Block* block = try_allocate_from_cache(size, device, stream);

    if (!block) {
        // No suitable cached block, allocate new one
        block = allocate_new_block(size, device, stream);
    } else {
        device_alloc.stats.num_cache_hits++;
    }

    // Mark block as allocated
    block->allocated = true;

    // Update statistics
    device_alloc.stats.allocated_bytes += block->size;
    if (device_alloc.stats.cached_bytes >= block->size) {
        device_alloc.stats.cached_bytes -= block->size;
    }

    return block->ptr;
}

void CachingAllocator::free(void* ptr, int device) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto& device_alloc = device_allocators_[device];
    device_alloc.stats.num_frees++;

    // Find the block
    auto it = device_alloc.all_blocks.find(ptr);
    if (it == device_alloc.all_blocks.end()) {
        throw std::runtime_error("Attempted to free pointer not allocated by CachingAllocator");
    }

    Block* block = it->second.get();
    if (!block->allocated) {
        throw std::runtime_error("Attempted to free already freed pointer");
    }

    // Mark as free
    block->allocated = false;

    // Update statistics
    if (device_alloc.stats.allocated_bytes >= block->size) {
        device_alloc.stats.allocated_bytes -= block->size;
    }
    device_alloc.stats.cached_bytes += block->size;

    // Try to merge with adjacent blocks
    if (merge_enabled_) {
        try_merge_blocks(block);
    }

    // Add to free blocks set
    device_alloc.free_blocks.insert(block);

    // Enforce cache limit if set
    if (max_cached_memory_ > 0) {
        enforce_cache_limit(device);
    }
}

void CachingAllocator::empty_cache(int device) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        // Empty all devices
        for (auto& pair : device_allocators_) {
            auto& device_alloc = pair.second;

            // Release all free blocks
            std::vector<Block*> blocks_to_release;
            for (Block* block : device_alloc.free_blocks) {
                blocks_to_release.push_back(block);
            }

            for (Block* block : blocks_to_release) {
                release_block(block);
            }
        }
    } else {
        // Empty specific device
        auto& device_alloc = device_allocators_[device];

        std::vector<Block*> blocks_to_release;
        for (Block* block : device_alloc.free_blocks) {
            blocks_to_release.push_back(block);
        }

        for (Block* block : blocks_to_release) {
            release_block(block);
        }
    }
}

size_t CachingAllocator::memory_allocated(int device) const {
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

size_t CachingAllocator::memory_reserved(int device) const {
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

size_t CachingAllocator::memory_cached(int device) const {
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

MemoryStats CachingAllocator::get_stats(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        MemoryStats total;
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
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats;
    }
    return MemoryStats();
}

void CachingAllocator::reset_stats() {
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

void CachingAllocator::set_alignment(size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    alignment_ = alignment;
}

void CachingAllocator::set_max_cached_memory(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_cached_memory_ = max_bytes;
}

void CachingAllocator::set_merge_enabled(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    merge_enabled_ = enable;
}

void CachingAllocator::set_min_split_size(size_t min_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_split_size_ = min_size;
}

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

    // Add to all_blocks
    device_alloc.all_blocks[ptr] = std::move(block);

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
    auto new_block = std::make_unique<Block>(new_ptr, remaining_size, block->device, block->stream);
    new_block->allocated = false;
    Block* new_block_ptr = new_block.get();

    // Add to all_blocks
    device_alloc.all_blocks[new_ptr] = std::move(new_block);

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

    // Try to find adjacent blocks to merge
    // Check if there's a block right after this one
    void* next_ptr = static_cast<char*>(block->ptr) + block->size;
    auto next_it = device_alloc.all_blocks.find(next_ptr);

    if (next_it != device_alloc.all_blocks.end()) {
        Block* next_block = next_it->second.get();
        if (!next_block->allocated) {
            // Merge with next block
            device_alloc.free_blocks.erase(next_block);

            // Update statistics
            device_alloc.stats.cached_bytes -= next_block->size;

            // Expand current block
            block->size += next_block->size;

            // Remove next block
            device_alloc.all_blocks.erase(next_it);

            device_alloc.stats.num_merges++;
            device_alloc.stats.cached_bytes += block->size;
            merged = true;
        }
    }

    // Try to find a block before this one (more complex)
    // We'd need to iterate through all blocks to find predecessor
    // For efficiency, we skip backward merging in this implementation
    // A more sophisticated approach would maintain a sorted map by address

    return merged;
}

size_t CachingAllocator::round_size(size_t size) const {
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void CachingAllocator::enforce_cache_limit(int device) {
    auto& device_alloc = device_allocators_[device];

    // Release blocks until we're under the limit
    while (device_alloc.stats.cached_bytes > max_cached_memory_ &&
           !device_alloc.free_blocks.empty()) {
        // Release largest block first
        auto it = device_alloc.free_blocks.rbegin();
        Block* block = *it;

        // Convert reverse iterator to forward iterator for erase
        auto forward_it = std::next(it).base();
        device_alloc.free_blocks.erase(forward_it);

        release_block(block);
    }
}

void CachingAllocator::release_block(Block* block) {
    auto& device_alloc = device_allocators_[block->device];

    // Remove from free blocks if present
    device_alloc.free_blocks.erase(block);

    // Free device memory
    cudaError_t err = cudaFree(block->ptr);
    if (err != cudaSuccess) {
        // Log error but don't throw in destructor context
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
