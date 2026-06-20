#include "tenzor/backend/rocm_caching_allocator.hip.hpp"

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
// HIP runtime fallback implementations for builds without HIP.
// Struct definitions must precede the stub functions that reference them,
// otherwise C++ rejects the undeclared parameter types (forward-reference
// compile error).
typedef enum { hipSuccess = 0, hipErrorOutOfMemory = 2 } hipError_t;

struct hipDeviceProp_t {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int multiProcessorCount;
    int warpSize;
};

struct hipPointerAttribute_t {
    int device;
};

inline hipError_t hipSetDevice(int) { return hipSuccess; }
inline hipError_t hipMalloc(void** ptr, size_t) { *ptr = nullptr; return hipSuccess; }
inline hipError_t hipFree(void*) { return hipSuccess; }
inline const char* hipGetErrorString(hipError_t) { return "HIP not available"; }
inline hipError_t hipMemGetInfo(size_t* free, size_t* total) { *free = 0; *total = 0; return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int) { return hipSuccess; }
inline hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attr, const void*) {
    attr->device = 0;
    return hipSuccess;
}
#endif

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdint>

namespace tenzor {
namespace backend {
namespace rocm {

// Default configuration optimized for AMD GPUs
constexpr size_t DEFAULT_ALIGNMENT = 256;        // HBM optimal alignment
constexpr size_t DEFAULT_MIN_SPLIT_SIZE = 512;   // Minimum split size
constexpr size_t DEFAULT_LARGE_ALLOC = 2ULL * 1024 * 1024 * 1024;  // 2GB

RocmCachingAllocator::RocmCachingAllocator()
    : alignment_(DEFAULT_ALIGNMENT),
      max_cached_memory_(0),  // Unlimited by default
      merge_enabled_(true),
      min_split_size_(DEFAULT_MIN_SPLIT_SIZE),
      logging_enabled_(false),
      large_alloc_threshold_(DEFAULT_LARGE_ALLOC) {
}

RocmCachingAllocator::~RocmCachingAllocator() {
    // Release all cached memory
    empty_cache(-1);
}

RocmCachingAllocator& RocmCachingAllocator::get() {
    static RocmCachingAllocator instance;
    return instance;
}

void* RocmCachingAllocator::allocate(size_t size, int device, hipStream_t stream) {
    if (size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Round size to alignment
    size = round_size(size);

    auto& device_alloc = device_allocators_[device];

    // Initialize device properties if not already done
    if (device_alloc.properties.total_memory == 0) {
        initialize_device_properties(device);
    }

    device_alloc.stats.num_allocations++;

    // Try to find a suitable block in cache
    Block* block = try_allocate_from_cache(size, device, stream);
    bool from_cache = (block != nullptr);

    if (!block) {
        // No suitable cached block, allocate new one
        int retry_count = 0;
        const int max_retries = 3;

        while (!block && retry_count < max_retries) {
            try {
                block = allocate_new_block(size, device, stream);
                break;
            } catch (const std::runtime_error& e) {
                if (handle_allocation_failure(size, device)) {
                    retry_count++;
                    log_message("Retrying allocation after garbage collection (attempt " +
                              std::to_string(retry_count) + "/" + std::to_string(max_retries) + ")");
                } else {
                    device_alloc.stats.num_oom_errors++;
                    throw;
                }
            }
        }

        if (!block) {
            device_alloc.stats.num_oom_errors++;
            throw std::runtime_error("Failed to allocate memory after " +
                                   std::to_string(max_retries) + " retries");
        }
    } else {
        device_alloc.stats.num_cache_hits++;
    }

    // Mark block as allocated
    block->allocated = true;

    // Update statistics
    device_alloc.stats.allocated_bytes += block->size;

    // cached_bytes was already decremented in try_allocate_from_cache when the
    // block was removed from the free pool (and split_block re-added any
    // remainder), so nothing to subtract here. Subtracting the post-split
    // block->size here as well would over-count by the split remainder on every
    // split (mirrors the CUDA reference allocator, caching_allocator.cpp:85-88).

    // Track peak memory usage
    if (device_alloc.stats.allocated_bytes > device_alloc.stats.peak_allocated) {
        device_alloc.stats.peak_allocated = device_alloc.stats.allocated_bytes;
    }
    if (device_alloc.stats.reserved_bytes > device_alloc.stats.peak_reserved) {
        device_alloc.stats.peak_reserved = device_alloc.stats.reserved_bytes;
    }

    // Track HBM allocations
    if (device_alloc.properties.has_hbm) {
        device_alloc.stats.hbm_bytes += block->size;
    }

    // Add to stream tracking
    device_alloc.stream_blocks[stream].push_back(block);

    log_message("Allocated " + std::to_string(size) + " bytes on device " +
                std::to_string(device) + " (cache hit: " +
                std::to_string(from_cache) + ")");

    return block->ptr;
}

void RocmCachingAllocator::free(void* ptr, int device) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto& device_alloc = device_allocators_[device];
    device_alloc.stats.num_frees++;

    // Find the block
    auto it = device_alloc.all_blocks.find(ptr);
    if (it == device_alloc.all_blocks.end()) {
        throw std::runtime_error("Attempted to free pointer not allocated by RocmCachingAllocator");
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

    // Update HBM tracking
    if (device_alloc.properties.has_hbm) {
        if (device_alloc.stats.hbm_bytes >= block->size) {
            device_alloc.stats.hbm_bytes -= block->size;
        }
    }

    // Remove from stream tracking
    auto& stream_blocks = device_alloc.stream_blocks[block->stream];
    stream_blocks.erase(std::remove(stream_blocks.begin(), stream_blocks.end(), block),
                       stream_blocks.end());

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

    log_message("Freed " + std::to_string(block->size) + " bytes on device " +
                std::to_string(device));
}

void RocmCachingAllocator::empty_cache(int device) {
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

    log_message("Emptied cache for device " + std::to_string(device));
}

void RocmCachingAllocator::garbage_collect(int device, bool aggressive) {
    std::lock_guard<std::mutex> lock(mutex_);
    garbage_collect_locked(device, aggressive);
}

void RocmCachingAllocator::garbage_collect_locked(int device, bool aggressive) {
    // Caller must hold mutex_.
    auto process_device = [this, aggressive](DeviceAllocator& device_alloc) {
        if (device_alloc.free_blocks.empty()) {
            return;
        }

        // Sort blocks by size (largest first)
        std::vector<Block*> blocks_to_consider;
        for (Block* block : device_alloc.free_blocks) {
            blocks_to_consider.push_back(block);
        }

        std::sort(blocks_to_consider.begin(), blocks_to_consider.end(),
                 [](const Block* a, const Block* b) { return a->size > b->size; });

        // In aggressive mode, free all blocks
        // In normal mode, free blocks larger than threshold or if cache is full
        size_t freed_bytes = 0;
        for (Block* block : blocks_to_consider) {
            bool should_free = aggressive ||
                              block->size >= large_alloc_threshold_ ||
                              (max_cached_memory_ > 0 &&
                               device_alloc.stats.cached_bytes > max_cached_memory_);

            if (should_free) {
                // release_block returns false for interior split-remainder
                // blocks that cannot be freed standalone; only count bytes that
                // were actually returned to the device.
                size_t block_size = block->size;
                if (release_block(block)) {
                    freed_bytes += block_size;
                }
            }
        }

        log_message("Garbage collected " + std::to_string(freed_bytes) + " bytes");
    };

    if (device == -1) {
        for (auto& pair : device_allocators_) {
            process_device(pair.second);
        }
    } else {
        auto it = device_allocators_.find(device);
        if (it != device_allocators_.end()) {
            process_device(it->second);
        }
    }
}

size_t RocmCachingAllocator::memory_allocated(int device) const {
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

size_t RocmCachingAllocator::memory_reserved(int device) const {
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

size_t RocmCachingAllocator::memory_cached(int device) const {
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

MemoryStats RocmCachingAllocator::get_stats(int device) const {
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
            total.peak_allocated = std::max(total.peak_allocated, stats.peak_allocated);
            total.peak_reserved = std::max(total.peak_reserved, stats.peak_reserved);
            total.num_oom_errors += stats.num_oom_errors;
            total.hbm_bytes += stats.hbm_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats;
    }
    return MemoryStats();
}

DeviceProperties RocmCachingAllocator::get_device_properties(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.properties;
    }

    // Initialize and return properties
    const_cast<RocmCachingAllocator*>(this)->initialize_device_properties(device);
    auto it2 = device_allocators_.find(device);
    if (it2 != device_allocators_.end()) {
        return it2->second.properties;
    }
    // Should not reach here after initialize, but return empty properties as fallback
    return DeviceProperties();
}

void RocmCachingAllocator::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : device_allocators_) {
        auto& stats = pair.second.stats;
        stats.num_allocations = 0;
        stats.num_frees = 0;
        stats.num_cache_hits = 0;
        stats.num_splits = 0;
        stats.num_merges = 0;
        stats.num_oom_errors = 0;
    }
}

void RocmCachingAllocator::set_alignment(size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    alignment_ = alignment;
}

void RocmCachingAllocator::set_max_cached_memory(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_cached_memory_ = max_bytes;
}

void RocmCachingAllocator::set_merge_enabled(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    merge_enabled_ = enable;
}

void RocmCachingAllocator::set_min_split_size(size_t min_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_split_size_ = min_size;
}

void RocmCachingAllocator::set_logging_enabled(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    logging_enabled_ = enable;
}

void RocmCachingAllocator::set_large_alloc_threshold(size_t threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    large_alloc_threshold_ = threshold;
}

void RocmCachingAllocator::synchronize_device(int device) {
    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to set HIP device: " +
                               std::string(hipGetErrorString(err)));
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to synchronize device: " +
                               std::string(hipGetErrorString(err)));
    }

    log_message("Synchronized device " + std::to_string(device));
}

Block* RocmCachingAllocator::try_allocate_from_cache(size_t size, int device, hipStream_t stream) {
    auto& device_alloc = device_allocators_[device];

    // Find smallest free block that fits (best-fit)
    Block search_block(nullptr, size, device, stream);
    auto it = device_alloc.free_blocks.lower_bound(&search_block);

    if (it != device_alloc.free_blocks.end()) {
        Block* block = *it;

        // Remove from free blocks
        device_alloc.free_blocks.erase(it);

        // The block leaves the free pool: drop its full (pre-split) size from
        // cached_bytes here. If it is then split, split_block re-adds the
        // remainder. This preserves the invariant cached_bytes == sum of
        // free_blocks sizes (subtracting the post-split size in allocate()
        // over-counted by the remainder on every split). Mirrors the CUDA
        // reference allocator (caching_allocator.cpp:348-357).
        if (device_alloc.stats.cached_bytes >= block->size) {
            device_alloc.stats.cached_bytes -= block->size;
        } else {
            device_alloc.stats.cached_bytes = 0;
        }

        // Try to split if block is too large
        if (block->size >= size + min_split_size_) {
            split_block(block, size);
        }

        return block;
    }

    return nullptr;
}

Block* RocmCachingAllocator::allocate_new_block(size_t size, int device, hipStream_t stream) {
    auto& device_alloc = device_allocators_[device];

    // Set device
    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to set HIP device: " +
                                 std::string(hipGetErrorString(err)));
    }

    // Check available memory
    size_t free_mem, total_mem;
    err = hipMemGetInfo(&free_mem, &total_mem);
    if (err == hipSuccess && free_mem < size) {
        log_message("Warning: Requested " + std::to_string(size) +
                   " bytes but only " + std::to_string(free_mem) + " bytes available");
    }

    // Allocate from device
    void* ptr = nullptr;
    err = hipMalloc(&ptr, size);
    if (err != hipSuccess) {
        throw std::runtime_error("HIP memory allocation failed: " +
                                 std::string(hipGetErrorString(err)) +
                                 " (requested " + std::to_string(size) + " bytes)");
    }

    // Create block with optimal alignment for HBM
    size_t alignment = device_alloc.properties.has_hbm ? 256 : 128;
    auto block = std::make_unique<Block>(ptr, size, device, stream, alignment);
    Block* block_ptr = block.get();

    // Add to all_blocks (and the address-ordered index)
    device_alloc.all_blocks[ptr] = std::move(block);
    device_alloc.blocks_by_addr[ptr] = block_ptr;

    // Update statistics
    device_alloc.stats.reserved_bytes += size;

    return block_ptr;
}

bool RocmCachingAllocator::split_block(Block* block, size_t size) {
    if (block->size < size + min_split_size_) {
        return false;
    }

    auto& device_alloc = device_allocators_[block->device];

    // Calculate split size
    size_t remaining_size = block->size - size;

    // Create new block for remaining memory
    void* new_ptr = static_cast<char*>(block->ptr) + size;
    auto new_block = std::make_unique<Block>(new_ptr, remaining_size,
                                             block->device, block->stream, block->alignment);
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

bool RocmCachingAllocator::try_merge_blocks(Block* block) {
    auto& device_alloc = device_allocators_[block->device];
    bool merged = false;

    // Try to find adjacent blocks to merge
    // Check if there's a block right after this one
    void* next_ptr = static_cast<char*>(block->ptr) + block->size;
    auto next_it = device_alloc.all_blocks.find(next_ptr);

    if (next_it != device_alloc.all_blocks.end()) {
        Block* next_block = next_it->second.get();
        // Only merge blocks from the same original hipMalloc allocation. Two
        // address-adjacent blocks from distinct hipMalloc calls must never be
        // fused: release_block() would only hipFree the first underlying
        // allocation (leaking the second), and the merged Block would span two
        // allocations and could be handed out for OOB access past the first.
        if (!next_block->allocated && next_block->original_ptr == block->original_ptr) {
            // Merge with next block
            device_alloc.free_blocks.erase(next_block);

            // When merging, next_block's size is already counted in cached_bytes
            // (added by free()). Only subtract it here; do NOT re-add the grown
            // block->size, or this block's pre-merge bytes get double-counted.
            // Guard the subtraction against size_t underflow (cached_bytes can
            // legitimately be < next_block->size after upstream clamping), and
            // mirror the backward merge below + the CUDA reference allocator
            // (caching_allocator.cpp:468-469).
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
        }
    }

    // Backward merge: find the immediately-preceding block via the address
    // index (O(log n)). `block` must remain the surviving object because the
    // caller still references it, so we absorb the predecessor INTO `block` and
    // re-key it to the predecessor's (lower) address. Mirrors the CUDA
    // reference allocator (caching_allocator.cpp:467-515).
    {
        auto addr_it = device_alloc.blocks_by_addr.find(block->ptr);
        if (addr_it != device_alloc.blocks_by_addr.end() &&
            addr_it != device_alloc.blocks_by_addr.begin()) {
            auto prev_it = std::prev(addr_it);
            Block* prev_block = prev_it->second;
            void* prev_end = static_cast<char*>(prev_block->ptr) + prev_block->size;
            // Only coalesce a truly adjacent, free predecessor from the same
            // original hipMalloc allocation (gate on original_ptr adjacency).
            if (prev_end == block->ptr && !prev_block->allocated &&
                prev_block->original_ptr == block->original_ptr) {
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
            }
        }
    }

    return merged;
}

size_t RocmCachingAllocator::round_size(size_t size) const {
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void RocmCachingAllocator::enforce_cache_limit(int device) {
    auto& device_alloc = device_allocators_[device];

    // Release blocks until we're under the limit. Only blocks that still own
    // their original hipMalloc pointer can actually be freed; interior split
    // remainder blocks are skipped (release_block returns false) — collect them
    // and re-insert afterwards so the largest-first scan makes progress instead
    // of repeatedly picking the same un-freeable interior block (infinite loop).
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

bool RocmCachingAllocator::release_block(Block* block) {
    auto& device_alloc = device_allocators_[block->device];

    // Only the block that still owns the original hipMalloc pointer may be
    // device-freed. A split remainder block has `ptr = original_ptr + offset`
    // (an interior pointer): calling hipFree on it returns hipErrorInvalidValue,
    // and freeing it would orphan/leak the rest of the underlying allocation
    // that sibling blocks still reference. Such interior sub-blocks are only
    // ever reclaimed by try_merge_blocks reassembling the full allocation back
    // into a single block whose ptr == original_ptr (backward merges re-key to
    // the lowest address). Skip interior blocks here, leaving them tracked.
    if (block->ptr != block->original_ptr) {
        return false;
    }

    // Remove from free blocks if present
    device_alloc.free_blocks.erase(block);

    // Free device memory. Surface the error rather than silently swallowing it
    // (a failed hipFree means the underlying allocation leaks).
    hipError_t err = hipFree(block->ptr);
    if (err != hipSuccess) {
        // Log error but don't throw — release_block runs in destructor/cache
        // teardown contexts where throwing would terminate.
        log_message("Warning: hipFree failed for ptr=" +
                    std::to_string(reinterpret_cast<uintptr_t>(block->ptr)) +
                    ": " + std::string(hipGetErrorString(err)));
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

void RocmCachingAllocator::initialize_device_properties(int device) {
    auto& device_alloc = device_allocators_[device];
    auto& props = device_alloc.properties;

    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to set HIP device: " +
                               std::string(hipGetErrorString(err)));
    }

    // Get device properties
    hipDeviceProp_t hip_props;
    err = hipGetDeviceProperties(&hip_props, device);
    if (err == hipSuccess) {
        props.device_name = hip_props.name;
        props.total_memory = hip_props.totalGlobalMem;
        props.compute_units = hip_props.multiProcessorCount;
        props.max_shared_memory = hip_props.sharedMemPerBlock;
        props.warp_size = hip_props.warpSize;
    }

    // Get memory info
    size_t free_mem, total_mem;
    err = hipMemGetInfo(&free_mem, &total_mem);
    if (err == hipSuccess) {
        props.available_memory = free_mem;
        if (props.total_memory == 0) {
            props.total_memory = total_mem;
        }
    }

    // Detect HBM based on device name
    props.has_hbm = is_hbm_device(device);

    log_message("Initialized device " + std::to_string(device) + ": " +
                props.device_name + " (" +
                std::to_string(props.total_memory / (1024*1024*1024)) + " GB" +
                (props.has_hbm ? ", HBM" : "") + ")");
}

bool RocmCachingAllocator::is_hbm_device(int device) {
    auto& props = device_allocators_[device].properties;
    const std::string name = props.device_name;

    // MI series GPUs have HBM
    // MI25, MI50, MI60, MI100, MI200 series (MI210, MI250, MI250X), MI300
    if (name.find("MI25") != std::string::npos ||
        name.find("MI50") != std::string::npos ||
        name.find("MI60") != std::string::npos ||
        name.find("MI100") != std::string::npos ||
        name.find("MI200") != std::string::npos ||
        name.find("MI210") != std::string::npos ||
        name.find("MI250") != std::string::npos ||
        name.find("MI300") != std::string::npos) {
        return true;
    }

    // Radeon Instinct series
    if (name.find("Instinct") != std::string::npos) {
        return true;
    }

    return false;
}

void RocmCachingAllocator::log_message(const std::string& message) {
    if (logging_enabled_) {
        std::cout << "[RocmCachingAllocator] " << message << std::endl;
    }
}

bool RocmCachingAllocator::handle_allocation_failure(size_t size, int device) {
    log_message("Allocation failure, attempting garbage collection");

    // First try normal garbage collection.
    // NOTE: called from allocate() with mutex_ already held, so we must use the
    // lock-free variant to avoid self-deadlock on the non-recursive mutex_.
    garbage_collect_locked(device, false);

    // Check if we freed enough memory
    auto& device_alloc = device_allocators_[device];
    if (device_alloc.stats.cached_bytes > 0) {
        return true;
    }

    // Try aggressive garbage collection
    garbage_collect_locked(device, true);

    return device_alloc.stats.cached_bytes > 0;
}

} // namespace rocm
} // namespace backend
} // namespace tenzor
