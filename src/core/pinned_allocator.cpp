/**
 * @file pinned_allocator.cpp
 * @brief Implementation of pinned memory allocator
 */

#include "tenzor/core/pinned_allocator.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cstring>

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tenzor::core {

namespace {

#ifdef TENZOR_USE_CUDA
/**
 * @brief Check CUDA error and throw exception if failed
 */
inline auto check_cuda_error(cudaError_t error, const char* msg) -> void {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA Error in " << msg << ": " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}
#endif

/**
 * @brief Align size to alignment boundary
 */
inline auto align_size(size_t size, size_t alignment = 256) -> size_t {
    if (size > SIZE_MAX - (alignment - 1)) {
        throw std::overflow_error(
            "Allocation size overflow in align_size: requested " +
            std::to_string(size) + " bytes with alignment " +
            std::to_string(alignment));
    }
    return (size + alignment - 1) & ~(alignment - 1);
}

} // anonymous namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

PinnedMemoryAllocator::PinnedMemoryAllocator(const Config& config)
    : config_(config)
    , free_list_head_(nullptr)
    , total_allocated_(0)
    , num_allocations_(0)
    , peak_allocated_(0)
    , num_defragmentations_(0)
    , is_initialized_(false) {

    if (config_.pool_size == 0) {
        throw std::invalid_argument("Pool size must be greater than 0");
    }

    if (config_.min_block_size == 0) {
        throw std::invalid_argument("Minimum block size must be greater than 0");
    }

    // Allocate initial pool
    try {
        void* base_ptr = allocate_cuda_pinned(config_.pool_size);

        // Create initial free block spanning entire pool
        auto initial_block = std::make_unique<MemoryBlock>(base_ptr, config_.pool_size, true);
        auto* initial_block_raw = initial_block.get();
        free_list_head_ = initial_block_raw;
        block_map_[base_ptr] = std::move(initial_block);

        // Store pool region
        PoolRegion region;
        region.base_ptr = base_ptr;
        region.size = config_.pool_size;
        region.head = initial_block_raw;
        pools_.push_back(region);

        is_initialized_ = true;

    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Failed to initialize PinnedMemoryAllocator: " << e.what();
        throw std::runtime_error(oss.str());
    }
}

PinnedMemoryAllocator::~PinnedMemoryAllocator() {
    if (!is_initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // unique_ptr handles block deallocation when map is cleared
    block_map_.clear();

    // Free all pools
    for (auto& pool : pools_) {
        try {
            free_cuda_pinned(pool.base_ptr);
        } catch (...) {
            // Ignore errors during cleanup
        }
    }
    pools_.clear();

    free_list_head_ = nullptr;
    is_initialized_ = false;
}

PinnedMemoryAllocator::PinnedMemoryAllocator(PinnedMemoryAllocator&& other) noexcept
    : config_(other.config_)
    , pools_(std::move(other.pools_))
    , free_list_head_(other.free_list_head_)
    , block_map_(std::move(other.block_map_))
    , total_allocated_(other.total_allocated_.load())
    , num_allocations_(other.num_allocations_.load())
    , peak_allocated_(other.peak_allocated_.load())
    , num_defragmentations_(other.num_defragmentations_.load())
    , is_initialized_(other.is_initialized_) {

    other.free_list_head_ = nullptr;
    other.is_initialized_ = false;
}

PinnedMemoryAllocator& PinnedMemoryAllocator::operator=(PinnedMemoryAllocator&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        if (is_initialized_) {
            block_map_.clear();
            for (auto& pool : pools_) {
                try {
                    free_cuda_pinned(pool.base_ptr);
                } catch (...) {}
            }
        }

        // Move from other
        config_ = other.config_;
        pools_ = std::move(other.pools_);
        free_list_head_ = other.free_list_head_;
        block_map_ = std::move(other.block_map_);
        total_allocated_ = other.total_allocated_.load();
        num_allocations_ = other.num_allocations_.load();
        peak_allocated_ = other.peak_allocated_.load();
        num_defragmentations_ = other.num_defragmentations_.load();
        is_initialized_ = other.is_initialized_;

        other.free_list_head_ = nullptr;
        other.is_initialized_ = false;
    }
    return *this;
}

// =============================================================================
// Allocation Interface
// =============================================================================

auto PinnedMemoryAllocator::allocate(size_t bytes) -> void* {
    if (!is_initialized_) {
        return nullptr;
    }

    if (bytes == 0) {
        return nullptr;
    }

    // Align allocation size
    size_t aligned_size = align_size(std::max(bytes, config_.min_block_size));

    std::lock_guard<std::mutex> lock(mutex_);

    // Find best-fit free block
    MemoryBlock* block = find_best_fit(aligned_size);

    // If no block found, try to grow pool
    if (block == nullptr) {
        if (config_.allow_growth) {
            size_t growth_size = std::max(aligned_size, config_.growth_increment);
            if (grow_pool(growth_size)) {
                block = find_best_fit(aligned_size);
            }
        }

        if (block == nullptr) {
            return nullptr;  // Out of memory
        }
    }

    // Split block if it's significantly larger than needed
    if (block->size > aligned_size + config_.min_block_size) {
        split_block(block, aligned_size);
    }

    // Mark block as allocated
    block->is_free = false;

    // Update statistics
    total_allocated_ += block->size;
    num_allocations_++;

    size_t current_allocated = total_allocated_.load();
    size_t current_peak = peak_allocated_.load();
    while (current_allocated > current_peak) {
        if (peak_allocated_.compare_exchange_weak(current_peak, current_allocated)) {
            break;
        }
    }

    return block->ptr;
}

auto PinnedMemoryAllocator::deallocate(void* ptr) -> void {
    if (!is_initialized_ || ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find block
    MemoryBlock* block = find_block(ptr);
    if (block == nullptr) {
        throw std::invalid_argument("Attempting to deallocate unknown pointer");
    }

    if (block->is_free) {
        throw std::invalid_argument("Double free detected");
    }

    // Mark as free
    block->is_free = true;

    // Update statistics
    total_allocated_ -= block->size;
    num_allocations_--;

    // Coalesce with adjacent free blocks
    coalesce_block(block);
}

// =============================================================================
// Pool Management
// =============================================================================

auto PinnedMemoryAllocator::defragment() -> size_t {
    if (!is_initialized_) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    size_t coalesce_count = 0;

    // Sort blocks by address for each pool
    for (auto& pool : pools_) {
        std::vector<MemoryBlock*> sorted_blocks;
        for (auto& [ptr, block] : block_map_) {
            // Check if block belongs to this pool
            if (block->ptr >= pool.base_ptr &&
                block->ptr < static_cast<char*>(pool.base_ptr) + pool.size) {
                sorted_blocks.push_back(block.get());
            }
        }

        std::sort(sorted_blocks.begin(), sorted_blocks.end(),
                  [](const MemoryBlock* a, const MemoryBlock* b) {
                      return a->ptr < b->ptr;
                  });

        // Coalesce adjacent free blocks
        for (size_t i = 0; i < sorted_blocks.size() - 1; ++i) {
            auto* current = sorted_blocks[i];
            auto* next = sorted_blocks[i + 1];

            if (current->is_free && next->is_free) {
                void* current_end = static_cast<char*>(current->ptr) + current->size;
                if (current_end == next->ptr) {
                    // Coalesce: merge next into current
                    current->size += next->size;
                    current->next = next->next;
                    if (next->next) {
                        next->next->prev = current;
                    }

                    // Remove next from block map (unique_ptr handles deallocation)
                    void* next_ptr = next->ptr;
                    block_map_.erase(next_ptr);

                    coalesce_count++;
                }
            }
        }
    }

    if (coalesce_count > 0) {
        num_defragmentations_++;
    }

    return coalesce_count;
}

auto PinnedMemoryAllocator::reset() -> void {
    if (!is_initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Delete all blocks except pool base blocks
    std::vector<void*> pool_bases;
    for (const auto& pool : pools_) {
        pool_bases.push_back(pool.base_ptr);
    }

    // Clear block map (unique_ptr handles deallocation)
    block_map_.clear();

    // Recreate initial blocks for each pool
    free_list_head_ = nullptr;
    MemoryBlock* prev_block = nullptr;

    for (auto& pool : pools_) {
        auto block = std::make_unique<MemoryBlock>(pool.base_ptr, pool.size, true);
        auto* block_raw = block.get();
        block_map_[pool.base_ptr] = std::move(block);
        pool.head = block_raw;

        if (prev_block) {
            prev_block->next = block_raw;
            block_raw->prev = prev_block;
        } else {
            free_list_head_ = block_raw;
        }

        prev_block = block_raw;
    }

    // Reset statistics
    total_allocated_ = 0;
    num_allocations_ = 0;
}

auto PinnedMemoryAllocator::grow_pool(size_t additional_bytes) -> bool {
    if (!config_.allow_growth) {
        return false;
    }

    // Calculate total size after growth
    size_t current_total = 0;
    for (const auto& pool : pools_) {
        current_total += pool.size;
    }

    if (current_total + additional_bytes > config_.max_pool_size) {
        return false;  // Would exceed maximum pool size
    }

    try {
        // Allocate new pool
        void* new_ptr = allocate_cuda_pinned(additional_bytes);

        // Create new block
        auto new_block = std::make_unique<MemoryBlock>(new_ptr, additional_bytes, true);
        auto* new_block_raw = new_block.get();
        block_map_[new_ptr] = std::move(new_block);

        // Add to free list
        if (free_list_head_) {
            new_block_raw->next = free_list_head_;
            free_list_head_->prev = new_block_raw;
        }
        free_list_head_ = new_block_raw;

        // Add pool region
        PoolRegion region;
        region.base_ptr = new_ptr;
        region.size = additional_bytes;
        region.head = new_block_raw;
        pools_.push_back(region);

        return true;

    } catch (...) {
        return false;
    }
}

// =============================================================================
// Statistics and Monitoring
// =============================================================================

auto PinnedMemoryAllocator::get_total_size() const -> size_t {
    size_t total = 0;
    for (const auto& pool : pools_) {
        total += pool.size;
    }
    return total;
}

auto PinnedMemoryAllocator::get_allocated_size() const -> size_t {
    return total_allocated_.load();
}

auto PinnedMemoryAllocator::get_free_size() const -> size_t {
    return get_total_size() - get_allocated_size();
}

auto PinnedMemoryAllocator::get_fragmentation_ratio() const -> float {
    std::lock_guard<std::mutex> lock(mutex_);
    return calculate_fragmentation();
}

auto PinnedMemoryAllocator::get_allocation_count() const -> size_t {
    return num_allocations_.load();
}

auto PinnedMemoryAllocator::get_stats() const -> PinnedMemoryStats {
    std::lock_guard<std::mutex> lock(mutex_);

    PinnedMemoryStats stats;
    stats.total_size = get_total_size();
    stats.allocated_size = total_allocated_.load();
    stats.free_size = stats.total_size - stats.allocated_size;
    stats.num_allocations = num_allocations_.load();
    stats.num_blocks = block_map_.size();
    stats.peak_allocated = peak_allocated_.load();
    stats.num_defragmentations = num_defragmentations_.load();

    // Count free blocks
    stats.num_free_blocks = 0;
    for (const auto& [ptr, block] : block_map_) {
        if (block->is_free) {
            stats.num_free_blocks++;
        }
    }

    stats.fragmentation_ratio = calculate_fragmentation();

    return stats;
}

auto PinnedMemoryAllocator::is_valid() const -> bool {
    return is_initialized_;
}

// =============================================================================
// Internal Implementation
// =============================================================================

auto PinnedMemoryAllocator::find_best_fit(size_t size) -> MemoryBlock* {
    MemoryBlock* best_fit = nullptr;
    size_t min_size_diff = SIZE_MAX;

    // Search all free blocks for best fit
    for (auto& [ptr, block] : block_map_) {
        if (block->is_free && block->size >= size) {
            size_t size_diff = block->size - size;
            if (size_diff < min_size_diff) {
                min_size_diff = size_diff;
                best_fit = block.get();

                // Perfect fit, stop searching
                if (size_diff == 0) {
                    break;
                }
            }
        }
    }

    return best_fit;
}

auto PinnedMemoryAllocator::split_block(MemoryBlock* block, size_t size) -> MemoryBlock* {
    if (block->size <= size) {
        return nullptr;  // No split needed
    }

    size_t remainder_size = block->size - size;
    if (remainder_size < config_.min_block_size) {
        return nullptr;  // Remainder too small
    }

    // Create new block for remainder
    void* remainder_ptr = static_cast<char*>(block->ptr) + size;
    auto remainder = std::make_unique<MemoryBlock>(remainder_ptr, remainder_size, true);
    auto* remainder_raw = remainder.get();

    // Update linked list
    remainder_raw->next = block->next;
    remainder_raw->prev = block;
    if (block->next) {
        block->next->prev = remainder_raw;
    }
    block->next = remainder_raw;

    // Update original block size
    block->size = size;

    // Add to block map (unique_ptr handles lifetime)
    block_map_[remainder_ptr] = std::move(remainder);

    return remainder_raw;
}

auto PinnedMemoryAllocator::coalesce_block(MemoryBlock* block) -> void {
    if (!block || !block->is_free) {
        return;
    }

    // Coalesce with next block
    while (block->next && block->next->is_free) {
        void* expected_next = static_cast<char*>(block->ptr) + block->size;
        if (expected_next == block->next->ptr) {
            MemoryBlock* next = block->next;

            // Merge next into block
            block->size += next->size;
            block->next = next->next;
            if (next->next) {
                next->next->prev = block;
            }

            // Remove next from map (unique_ptr handles deallocation)
            void* next_ptr_key = next->ptr;
            block_map_.erase(next_ptr_key);
        } else {
            break;  // Not adjacent
        }
    }

    // Coalesce with previous block
    while (block->prev && block->prev->is_free) {
        void* expected_next = static_cast<char*>(block->prev->ptr) + block->prev->size;
        if (expected_next == block->ptr) {
            MemoryBlock* prev = block->prev;

            // Merge block into prev
            prev->size += block->size;
            prev->next = block->next;
            if (block->next) {
                block->next->prev = prev;
            }

            // Remove current from map (unique_ptr handles deallocation)
            void* block_ptr_key = block->ptr;
            block_map_.erase(block_ptr_key);

            block = prev;  // Continue with merged block
        } else {
            break;  // Not adjacent
        }
    }
}

auto PinnedMemoryAllocator::find_block(void* ptr) -> MemoryBlock* {
    auto it = block_map_.find(ptr);
    if (it != block_map_.end()) {
        return it->second.get();
    }
    return nullptr;
}

auto PinnedMemoryAllocator::allocate_cuda_pinned(size_t size) -> void* {
#ifdef TENZOR_USE_CUDA
    void* ptr = nullptr;
    cudaError_t error = cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
    check_cuda_error(error, "cudaHostAlloc");
    return ptr;
#else
    // Fallback to regular allocation if CUDA not available
    void* ptr = std::malloc(size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
#endif
}

auto PinnedMemoryAllocator::free_cuda_pinned(void* ptr) -> void {
    if (!ptr) {
        return;
    }

#ifdef TENZOR_USE_CUDA
    cudaError_t error = cudaFreeHost(ptr);
    check_cuda_error(error, "cudaFreeHost");
#else
    std::free(ptr);
#endif
}

auto PinnedMemoryAllocator::calculate_fragmentation() const -> float {
    if (block_map_.empty()) {
        return 0.0f;
    }

    // Count free blocks
    size_t num_free_blocks = 0;
    for (const auto& [ptr, block] : block_map_) {
        if (block->is_free) {
            num_free_blocks++;
        }
    }

    if (num_free_blocks <= 1) {
        return 0.0f;  // No fragmentation
    }

    // Calculate fragmentation ratio
    // More free blocks = more fragmentation
    size_t max_fragments = block_map_.size();
    float ratio = static_cast<float>(num_free_blocks - 1) /
                  static_cast<float>(max_fragments);

    return std::min(ratio, 1.0f);
}

} // namespace tenzor::core
