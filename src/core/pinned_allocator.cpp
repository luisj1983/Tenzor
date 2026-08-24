/**
 * @file pinned_allocator.cpp
 * @brief Implementation of pinned memory allocator
 */

#include "tenzor/core/pinned_allocator.hpp"
#include "tenzor/core/rocm_transfer.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <cstdlib>

#ifdef TENZOR_USE_CUDA
// CUDA-provided page-locked allocator.
#else
// Non-CUDA build: use OS-level page locking.
#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <unistd.h>
#endif
#endif

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

#ifndef TENZOR_USE_CUDA
/**
 * @brief OS page size, for the non-CUDA/non-ROCm mlock/VirtualLock fallback
 * path below. sysconf(_SC_PAGESIZE) is POSIX-only; Windows has no equivalent
 * libc call, so query it via GetSystemInfo instead.
 */
inline auto os_page_size() -> size_t {
#if defined(_WIN32)
    SYSTEM_INFO info;
    ::GetSystemInfo(&info);
    return static_cast<size_t>(info.dwPageSize);
#else
    return static_cast<size_t>(::sysconf(_SC_PAGESIZE));
#endif
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
        bool via_rocm = false;
        void* base_ptr = allocate_cuda_pinned(config_.pool_size, via_rocm);

        // Create initial free block spanning entire pool
        auto initial_block = std::make_unique<MemoryBlock>(base_ptr, config_.pool_size, true);
        auto* initial_block_raw = initial_block.get();
        free_list_head_ = initial_block_raw;
        block_map_[base_ptr] = std::move(initial_block);
        index_free_block(initial_block_raw);

        // Store pool region
        PoolRegion region;
        region.base_ptr = base_ptr;
        region.size = config_.pool_size;
        region.head = initial_block_raw;
        region.via_rocm = via_rocm;
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
    free_blocks_.clear();
    block_map_.clear();

    // Free all pools
    for (auto& pool : pools_) {
        try {
            free_cuda_pinned(pool.base_ptr, pool.size, pool.via_rocm);
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
    , free_blocks_(std::move(other.free_blocks_))
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
            free_blocks_.clear();
            block_map_.clear();
            for (auto& pool : pools_) {
                try {
                    free_cuda_pinned(pool.base_ptr, pool.size, pool.via_rocm);
                } catch (...) {}
            }
        }

        // Move from other
        config_ = other.config_;
        pools_ = std::move(other.pools_);
        free_list_head_ = other.free_list_head_;
        block_map_ = std::move(other.block_map_);
        free_blocks_ = std::move(other.free_blocks_);
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
            if (config_.throw_on_oom) {
                throw std::runtime_error(
                    "PinnedMemoryAllocator: pool exhausted, requested " +
                    std::to_string(bytes) + " bytes (" +
                    std::to_string(aligned_size) + " aligned)");
            }
            return nullptr;  // Out of memory
        }
    }

    // Remove the chosen block from the free index before mutating its size
    // (split_block) or its free state below.
    unindex_free_block(block);

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

    // Bucket every block into its owning pool. Pool regions are disjoint
    // address ranges, so sort their indices once by base_ptr (O(P log P)) and
    // binary-search the owning pool for each block (O(N log P)) rather than
    // rescanning every pool per block (O(N*P)). block_map_ is keyed by pointer,
    // so each pool's bucket is already sorted by address on insertion.
    std::vector<size_t> pool_order(pools_.size());
    std::iota(pool_order.begin(), pool_order.end(), size_t{0});
    // Use std::less<void*> for a well-defined total order over pointers,
    // matching the ordering std::map<void*,...> uses for block_map_.
    const std::less<void*> ptr_less{};
    std::sort(pool_order.begin(), pool_order.end(),
              [this, &ptr_less](size_t a, size_t b) {
                  return ptr_less(pools_[a].base_ptr, pools_[b].base_ptr);
              });

    std::vector<std::vector<MemoryBlock*>> pool_blocks(pools_.size());
    for (auto& [ptr, block] : block_map_) {
        // Find the last pool whose base_ptr is <= block->ptr (the only pool
        // that could contain it, since ranges are disjoint and ordered).
        auto it = std::upper_bound(
            pool_order.begin(), pool_order.end(), block->ptr,
            [this, &ptr_less](void* value, size_t idx) {
                return ptr_less(value, pools_[idx].base_ptr);
            });
        if (it == pool_order.begin()) {
            continue;  // Below the lowest pool base; not owned by any pool.
        }
        const size_t p = *std::prev(it);
        const auto& pool = pools_[p];
        if (block->ptr >= pool.base_ptr &&
            block->ptr < static_cast<char*>(pool.base_ptr) + pool.size) {
            pool_blocks[p].push_back(block.get());
        }
    }

    for (auto& sorted_blocks : pool_blocks) {
        // Coalesce adjacent free blocks. `current` always survives a merge, so
        // advancing only when no merge happened lets a run of free blocks fold
        // into a single block.
        for (size_t i = 0; i + 1 < sorted_blocks.size(); ) {
            auto* current = sorted_blocks[i];
            auto* next = sorted_blocks[i + 1];

            bool merged = false;
            if (current->is_free && next->is_free) {
                void* current_end = static_cast<char*>(current->ptr) + current->size;
                if (current_end == next->ptr) {
                    // Both neighbours are indexed; drop their stale entries
                    // before mutating sizes, then re-index the survivor.
                    unindex_free_block(current);
                    unindex_free_block(next);

                    // Coalesce: merge next into current
                    current->size += next->size;
                    current->next = next->next;
                    if (next->next) {
                        next->next->prev = current;
                    }

                    // Remove next from block map (unique_ptr handles deallocation)
                    void* next_ptr = next->ptr;
                    block_map_.erase(next_ptr);

                    index_free_block(current);

                    // Drop `next` from the local bucket and retry merging
                    // `current` with the following block.
                    sorted_blocks.erase(sorted_blocks.begin() +
                                        static_cast<std::ptrdiff_t>(i + 1));
                    coalesce_count++;
                    merged = true;
                }
            }
            if (!merged) {
                ++i;
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

    // Clear block map (unique_ptr handles deallocation) and the free index.
    free_blocks_.clear();
    block_map_.clear();

    // Recreate initial blocks for each pool
    free_list_head_ = nullptr;
    MemoryBlock* prev_block = nullptr;

    for (auto& pool : pools_) {
        auto block = std::make_unique<MemoryBlock>(pool.base_ptr, pool.size, true);
        auto* block_raw = block.get();
        block_map_[pool.base_ptr] = std::move(block);
        index_free_block(block_raw);
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
        bool via_rocm = false;
        void* new_ptr = allocate_cuda_pinned(additional_bytes, via_rocm);

        // Create new block
        auto new_block = std::make_unique<MemoryBlock>(new_ptr, additional_bytes, true);
        auto* new_block_raw = new_block.get();
        block_map_[new_ptr] = std::move(new_block);
        index_free_block(new_block_raw);

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
        region.via_rocm = via_rocm;
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

auto PinnedMemoryAllocator::index_free_block(MemoryBlock* block) -> void {
    if (block == nullptr || block->indexed) {
        return;
    }
    block->free_it = free_blocks_.emplace(block->size, block);
    block->indexed = true;
}

auto PinnedMemoryAllocator::unindex_free_block(MemoryBlock* block) -> void {
    if (block == nullptr || !block->indexed) {
        return;
    }
    free_blocks_.erase(block->free_it);
    block->indexed = false;
}

auto PinnedMemoryAllocator::find_best_fit(size_t size) -> MemoryBlock* {
    // free_blocks_ is ordered by block size, so the first entry whose key is
    // >= size is the smallest free block that fits (best fit) -- O(log N).
    auto it = free_blocks_.lower_bound(size);
    if (it == free_blocks_.end()) {
        return nullptr;
    }
    return it->second;
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

    // The remainder is free -- register it in the size-ordered free index.
    index_free_block(remainder_raw);

    return remainder_raw;
}

auto PinnedMemoryAllocator::coalesce_block(MemoryBlock* block) -> void {
    if (!block || !block->is_free) {
        return;
    }

    // The incoming `block` must NOT be present in free_blocks_ when this is
    // called (deallocate marks it free but leaves it unindexed). We merge any
    // adjacent free neighbours -- each of which IS indexed -- removing their
    // index entries as they are absorbed, then index the final merged block
    // exactly once at the end. This keeps free_blocks_ consistent with a single
    // O(log N) update per merge instead of an O(N) rebuild.

    // Coalesce with next block
    while (block->next && block->next->is_free) {
        void* expected_next = static_cast<char*>(block->ptr) + block->size;
        if (expected_next == block->next->ptr) {
            MemoryBlock* next = block->next;

            // Drop the absorbed neighbour from the size-ordered free index.
            unindex_free_block(next);

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

            // The surviving block becomes `prev`; remove its stale index entry
            // since its size is about to change.
            unindex_free_block(prev);

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

    // Index the final merged block under its final size.
    index_free_block(block);
}

auto PinnedMemoryAllocator::find_block(void* ptr) -> MemoryBlock* {
    auto it = block_map_.find(ptr);
    if (it != block_map_.end()) {
        return it->second.get();
    }
    return nullptr;
}

auto PinnedMemoryAllocator::allocate_cuda_pinned(size_t size, bool& out_via_rocm) -> void* {
    out_via_rocm = false;
#ifdef TENZOR_USE_CUDA
    void* ptr = nullptr;
    cudaError_t error = cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
    check_cuda_error(error, "cudaHostAlloc");
    return ptr;
#else
    // ROCm build: hipHostMalloc gives real DMA-capable pinned memory (like
    // cudaHostAlloc), so try it before falling back to OS-level mlock (which
    // merely prevents paging — no DMA acceleration). No-op / returns nullptr
    // when ROCm isn't linked in.
    if (void* rocm_ptr = rocm_transfer::host_malloc(size)) {
        out_via_rocm = true;
        return rocm_ptr;
    }
    // Non-CUDA, non-ROCm build: provide REAL page-locked host memory via OS-level
    // page locking (POSIX mlock / Windows VirtualLock) rather than
    // silently returning a regular std::malloc that pretends to be
    // pinned but isn't (audit item F.9).  Users get the async-transfer
    // guarantees they asked for; if the OS refuses (RLIMIT_MEMLOCK,
    // unprivileged, etc.) we error rather than degrading silently.
    //
    // Allocate page-aligned so mlock can lock the whole region.
    const size_t page_size = os_page_size();
    const size_t aligned_size = (size + page_size - 1) / page_size * page_size;

#if defined(_WIN32)
    void* ptr = ::VirtualAlloc(nullptr, aligned_size,
                                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ptr) throw std::bad_alloc();
    if (!::VirtualLock(ptr, aligned_size)) {
        ::VirtualFree(ptr, 0, MEM_RELEASE);
        throw std::runtime_error(
            "PinnedMemoryAllocator: VirtualLock failed — increase the "
            "process working-set size or build with CUDA support.");
    }
#else
    void* ptr = nullptr;
    int rc = ::posix_memalign(&ptr, page_size, aligned_size);
    if (rc != 0 || !ptr) throw std::bad_alloc();
    if (::mlock(ptr, aligned_size) != 0) {
        std::free(ptr);
        throw std::runtime_error(
            std::string("PinnedMemoryAllocator: mlock failed (") +
            std::strerror(errno) +
            ") — raise RLIMIT_MEMLOCK or build with CUDA support.");
    }
#endif
    return ptr;
#endif
}

auto PinnedMemoryAllocator::free_cuda_pinned(void* ptr, size_t size, bool via_rocm) -> void {
    if (!ptr) {
        return;
    }

#ifdef TENZOR_USE_CUDA
    (void)size;
    (void)via_rocm;
    cudaError_t error = cudaFreeHost(ptr);
    check_cuda_error(error, "cudaFreeHost");
#else
    if (via_rocm) {
        (void)size;
        rocm_transfer::host_free(ptr);
        return;
    }
    // Mirror the non-CUDA, non-ROCm allocation path: unlock the WHOLE region then free.
    // allocate_cuda_pinned() locked [ptr, ptr+aligned_size) with
    // mlock/VirtualLock, so we must unlock the same span here. munlock(2) only
    // unlocks the pages in [addr, addr+len); passing len==1 unlocks just the
    // first page and leaves the rest locked, slowly draining the
    // RLIMIT_MEMLOCK budget when free() returns the region to the heap
    // free-list instead of unmapping it. Recompute the same page-aligned size
    // the allocation used so the unlock span matches the lock span exactly.
    const size_t page_size = os_page_size();
    const size_t aligned_size = (size + page_size - 1) / page_size * page_size;
#if defined(_WIN32)
    // VirtualUnlock undoes the VirtualLock; VirtualFree(_, 0, RELEASE)
    // releases the reservation regardless of size.
    ::VirtualUnlock(ptr, aligned_size);
    ::VirtualFree(ptr, 0, MEM_RELEASE);
#else
    ::munlock(ptr, aligned_size);
    std::free(ptr);
#endif
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
