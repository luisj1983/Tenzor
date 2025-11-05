#include "tenzor/core/caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include <algorithm>
#include <sstream>

namespace tenzor {

CachingAllocator::CachingAllocator(Backend* backend, Device device)
    : backend_(backend), device_(device) {
    if (!backend) {
        throw std::invalid_argument("CachingAllocator: backend cannot be null");
    }
}

CachingAllocator::~CachingAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free all cached blocks
    free_cached_blocks();

    // Free any remaining allocated blocks (leaked memory)
    for (const auto& [ptr, size] : allocated_blocks_) {
        if (ptr) {
            backend_->deallocate(ptr);
        }
    }
    allocated_blocks_.clear();
}

CachingAllocator::CachingAllocator(CachingAllocator&& other) noexcept
    : backend_(other.backend_),
      device_(other.device_),
      free_blocks_(std::move(other.free_blocks_)),
      allocated_blocks_(std::move(other.allocated_blocks_)),
      total_allocations_(other.total_allocations_),
      cache_hits_(other.cache_hits_),
      backend_allocations_(other.backend_allocations_),
      total_allocated_bytes_(other.total_allocated_bytes_),
      total_cached_bytes_(other.total_cached_bytes_) {
    other.backend_ = nullptr;
    other.total_allocations_ = 0;
    other.cache_hits_ = 0;
    other.backend_allocations_ = 0;
    other.total_allocated_bytes_ = 0;
    other.total_cached_bytes_ = 0;
}

CachingAllocator& CachingAllocator::operator=(CachingAllocator&& other) noexcept {
    if (this != &other) {
        // Lock both mutexes
        std::scoped_lock lock(mutex_, other.mutex_);

        // Free current resources
        free_cached_blocks();
        for (const auto& [ptr, size] : allocated_blocks_) {
            if (ptr && backend_) {
                backend_->deallocate(ptr);
            }
        }

        // Move from other
        backend_ = other.backend_;
        device_ = other.device_;
        free_blocks_ = std::move(other.free_blocks_);
        allocated_blocks_ = std::move(other.allocated_blocks_);
        total_allocations_ = other.total_allocations_;
        cache_hits_ = other.cache_hits_;
        backend_allocations_ = other.backend_allocations_;
        total_allocated_bytes_ = other.total_allocated_bytes_;
        total_cached_bytes_ = other.total_cached_bytes_;

        // Reset other
        other.backend_ = nullptr;
        other.total_allocations_ = 0;
        other.cache_hits_ = 0;
        other.backend_allocations_ = 0;
        other.total_allocated_bytes_ = 0;
        other.total_cached_bytes_ = 0;
    }
    return *this;
}

auto CachingAllocator::allocate(size_t bytes) -> void* {
    if (bytes == 0) {
        throw std::invalid_argument("CachingAllocator::allocate: bytes must be > 0");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    ++total_allocations_;

    // Try to reuse cached block
    if (void* cached_ptr = find_free_block(bytes)) {
        ++cache_hits_;
        return cached_ptr;
    }

    // No suitable cached block, allocate from backend
    void* ptr = nullptr;
    try {
        ptr = backend_->allocate(bytes, device_.index);
        if (!ptr) {
            throw std::runtime_error("Backend returned null pointer");
        }
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "CachingAllocator::allocate: Failed to allocate " << bytes
            << " bytes on device " << device_.to_string() << ": " << e.what();
        throw std::runtime_error(oss.str());
    }

    // Track allocation
    allocated_blocks_[ptr] = bytes;
    total_allocated_bytes_ += bytes;
    ++backend_allocations_;

    return ptr;
}

auto CachingAllocator::deallocate(void* ptr) -> void {
    if (!ptr) {
        return;  // Deallocating null is a no-op
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find the block in allocated_blocks_
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        throw std::runtime_error(
            "CachingAllocator::deallocate: Pointer not allocated by this allocator"
        );
    }

    size_t size = it->second;

    // Move from allocated to free pool instead of freeing immediately
    allocated_blocks_.erase(it);
    free_blocks_.insert({size, ptr});
    total_cached_bytes_ += size;
}

auto CachingAllocator::defragment() -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    free_cached_blocks();
}

auto CachingAllocator::total_allocated_bytes() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_bytes_;
}

auto CachingAllocator::total_cached_bytes() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_cached_bytes_;
}

auto CachingAllocator::allocated_block_count() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_blocks_.size();
}

auto CachingAllocator::cached_block_count() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
}

auto CachingAllocator::cache_hit_rate() const -> double {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_allocations_ == 0) {
        return 0.0;
    }
    return (static_cast<double>(cache_hits_) / static_cast<double>(total_allocations_)) * 100.0;
}

auto CachingAllocator::find_free_block(size_t bytes) -> void* {
    // Use lower_bound to find first block with size >= bytes
    // This implements a best-fit strategy for efficient memory reuse
    auto it = free_blocks_.lower_bound(bytes);

    if (it == free_blocks_.end()) {
        return nullptr;  // No suitable block found
    }

    // Found a suitable block
    void* ptr = it->second;
    size_t block_size = it->first;

    // Remove from free blocks
    free_blocks_.erase(it);
    total_cached_bytes_ -= block_size;

    // Add back to allocated_blocks_ since we're reusing it
    allocated_blocks_[ptr] = block_size;

    return ptr;
}

auto CachingAllocator::size_of(void* ptr) const -> size_t {
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        throw std::runtime_error(
            "CachingAllocator::size_of: Pointer not found in allocated blocks"
        );
    }
    return it->second;
}

auto CachingAllocator::free_cached_blocks() -> void {
    // Free all blocks in the free pool
    for (const auto& [size, ptr] : free_blocks_) {
        if (ptr && backend_) {
            backend_->deallocate(ptr);

            // Remove from allocated_blocks_
            auto alloc_it = allocated_blocks_.find(ptr);
            if (alloc_it != allocated_blocks_.end()) {
                total_allocated_bytes_ -= alloc_it->second;
                allocated_blocks_.erase(alloc_it);
            }
        }
    }

    free_blocks_.clear();
    total_cached_bytes_ = 0;
}

} // namespace tenzor
