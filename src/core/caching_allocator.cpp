#include "tenzor/core/caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/utils/logging.hpp"
#include "tenzor/utils/memory_profiler.hpp"
#include <cstdlib>
#include <cstdio>
#include <sstream>

namespace tenzor {

namespace {

/// Check TENZOR_ALLOCATOR_DEBUG environment variable (cached after first call)
bool is_allocator_debug_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("TENZOR_ALLOCATOR_DEBUG");
        return env && std::string_view(env) == "1";
    }();
    return enabled;
}

/// Interval between diagnostic log messages (in allocation count)
constexpr size_t kDebugLogInterval = 1000;

} // anonymous namespace

CachingAllocator::CachingAllocator(Backend* backend, Device device)
    : backend_(backend), device_(device), config_() {
    if (!backend) {
        throw std::invalid_argument("CachingAllocator: backend cannot be null");
    }
}

CachingAllocator::CachingAllocator(Backend* backend, Device device, const Config& config)
    : backend_(backend), device_(device), config_(config) {
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
      config_(other.config_),
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
        // Use swap idiom for exception safety: old resources transfer to
        // 'other' and get cleaned up when 'other' is destroyed (in a
        // non-noexcept destructor context, avoiding std::terminate).
        std::scoped_lock lock(mutex_, other.mutex_);

        std::swap(backend_, other.backend_);
        std::swap(device_, other.device_);
        std::swap(config_, other.config_);
        std::swap(free_blocks_, other.free_blocks_);
        std::swap(allocated_blocks_, other.allocated_blocks_);
        std::swap(total_allocations_, other.total_allocations_);
        std::swap(cache_hits_, other.cache_hits_);
        std::swap(backend_allocations_, other.backend_allocations_);
        std::swap(total_allocated_bytes_, other.total_allocated_bytes_);
        std::swap(total_cached_bytes_, other.total_cached_bytes_);
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
    void* result_ptr = find_free_block(bytes);
    if (result_ptr) {
        ++cache_hits_;
        // The reused block becomes live again; record it with the memory
        // profiler so on_allocate/on_deallocate stay balanced across reuse
        // (deallocate() always emits on_deallocate). Use the actual block
        // size that find_free_block recorded in allocated_blocks_.
        MemoryProfiler::instance().on_allocate(allocated_blocks_[result_ptr]);
    } else {
        // No suitable cached block, allocate from backend
        try {
            result_ptr = backend_->allocate(bytes, device_.index);
            if (!result_ptr) {
                throw std::runtime_error("Backend returned null pointer");
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CachingAllocator::allocate: Failed to allocate " << bytes
                << " bytes on device " << device_.to_string() << ": " << e.what();
            throw std::runtime_error(oss.str());
        }

        // Track allocation
        allocated_blocks_[result_ptr] = bytes;
        total_allocated_bytes_ += bytes;
        ++backend_allocations_;

        // Record in global memory profiler
        MemoryProfiler::instance().on_allocate(bytes);
    }

    // Periodic diagnostics when TENZOR_ALLOCATOR_DEBUG=1
    if (is_allocator_debug_enabled() && (total_allocations_ % kDebugLogInterval) == 0) {
        double hit_rate = (total_allocations_ > 0)
            ? (static_cast<double>(cache_hits_) / static_cast<double>(total_allocations_)) * 100.0
            : 0.0;
        double frag_ratio = (total_allocated_bytes_ > 0)
            ? (static_cast<double>(total_cached_bytes_) / static_cast<double>(total_allocated_bytes_)) * 100.0
            : 0.0;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "CachingAllocator[%s] allocs=%zu hit_rate=%.1f%% peak=%zuMB "
            "cached=%zuMB fragmentation=%.1f%% blocks(alloc=%zu free=%zu)",
            device_.to_string().c_str(),
            total_allocations_, hit_rate,
            total_allocated_bytes_ / (1024 * 1024),
            total_cached_bytes_ / (1024 * 1024),
            frag_ratio,
            allocated_blocks_.size(), free_blocks_.size());
        TENZOR_LOG_INFO(std::string(buf));
    }

    return result_ptr;
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

    // Auto-defragment if cached memory exceeds configured limit
    if (config_.auto_defragment && config_.max_cached_bytes > 0 &&
        total_cached_bytes_ > config_.max_cached_bytes) {
        free_cached_blocks();
    }

    // Record in global memory profiler
    MemoryProfiler::instance().on_deallocate(size);
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

    // Return the full block — splitting would create sub-pointers that
    // cannot be individually freed back to the backend allocator.
    allocated_blocks_[ptr] = block_size;

    return ptr;
}

auto CachingAllocator::size_of(void* ptr) const -> size_t {
    // Lock like the other accessors: a concurrent allocate()/deallocate() can
    // rehash allocated_blocks_ and invalidate a lock-free find().
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        throw std::runtime_error(
            "CachingAllocator::size_of: Pointer not found in allocated blocks"
        );
    }
    return it->second;
}

auto CachingAllocator::free_cached_blocks() -> void {
    // Free all blocks in the free pool. deallocate() erases the pointer from
    // allocated_blocks_ before inserting it into free_blocks_, so the two maps
    // are disjoint here — looking the pointer up in allocated_blocks_ would
    // always miss and leave total_allocated_bytes_ growing monotonically.
    // Subtract using the size key already held by the free_blocks_ entry.
    for (const auto& [size, ptr] : free_blocks_) {
        if (ptr && backend_) {
            backend_->deallocate(ptr);
            total_allocated_bytes_ -= size;
        }
    }

    free_blocks_.clear();
    total_cached_bytes_ = 0;
}

} // namespace tenzor
