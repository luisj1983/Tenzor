/**
 * @file test_caching_allocator.cpp
 * @brief Unit tests for CachingAllocator memory pooling
 */

#include <gtest/gtest.h>
#include "tenzor/core/caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include <cstring>
#include "tenzor/core/device.hpp"
#include <memory>
#include <vector>
#include <unordered_set>
#include <thread>

namespace tenzor {
namespace test {

/**
 * @brief Mock backend for testing CachingAllocator
 *
 * Tracks allocation/deallocation calls and simulates memory operations
 * without requiring actual GPU hardware.
 */
class MockBackend : public Backend {
public:
    // Track allocations and deallocations
    size_t allocation_count{0};
    size_t deallocation_count{0};
    std::unordered_set<void*> active_allocations;

    auto name() const -> std::string_view override {
        return "mock";
    }

    auto device_count() const -> int32_t override {
        return 1;
    }

    auto is_available() const -> bool override {
        return true;
    }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        (void)device_id;  // Unused
        void* ptr = ::operator new(bytes);  // Use standard new
        active_allocations.insert(ptr);
        ++allocation_count;
        return ptr;
    }

    auto deallocate(void* ptr) -> void override {
        auto it = active_allocations.find(ptr);
        if (it != active_allocations.end()) {
            ::operator delete(ptr);  // Use standard delete
            active_allocations.erase(it);
            ++deallocation_count;
        }
    }

    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override {
        (void)kind;  // Unused
        memcpy(dst, src, bytes);
    }

    auto synchronize(int32_t device_id) -> void override {
        (void)device_id;  // Unused
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        (void)device_id;  // Unused
        return nullptr;
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        (void)stream;  // Unused
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        (void)stream;  // Unused
    }

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        (void)op_name;
        (void)inputs;
        (void)attrs;
        return {};
    }
};

class CachingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = std::make_unique<MockBackend>();
        device_ = Device::cpu();
    }

    void TearDown() override {
        allocator_.reset();
        backend_.reset();
    }

    std::unique_ptr<MockBackend> backend_;
    Device device_;
    std::unique_ptr<CachingAllocator> allocator_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(CachingAllocatorTest, ConstructorValidBackend) {
    EXPECT_NO_THROW({
        allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);
    });
    EXPECT_EQ(allocator_->device(), device_);
}

TEST_F(CachingAllocatorTest, ConstructorNullBackend) {
    EXPECT_THROW(
        CachingAllocator(nullptr, device_),
        std::invalid_argument
    );
}

TEST_F(CachingAllocatorTest, AllocateBasic) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    void* ptr = allocator_->allocate(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(backend_->allocation_count, 1);
    EXPECT_EQ(allocator_->allocated_block_count(), 1);
    EXPECT_EQ(allocator_->total_allocated_bytes(), 1024);
}

TEST_F(CachingAllocatorTest, AllocateZeroBytes) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    EXPECT_THROW(
        allocator_->allocate(0),
        std::invalid_argument
    );
}

TEST_F(CachingAllocatorTest, DeallocateBasic) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    void* ptr = allocator_->allocate(1024);
    EXPECT_NO_THROW(allocator_->deallocate(ptr));

    // Should be cached, not freed
    EXPECT_EQ(backend_->deallocation_count, 0);
    EXPECT_EQ(allocator_->cached_block_count(), 1);
    EXPECT_EQ(allocator_->total_cached_bytes(), 1024);
}

TEST_F(CachingAllocatorTest, DeallocateNull) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Deallocating null should be a no-op
    EXPECT_NO_THROW(allocator_->deallocate(nullptr));
}

TEST_F(CachingAllocatorTest, DeallocateInvalidPointer) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    int dummy;
    void* invalid_ptr = &dummy;

    EXPECT_THROW(
        allocator_->deallocate(invalid_ptr),
        std::runtime_error
    );
}

// ============================================================================
// Memory Reuse Tests
// ============================================================================

TEST_F(CachingAllocatorTest, ReuseExactSize) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Allocate and deallocate
    void* ptr1 = allocator_->allocate(1024);
    allocator_->deallocate(ptr1);

    EXPECT_EQ(backend_->allocation_count, 1);

    // Second allocation should reuse cached block
    void* ptr2 = allocator_->allocate(1024);
    EXPECT_EQ(ptr2, ptr1);  // Should be the same pointer
    EXPECT_EQ(backend_->allocation_count, 1);  // No new allocation
    EXPECT_EQ(allocator_->cached_block_count(), 0);  // Cache consumed
}

TEST_F(CachingAllocatorTest, ReuseLargerCachedBlock) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Allocate and deallocate larger block
    void* ptr1 = allocator_->allocate(2048);
    allocator_->deallocate(ptr1);

    // Request smaller size - should reuse larger block
    void* ptr2 = allocator_->allocate(1024);
    EXPECT_EQ(ptr2, ptr1);
    EXPECT_EQ(backend_->allocation_count, 1);
}

TEST_F(CachingAllocatorTest, NoReuseWhenSizeTooSmall) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Allocate and deallocate small block
    void* ptr1 = allocator_->allocate(512);
    allocator_->deallocate(ptr1);

    // Request larger size - cannot reuse
    void* ptr2 = allocator_->allocate(1024);
    EXPECT_NE(ptr2, ptr1);
    EXPECT_EQ(backend_->allocation_count, 2);
    EXPECT_EQ(allocator_->cached_block_count(), 1);  // Original still cached
}

TEST_F(CachingAllocatorTest, BestFitStrategy) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Create blocks of different sizes
    void* ptr512 = allocator_->allocate(512);
    void* ptr1024 = allocator_->allocate(1024);
    void* ptr2048 = allocator_->allocate(2048);

    // Deallocate all
    allocator_->deallocate(ptr512);
    allocator_->deallocate(ptr1024);
    allocator_->deallocate(ptr2048);

    EXPECT_EQ(allocator_->cached_block_count(), 3);

    // Request 1024 - should get exact match (best fit)
    void* reused = allocator_->allocate(1024);
    EXPECT_EQ(reused, ptr1024);

    // Should still have 512 and 2048 cached
    EXPECT_EQ(allocator_->cached_block_count(), 2);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(CachingAllocatorTest, CacheHitRate) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Initially 0.0
    EXPECT_DOUBLE_EQ(allocator_->cache_hit_rate(), 0.0);

    // First allocation - miss
    void* ptr1 = allocator_->allocate(1024);
    EXPECT_DOUBLE_EQ(allocator_->cache_hit_rate(), 0.0);

    // Deallocate and reallocate - hit
    allocator_->deallocate(ptr1);
    void* ptr2 = allocator_->allocate(1024);
    EXPECT_DOUBLE_EQ(allocator_->cache_hit_rate(), 50.0);  // 1 hit out of 2

    // Another miss
    void* ptr3 = allocator_->allocate(2048);
    EXPECT_DOUBLE_EQ(allocator_->cache_hit_rate(), 100.0 / 3.0);  // 1 hit out of 3

    (void)ptr2;
    (void)ptr3;
}

TEST_F(CachingAllocatorTest, TotalAllocatedBytes) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    allocator_->allocate(1024);
    EXPECT_EQ(allocator_->total_allocated_bytes(), 1024);

    allocator_->allocate(2048);
    EXPECT_EQ(allocator_->total_allocated_bytes(), 3072);

    allocator_->allocate(512);
    EXPECT_EQ(allocator_->total_allocated_bytes(), 3584);
}

TEST_F(CachingAllocatorTest, TotalCachedBytes) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    void* ptr1 = allocator_->allocate(1024);
    void* ptr2 = allocator_->allocate(2048);

    EXPECT_EQ(allocator_->total_cached_bytes(), 0);

    allocator_->deallocate(ptr1);
    EXPECT_EQ(allocator_->total_cached_bytes(), 1024);

    allocator_->deallocate(ptr2);
    EXPECT_EQ(allocator_->total_cached_bytes(), 3072);
}

TEST_F(CachingAllocatorTest, BlockCounts) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    void* ptr1 = allocator_->allocate(1024);
    void* ptr2 = allocator_->allocate(2048);
    void* ptr3 = allocator_->allocate(512);

    EXPECT_EQ(allocator_->allocated_block_count(), 3);
    EXPECT_EQ(allocator_->cached_block_count(), 0);

    allocator_->deallocate(ptr1);
    EXPECT_EQ(allocator_->allocated_block_count(), 3);
    EXPECT_EQ(allocator_->cached_block_count(), 1);

    allocator_->deallocate(ptr2);
    allocator_->deallocate(ptr3);
    EXPECT_EQ(allocator_->allocated_block_count(), 3);
    EXPECT_EQ(allocator_->cached_block_count(), 3);
}

// ============================================================================
// Defragmentation Tests
// ============================================================================

TEST_F(CachingAllocatorTest, DefragmentBasic) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    void* ptr1 = allocator_->allocate(1024);
    void* ptr2 = allocator_->allocate(2048);

    allocator_->deallocate(ptr1);
    allocator_->deallocate(ptr2);

    EXPECT_EQ(allocator_->cached_block_count(), 2);
    EXPECT_EQ(backend_->deallocation_count, 0);

    // Defragment should free all cached blocks
    allocator_->defragment();

    EXPECT_EQ(allocator_->cached_block_count(), 0);
    EXPECT_EQ(allocator_->total_cached_bytes(), 0);
    EXPECT_EQ(backend_->deallocation_count, 2);
}

TEST_F(CachingAllocatorTest, DefragmentDoesNotFreeActiveBlocks) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    void* ptr1 = allocator_->allocate(1024);
    void* ptr2 = allocator_->allocate(2048);

    allocator_->deallocate(ptr1);  // Only deallocate one

    allocator_->defragment();

    // Only the deallocated block should be freed
    EXPECT_EQ(backend_->deallocation_count, 1);
    EXPECT_EQ(allocator_->allocated_block_count(), 1);  // ptr2 still allocated

    (void)ptr2;
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(CachingAllocatorTest, MoveConstructor) {
    auto alloc1 = std::make_unique<CachingAllocator>(backend_.get(), device_);
    void* ptr = alloc1->allocate(1024);

    size_t initial_allocs = backend_->allocation_count;

    // Move construct
    auto alloc2 = std::make_unique<CachingAllocator>(std::move(*alloc1));

    EXPECT_EQ(alloc2->allocated_block_count(), 1);
    EXPECT_EQ(backend_->allocation_count, initial_allocs);

    // Original should be in moved-from state
    EXPECT_EQ(alloc1->allocated_block_count(), 0);

    (void)ptr;
}

TEST_F(CachingAllocatorTest, MoveAssignment) {
    auto alloc1 = std::make_unique<CachingAllocator>(backend_.get(), device_);
    void* ptr1 = alloc1->allocate(1024);

    auto alloc2 = std::make_unique<CachingAllocator>(backend_.get(), device_);
    void* ptr2 = alloc2->allocate(2048);

    // Move assign
    *alloc2 = std::move(*alloc1);

    EXPECT_EQ(alloc2->allocated_block_count(), 1);

    (void)ptr1;
    (void)ptr2;
}

// ============================================================================
// Destructor Tests
// ============================================================================

TEST_F(CachingAllocatorTest, DestructorFreesAllMemory) {
    auto alloc = std::make_unique<CachingAllocator>(backend_.get(), device_);

    alloc->allocate(1024);
    alloc->allocate(2048);
    alloc->allocate(512);

    size_t initial_allocs = backend_->allocation_count;
    EXPECT_EQ(initial_allocs, 3);

    // Destructor should free all memory
    alloc.reset();

    EXPECT_EQ(backend_->deallocation_count, 3);
    EXPECT_EQ(backend_->active_allocations.size(), 0);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(CachingAllocatorTest, ConcurrentAllocations) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    constexpr int num_threads = 4;
    constexpr int allocs_per_thread = 100;

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < allocs_per_thread; ++i) {
                void* ptr = allocator_->allocate(1024 + i * 16);
                thread_ptrs[t].push_back(ptr);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(allocator_->allocated_block_count(), num_threads * allocs_per_thread);
}

TEST_F(CachingAllocatorTest, ConcurrentDeallocations) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    constexpr int num_threads = 4;
    constexpr int allocs_per_thread = 100;

    // Pre-allocate pointers
    std::vector<std::vector<void*>> thread_ptrs(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < allocs_per_thread; ++i) {
            thread_ptrs[t].push_back(allocator_->allocate(1024));
        }
    }

    // Deallocate concurrently
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (void* ptr : thread_ptrs[t]) {
                allocator_->deallocate(ptr);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(allocator_->cached_block_count(), num_threads * allocs_per_thread);
}

TEST_F(CachingAllocatorTest, ConcurrentMixedOperations) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    constexpr int num_threads = 4;
    constexpr int ops_per_thread = 50;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            std::vector<void*> local_ptrs;
            for (int i = 0; i < ops_per_thread; ++i) {
                // Allocate
                void* ptr = allocator_->allocate(512 + i * 32);
                local_ptrs.push_back(ptr);

                // Deallocate half
                if (i % 2 == 0 && !local_ptrs.empty()) {
                    allocator_->deallocate(local_ptrs.back());
                    local_ptrs.pop_back();
                }
            }
            // Cleanup
            for (void* ptr : local_ptrs) {
                allocator_->deallocate(ptr);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All allocations should be tracked
    EXPECT_GT(allocator_->allocated_block_count(), 0);
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(CachingAllocatorTest, LargeAllocation) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    constexpr size_t large_size = 1024 * 1024 * 100;  // 100 MB

    void* ptr = allocator_->allocate(large_size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(allocator_->total_allocated_bytes(), large_size);

    allocator_->deallocate(ptr);
    EXPECT_EQ(allocator_->total_cached_bytes(), large_size);
}

TEST_F(CachingAllocatorTest, ManySmallAllocations) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    constexpr int count = 10000;
    std::vector<void*> ptrs;

    for (int i = 0; i < count; ++i) {
        ptrs.push_back(allocator_->allocate(64));
    }

    EXPECT_EQ(allocator_->allocated_block_count(), count);
    EXPECT_EQ(backend_->allocation_count, static_cast<size_t>(count));

    // Deallocate all
    for (void* ptr : ptrs) {
        allocator_->deallocate(ptr);
    }

    EXPECT_EQ(allocator_->cached_block_count(), count);
}

TEST_F(CachingAllocatorTest, FragmentationScenario) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Create fragmented memory pattern
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator_->allocate(1024 * (i + 1)));
    }

    // Deallocate every other block
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        allocator_->deallocate(ptrs[i]);
    }

    EXPECT_EQ(allocator_->cached_block_count(), 5);

    // Allocate sizes that could fit in cached blocks
    void* new_ptr1 = allocator_->allocate(1024);  // Should reuse best fit (1024 bytes)
    EXPECT_EQ(new_ptr1, ptrs[0]);

    // Allocate 3072 - should find best fit block >= 3072
    void* new_ptr2 = allocator_->allocate(3072);
    EXPECT_NE(new_ptr2, nullptr);  // Just verify it succeeded

    EXPECT_EQ(allocator_->cached_block_count(), 3);
}

} // namespace test
} // namespace tenzor
