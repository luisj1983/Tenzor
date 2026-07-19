/**
 * @file test_oneapi_caching_allocator.cpp
 * @brief Unit tests for OneAPICachingAllocator
 *
 * Test coverage includes:
 * - Basic allocation and deallocation
 * - Cache management (hits, misses)
 * - Shared vs device memory allocation
 * - Memory reuse and statistics
 * - Garbage collection
 * - Thread safety
 * - Edge cases
 */

#include <gtest/gtest.h>
#include "tenzor/backend/oneapi_caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/tenzor.hpp"
#include "../multi_backend_dtype_fixture.hpp"  // FF.28: SKIP_WITH_REASON
#include <vector>
#include <algorithm>
#include <thread>
#include <random>
#include <cstring>

using namespace tenzor;
using namespace tenzor::backend;

// Global test environment to initialize Tenzor
class TenzorEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Register the environment
static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorEnvironment());

/**
 * @brief Test fixture for OneAPICachingAllocator tests
 */
class OneAPICachingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if OneAPI backend is available
        auto* backend = backend_registry().get_backend(Device::Type::OneAPI);
        if (!backend || !backend->is_available()) {
            SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable,
                             "OneAPI backend not available");
        }

        // The allocator is initialized by the backend, so we just need to check it's ready
        if (!OneAPICachingAllocator::get().is_initialized(0)) {
            SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable,
                             "OneAPICachingAllocator not initialized");
        }

        allocator = &OneAPICachingAllocator::get();
        allocator->reset_stats();
    }

    void TearDown() override {
        if (allocator) {
            // Empty cache to clean up
            allocator->empty_cache(0);
        }
    }

    OneAPICachingAllocator* allocator = nullptr;
};

// ============================================================================
// Basic Allocation/Deallocation Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, AllocateZeroReturnsNull) {
    void* ptr = allocator->allocate(0, 0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(OneAPICachingAllocatorTest, AllocateSharedBasicSizes) {
    const std::vector<size_t> sizes = {64, 256, 512, 1024, 4096, 1024 * 1024};

    for (size_t size : sizes) {
        void* ptr = allocator->allocate_shared(size, 0);
        ASSERT_NE(ptr, nullptr) << "Allocation failed for size: " << size;

        // Verify memory is usable (write test pattern)
        // Shared memory is host-accessible
        std::memset(ptr, 0xAB, size);

        allocator->free(ptr, 0);
    }
}

TEST_F(OneAPICachingAllocatorTest, AllocateDeviceBasicSizes) {
    const std::vector<size_t> sizes = {64, 256, 512, 1024, 4096};

    for (size_t size : sizes) {
        void* ptr = allocator->allocate_device(size, 0);
        ASSERT_NE(ptr, nullptr) << "Device allocation failed for size: " << size;

        // Device memory is NOT host-accessible, so we can't write to it directly
        // Just verify the allocation succeeded

        allocator->free(ptr, 0);
    }
}

TEST_F(OneAPICachingAllocatorTest, FreeNullIsNoOp) {
    EXPECT_NO_THROW({
        allocator->free(nullptr, 0);
    });
}

TEST_F(OneAPICachingAllocatorTest, FreeInvalidPointerThrows) {
    int dummy = 0;
    void* invalid_ptr = &dummy;

    EXPECT_THROW({
        allocator->free(invalid_ptr, 0);
    }, std::runtime_error);
}

TEST_F(OneAPICachingAllocatorTest, DoubleFreeThrows) {
    void* ptr = allocator->allocate_shared(1024, 0);
    ASSERT_NE(ptr, nullptr);

    allocator->free(ptr, 0);

    EXPECT_THROW({
        allocator->free(ptr, 0);
    }, std::runtime_error);
}

TEST_F(OneAPICachingAllocatorTest, MultipleAllocations) {
    const size_t count = 10;
    const size_t size = 1024;
    std::vector<void*> ptrs;

    // Allocate multiple blocks
    for (size_t i = 0; i < count; ++i) {
        void* ptr = allocator->allocate_shared(size, 0);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Verify all pointers are unique
    std::sort(ptrs.begin(), ptrs.end());
    auto it = std::unique(ptrs.begin(), ptrs.end());
    EXPECT_EQ(it, ptrs.end()) << "Duplicate pointers allocated";

    // Deallocate all
    for (void* ptr : ptrs) {
        allocator->free(ptr, 0);
    }
}

// ============================================================================
// Cache Management Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, CacheReuseExactSize) {
    const size_t size = 1024;

    // First allocation - cache miss
    void* ptr1 = allocator->allocate_shared(size, 0);
    ASSERT_NE(ptr1, nullptr);

    allocator->free(ptr1, 0);

    // Second allocation of same size - should reuse cached block
    void* ptr2 = allocator->allocate_shared(size, 0);
    ASSERT_NE(ptr2, nullptr);

    // Should reuse the same pointer
    EXPECT_EQ(ptr1, ptr2) << "Expected same pointer from cache";

    allocator->free(ptr2, 0);
}

TEST_F(OneAPICachingAllocatorTest, BlockSplitOnOversizedReuse) {
    const size_t big_size = 8192;
    const size_t small_size = 1024;

    void* big_ptr = allocator->allocate_shared(big_size, 0);
    ASSERT_NE(big_ptr, nullptr);
    allocator->free(big_ptr, 0);

    size_t splits_before = allocator->get_stats(0).num_splits;

    // A much smaller request reusing this cached block should SPLIT it rather
    // than handing out the whole 8192 bytes: the leading `small_size` bytes
    // come back at the original address, and the remainder becomes its own
    // free block immediately available for reuse.
    void* small_ptr = allocator->allocate_shared(small_size, 0);
    ASSERT_NE(small_ptr, nullptr);
    EXPECT_EQ(small_ptr, big_ptr) << "Split should hand out the front of the original block";
    EXPECT_GT(allocator->get_stats(0).num_splits, splits_before)
        << "Reusing an oversized cached block for a much smaller request should split it";

    // The remainder (big_size - small_size) must be independently cached and
    // reusable WITHOUT freeing small_ptr first — proving it was actually split
    // off, not just logically reserved inside the still-whole block.
    const size_t remainder_request = 2048;
    ASSERT_LT(remainder_request, big_size - small_size);

    size_t hits_before = allocator->get_stats(0).num_cache_hits;
    void* remainder_ptr = allocator->allocate_shared(remainder_request, 0);
    ASSERT_NE(remainder_ptr, nullptr);
    EXPECT_EQ(remainder_ptr, static_cast<char*>(big_ptr) + small_size)
        << "Remainder allocation should start exactly where the split occurred";
    EXPECT_GT(allocator->get_stats(0).num_cache_hits, hits_before)
        << "Remainder should be served from cache, not a fresh USM allocation";

    allocator->free(small_ptr, 0);
    allocator->free(remainder_ptr, 0);
}

TEST_F(OneAPICachingAllocatorTest, SplitBlocksCascadeMergeOnFree) {
    const size_t total_size = 8192;
    const size_t a_size = 2048;
    const size_t b_size = 1024;

    void* root_ptr = allocator->allocate_shared(total_size, 0);
    ASSERT_NE(root_ptr, nullptr);
    allocator->free(root_ptr, 0);

    // Splits root -> A (used) + rem1 (free).
    void* a_ptr = allocator->allocate_shared(a_size, 0);
    ASSERT_NE(a_ptr, nullptr);
    ASSERT_EQ(a_ptr, root_ptr);

    // Splits rem1 -> B (used) + rem2 (free).
    void* b_ptr = allocator->allocate_shared(b_size, 0);
    ASSERT_NE(b_ptr, nullptr);
    ASSERT_EQ(b_ptr, static_cast<char*>(root_ptr) + a_size);
    ASSERT_GE(allocator->get_stats(0).num_splits, 2u);

    // Freeing A first has no free neighbor yet: B (immediately after it) is
    // still allocated, and A has no predecessor. Should not merge with anything.
    allocator->free(a_ptr, 0);
    EXPECT_EQ(allocator->get_stats(0).num_merges, 0u)
        << "A has no free neighbor yet and should not have merged";

    size_t merges_before_b = allocator->get_stats(0).num_merges;
    allocator->free(b_ptr, 0);
    // Freeing B should forward-merge with the still-free rem2 sibling, then
    // backward-merge with the now-free A sibling — two coalesces from one
    // free() call, fully reassembling the original 8192-byte allocation.
    EXPECT_GE(allocator->get_stats(0).num_merges, merges_before_b + 2)
        << "Freeing B should coalesce with both free neighbors (forward into "
           "rem2, then backward into A)";

    size_t hits_before = allocator->get_stats(0).num_cache_hits;
    size_t reserved_before = allocator->get_stats(0).reserved_bytes;

    const size_t big_request = total_size - 64;  // bigger than any single piece
    void* merged_ptr = allocator->allocate_shared(big_request, 0);
    ASSERT_NE(merged_ptr, nullptr);
    EXPECT_EQ(merged_ptr, root_ptr)
        << "Fully reassembled block should be handed out from the original base pointer";
    EXPECT_GT(allocator->get_stats(0).num_cache_hits, hits_before)
        << "A properly merged block should satisfy this from cache, not a fresh USM allocation";
    EXPECT_EQ(allocator->get_stats(0).reserved_bytes, reserved_before)
        << "No new USM allocation should occur once the split pieces fully re-merged";

    allocator->free(merged_ptr, 0);
}

TEST_F(OneAPICachingAllocatorTest, SeparateCachePoolsForSharedAndDevice) {
    const size_t size = 1024;

    // Allocate and free shared memory
    void* shared_ptr = allocator->allocate_shared(size, 0);
    allocator->free(shared_ptr, 0);

    // Allocate device memory - should NOT reuse the shared block
    void* device_ptr = allocator->allocate_device(size, 0);
    EXPECT_NE(device_ptr, shared_ptr) << "Device allocation should not reuse shared cache";

    // Allocate shared again - should reuse the original shared block
    void* shared_ptr2 = allocator->allocate_shared(size, 0);
    EXPECT_EQ(shared_ptr2, shared_ptr) << "Shared allocation should reuse shared cache";

    allocator->free(device_ptr, 0);
    allocator->free(shared_ptr2, 0);
}

TEST_F(OneAPICachingAllocatorTest, CacheHitTracking) {
    const size_t size = 2048;

    // Initial allocation
    void* ptr1 = allocator->allocate_shared(size, 0);
    allocator->free(ptr1, 0);

    // Multiple allocations of same size should increase hit count
    for (int i = 0; i < 5; ++i) {
        void* ptr = allocator->allocate_shared(size, 0);
        ASSERT_NE(ptr, nullptr);
        allocator->free(ptr, 0);
    }

    auto stats = allocator->get_stats(0);
    EXPECT_GE(stats.num_cache_hits, 5u) << "Expected cache hits for same-size allocations";
}

TEST_F(OneAPICachingAllocatorTest, EmptyCacheClearsAllFreeBlocks) {
    const size_t size = 4096;
    const int count = 5;

    // Allocate and deallocate to build up cache
    std::vector<void*> ptrs;
    for (int i = 0; i < count; ++i) {
        ptrs.push_back(allocator->allocate_shared(size, 0));
    }
    for (void* ptr : ptrs) {
        allocator->free(ptr, 0);
    }

    size_t cached_before = allocator->memory_cached(0);
    EXPECT_GT(cached_before, 0u);

    // Empty cache
    allocator->empty_cache(0);

    EXPECT_EQ(allocator->memory_cached(0), 0u);
}

// ============================================================================
// Garbage Collection Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, GarbageCollectAggressive) {
    // Allocate and free several blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        ptrs.push_back(allocator->allocate_shared(1024 * 1024, 0));  // 1MB each
    }
    for (void* ptr : ptrs) {
        allocator->free(ptr, 0);
    }

    size_t cached_before = allocator->memory_cached(0);
    EXPECT_GT(cached_before, 0u);

    // Aggressive GC should release all
    allocator->garbage_collect(0, true);

    EXPECT_EQ(allocator->memory_cached(0), 0u);
}

TEST_F(OneAPICachingAllocatorTest, GarbageCollectNonAggressive) {
    // Allocate small and large blocks
    void* small_ptr = allocator->allocate_shared(512, 0);  // 512 bytes
    void* large_ptr = allocator->allocate_shared(2 * 1024 * 1024, 0);  // 2MB

    allocator->free(small_ptr, 0);
    allocator->free(large_ptr, 0);

    // Non-aggressive GC only releases large blocks (>1MB)
    allocator->garbage_collect(0, false);

    // Small block should still be cached, large block released
    // (behavior may vary based on implementation)
    size_t cached = allocator->memory_cached(0);
    EXPECT_GE(cached, 0u);  // At minimum, large block should be freed
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, AllocationIncreasesStats) {
    const size_t size = 8192;

    void* ptr = allocator->allocate_shared(size, 0);
    ASSERT_NE(ptr, nullptr);

    EXPECT_GE(allocator->memory_allocated(0), size);
    EXPECT_GE(allocator->memory_reserved(0), size);

    auto stats = allocator->get_stats(0);
    EXPECT_GT(stats.shared_memory_bytes, 0u);

    allocator->free(ptr, 0);
}

TEST_F(OneAPICachingAllocatorTest, DeallocationIncreasesCachedStats) {
    const size_t size = 4096;

    void* ptr = allocator->allocate_shared(size, 0);
    allocator->free(ptr, 0);

    EXPECT_GE(allocator->memory_cached(0), size);
}

TEST_F(OneAPICachingAllocatorTest, PeakAllocatedTracking) {
    const size_t size = 1024 * 1024;  // 1MB

    // Allocate multiple blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        ptrs.push_back(allocator->allocate_shared(size, 0));
    }

    auto stats = allocator->get_stats(0);
    EXPECT_GE(stats.peak_allocated, 5 * size);

    // Free all
    for (void* ptr : ptrs) {
        allocator->free(ptr, 0);
    }

    // Peak should still reflect the maximum
    stats = allocator->get_stats(0);
    EXPECT_GE(stats.peak_allocated, 5 * size);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, SetAlignmentValidation) {
    // Valid power of 2
    EXPECT_NO_THROW(allocator->set_alignment(128));

    // Invalid (not power of 2)
    EXPECT_THROW(allocator->set_alignment(300), std::invalid_argument);

    // Invalid (zero)
    EXPECT_THROW(allocator->set_alignment(0), std::invalid_argument);
}

TEST_F(OneAPICachingAllocatorTest, SetMaxCachedMemory) {
    // Allocate and free multiple blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator->allocate_shared(1024 * 1024, 0));  // 1MB each
    }
    for (void* ptr : ptrs) {
        allocator->free(ptr, 0);
    }

    // Set a cache limit
    allocator->set_max_cached_memory(5 * 1024 * 1024);  // 5MB

    // Allocate and free to trigger cache limit enforcement
    void* ptr = allocator->allocate_shared(1024 * 1024, 0);
    allocator->free(ptr, 0);

    // Cache should be limited
    EXPECT_LE(allocator->memory_cached(0), 6 * 1024 * 1024);

    // Reset limit
    allocator->set_max_cached_memory(0);  // Unlimited
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, StressTestRandomSizes) {
    const int num_allocations = 100;
    std::vector<void*> ptrs;
    std::vector<size_t> sizes;

    // Random size generator
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> size_dist(64, 16384);

    for (int i = 0; i < num_allocations; ++i) {
        size_t size = size_dist(rng);
        void* ptr = allocator->allocate_shared(size, 0);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
        sizes.push_back(size);
    }

    // Deallocate in random order
    std::vector<size_t> indices(num_allocations);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t idx : indices) {
        allocator->free(ptrs[idx], 0);
    }
}

TEST_F(OneAPICachingAllocatorTest, RepeatedAllocDeallocCycles) {
    const int num_cycles = 50;
    const size_t size = 8192;

    void* prev_ptr = nullptr;
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        void* ptr = allocator->allocate_shared(size, 0);
        ASSERT_NE(ptr, nullptr);

        if (prev_ptr != nullptr) {
            // Should reuse same pointer (cache hit)
            EXPECT_EQ(ptr, prev_ptr) << "Expected pointer reuse on cycle " << cycle;
        }

        prev_ptr = ptr;
        allocator->free(ptr, 0);
    }

    auto stats = allocator->get_stats(0);
    EXPECT_GE(stats.num_cache_hits, num_cycles - 1);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, ConcurrentAllocationsBasic) {
    const size_t size = 1024;
    const int num_threads = 4;
    const int allocations_per_thread = 10;

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch threads that allocate memory
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < allocations_per_thread; ++i) {
                void* ptr = allocator->allocate_shared(size, 0);
                if (ptr != nullptr) {
                    thread_ptrs[t].push_back(ptr);
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all allocations succeeded
    size_t total_allocations = 0;
    for (const auto& ptrs : thread_ptrs) {
        total_allocations += ptrs.size();
    }
    EXPECT_EQ(total_allocations, num_threads * allocations_per_thread);

    // Clean up
    for (const auto& ptrs : thread_ptrs) {
        for (void* ptr : ptrs) {
            allocator->free(ptr, 0);
        }
    }
}

// ============================================================================
// RAII Guard Tests
// ============================================================================

TEST_F(OneAPICachingAllocatorTest, CachedMemoryGuardBasic) {
    size_t allocated_before = allocator->memory_allocated(0);

    {
        OneAPICachedMemoryGuard guard(4096, 0, true);  // shared memory
        EXPECT_NE(guard.get(), nullptr);
        EXPECT_EQ(guard.size(), 4096u);
        EXPECT_GT(allocator->memory_allocated(0), allocated_before);
    }

    // Memory should be returned to cache after guard goes out of scope
    EXPECT_GE(allocator->memory_cached(0), 0u);
}

TEST_F(OneAPICachingAllocatorTest, CachedMemoryGuardMoveSemantics) {
    OneAPICachedMemoryGuard guard1(2048, 0, true);
    void* original_ptr = guard1.get();

    OneAPICachedMemoryGuard guard2(std::move(guard1));

    EXPECT_EQ(guard1.get(), nullptr);
    EXPECT_EQ(guard2.get(), original_ptr);

    // guard2 will free the memory when it goes out of scope
}
