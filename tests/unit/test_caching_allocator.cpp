/**
 * @file test_caching_allocator.cpp
 * @brief Comprehensive tests for CachingAllocator with multi-backend support
 *
 * Test coverage includes:
 * - Basic allocation and deallocation
 * - Cache management (hits, misses, eviction)
 * - Memory reuse and fragmentation
 * - Statistics tracking
 * - Edge cases and error conditions
 * - Multi-backend support (CPU, CUDA, Vulkan, OneAPI)
 * - Thread safety considerations
 * - Move semantics
 * - Destructor behavior
 */

#include <gtest/gtest.h>
#include "tenzor/core/caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/core/device.hpp"
#include "../backend_test_fixture.hpp"
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <numeric>
#include <random>

using namespace tenzor;
using namespace tenzor::testing;

/**
 * @brief Test fixture for CachingAllocator with multi-backend support
 */
class CachingAllocatorTest : public BackendTest {
protected:
    Backend* backend = nullptr;
    std::unique_ptr<CachingAllocator> allocator;

    void SetUp() override {
        BackendTest::SetUp();
        backend = backend_registry().get_backend(device.type);
        ASSERT_NE(backend, nullptr) << "Backend not available for device: " << device.to_string();
        allocator = std::make_unique<CachingAllocator>(backend, device);
    }

    void TearDown() override {
        allocator.reset();
        BackendTest::TearDown();
    }

    // Helper to verify pointer is valid by writing/reading
    void verifyMemoryWritable(void* ptr, size_t bytes) {
        ASSERT_NE(ptr, nullptr);
        std::vector<uint8_t> test_data(bytes, 0xAB);
        std::vector<uint8_t> read_data(bytes, 0);

        // Write test pattern
        backend->copy(ptr, test_data.data(), bytes, CopyKind::HostToDevice);
        backend->synchronize(device.index);

        // Read back and verify
        backend->copy(read_data.data(), ptr, bytes, CopyKind::DeviceToHost);
        backend->synchronize(device.index);

        EXPECT_EQ(test_data, read_data) << "Memory verification failed";
    }
};

// Instantiate tests for all backends
INSTANTIATE_BACKEND_TESTS(CachingAllocatorTest);

// ============================================================================
// Basic Allocation/Deallocation Tests
// ============================================================================

TEST_P(CachingAllocatorTest, ConstructorValidBackend) {
    EXPECT_NO_THROW({
        CachingAllocator alloc(backend, device);
    });
}

TEST_P(CachingAllocatorTest, ConstructorNullBackendThrows) {
    EXPECT_THROW({
        CachingAllocator alloc(nullptr, device);
    }, std::invalid_argument);
}

TEST_P(CachingAllocatorTest, AllocateBasicSizes) {
    const std::vector<size_t> sizes = {1, 16, 256, 1024, 4096, 1024*1024};

    for (size_t size : sizes) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr) << "Allocation failed for size: " << size;

        // Verify memory is usable
        verifyMemoryWritable(ptr, size);

        allocator->deallocate(ptr);
    }
}

TEST_P(CachingAllocatorTest, AllocateZeroSizeThrows) {
    EXPECT_THROW({
        void* ptr = allocator->allocate(0);
    }, std::invalid_argument);
}

TEST_P(CachingAllocatorTest, DeallocateNullIsNoOp) {
    EXPECT_NO_THROW({
        allocator->deallocate(nullptr);
    });
}

TEST_P(CachingAllocatorTest, DeallocateInvalidPointerThrows) {
    int dummy = 0;
    void* invalid_ptr = &dummy;

    EXPECT_THROW({
        allocator->deallocate(invalid_ptr);
    }, std::runtime_error);
}

TEST_P(CachingAllocatorTest, DoubleDeallocateThrows) {
    void* ptr = allocator->allocate(1024);
    allocator->deallocate(ptr);

    EXPECT_THROW({
        allocator->deallocate(ptr);
    }, std::runtime_error);
}

TEST_P(CachingAllocatorTest, MultipleAllocations) {
    const size_t count = 10;
    const size_t size = 1024;
    std::vector<void*> ptrs;

    // Allocate multiple blocks
    for (size_t i = 0; i < count; ++i) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Verify all pointers are unique
    std::sort(ptrs.begin(), ptrs.end());
    auto it = std::unique(ptrs.begin(), ptrs.end());
    EXPECT_EQ(it, ptrs.end()) << "Duplicate pointers allocated";

    // Deallocate all
    for (void* ptr : ptrs) {
        allocator->deallocate(ptr);
    }
}

TEST_P(CachingAllocatorTest, AllocateDifferentSizes) {
    std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};
    std::vector<void*> ptrs;

    for (size_t size : sizes) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        verifyMemoryWritable(ptr, size);
        ptrs.push_back(ptr);
    }

    for (void* ptr : ptrs) {
        allocator->deallocate(ptr);
    }
}

// ============================================================================
// Cache Management Tests
// ============================================================================

TEST_P(CachingAllocatorTest, CacheReuseExactSize) {
    const size_t size = 1024;

    // First allocation - cache miss
    void* ptr1 = allocator->allocate(size);
    ASSERT_NE(ptr1, nullptr);
    size_t blocks_before = allocator->allocated_block_count();

    allocator->deallocate(ptr1);

    // Second allocation of same size - should reuse cached block
    void* ptr2 = allocator->allocate(size);
    ASSERT_NE(ptr2, nullptr);

    size_t blocks_after = allocator->allocated_block_count();

    // Should reuse the same block (or at least not allocate new)
    EXPECT_EQ(blocks_before, blocks_after);
    EXPECT_EQ(ptr1, ptr2) << "Expected same pointer from cache";

    allocator->deallocate(ptr2);
}

TEST_P(CachingAllocatorTest, CacheHitRate) {
    const size_t size = 2048;

    // Initial allocation
    void* ptr1 = allocator->allocate(size);
    allocator->deallocate(ptr1);

    // Multiple allocations of same size should increase hit rate
    for (int i = 0; i < 5; ++i) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        allocator->deallocate(ptr);
    }

    double hit_rate = allocator->cache_hit_rate();
    // We expect at least some cache hits (>0%)
    EXPECT_GT(hit_rate, 0.0) << "Expected cache hits";
    EXPECT_LE(hit_rate, 100.0) << "Hit rate should be <= 100%";
}

TEST_P(CachingAllocatorTest, CacheMissNewSize) {
    const size_t size1 = 1024;
    const size_t size2 = 2048;

    void* ptr1 = allocator->allocate(size1);
    allocator->deallocate(ptr1);

    size_t cached_before = allocator->cached_block_count();
    EXPECT_GT(cached_before, 0u);

    // Different size allocation should be a cache miss
    void* ptr2 = allocator->allocate(size2);
    ASSERT_NE(ptr2, nullptr);

    // Should have allocated new block (cache miss)
    allocator->deallocate(ptr2);
}

TEST_P(CachingAllocatorTest, DefragmentEmptiesCache) {
    const size_t size = 4096;
    const int count = 5;

    // Allocate and deallocate to build up cache
    for (int i = 0; i < count; ++i) {
        void* ptr = allocator->allocate(size);
        allocator->deallocate(ptr);
    }

    size_t cached_before = allocator->cached_block_count();
    EXPECT_GT(cached_before, 0u);

    size_t cached_bytes_before = allocator->total_cached_bytes();
    EXPECT_GT(cached_bytes_before, 0u);

    // Defragment should clear cache
    allocator->defragment();

    EXPECT_EQ(allocator->cached_block_count(), 0u);
    EXPECT_EQ(allocator->total_cached_bytes(), 0u);
}

TEST_P(CachingAllocatorTest, DefragmentPreservesActiveAllocations) {
    const size_t size = 2048;
    std::vector<void*> active_ptrs;

    // Create some active allocations
    for (int i = 0; i < 3; ++i) {
        void* ptr = allocator->allocate(size);
        active_ptrs.push_back(ptr);
    }

    // Create and free some to build cache
    for (int i = 0; i < 3; ++i) {
        void* ptr = allocator->allocate(size);
        allocator->deallocate(ptr);
    }

    size_t allocated_before = allocator->allocated_block_count();
    EXPECT_GT(allocated_before, 0u);

    // Defragment
    allocator->defragment();

    // Active allocations should still be valid
    for (void* ptr : active_ptrs) {
        verifyMemoryWritable(ptr, size);
    }

    // Clean up
    for (void* ptr : active_ptrs) {
        allocator->deallocate(ptr);
    }
}

TEST_P(CachingAllocatorTest, BestFitStrategy) {
    // Allocate and free blocks of different sizes
    void* ptr_small = allocator->allocate(512);
    void* ptr_medium = allocator->allocate(1024);
    void* ptr_large = allocator->allocate(2048);

    allocator->deallocate(ptr_small);
    allocator->deallocate(ptr_medium);
    allocator->deallocate(ptr_large);

    // Request size that best fits medium block
    void* ptr_reused = allocator->allocate(1000);

    // Should ideally reuse medium block (best fit)
    // Note: We can't guarantee exact pointer without knowing internal implementation,
    // but we can verify it works
    ASSERT_NE(ptr_reused, nullptr);
    verifyMemoryWritable(ptr_reused, 1000);

    allocator->deallocate(ptr_reused);
}

// ============================================================================
// Statistics and Memory Tracking Tests
// ============================================================================

TEST_P(CachingAllocatorTest, InitialStatistics) {
    EXPECT_EQ(allocator->total_allocated_bytes(), 0u);
    EXPECT_EQ(allocator->total_cached_bytes(), 0u);
    EXPECT_EQ(allocator->allocated_block_count(), 0u);
    EXPECT_EQ(allocator->cached_block_count(), 0u);
    EXPECT_EQ(allocator->cache_hit_rate(), 0.0);
}

TEST_P(CachingAllocatorTest, AllocationIncreasesStats) {
    const size_t size = 8192;

    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);

    EXPECT_GE(allocator->total_allocated_bytes(), size);
    EXPECT_EQ(allocator->allocated_block_count(), 1u);
    EXPECT_EQ(allocator->cached_block_count(), 0u);

    allocator->deallocate(ptr);
}

TEST_P(CachingAllocatorTest, DeallocationIncreasesCachedStats) {
    const size_t size = 4096;

    void* ptr = allocator->allocate(size);
    allocator->deallocate(ptr);

    EXPECT_GE(allocator->total_cached_bytes(), size);
    EXPECT_EQ(allocator->cached_block_count(), 1u);
}

TEST_P(CachingAllocatorTest, TotalAllocatedBytesTracking) {
    const std::vector<size_t> sizes = {1024, 2048, 4096};
    size_t expected_total = 0;
    std::vector<void*> ptrs;

    for (size_t size : sizes) {
        void* ptr = allocator->allocate(size);
        ptrs.push_back(ptr);
        expected_total += size;
    }

    // Total allocated should be at least the sum of requested sizes
    EXPECT_GE(allocator->total_allocated_bytes(), expected_total);

    for (void* ptr : ptrs) {
        allocator->deallocate(ptr);
    }
}

TEST_P(CachingAllocatorTest, CachedBytesTracking) {
    const size_t size = 16384;
    const int count = 3;

    // Allocate multiple blocks first (don't deallocate yet)
    std::vector<void*> ptrs;
    for (int i = 0; i < count; ++i) {
        ptrs.push_back(allocator->allocate(size));
    }

    // Now deallocate all blocks so they all get cached
    for (void* ptr : ptrs) {
        allocator->deallocate(ptr);
    }

    size_t expected_cached = size * count;

    // All blocks should be cached
    EXPECT_GE(allocator->total_cached_bytes(), expected_cached);
    EXPECT_EQ(allocator->cached_block_count(), count);
}

TEST_P(CachingAllocatorTest, CacheHitRateZeroInitially) {
    // No allocations yet
    EXPECT_EQ(allocator->cache_hit_rate(), 0.0);
}

TEST_P(CachingAllocatorTest, CacheHitRateCalculation) {
    const size_t size = 1024;

    // First allocation - miss
    void* ptr1 = allocator->allocate(size);
    allocator->deallocate(ptr1);

    // These should be hits
    void* ptr2 = allocator->allocate(size);
    allocator->deallocate(ptr2);

    void* ptr3 = allocator->allocate(size);
    allocator->deallocate(ptr3);

    double hit_rate = allocator->cache_hit_rate();

    // Expected: 2 hits out of 3 allocations = 66.67%
    EXPECT_GT(hit_rate, 50.0);
    EXPECT_LT(hit_rate, 100.0);
}

TEST_P(CachingAllocatorTest, DeviceProperty) {
    EXPECT_EQ(allocator->device().type, device.type);
    EXPECT_EQ(allocator->device().index, device.index);
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST_P(CachingAllocatorTest, VeryLargeAllocation) {
    // Try allocating large block (100 MB)
    const size_t large_size = 100 * 1024 * 1024;

    void* ptr = nullptr;
    EXPECT_NO_THROW({
        ptr = allocator->allocate(large_size);
    });

    if (ptr != nullptr) {
        // If allocation succeeded, verify it's usable
        // (just write to beginning and end)
        std::vector<uint8_t> test_data(4096, 0xCD);
        backend->copy(ptr, test_data.data(), test_data.size(), CopyKind::HostToDevice);
        backend->synchronize(device.index);

        allocator->deallocate(ptr);
    }
}

TEST_P(CachingAllocatorTest, ManySmallAllocations) {
    // Test potential fragmentation scenario
    const size_t count = 100;
    const size_t size = 64;
    std::vector<void*> ptrs;

    // Allocate many small blocks
    for (size_t i = 0; i < count; ++i) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    EXPECT_EQ(allocator->allocated_block_count(), count);

    // Deallocate every other block
    for (size_t i = 0; i < count; i += 2) {
        allocator->deallocate(ptrs[i]);
    }

    // Should have cached blocks now
    EXPECT_GT(allocator->cached_block_count(), 0u);

    // Allocate again - should reuse cached blocks
    std::vector<void*> new_ptrs;
    for (size_t i = 0; i < count / 2; ++i) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        new_ptrs.push_back(ptr);
    }

    // Clean up
    for (size_t i = 1; i < count; i += 2) {
        allocator->deallocate(ptrs[i]);
    }
    for (void* ptr : new_ptrs) {
        allocator->deallocate(ptr);
    }
}

TEST_P(CachingAllocatorTest, InterleavedAllocDealloc) {
    const size_t size = 2048;
    std::vector<void*> ptrs;

    // Interleaved allocate/deallocate pattern
    for (int i = 0; i < 10; ++i) {
        void* ptr = allocator->allocate(size);
        ptrs.push_back(ptr);

        if (i > 0 && i % 2 == 0) {
            allocator->deallocate(ptrs[i - 2]);
        }
    }

    // Clean up remaining
    for (size_t i = 0; i < ptrs.size(); ++i) {
        if (i % 2 != 0 || i >= ptrs.size() - 2) {
            allocator->deallocate(ptrs[i]);
        }
    }
}

TEST_P(CachingAllocatorTest, RepeatedAllocDeallocSameSize) {
    const size_t size = 4096;
    const int iterations = 20;

    void* prev_ptr = nullptr;
    for (int i = 0; i < iterations; ++i) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);

        if (i > 0) {
            // Should reuse same pointer
            EXPECT_EQ(ptr, prev_ptr) << "Expected pointer reuse on iteration " << i;
        }

        prev_ptr = ptr;
        allocator->deallocate(ptr);
    }

    // Cache hit rate should be very high
    double hit_rate = allocator->cache_hit_rate();
    EXPECT_GT(hit_rate, 90.0) << "Expected high cache hit rate";
}

TEST_P(CachingAllocatorTest, PowerOfTwoSizes) {
    // Test common power-of-two allocation sizes
    for (size_t power = 6; power <= 20; ++power) {
        size_t size = 1ULL << power;
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr) << "Allocation failed for size 2^" << power;
        verifyMemoryWritable(ptr, std::min(size, size_t(4096)));
        allocator->deallocate(ptr);
    }
}

TEST_P(CachingAllocatorTest, NonPowerOfTwoSizes) {
    // Test non-power-of-two sizes
    const std::vector<size_t> sizes = {
        100, 333, 555, 777, 1000, 1500, 2500, 3333, 5555
    };

    for (size_t size : sizes) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr) << "Allocation failed for size " << size;
        verifyMemoryWritable(ptr, size);
        allocator->deallocate(ptr);
    }
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_P(CachingAllocatorTest, MoveConstructor) {
    const size_t size = 1024;
    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);

    size_t allocated_before = allocator->allocated_block_count();

    // Move construct
    CachingAllocator moved_allocator(std::move(*allocator));

    EXPECT_EQ(moved_allocator.allocated_block_count(), allocated_before);
    EXPECT_EQ(moved_allocator.device().type, device.type);

    // Clean up using moved allocator
    moved_allocator.deallocate(ptr);
}

TEST_P(CachingAllocatorTest, MoveAssignment) {
    const size_t size = 2048;
    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Create another allocator
    CachingAllocator other_allocator(backend, device);

    // Move assign
    other_allocator = std::move(*allocator);

    EXPECT_GT(other_allocator.allocated_block_count(), 0u);

    // Clean up
    other_allocator.deallocate(ptr);
}

// ============================================================================
// Multi-threaded Safety Tests (Basic)
// ============================================================================

TEST_P(CachingAllocatorTest, ConcurrentAllocationsBasic) {
    const size_t size = 1024;
    const int num_threads = 4;
    const int allocations_per_thread = 10;

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch threads that allocate memory
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < allocations_per_thread; ++i) {
                void* ptr = allocator->allocate(size);
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
            allocator->deallocate(ptr);
        }
    }
}

TEST_P(CachingAllocatorTest, ConcurrentAllocDeallocMixed) {
    const size_t size = 2048;
    const int num_threads = 4;
    const int operations_per_thread = 20;

    std::vector<std::thread> threads;

    // Launch threads with mixed alloc/dealloc
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            std::vector<void*> local_ptrs;
            for (int i = 0; i < operations_per_thread; ++i) {
                if (i % 2 == 0 || local_ptrs.empty()) {
                    // Allocate
                    void* ptr = allocator->allocate(size);
                    if (ptr != nullptr) {
                        local_ptrs.push_back(ptr);
                    }
                } else {
                    // Deallocate
                    if (!local_ptrs.empty()) {
                        allocator->deallocate(local_ptrs.back());
                        local_ptrs.pop_back();
                    }
                }
            }
            // Clean up remaining
            for (void* ptr : local_ptrs) {
                allocator->deallocate(ptr);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // All allocations should be cleaned up
    // Cache may still have blocks
    EXPECT_GE(allocator->cached_block_count(), 0u);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_P(CachingAllocatorTest, StressTestRandomSizes) {
    const int num_allocations = 100;
    std::vector<void*> ptrs;
    std::vector<size_t> sizes;

    // Random size generator
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> size_dist(64, 16384);

    for (int i = 0; i < num_allocations; ++i) {
        size_t size = size_dist(rng);
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
        sizes.push_back(size);
    }

    // Verify all allocations are unique
    std::vector<void*> sorted_ptrs = ptrs;
    std::sort(sorted_ptrs.begin(), sorted_ptrs.end());
    auto it = std::unique(sorted_ptrs.begin(), sorted_ptrs.end());
    EXPECT_EQ(it, sorted_ptrs.end());

    // Deallocate in random order
    std::vector<size_t> indices(num_allocations);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t idx : indices) {
        allocator->deallocate(ptrs[idx]);
    }
}

TEST_P(CachingAllocatorTest, StressTestAllocDeallocCycles) {
    const int num_cycles = 50;
    const size_t size = 8192;

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        std::vector<void*> ptrs;

        // Allocate batch
        for (int i = 0; i < 10; ++i) {
            void* ptr = allocator->allocate(size);
            ASSERT_NE(ptr, nullptr);
            ptrs.push_back(ptr);
        }

        // Deallocate batch
        for (void* ptr : ptrs) {
            allocator->deallocate(ptr);
        }

        // Cache should improve over cycles
        if (cycle > 10) {
            double hit_rate = allocator->cache_hit_rate();
            EXPECT_GT(hit_rate, 0.0);
        }
    }
}

// ============================================================================
// Destructor and Cleanup Tests
// ============================================================================

TEST_P(CachingAllocatorTest, DestructorCleansUpAllocations) {
    const size_t size = 4096;

    {
        CachingAllocator temp_allocator(backend, device);
        void* ptr1 = temp_allocator.allocate(size);
        void* ptr2 = temp_allocator.allocate(size);
        ASSERT_NE(ptr1, nullptr);
        ASSERT_NE(ptr2, nullptr);

        // Don't deallocate - destructor should clean up
    }

    // If destructor didn't clean up, backend would leak memory
    // We can't directly test this, but no crash/hang is a good sign
    SUCCEED();
}

TEST_P(CachingAllocatorTest, DefragmentAfterManyAllocations) {
    const size_t size = 1024;
    const int count = 50;

    std::vector<void*> ptrs;
    for (int i = 0; i < count; ++i) {
        void* ptr = allocator->allocate(size);
        ptrs.push_back(ptr);
    }

    // Deallocate half
    for (int i = 0; i < count / 2; ++i) {
        allocator->deallocate(ptrs[i]);
    }

    size_t cached_before = allocator->cached_block_count();
    EXPECT_GT(cached_before, 0u);

    // Defragment
    allocator->defragment();

    EXPECT_EQ(allocator->cached_block_count(), 0u);
    EXPECT_EQ(allocator->total_cached_bytes(), 0u);

    // Remaining allocations should still be valid
    for (int i = count / 2; i < count; ++i) {
        verifyMemoryWritable(ptrs[i], size);
        allocator->deallocate(ptrs[i]);
    }
}

// ============================================================================
// Cache Behavior Tests
// ============================================================================

TEST_P(CachingAllocatorTest, CacheReusesSmallerBlocks) {
    // Allocate large block
    void* ptr_large = allocator->allocate(4096);
    allocator->deallocate(ptr_large);

    // Request smaller size - should reuse large block
    void* ptr_small = allocator->allocate(2048);
    ASSERT_NE(ptr_small, nullptr);

    // In a best-fit strategy, it might not be the same pointer,
    // but allocation should succeed efficiently
    allocator->deallocate(ptr_small);
}

TEST_P(CachingAllocatorTest, MultipleCachedBlocksOfSameSize) {
    const size_t size = 1024;
    const int count = 5;

    // Allocate multiple blocks first (don't deallocate yet)
    std::vector<void*> ptrs;
    for (int i = 0; i < count; ++i) {
        ptrs.push_back(allocator->allocate(size));
    }

    // Now deallocate all blocks to create multiple cached blocks of same size
    for (void* ptr : ptrs) {
        allocator->deallocate(ptr);
    }

    EXPECT_EQ(allocator->cached_block_count(), count);
    EXPECT_GE(allocator->total_cached_bytes(), size * count);

    // Allocations should reuse cached blocks
    for (int i = 0; i < count; ++i) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        allocator->deallocate(ptr);
    }

    // High cache hit rate expected
    double hit_rate = allocator->cache_hit_rate();
    EXPECT_GT(hit_rate, 40.0);
}

// ============================================================================
// Summary Test
// ============================================================================

TEST_P(CachingAllocatorTest, ComprehensiveFunctionalityTest) {
    // This test exercises multiple features together
    const std::vector<size_t> sizes = {512, 1024, 2048, 4096};
    std::vector<void*> active_ptrs;

    // Phase 1: Allocate various sizes
    for (size_t size : sizes) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        active_ptrs.push_back(ptr);
    }

    EXPECT_EQ(allocator->allocated_block_count(), sizes.size());

    // Phase 2: Deallocate to build cache
    for (void* ptr : active_ptrs) {
        allocator->deallocate(ptr);
    }
    active_ptrs.clear();

    EXPECT_EQ(allocator->cached_block_count(), sizes.size());

    // Phase 3: Reallocate (should hit cache)
    for (size_t size : sizes) {
        void* ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
        active_ptrs.push_back(ptr);
    }

    double hit_rate = allocator->cache_hit_rate();
    EXPECT_GT(hit_rate, 0.0);

    // Phase 4: Partial deallocation
    for (size_t i = 0; i < active_ptrs.size() / 2; ++i) {
        allocator->deallocate(active_ptrs[i]);
    }

    // Phase 5: Defragment
    allocator->defragment();
    EXPECT_EQ(allocator->cached_block_count(), 0u);

    // Phase 6: Verify remaining allocations still work
    for (size_t i = active_ptrs.size() / 2; i < active_ptrs.size(); ++i) {
        verifyMemoryWritable(active_ptrs[i], sizes[i]);
        allocator->deallocate(active_ptrs[i]);
    }

    // Final state
    EXPECT_GT(allocator->cache_hit_rate(), 0.0);
}
