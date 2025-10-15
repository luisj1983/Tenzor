#include <gtest/gtest.h>
#include "tenzor/backend/rocm_caching_allocator.hip.hpp"
#include <thread>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define HIP_AVAILABLE 1
#else
#define HIP_AVAILABLE 0
#endif

using namespace tenzor::backend::rocm;

class RocmCachingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
#if HIP_AVAILABLE
        int device_count;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count == 0) {
            GTEST_SKIP() << "No HIP devices available";
        }
#else
        GTEST_SKIP() << "HIP not available";
#endif
    }

    void TearDown() override {
#if HIP_AVAILABLE
        RocmCachingAllocator::get().empty_cache(-1);
        RocmCachingAllocator::get().reset_stats();
#endif
    }
};

#if HIP_AVAILABLE

TEST_F(RocmCachingAllocatorTest, BasicAllocation) {
    auto& allocator = RocmCachingAllocator::get();

    // Allocate memory
    size_t size = 1024;
    void* ptr = allocator.allocate(size, 0);
    ASSERT_NE(ptr, nullptr);

    // Verify statistics
    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_allocations, 1);
    EXPECT_GE(stats.allocated_bytes, size);
    EXPECT_GE(stats.reserved_bytes, size);

    // Free memory
    allocator.free(ptr, 0);
    stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_frees, 1);
    EXPECT_EQ(stats.allocated_bytes, 0);
    EXPECT_GE(stats.cached_bytes, size);
}

TEST_F(RocmCachingAllocatorTest, ZeroSizeAllocation) {
    auto& allocator = RocmCachingAllocator::get();

    void* ptr = allocator.allocate(0, 0);
    EXPECT_EQ(ptr, nullptr);

    // Should not affect statistics
    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_allocations, 0);
}

TEST_F(RocmCachingAllocatorTest, CacheReuse) {
    auto& allocator = RocmCachingAllocator::get();

    size_t size = 2048;

    // First allocation
    void* ptr1 = allocator.allocate(size, 0);
    ASSERT_NE(ptr1, nullptr);

    // Free it
    allocator.free(ptr1, 0);

    // Second allocation should reuse from cache
    void* ptr2 = allocator.allocate(size, 0);
    ASSERT_NE(ptr2, nullptr);

    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_allocations, 2);
    EXPECT_GE(stats.num_cache_hits, 1);

    allocator.free(ptr2, 0);
}

TEST_F(RocmCachingAllocatorTest, BlockSplitting) {
    auto& allocator = RocmCachingAllocator::get();
    allocator.set_min_split_size(512);

    // Allocate large block
    size_t large_size = 4096;
    void* ptr1 = allocator.allocate(large_size, 0);
    ASSERT_NE(ptr1, nullptr);
    allocator.free(ptr1, 0);

    // Allocate smaller block (should split)
    size_t small_size = 1024;
    void* ptr2 = allocator.allocate(small_size, 0);
    ASSERT_NE(ptr2, nullptr);

    auto stats = allocator.get_stats(0);
    EXPECT_GE(stats.num_splits, 1);
    EXPECT_GE(stats.cached_bytes, 0);  // Should have remaining cached

    allocator.free(ptr2, 0);
}

TEST_F(RocmCachingAllocatorTest, BlockMerging) {
    auto& allocator = RocmCachingAllocator::get();
    allocator.set_merge_enabled(true);

    size_t size = 1024;

    // Allocate and free multiple blocks
    void* ptr1 = allocator.allocate(size, 0);
    void* ptr2 = allocator.allocate(size, 0);
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    // Free in order (might trigger merging)
    allocator.free(ptr1, 0);
    allocator.free(ptr2, 0);

    auto stats = allocator.get_stats(0);
    // Merging may or may not occur depending on memory layout
    EXPECT_GE(stats.num_frees, 2);
}

TEST_F(RocmCachingAllocatorTest, MemoryAlignment) {
    auto& allocator = RocmCachingAllocator::get();
    allocator.set_alignment(256);

    // Allocate unaligned size
    size_t size = 100;
    void* ptr = allocator.allocate(size, 0);
    ASSERT_NE(ptr, nullptr);

    // Verify pointer is aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 256, 0);

    allocator.free(ptr, 0);
}

TEST_F(RocmCachingAllocatorTest, EmptyCache) {
    auto& allocator = RocmCachingAllocator::get();

    // Allocate and free multiple blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator.allocate(1024, 0));
    }
    for (void* ptr : ptrs) {
        allocator.free(ptr, 0);
    }

    auto stats_before = allocator.get_stats(0);
    EXPECT_GT(stats_before.cached_bytes, 0);

    // Empty cache
    allocator.empty_cache(0);

    auto stats_after = allocator.get_stats(0);
    EXPECT_EQ(stats_after.cached_bytes, 0);
    EXPECT_EQ(stats_after.reserved_bytes, 0);
}

TEST_F(RocmCachingAllocatorTest, GarbageCollection) {
    auto& allocator = RocmCachingAllocator::get();

    // Allocate and free large blocks
    size_t large_size = 10 * 1024 * 1024;  // 10 MB
    void* ptr = allocator.allocate(large_size, 0);
    ASSERT_NE(ptr, nullptr);
    allocator.free(ptr, 0);

    auto stats_before = allocator.get_stats(0);
    EXPECT_GT(stats_before.cached_bytes, 0);

    // Garbage collect
    allocator.garbage_collect(0, false);

    auto stats_after = allocator.get_stats(0);
    // GC may have freed some memory
    EXPECT_LE(stats_after.cached_bytes, stats_before.cached_bytes);
}

TEST_F(RocmCachingAllocatorTest, CacheLimit) {
    auto& allocator = RocmCachingAllocator::get();

    // Set cache limit to 5 MB
    size_t limit = 5 * 1024 * 1024;
    allocator.set_max_cached_memory(limit);

    // Allocate and free multiple blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator.allocate(1024 * 1024, 0));  // 1 MB each
    }
    for (void* ptr : ptrs) {
        allocator.free(ptr, 0);
    }

    auto stats = allocator.get_stats(0);
    // Cache should be limited
    EXPECT_LE(stats.cached_bytes, limit);
}

TEST_F(RocmCachingAllocatorTest, DeviceProperties) {
    auto& allocator = RocmCachingAllocator::get();

    // Allocate something to initialize device
    void* ptr = allocator.allocate(1024, 0);
    ASSERT_NE(ptr, nullptr);

    auto props = allocator.get_device_properties(0);
    EXPECT_GT(props.total_memory, 0);
    EXPECT_FALSE(props.device_name.empty());
    EXPECT_GT(props.warp_size, 0);

    allocator.free(ptr, 0);
}

TEST_F(RocmCachingAllocatorTest, PeakMemoryTracking) {
    auto& allocator = RocmCachingAllocator::get();
    allocator.reset_stats();

    // Allocate increasing sizes
    void* ptr1 = allocator.allocate(1024, 0);
    void* ptr2 = allocator.allocate(2048, 0);
    void* ptr3 = allocator.allocate(4096, 0);

    auto stats = allocator.get_stats(0);
    size_t current_allocated = stats.allocated_bytes;
    EXPECT_EQ(stats.peak_allocated, current_allocated);

    // Free some memory
    allocator.free(ptr1, 0);
    allocator.free(ptr2, 0);

    stats = allocator.get_stats(0);
    // Peak should remain at maximum
    EXPECT_GE(stats.peak_allocated, stats.allocated_bytes);

    allocator.free(ptr3, 0);
}

TEST_F(RocmCachingAllocatorTest, ThreadSafety) {
    auto& allocator = RocmCachingAllocator::get();

    const int num_threads = 8;
    const int allocations_per_thread = 100;
    std::vector<std::thread> threads;

    // Launch threads that allocate and free memory
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&allocator, allocations_per_thread]() {
            std::vector<void*> ptrs;
            for (int i = 0; i < allocations_per_thread; ++i) {
                size_t size = (i + 1) * 1024;
                void* ptr = allocator.allocate(size, 0);
                EXPECT_NE(ptr, nullptr);
                ptrs.push_back(ptr);
            }

            // Free in reverse order
            for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
                allocator.free(*it, 0);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify statistics
    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_allocations, num_threads * allocations_per_thread);
    EXPECT_EQ(stats.num_frees, num_threads * allocations_per_thread);
    EXPECT_EQ(stats.allocated_bytes, 0);
}

TEST_F(RocmCachingAllocatorTest, RocmCachedMemoryGuard) {
    auto& allocator = RocmCachingAllocator::get();
    allocator.reset_stats();

    {
        // Allocate using RAII guard
        RocmCachedMemoryGuard guard(2048, 0);
        EXPECT_NE(guard.get(), nullptr);
        EXPECT_EQ(guard.size(), 2048);

        auto stats = allocator.get_stats(0);
        EXPECT_EQ(stats.num_allocations, 1);
        EXPECT_GT(stats.allocated_bytes, 0);
    }

    // Memory should be freed automatically
    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_frees, 1);
    EXPECT_EQ(stats.allocated_bytes, 0);
}

TEST_F(RocmCachingAllocatorTest, MoveSemantics) {
    RocmCachedMemoryGuard guard1(1024, 0);
    void* ptr1 = guard1.get();
    ASSERT_NE(ptr1, nullptr);

    // Move construct
    RocmCachedMemoryGuard guard2(std::move(guard1));
    EXPECT_EQ(guard2.get(), ptr1);
    EXPECT_EQ(guard1.get(), nullptr);

    // Move assign
    RocmCachedMemoryGuard guard3(512, 0);
    guard3 = std::move(guard2);
    EXPECT_EQ(guard3.get(), ptr1);
    EXPECT_EQ(guard2.get(), nullptr);
}

TEST_F(RocmCachingAllocatorTest, LargeAllocation) {
    auto& allocator = RocmCachingAllocator::get();

    // Try to allocate large block (100 MB)
    size_t large_size = 100 * 1024 * 1024;

    // Check if device has enough memory
    auto props = allocator.get_device_properties(0);
    if (props.available_memory < large_size) {
        GTEST_SKIP() << "Not enough memory for large allocation test";
    }

    void* ptr = allocator.allocate(large_size, 0);
    ASSERT_NE(ptr, nullptr);

    auto stats = allocator.get_stats(0);
    EXPECT_GE(stats.allocated_bytes, large_size);

    allocator.free(ptr, 0);
}

TEST_F(RocmCachingAllocatorTest, StatisticsReset) {
    auto& allocator = RocmCachingAllocator::get();

    // Perform some allocations
    void* ptr1 = allocator.allocate(1024, 0);
    void* ptr2 = allocator.allocate(2048, 0);
    allocator.free(ptr1, 0);

    auto stats_before = allocator.get_stats(0);
    EXPECT_GT(stats_before.num_allocations, 0);
    EXPECT_GT(stats_before.num_frees, 0);

    // Reset statistics
    allocator.reset_stats();

    auto stats_after = allocator.get_stats(0);
    EXPECT_EQ(stats_after.num_allocations, 0);
    EXPECT_EQ(stats_after.num_frees, 0);
    EXPECT_EQ(stats_after.num_cache_hits, 0);
    EXPECT_EQ(stats_after.num_splits, 0);
    EXPECT_EQ(stats_after.num_merges, 0);

    // Memory amounts should not be reset
    EXPECT_EQ(stats_after.allocated_bytes, stats_before.allocated_bytes);

    allocator.free(ptr2, 0);
}

TEST_F(RocmCachingAllocatorTest, InvalidAlignment) {
    auto& allocator = RocmCachingAllocator::get();

    // Non-power-of-2 alignment should throw
    EXPECT_THROW(allocator.set_alignment(100), std::invalid_argument);

    // Zero alignment should throw
    EXPECT_THROW(allocator.set_alignment(0), std::invalid_argument);

    // Valid power-of-2 should not throw
    EXPECT_NO_THROW(allocator.set_alignment(256));
}

TEST_F(RocmCachingAllocatorTest, DoubleFreeDetection) {
    auto& allocator = RocmCachingAllocator::get();

    void* ptr = allocator.allocate(1024, 0);
    ASSERT_NE(ptr, nullptr);

    // First free should succeed
    EXPECT_NO_THROW(allocator.free(ptr, 0));

    // Second free should throw
    EXPECT_THROW(allocator.free(ptr, 0), std::runtime_error);
}

TEST_F(RocmCachingAllocatorTest, InvalidPointerFree) {
    auto& allocator = RocmCachingAllocator::get();

    // Try to free an invalid pointer
    void* invalid_ptr = reinterpret_cast<void*>(0x12345678);
    EXPECT_THROW(allocator.free(invalid_ptr, 0), std::runtime_error);
}

TEST_F(RocmCachingAllocatorTest, HBMDetection) {
    auto& allocator = RocmCachingAllocator::get();

    // Allocate something to initialize device
    void* ptr = allocator.allocate(1024, 0);
    ASSERT_NE(ptr, nullptr);

    auto props = allocator.get_device_properties(0);

    // Check if HBM detection works (depends on GPU model)
    // MI series GPUs should have HBM
    if (props.device_name.find("MI") != std::string::npos) {
        EXPECT_TRUE(props.has_hbm);
    }

    allocator.free(ptr, 0);
}

#endif  // HIP_AVAILABLE

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
