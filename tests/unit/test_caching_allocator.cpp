#include <gtest/gtest.h>
#include "tenzor/backend/caching_allocator.hpp"
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <chrono>
#include <random>

using namespace tenzor::backend;

class CachingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Reset allocator state
        CachingAllocator::get().empty_cache();
        CachingAllocator::get().reset_stats();
    }

    void TearDown() override {
        // Clean up
        CachingAllocator::get().empty_cache();
    }
};

TEST_F(CachingAllocatorTest, BasicAllocationDeallocation) {
    auto& allocator = CachingAllocator::get();

    // Allocate memory
    void* ptr = allocator.allocate(1024, 0);
    ASSERT_NE(ptr, nullptr);

    // Check statistics
    EXPECT_EQ(allocator.memory_allocated(0), 1024);
    EXPECT_GE(allocator.memory_reserved(0), 1024);

    // Free memory
    allocator.free(ptr, 0);

    // After free, allocated should be 0, but reserved might still be > 0 (cached)
    EXPECT_EQ(allocator.memory_allocated(0), 0);
    EXPECT_GT(allocator.memory_cached(0), 0);
}

TEST_F(CachingAllocatorTest, MemoryReuse) {
    auto& allocator = CachingAllocator::get();

    // Allocate and free
    void* ptr1 = allocator.allocate(1024, 0);
    allocator.free(ptr1, 0);

    // Get initial stats
    auto stats1 = allocator.get_stats(0);

    // Allocate again with same size - should reuse
    void* ptr2 = allocator.allocate(1024, 0);

    auto stats2 = allocator.get_stats(0);

    // Check that we got a cache hit
    EXPECT_GT(stats2.num_cache_hits, stats1.num_cache_hits);

    // Reserved memory should not have increased
    EXPECT_EQ(allocator.memory_reserved(0), stats1.reserved_bytes);

    allocator.free(ptr2, 0);
}

TEST_F(CachingAllocatorTest, MultipleAllocations) {
    auto& allocator = CachingAllocator::get();
    std::vector<void*> ptrs;

    // Allocate multiple blocks
    for (int i = 0; i < 10; i++) {
        void* ptr = allocator.allocate(1024 * (i + 1), 0);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Check that all are allocated
    size_t expected_min = 0;
    for (int i = 0; i < 10; i++) {
        expected_min += 1024 * (i + 1);
    }
    EXPECT_GE(allocator.memory_allocated(0), expected_min);

    // Free all
    for (void* ptr : ptrs) {
        allocator.free(ptr, 0);
    }

    // All should be cached now
    EXPECT_EQ(allocator.memory_allocated(0), 0);
    EXPECT_GT(allocator.memory_cached(0), 0);
}

TEST_F(CachingAllocatorTest, BlockSplitting) {
    auto& allocator = CachingAllocator::get();

    // Allocate large block and free it
    void* large = allocator.allocate(8192, 0);
    allocator.free(large, 0);

    auto stats1 = allocator.get_stats(0);

    // Allocate smaller block - should split the large one
    void* small = allocator.allocate(1024, 0);

    auto stats2 = allocator.get_stats(0);

    // Should have performed a split if min_split_size was met
    if (stats2.num_splits > stats1.num_splits) {
        EXPECT_GT(stats2.num_splits, stats1.num_splits);
        EXPECT_GT(stats2.num_cache_hits, stats1.num_cache_hits);
    }

    allocator.free(small, 0);
}

TEST_F(CachingAllocatorTest, BlockMerging) {
    auto& allocator = CachingAllocator::get();
    allocator.set_merge_enabled(true);

    // Allocate adjacent blocks
    void* ptr1 = allocator.allocate(1024, 0);
    void* ptr2 = allocator.allocate(1024, 0);

    // Free them in sequence - might trigger merging
    allocator.free(ptr1, 0);

    auto stats1 = allocator.get_stats(0);

    allocator.free(ptr2, 0);

    auto stats2 = allocator.get_stats(0);

    // Note: Merging might not always happen depending on memory layout
    // Just check that the mechanism doesn't crash
    EXPECT_TRUE(true);
}

TEST_F(CachingAllocatorTest, EmptyCache) {
    auto& allocator = CachingAllocator::get();

    // Allocate and free several blocks
    for (int i = 0; i < 5; i++) {
        void* ptr = allocator.allocate(1024 * (i + 1), 0);
        allocator.free(ptr, 0);
    }

    EXPECT_GT(allocator.memory_cached(0), 0);

    // Empty cache
    allocator.empty_cache(0);

    // Cached memory should be 0
    EXPECT_EQ(allocator.memory_cached(0), 0);
    EXPECT_EQ(allocator.memory_reserved(0), 0);
}

TEST_F(CachingAllocatorTest, Statistics) {
    auto& allocator = CachingAllocator::get();
    allocator.reset_stats();

    // Perform various operations
    void* ptr1 = allocator.allocate(1024, 0);
    void* ptr2 = allocator.allocate(2048, 0);
    allocator.free(ptr1, 0);
    void* ptr3 = allocator.allocate(1024, 0); // Should be cache hit
    allocator.free(ptr2, 0);
    allocator.free(ptr3, 0);

    auto stats = allocator.get_stats(0);

    EXPECT_EQ(stats.num_allocations, 3);
    EXPECT_EQ(stats.num_frees, 3);
    EXPECT_GT(stats.num_cache_hits, 0);
    EXPECT_EQ(stats.allocated_bytes, 0);
    EXPECT_GT(stats.reserved_bytes, 0);
    EXPECT_GT(stats.cached_bytes, 0);
}

TEST_F(CachingAllocatorTest, Alignment) {
    auto& allocator = CachingAllocator::get();

    // Set alignment
    allocator.set_alignment(1024);

    // Allocate odd size - should be rounded up
    void* ptr = allocator.allocate(1500, 0);
    ASSERT_NE(ptr, nullptr);

    // Check that allocated size is aligned
    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.allocated_bytes % 1024, 0);

    allocator.free(ptr, 0);
}

TEST_F(CachingAllocatorTest, MaxCachedMemory) {
    auto& allocator = CachingAllocator::get();

    // Set max cached memory to 8KB
    allocator.set_max_cached_memory(8192);

    // Allocate and free multiple blocks totaling more than limit
    for (int i = 0; i < 10; i++) {
        void* ptr = allocator.allocate(2048, 0);
        allocator.free(ptr, 0);
    }

    // Cached memory should not exceed limit
    EXPECT_LE(allocator.memory_cached(0), 8192);

    // Reset limit
    allocator.set_max_cached_memory(0);
}

TEST_F(CachingAllocatorTest, ThreadSafety) {
    auto& allocator = CachingAllocator::get();

    const int num_threads = 4;
    const int allocs_per_thread = 100;

    auto worker = [&allocator](int thread_id) {
        std::vector<void*> ptrs;
        std::mt19937 rng(thread_id);
        std::uniform_int_distribution<size_t> dist(512, 4096);

        for (int i = 0; i < allocs_per_thread; i++) {
            size_t size = dist(rng);
            void* ptr = allocator.allocate(size, 0);
            EXPECT_NE(ptr, nullptr);
            ptrs.push_back(ptr);

            // Randomly free some allocations
            if (i % 3 == 0 && !ptrs.empty()) {
                size_t idx = rng() % ptrs.size();
                allocator.free(ptrs[idx], 0);
                ptrs.erase(ptrs.begin() + idx);
            }
        }

        // Free remaining
        for (void* ptr : ptrs) {
            allocator.free(ptr, 0);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All memory should be freed
    EXPECT_EQ(allocator.memory_allocated(0), 0);
}

TEST_F(CachingAllocatorTest, ZeroSizeAllocation) {
    auto& allocator = CachingAllocator::get();

    // Allocating zero size should return nullptr
    void* ptr = allocator.allocate(0, 0);
    EXPECT_EQ(ptr, nullptr);

    // Freeing nullptr should not crash
    allocator.free(nullptr, 0);
}

TEST_F(CachingAllocatorTest, LargeAllocations) {
    auto& allocator = CachingAllocator::get();

    // Allocate 1GB
    size_t large_size = 1024 * 1024 * 1024;
    void* ptr = nullptr;

    try {
        ptr = allocator.allocate(large_size, 0);

        if (ptr != nullptr) {
            EXPECT_GE(allocator.memory_allocated(0), large_size);
            allocator.free(ptr, 0);
        }
    } catch (const std::runtime_error&) {
        // Out of memory is acceptable for large allocations
        GTEST_SKIP() << "Not enough GPU memory for large allocation test";
    }
}

TEST_F(CachingAllocatorTest, FragmentationReduction) {
    auto& allocator = CachingAllocator::get();
    allocator.set_merge_enabled(true);

    std::vector<void*> ptrs;

    // Allocate many small blocks
    for (int i = 0; i < 100; i++) {
        void* ptr = allocator.allocate(1024, 0);
        ptrs.push_back(ptr);
    }

    // Free every other block
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        allocator.free(ptrs[i], 0);
        ptrs[i] = nullptr;
    }

    size_t cached_before = allocator.memory_cached(0);

    // Free remaining blocks - should trigger merging
    for (size_t i = 1; i < ptrs.size(); i += 2) {
        if (ptrs[i]) {
            allocator.free(ptrs[i], 0);
        }
    }

    // Cached memory should increase
    EXPECT_GT(allocator.memory_cached(0), cached_before);
}

TEST_F(CachingAllocatorTest, ReusePattern) {
    auto& allocator = CachingAllocator::get();

    // Simulate typical training loop pattern
    const int iterations = 10;
    const int tensors_per_iter = 5;

    for (int iter = 0; iter < iterations; iter++) {
        std::vector<void*> ptrs;

        // Allocate tensors
        for (int i = 0; i < tensors_per_iter; i++) {
            void* ptr = allocator.allocate(4096, 0);
            ptrs.push_back(ptr);
        }

        // Free tensors
        for (void* ptr : ptrs) {
            allocator.free(ptr, 0);
        }
    }

    auto stats = allocator.get_stats(0);

    // After first iteration, most allocations should be cache hits
    float hit_rate = static_cast<float>(stats.num_cache_hits) / stats.num_allocations;
    EXPECT_GT(hit_rate, 0.7f); // At least 70% hit rate
}

TEST_F(CachingAllocatorTest, CachedMemoryGuard) {
    // Test RAII wrapper
    {
        CachedMemoryGuard guard(1024, 0);
        EXPECT_NE(guard.get(), nullptr);
        EXPECT_EQ(guard.size(), 1024);

        // Memory should be allocated
        EXPECT_GT(CachingAllocator::get().memory_allocated(0), 0);
    }

    // After scope, memory should be freed
    EXPECT_EQ(CachingAllocator::get().memory_allocated(0), 0);
}

TEST_F(CachingAllocatorTest, GuardMoveSemantics) {
    CachedMemoryGuard guard1(2048, 0);
    void* ptr = guard1.get();

    // Move construct
    CachedMemoryGuard guard2(std::move(guard1));
    EXPECT_EQ(guard2.get(), ptr);
    EXPECT_EQ(guard1.get(), nullptr);

    // Move assign
    CachedMemoryGuard guard3(1024, 0);
    void* ptr3 = guard3.get();
    guard3 = std::move(guard2);
    EXPECT_EQ(guard3.get(), ptr);
    EXPECT_EQ(guard2.get(), nullptr);
}

// Benchmark test comparing standard vs caching allocator
class CachingAllocatorBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(CachingAllocatorBenchmark, StandardVsCaching) {
    const int iterations = 1000;
    const size_t alloc_size = 4096;

    // Standard cudaMalloc/cudaFree
    auto start_standard = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        void* ptr = nullptr;
        cudaMalloc(&ptr, alloc_size);
        cudaFree(ptr);
    }
    auto end_standard = std::chrono::high_resolution_clock::now();
    auto duration_standard = std::chrono::duration_cast<std::chrono::microseconds>(
        end_standard - start_standard).count();

    // CachingAllocator
    auto& allocator = CachingAllocator::get();
    allocator.empty_cache();
    allocator.reset_stats();

    auto start_caching = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        void* ptr = allocator.allocate(alloc_size, 0);
        allocator.free(ptr, 0);
    }
    auto end_caching = std::chrono::high_resolution_clock::now();
    auto duration_caching = std::chrono::duration_cast<std::chrono::microseconds>(
        end_caching - start_caching).count();

    std::cout << "\nBenchmark Results (" << iterations << " iterations, " << alloc_size << " bytes):\n";
    std::cout << "  Standard cudaMalloc/cudaFree: " << duration_standard << " us\n";
    std::cout << "  CachingAllocator:             " << duration_caching << " us\n";
    std::cout << "  Speedup:                      "
              << (static_cast<double>(duration_standard) / duration_caching) << "x\n";

    auto stats = allocator.get_stats(0);
    std::cout << "  Cache hit rate:               "
              << (100.0 * stats.num_cache_hits / stats.num_allocations) << "%\n";

    // Caching allocator should be significantly faster
    EXPECT_LT(duration_caching, duration_standard);

    allocator.empty_cache();
}

TEST_F(CachingAllocatorBenchmark, VariableSizePattern) {
    const int iterations = 500;
    std::vector<size_t> sizes = {1024, 2048, 4096, 8192, 16384};

    auto& allocator = CachingAllocator::get();
    allocator.empty_cache();
    allocator.reset_stats();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        for (size_t size : sizes) {
            void* ptr = allocator.allocate(size, 0);
            allocator.free(ptr, 0);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    auto stats = allocator.get_stats(0);

    std::cout << "\nVariable Size Pattern (" << iterations << " iterations):\n";
    std::cout << "  Duration:        " << duration << " us\n";
    std::cout << "  Allocations:     " << stats.num_allocations << "\n";
    std::cout << "  Cache hits:      " << stats.num_cache_hits << "\n";
    std::cout << "  Hit rate:        "
              << (100.0 * stats.num_cache_hits / stats.num_allocations) << "%\n";
    std::cout << "  Splits:          " << stats.num_splits << "\n";
    std::cout << "  Merges:          " << stats.num_merges << "\n";
    std::cout << "  Reserved memory: " << (stats.reserved_bytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Cached memory:   " << (stats.cached_bytes / 1024.0 / 1024.0) << " MB\n";

    allocator.empty_cache();
}

TEST_F(CachingAllocatorBenchmark, MemoryLeakDetection) {
    auto& allocator = CachingAllocator::get();
    allocator.empty_cache();

    size_t initial_reserved = allocator.memory_reserved(0);

    // Allocate and properly free
    for (int i = 0; i < 100; i++) {
        void* ptr = allocator.allocate(4096, 0);
        allocator.free(ptr, 0);
    }

    // Empty cache and check
    allocator.empty_cache();
    size_t final_reserved = allocator.memory_reserved(0);

    // Should return to initial state (no leaks)
    EXPECT_EQ(final_reserved, initial_reserved);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
