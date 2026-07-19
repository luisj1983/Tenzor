#include <gtest/gtest.h>
#include "tenzor/backend/caching_allocator.hpp"
#include <cuda_runtime.h>
#include <thread>
#include <vector>

using namespace tenzor::backend;

class CudaCachingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    void TearDown() override {
        CachingAllocator::get().empty_cache(-1);
        CachingAllocator::get().reset_stats();
    }
};

TEST_F(CudaCachingAllocatorTest, BasicAllocation) {
    auto& allocator = CachingAllocator::get();

    size_t size = 1024;
    void* ptr = allocator.allocate(size, 0);
    ASSERT_NE(ptr, nullptr);

    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_allocations, 1u);
    EXPECT_GE(stats.allocated_bytes, size);
    EXPECT_GE(stats.reserved_bytes, size);

    allocator.free(ptr, 0);
    stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_frees, 1u);
    EXPECT_EQ(stats.allocated_bytes, 0u);
    EXPECT_GE(stats.cached_bytes, size);
}

TEST_F(CudaCachingAllocatorTest, ZeroSizeAllocation) {
    auto& allocator = CachingAllocator::get();

    void* ptr = allocator.allocate(0, 0);
    EXPECT_EQ(ptr, nullptr);

    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_allocations, 0u);
}

TEST_F(CudaCachingAllocatorTest, CacheReuseExactSize) {
    auto& allocator = CachingAllocator::get();

    size_t size = 2048;

    void* ptr1 = allocator.allocate(size, 0);
    ASSERT_NE(ptr1, nullptr);
    allocator.free(ptr1, 0);

    size_t hits_before = allocator.get_stats(0).num_cache_hits;
    void* ptr2 = allocator.allocate(size, 0);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_EQ(ptr1, ptr2) << "Exact-size reuse should hand back the same cached pointer";
    EXPECT_GT(allocator.get_stats(0).num_cache_hits, hits_before);

    allocator.free(ptr2, 0);
}

TEST_F(CudaCachingAllocatorTest, BlockSplitOnOversizedReuse) {
    auto& allocator = CachingAllocator::get();
    allocator.set_min_split_size(512);

    const size_t big_size = 8192;
    const size_t small_size = 1024;

    void* big_ptr = allocator.allocate(big_size, 0);
    ASSERT_NE(big_ptr, nullptr);
    allocator.free(big_ptr, 0);

    size_t splits_before = allocator.get_stats(0).num_splits;

    // A much smaller request reusing this cached block should SPLIT it rather
    // than handing out the whole 8192 bytes: the leading `small_size` bytes
    // come back at the original address, and the remainder becomes its own
    // free block immediately available for reuse.
    void* small_ptr = allocator.allocate(small_size, 0);
    ASSERT_NE(small_ptr, nullptr);
    EXPECT_EQ(small_ptr, big_ptr) << "Split should hand out the front of the original block";
    EXPECT_GT(allocator.get_stats(0).num_splits, splits_before)
        << "Reusing an oversized cached block for a much smaller request should split it";

    // The remainder must be independently cached and reusable WITHOUT freeing
    // small_ptr first — proving it was actually split off, not just logically
    // reserved inside the still-whole block.
    const size_t remainder_request = 2048;
    ASSERT_LT(remainder_request, big_size - small_size);

    size_t hits_before = allocator.get_stats(0).num_cache_hits;
    void* remainder_ptr = allocator.allocate(remainder_request, 0);
    ASSERT_NE(remainder_ptr, nullptr);
    EXPECT_EQ(remainder_ptr, static_cast<char*>(big_ptr) + small_size)
        << "Remainder allocation should start exactly where the split occurred";
    EXPECT_GT(allocator.get_stats(0).num_cache_hits, hits_before)
        << "Remainder should be served from cache, not a fresh cudaMalloc";

    allocator.free(small_ptr, 0);
    allocator.free(remainder_ptr, 0);
}

TEST_F(CudaCachingAllocatorTest, SplitBlocksCascadeMergeOnFree) {
    auto& allocator = CachingAllocator::get();
    allocator.set_merge_enabled(true);
    allocator.set_min_split_size(512);

    const size_t total_size = 8192;
    const size_t a_size = 2048;
    const size_t b_size = 1024;

    void* root_ptr = allocator.allocate(total_size, 0);
    ASSERT_NE(root_ptr, nullptr);
    allocator.free(root_ptr, 0);

    // Splits root -> A (used) + rem1 (free).
    void* a_ptr = allocator.allocate(a_size, 0);
    ASSERT_NE(a_ptr, nullptr);
    ASSERT_EQ(a_ptr, root_ptr);

    // Splits rem1 -> B (used) + rem2 (free).
    void* b_ptr = allocator.allocate(b_size, 0);
    ASSERT_NE(b_ptr, nullptr);
    ASSERT_EQ(b_ptr, static_cast<char*>(root_ptr) + a_size);
    ASSERT_GE(allocator.get_stats(0).num_splits, 2u);

    // Freeing A first has no free neighbor yet: B (immediately after it) is
    // still allocated, and A has no predecessor. Should not merge with anything.
    size_t merges_baseline = allocator.get_stats(0).num_merges;
    allocator.free(a_ptr, 0);
    EXPECT_EQ(allocator.get_stats(0).num_merges, merges_baseline)
        << "A has no free neighbor yet and should not have merged";

    size_t merges_before_b = allocator.get_stats(0).num_merges;
    allocator.free(b_ptr, 0);
    // Freeing B should forward-merge with the still-free rem2 sibling, then
    // backward-merge with the now-free A sibling — two coalesces from one
    // free() call, fully reassembling the original 8192-byte allocation.
    EXPECT_GE(allocator.get_stats(0).num_merges, merges_before_b + 2)
        << "Freeing B should coalesce with both free neighbors (forward into "
           "rem2, then backward into A)";

    size_t hits_before = allocator.get_stats(0).num_cache_hits;
    size_t reserved_before = allocator.get_stats(0).reserved_bytes;

    const size_t big_request = total_size - 64;  // bigger than any single piece
    void* merged_ptr = allocator.allocate(big_request, 0);
    ASSERT_NE(merged_ptr, nullptr);
    EXPECT_EQ(merged_ptr, root_ptr)
        << "Fully reassembled block should be handed out from the original base pointer";
    EXPECT_GT(allocator.get_stats(0).num_cache_hits, hits_before)
        << "A properly merged block should satisfy this from cache, not a fresh cudaMalloc";
    EXPECT_EQ(allocator.get_stats(0).reserved_bytes, reserved_before)
        << "No new cudaMalloc should occur once the split pieces fully re-merged";

    allocator.free(merged_ptr, 0);
}

TEST_F(CudaCachingAllocatorTest, MemoryAlignment) {
    auto& allocator = CachingAllocator::get();
    allocator.set_alignment(256);

    size_t size = 100;
    void* ptr = allocator.allocate(size, 0);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 256, 0u);

    allocator.free(ptr, 0);
    allocator.set_alignment(512);  // restore default for later tests
}

TEST_F(CudaCachingAllocatorTest, EmptyCacheReleasesFreeBlocks) {
    auto& allocator = CachingAllocator::get();

    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator.allocate(1024, 0));
    }
    for (void* ptr : ptrs) {
        allocator.free(ptr, 0);
    }

    auto stats_before = allocator.get_stats(0);
    EXPECT_GT(stats_before.cached_bytes, 0u);

    allocator.empty_cache(0);

    auto stats_after = allocator.get_stats(0);
    EXPECT_EQ(stats_after.cached_bytes, 0u);
    EXPECT_EQ(stats_after.reserved_bytes, 0u);
}

TEST_F(CudaCachingAllocatorTest, CacheLimit) {
    auto& allocator = CachingAllocator::get();

    size_t limit = 5 * 1024 * 1024;  // 5 MB
    allocator.set_max_cached_memory(limit);

    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator.allocate(1024 * 1024, 0));  // 1 MB each
    }
    for (void* ptr : ptrs) {
        allocator.free(ptr, 0);
    }

    auto stats = allocator.get_stats(0);
    EXPECT_LE(stats.cached_bytes, limit);

    allocator.set_max_cached_memory(0);  // restore unlimited for later tests
}

TEST_F(CudaCachingAllocatorTest, StatisticsReset) {
    auto& allocator = CachingAllocator::get();

    void* ptr1 = allocator.allocate(1024, 0);
    void* ptr2 = allocator.allocate(2048, 0);
    allocator.free(ptr1, 0);

    auto stats_before = allocator.get_stats(0);
    EXPECT_GT(stats_before.num_allocations, 0u);
    EXPECT_GT(stats_before.num_frees, 0u);

    allocator.reset_stats();

    auto stats_after = allocator.get_stats(0);
    EXPECT_EQ(stats_after.num_allocations, 0u);
    EXPECT_EQ(stats_after.num_frees, 0u);
    EXPECT_EQ(stats_after.num_cache_hits, 0u);
    EXPECT_EQ(stats_after.num_splits, 0u);
    EXPECT_EQ(stats_after.num_merges, 0u);

    // Memory amounts should not be reset
    EXPECT_EQ(stats_after.allocated_bytes, stats_before.allocated_bytes);

    allocator.free(ptr2, 0);
}

TEST_F(CudaCachingAllocatorTest, InvalidAlignmentThrows) {
    auto& allocator = CachingAllocator::get();

    EXPECT_THROW(allocator.set_alignment(100), std::invalid_argument);
    EXPECT_THROW(allocator.set_alignment(0), std::invalid_argument);
    EXPECT_NO_THROW(allocator.set_alignment(256));
    allocator.set_alignment(512);  // restore default
}

TEST_F(CudaCachingAllocatorTest, DoubleFreeThrows) {
    auto& allocator = CachingAllocator::get();

    void* ptr = allocator.allocate(1024, 0);
    ASSERT_NE(ptr, nullptr);

    EXPECT_NO_THROW(allocator.free(ptr, 0));
    EXPECT_THROW(allocator.free(ptr, 0), std::runtime_error);
}

TEST_F(CudaCachingAllocatorTest, InvalidPointerFreeThrows) {
    auto& allocator = CachingAllocator::get();

    void* invalid_ptr = reinterpret_cast<void*>(0x12345678);
    EXPECT_THROW(allocator.free(invalid_ptr, 0), std::runtime_error);
}

TEST_F(CudaCachingAllocatorTest, CachedMemoryGuardBasic) {
    auto& allocator = CachingAllocator::get();
    allocator.reset_stats();

    {
        CachedMemoryGuard guard(2048, 0);
        EXPECT_NE(guard.get(), nullptr);
        EXPECT_EQ(guard.size(), 2048u);

        auto stats = allocator.get_stats(0);
        EXPECT_EQ(stats.num_allocations, 1u);
        EXPECT_GT(stats.allocated_bytes, 0u);
    }

    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_frees, 1u);
    EXPECT_EQ(stats.allocated_bytes, 0u);
}

TEST_F(CudaCachingAllocatorTest, CachedMemoryGuardMoveSemantics) {
    CachedMemoryGuard guard1(1024, 0);
    void* ptr1 = guard1.get();
    ASSERT_NE(ptr1, nullptr);

    CachedMemoryGuard guard2(std::move(guard1));
    EXPECT_EQ(guard2.get(), ptr1);
    EXPECT_EQ(guard1.get(), nullptr);

    CachedMemoryGuard guard3(512, 0);
    guard3 = std::move(guard2);
    EXPECT_EQ(guard3.get(), ptr1);
    EXPECT_EQ(guard2.get(), nullptr);
}

TEST_F(CudaCachingAllocatorTest, ThreadSafety) {
    auto& allocator = CachingAllocator::get();

    const int num_threads = 8;
    const int allocations_per_thread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&allocator, allocations_per_thread]() {
            std::vector<void*> ptrs;
            for (int i = 0; i < allocations_per_thread; ++i) {
                size_t size = (i + 1) * 1024;
                void* ptr = allocator.allocate(size, 0);
                EXPECT_NE(ptr, nullptr);
                ptrs.push_back(ptr);
            }
            for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
                allocator.free(*it, 0);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto stats = allocator.get_stats(0);
    EXPECT_EQ(stats.num_allocations, static_cast<size_t>(num_threads * allocations_per_thread));
    EXPECT_EQ(stats.num_frees, static_cast<size_t>(num_threads * allocations_per_thread));
    EXPECT_EQ(stats.allocated_bytes, 0u);
}

TEST_F(CudaCachingAllocatorTest, LargeAllocation) {
    auto& allocator = CachingAllocator::get();

    size_t large_size = 100 * 1024 * 1024;  // 100 MB

    size_t free_mem = 0, total_mem = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_mem, &total_mem), cudaSuccess);
    if (free_mem < large_size) {
        GTEST_SKIP() << "Not enough memory for large allocation test";
    }

    void* ptr = allocator.allocate(large_size, 0);
    ASSERT_NE(ptr, nullptr);

    auto stats = allocator.get_stats(0);
    EXPECT_GE(stats.allocated_bytes, large_size);

    allocator.free(ptr, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
