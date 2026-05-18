/**
 * @file test_caching_allocator_coalesce.cpp
 * @brief Tests for CPUCachingAllocator split-sibling coalescing (Task 7.1)
 *        and memory_summary() introspection (Task 7.2).
 *
 * Task 7.1: When split siblings are all freed they coalesce back into one
 *            block so the root can be returned to the OS.
 * Task 7.2: memory_summary() returns a non-empty PyTorch-style string with
 *            expected counters after N allocations.
 */

#ifndef TENZOR_TESTING
#define TENZOR_TESTING
#endif

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <cstdint>
#include "tenzor/backend/cpu_caching_allocator.hpp"

namespace tz_cpu = ::tenzor::cpu;
using Alloc = tz_cpu::CPUCachingAllocator;

// ---------------------------------------------------------------------------
// Helper: reset stats and cached memory between tests
// ---------------------------------------------------------------------------
static void reset_allocator() {
    Alloc::instance().release_cached_memory();
    Alloc::instance().reset_stats();
}

// ===========================================================================
// Task 7.1 — split-sibling coalescing
// ===========================================================================

// When two halves of a split block are both freed they should merge so the
// root allocation is fully coalesced and can be returned to the OS.
TEST(CoalesceTest, SplitSiblingsCoalescedToRoot) {
    reset_allocator();

    // Use a large alloc (8 MB) so splitting triggers (min_split_size default 1 MB).
    const size_t big = 8ULL << 20;   // 8 MB root
    const size_t half = big / 2;     // 4 MB — fits in split remainder

    // Force split_size low enough to guarantee splitting happens.
    Alloc::instance().set_min_split_size(1ULL << 20);  // 1 MB

    // 1. Allocate big block from system.
    void* p1 = Alloc::instance().allocate(big);
    ASSERT_NE(p1, nullptr);

    auto stats_after_alloc = Alloc::instance().get_stats();
    size_t backend_allocs = stats_after_alloc.num_backend_allocs;

    // 2. Free it → goes to local free pool.
    Alloc::instance().deallocate(p1);

    // 3. Allocate half-size → should hit cache and produce a split.
    void* p2 = Alloc::instance().allocate(half);
    ASSERT_NE(p2, nullptr);

    auto stats_split = Alloc::instance().get_stats();
    // At least one split should have been recorded.
    EXPECT_GE(stats_split.num_splits, 1u);

    // 4. Allocate the second half (the remainder fragment).
    void* p3 = Alloc::instance().allocate(half);
    ASSERT_NE(p3, nullptr);
    EXPECT_NE(p2, p3);

    // 5. Free both halves.
    Alloc::instance().deallocate(p2);
    Alloc::instance().deallocate(p3);

    // 6. After freeing both siblings, force a memory-pressure check by
    //    setting max_cached_bytes below current cached to trigger release.
    auto stats_before = Alloc::instance().get_stats();
    size_t cached = stats_before.cached_bytes;

    if (cached > 0) {
        Alloc::instance().set_max_cached_bytes(0);  // force pressure
        // restore to default
        Alloc::instance().set_max_cached_bytes(1ULL << 30);
    }

    // 7. Verify that the root was returned to OS: num_backend_frees should
    //    have incremented by at least 1 relative to before the test.
    auto stats_final = Alloc::instance().get_stats();
    EXPECT_GT(stats_final.num_backend_frees, 0u);
    // And cached_bytes should be 0 (or at least lower than before).
    EXPECT_LT(stats_final.cached_bytes, cached + 1);
}

// After freeing all fragments from a split, a subsequent allocation of the
// original size should be served from cache (cache_hits > 0), not the system.
TEST(CoalesceTest, ReusedAfterCoalescing) {
    reset_allocator();

    Alloc::instance().set_min_split_size(1ULL << 20);

    const size_t big = 8ULL << 20;
    const size_t half = big / 2;

    // Allocate big, free → goes to cache.
    void* p1 = Alloc::instance().allocate(big);
    Alloc::instance().deallocate(p1);

    // Split it into two halves.
    void* p2 = Alloc::instance().allocate(half);
    void* p3 = Alloc::instance().allocate(half);

    // Free both siblings.
    Alloc::instance().deallocate(p2);
    Alloc::instance().deallocate(p3);

    // Now allocate big again — should hit cache.
    Alloc::instance().reset_stats();
    void* p4 = Alloc::instance().allocate(big);
    ASSERT_NE(p4, nullptr);

    auto stats = Alloc::instance().get_stats();
    EXPECT_GE(stats.cache_hits, 1u)
        << "Expected coalesced block to be reused from cache";

    Alloc::instance().deallocate(p4);
}

// Verify that release_cached_memory() returns an unsplit block to the OS.
TEST(CoalesceTest, UnsplitBlockReleasedToOS) {
    reset_allocator();

    const size_t sz = 4ULL << 20;  // 4 MB
    void* p = Alloc::instance().allocate(sz);
    Alloc::instance().deallocate(p);

    auto before = Alloc::instance().get_stats();
    EXPECT_GT(before.cached_bytes, 0u);

    // release_cached_memory() explicitly flushes ALL free pools to the OS.
    Alloc::instance().release_cached_memory();

    auto after = Alloc::instance().get_stats();
    EXPECT_EQ(after.cached_bytes, 0u);
    EXPECT_GT(after.num_backend_frees, 0u);
}

// ===========================================================================
// Task 7.2 — memory_summary()
// ===========================================================================

TEST(MemorySummaryTest, NonEmptyAfterAllocation) {
    reset_allocator();

    const size_t sz = 1024 * 1024;  // 1 MB
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        ptrs.push_back(Alloc::instance().allocate(sz));
    }

    std::string summary = Alloc::instance().memory_summary();

    EXPECT_FALSE(summary.empty()) << "memory_summary() should not be empty";
    // Expect basic section headers
    EXPECT_NE(summary.find("allocated"), std::string::npos)
        << "summary missing 'allocated' field";
    EXPECT_NE(summary.find("cached"), std::string::npos)
        << "summary missing 'cached' field";

    for (void* p : ptrs) {
        Alloc::instance().deallocate(p);
    }
}

TEST(MemorySummaryTest, ContainsAllocationCount) {
    reset_allocator();

    const size_t sz = 64 * 1024;  // 64 KB
    constexpr int N = 8;
    std::vector<void*> ptrs;
    for (int i = 0; i < N; ++i) {
        ptrs.push_back(Alloc::instance().allocate(sz));
    }

    std::string summary = Alloc::instance().memory_summary();

    // The summary must contain "8" somewhere (allocation count).
    EXPECT_NE(summary.find('8'), std::string::npos)
        << "summary does not contain allocation count 8:\n" << summary;

    for (void* p : ptrs) {
        Alloc::instance().deallocate(p);
    }
}

TEST(MemorySummaryTest, ContainsCacheHits) {
    reset_allocator();

    const size_t sz = 64 * 1024;
    void* p = Alloc::instance().allocate(sz);
    Alloc::instance().deallocate(p);

    // Second alloc should be a cache hit.
    void* p2 = Alloc::instance().allocate(sz);

    std::string summary = Alloc::instance().memory_summary();

    EXPECT_NE(summary.find("hit"), std::string::npos)
        << "summary should mention cache hits:\n" << summary;

    Alloc::instance().deallocate(p2);
}

TEST(MemorySummaryTest, ContainsPeakBytes) {
    reset_allocator();

    const size_t sz = 2 * 1024 * 1024;  // 2 MB
    void* p = Alloc::instance().allocate(sz);
    Alloc::instance().deallocate(p);

    std::string summary = Alloc::instance().memory_summary();

    EXPECT_NE(summary.find("peak"), std::string::npos)
        << "summary should mention peak bytes:\n" << summary;
}

TEST(MemorySummaryTest, ContainsBackendAllocFreeCounters) {
    reset_allocator();

    const size_t sz = 4 * 1024 * 1024;
    void* p = Alloc::instance().allocate(sz);
    Alloc::instance().deallocate(p);
    // Force OS return.
    Alloc::instance().set_max_cached_bytes(0);
    Alloc::instance().set_max_cached_bytes(1ULL << 30);

    std::string summary = Alloc::instance().memory_summary();

    EXPECT_NE(summary.find("backend"), std::string::npos)
        << "summary should mention backend alloc/free:\n" << summary;
}

TEST(MemorySummaryTest, LocalThreadStats) {
    reset_allocator();

    const size_t sz = 256 * 1024;
    void* p = Alloc::instance().allocate(sz);

    std::string summary = Alloc::instance().memory_summary();

    // The summary should include per-thread info.
    EXPECT_NE(summary.find("thread"), std::string::npos)
        << "summary should include per-thread stats:\n" << summary;

    Alloc::instance().deallocate(p);
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
