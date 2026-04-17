/**
 * @file test_memory_profiler_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for the global memory profiler
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/utils/memory_profiler.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <thread>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

class MemoryProfilerMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(MemoryProfilerMultiDTypeTest, SingletonAccess) {
    auto& profiler1 = MemoryProfiler::instance();
    auto& profiler2 = MemoryProfiler::instance();
    EXPECT_EQ(&profiler1, &profiler2);
}

TEST_P(MemoryProfilerMultiDTypeTest, AllocateUpdatesCounters) {
    auto& profiler = MemoryProfiler::instance();

    auto before = profiler.memory_stats();
    profiler.on_allocate(1024);
    auto after = profiler.memory_stats();

    EXPECT_EQ(after.current_allocated_bytes, before.current_allocated_bytes + 1024);
    EXPECT_EQ(after.total_allocated_bytes, before.total_allocated_bytes + 1024);
    EXPECT_EQ(after.allocation_count, before.allocation_count + 1);

    profiler.on_deallocate(1024);
}

TEST_P(MemoryProfilerMultiDTypeTest, DeallocateUpdatesCounters) {
    auto& profiler = MemoryProfiler::instance();

    profiler.on_allocate(2048);
    auto before = profiler.memory_stats();
    profiler.on_deallocate(2048);
    auto after = profiler.memory_stats();

    EXPECT_EQ(after.current_allocated_bytes, before.current_allocated_bytes - 2048);
    EXPECT_EQ(after.deallocation_count, before.deallocation_count + 1);
}

TEST_P(MemoryProfilerMultiDTypeTest, PeakTracking) {
    auto& profiler = MemoryProfiler::instance();

    profiler.on_allocate(1000000);
    auto peak_after_alloc = profiler.memory_stats().peak_allocated_bytes;

    profiler.on_deallocate(500000);
    auto peak_after_dealloc = profiler.memory_stats().peak_allocated_bytes;

    EXPECT_EQ(peak_after_alloc, peak_after_dealloc);

    profiler.on_deallocate(500000);
}

TEST_P(MemoryProfilerMultiDTypeTest, MemorySummaryNotEmpty) {
    auto& profiler = MemoryProfiler::instance();
    auto summary = profiler.memory_summary();
    EXPECT_FALSE(summary.empty());
}

TEST_P(MemoryProfilerMultiDTypeTest, TensorAllocationOnDevice) {
    // Verify that tensor creation on the target device works and
    // produces a tensor with the expected dtype and device.
    auto t = createRandn({8, 8});
    expectDType(t);
    expectDevice(t);
    EXPECT_EQ(t.numel(), 64);
}

TEST_P(MemoryProfilerMultiDTypeTest, ThreadSafety) {
    auto& profiler = MemoryProfiler::instance();

    auto before = profiler.memory_stats();

    constexpr int num_threads = 4;
    constexpr int ops_per_thread = 100;
    constexpr size_t alloc_size = 64;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&]() {
            for (int i = 0; i < ops_per_thread; i++) {
                profiler.on_allocate(alloc_size);
            }
            for (int i = 0; i < ops_per_thread; i++) {
                profiler.on_deallocate(alloc_size);
            }
        });
    }
    for (auto& t : threads) t.join();

    auto after = profiler.memory_stats();

    EXPECT_EQ(after.current_allocated_bytes, before.current_allocated_bytes);
    EXPECT_EQ(after.allocation_count, before.allocation_count + num_threads * ops_per_thread);
    EXPECT_EQ(after.deallocation_count, before.deallocation_count + num_threads * ops_per_thread);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MemoryProfilerMultiDTypeTest);
