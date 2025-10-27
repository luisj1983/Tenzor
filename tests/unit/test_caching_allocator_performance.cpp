/**
 * @file test_caching_allocator_performance.cpp
 * @brief Performance benchmarks and verification for CachingAllocator
 */

#include <gtest/gtest.h>
#include "tenzor/core/caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/core/device.hpp"
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

namespace tenzor {
namespace test {

/**
 * @brief Simple backend for performance testing
 */
class SimpleBenchmarkBackend : public Backend {
public:
    size_t allocation_count{0};
    size_t deallocation_count{0};
    size_t total_bytes_allocated{0};

    auto name() const -> std::string_view override { return "benchmark"; }
    auto device_count() const -> int32_t override { return 1; }
    auto is_available() const -> bool override { return true; }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        (void)device_id;
        void* ptr = ::operator new(bytes);
        ++allocation_count;
        total_bytes_allocated += bytes;
        return ptr;
    }

    auto deallocate(void* ptr) -> void override {
        if (ptr) {
            ::operator delete(ptr);
            ++deallocation_count;
        }
    }

    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override {
        (void)kind;
        memcpy(dst, src, bytes);
    }

    auto synchronize(int32_t device_id) -> void override { (void)device_id; }
    auto create_stream(int32_t device_id) -> StreamHandle override {
        (void)device_id;
        return nullptr;
    }
    auto destroy_stream(StreamHandle stream) -> void override { (void)stream; }
    auto synchronize_stream(StreamHandle stream) -> void override { (void)stream; }

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        (void)op_name; (void)inputs; (void)attrs;
        return {};
    }

    auto reset_stats() -> void {
        allocation_count = 0;
        deallocation_count = 0;
        total_bytes_allocated = 0;
    }
};

class CachingAllocatorPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = std::make_unique<SimpleBenchmarkBackend>();
        device_ = Device::cpu();
    }

    void TearDown() override {
        allocator_.reset();
        backend_.reset();
    }

    std::unique_ptr<SimpleBenchmarkBackend> backend_;
    Device device_;
    std::unique_ptr<CachingAllocator> allocator_;
};

// ============================================================================
// Cache Hit Rate Tests
// ============================================================================

TEST_F(CachingAllocatorPerformanceTest, CacheHitRateExactSizeReuse) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Allocate and deallocate same size many times
    constexpr size_t size = 1024;
    constexpr int iterations = 1000;

    for (int i = 0; i < iterations; ++i) {
        void* ptr = allocator_->allocate(size);
        allocator_->deallocate(ptr);
    }

    // After first allocation, all should be cache hits
    // Expected: 1 backend allocation, 999 cache hits out of 1000 allocations
    double hit_rate = allocator_->cache_hit_rate();
    EXPECT_GE(hit_rate, 99.0) << "Hit rate should be ~99.9% for exact size reuse";
    EXPECT_EQ(backend_->allocation_count, 1) << "Should only allocate once from backend";
}

TEST_F(CachingAllocatorPerformanceTest, CacheHitRateVariedSizes) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Allocate varied sizes
    std::vector<size_t> sizes = {512, 1024, 2048, 4096, 8192};
    constexpr int iterations_per_size = 100;

    // Allocate each size multiple times
    std::vector<void*> ptrs;
    for (size_t size : sizes) {
        for (int i = 0; i < iterations_per_size; ++i) {
            ptrs.push_back(allocator_->allocate(size));
        }
    }

    // Deallocate all
    for (void* ptr : ptrs) {
        allocator_->deallocate(ptr);
    }

    // Reallocate - should get high cache hit rate
    size_t initial_backend_allocs = backend_->allocation_count;
    ptrs.clear();

    for (size_t size : sizes) {
        for (int i = 0; i < iterations_per_size; ++i) {
            ptrs.push_back(allocator_->allocate(size));
        }
    }

    // Should have no new backend allocations
    EXPECT_EQ(backend_->allocation_count, initial_backend_allocs)
        << "All allocations should come from cache";

    // Cleanup
    for (void* ptr : ptrs) {
        allocator_->deallocate(ptr);
    }
}

// ============================================================================
// Defragmentation Tests
// ============================================================================

TEST_F(CachingAllocatorPerformanceTest, DefragmentationReducesMemory) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Allocate many blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        ptrs.push_back(allocator_->allocate(1024 * (i + 1)));
    }

    size_t allocated_before = allocator_->total_allocated_bytes();

    // Deallocate all - they go into cache
    for (void* ptr : ptrs) {
        allocator_->deallocate(ptr);
    }

    size_t cached = allocator_->total_cached_bytes();
    EXPECT_EQ(cached, allocated_before) << "All memory should be cached";

    size_t backend_deallocs_before = backend_->deallocation_count;

    // Defragment - should free cached memory
    allocator_->defragment();

    EXPECT_EQ(allocator_->total_cached_bytes(), 0) << "Cache should be empty after defrag";
    EXPECT_GT(backend_->deallocation_count, backend_deallocs_before)
        << "Defrag should call backend deallocate";
}

TEST_F(CachingAllocatorPerformanceTest, DefragmentationDoesNotAffectActiveMemory) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Allocate some blocks
    void* active1 = allocator_->allocate(1024);
    void* active2 = allocator_->allocate(2048);

    // Allocate and free some blocks (these go to cache)
    void* temp1 = allocator_->allocate(512);
    void* temp2 = allocator_->allocate(1536);
    allocator_->deallocate(temp1);
    allocator_->deallocate(temp2);

    size_t cached_before = allocator_->total_cached_bytes();
    EXPECT_GT(cached_before, 0);

    size_t active_blocks_before = allocator_->allocated_block_count();

    // Defragment
    allocator_->defragment();

    // Active blocks should remain
    EXPECT_EQ(allocator_->allocated_block_count(), active_blocks_before);
    EXPECT_EQ(allocator_->total_cached_bytes(), 0);

    // Active pointers should still be valid
    EXPECT_NE(active1, nullptr);
    EXPECT_NE(active2, nullptr);

    // Cleanup
    allocator_->deallocate(active1);
    allocator_->deallocate(active2);
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST_F(CachingAllocatorPerformanceTest, DISABLED_AllocationSpeedBenchmark) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    constexpr int iterations = 10000;
    constexpr size_t size = 1024;

    // Warm up
    for (int i = 0; i < 100; ++i) {
        void* ptr = allocator_->allocate(size);
        allocator_->deallocate(ptr);
    }

    // Benchmark cached allocations
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        void* ptr = allocator_->allocate(size);
        allocator_->deallocate(ptr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double ops_per_sec = (iterations * 2.0 / duration.count()) * 1e6;  // *2 for alloc+dealloc

    std::cout << "\nCaching Allocator Performance:\n";
    std::cout << "  Operations: " << (iterations * 2) << " (alloc + dealloc pairs)\n";
    std::cout << "  Time: " << duration.count() << " μs\n";
    std::cout << "  Throughput: " << ops_per_sec << " ops/sec\n";
    std::cout << "  Cache hit rate: " << allocator_->cache_hit_rate() << "%\n";
    std::cout << "  Backend allocations: " << backend_->allocation_count << "\n";

    // Expect high throughput from caching
    EXPECT_GT(ops_per_sec, 1e6) << "Should achieve >1M ops/sec with caching";
}

TEST_F(CachingAllocatorPerformanceTest, DISABLED_MemoryFragmentationBenchmark) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Simulate fragmented allocation pattern
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(64, 4096);

    std::vector<void*> ptrs;
    constexpr int total_allocs = 1000;

    // Random allocations
    for (int i = 0; i < total_allocs; ++i) {
        size_t size = size_dist(gen);
        ptrs.push_back(allocator_->allocate(size));
    }

    // Random deallocations (50%)
    std::shuffle(ptrs.begin(), ptrs.end(), gen);
    for (size_t i = 0; i < ptrs.size() / 2; ++i) {
        allocator_->deallocate(ptrs[i]);
    }
    ptrs.erase(ptrs.begin(), ptrs.begin() + ptrs.size() / 2);

    size_t cached_before = allocator_->total_cached_bytes();
    size_t allocated_before = allocator_->total_allocated_bytes();

    std::cout << "\nFragmentation Test:\n";
    std::cout << "  Total allocated: " << allocated_before << " bytes\n";
    std::cout << "  Cached (free): " << cached_before << " bytes\n";
    std::cout << "  Fragmentation: " << (100.0 * cached_before / allocated_before) << "%\n";
    std::cout << "  Cached blocks: " << allocator_->cached_block_count() << "\n";

    // Defragment
    auto start = std::chrono::high_resolution_clock::now();
    allocator_->defragment();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "  Defragmentation time: " << duration.count() << " μs\n";
    std::cout << "  Cached after defrag: " << allocator_->total_cached_bytes() << " bytes\n";

    EXPECT_EQ(allocator_->total_cached_bytes(), 0);

    // Cleanup
    for (void* ptr : ptrs) {
        allocator_->deallocate(ptr);
    }
}

// ============================================================================
// Best-Fit Strategy Tests
// ============================================================================

TEST_F(CachingAllocatorPerformanceTest, BestFitMinimizesFragmentation) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    // Create blocks of different sizes
    void* p512 = allocator_->allocate(512);
    void* p1024 = allocator_->allocate(1024);
    void* p2048 = allocator_->allocate(2048);
    void* p4096 = allocator_->allocate(4096);

    // Free all
    allocator_->deallocate(p512);
    allocator_->deallocate(p1024);
    allocator_->deallocate(p2048);
    allocator_->deallocate(p4096);

    EXPECT_EQ(allocator_->cached_block_count(), 4);

    size_t backend_allocs_before = backend_->allocation_count;

    // Request 1024 - should get exact match (best fit)
    void* p1024_reused = allocator_->allocate(1024);
    EXPECT_EQ(p1024_reused, p1024) << "Best-fit should return exact size match";
    EXPECT_EQ(backend_->allocation_count, backend_allocs_before) << "Should reuse cached block";
    EXPECT_EQ(allocator_->cached_block_count(), 3) << "One block should be consumed";

    // Request 600 - should get 1024 block (smallest that fits)
    void* p600 = allocator_->allocate(600);
    EXPECT_NE(p600, nullptr);
    EXPECT_EQ(backend_->allocation_count, backend_allocs_before) << "Should reuse cached block";
    EXPECT_EQ(allocator_->cached_block_count(), 2);

    // Cleanup
    allocator_->deallocate(p1024_reused);
    allocator_->deallocate(p600);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(CachingAllocatorPerformanceTest, HighVolumeAllocationDeallocation) {
    allocator_ = std::make_unique<CachingAllocator>(backend_.get(), device_);

    constexpr int iterations = 10000;
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_idx_dist(0, sizes.size() - 1);

    std::vector<void*> active_ptrs;
    active_ptrs.reserve(1000);

    for (int i = 0; i < iterations; ++i) {
        // 70% chance to allocate, 30% to deallocate
        if (active_ptrs.empty() || (gen() % 10) < 7) {
            // Allocate
            size_t size = sizes[size_idx_dist(gen)];
            void* ptr = allocator_->allocate(size);
            ASSERT_NE(ptr, nullptr);
            active_ptrs.push_back(ptr);
        } else {
            // Deallocate random pointer
            std::uniform_int_distribution<> ptr_dist(0, active_ptrs.size() - 1);
            size_t idx = ptr_dist(gen);
            allocator_->deallocate(active_ptrs[idx]);
            active_ptrs.erase(active_ptrs.begin() + idx);
        }
    }

    // Verify cache effectiveness
    double hit_rate = allocator_->cache_hit_rate();
    std::cout << "\nStress Test Results:\n";
    std::cout << "  Cache hit rate: " << hit_rate << "%\n";
    std::cout << "  Backend allocations: " << backend_->allocation_count << "\n";
    std::cout << "  Total operations: " << iterations << "\n";
    std::cout << "  Active pointers: " << active_ptrs.size() << "\n";

    // Cleanup
    for (void* ptr : active_ptrs) {
        allocator_->deallocate(ptr);
    }
}

// ============================================================================
// Memory Leak Detection
// ============================================================================

TEST_F(CachingAllocatorPerformanceTest, NoMemoryLeaksOnDestruction) {
    {
        auto temp_allocator = std::make_unique<CachingAllocator>(backend_.get(), device_);

        // Allocate various sizes
        for (int i = 0; i < 100; ++i) {
            temp_allocator->allocate(1024 * (i + 1));
        }

        size_t allocs = backend_->allocation_count;
        EXPECT_GT(allocs, 0);

        // Allocator destructor should free everything
    }

    // All memory should be freed
    EXPECT_EQ(backend_->allocation_count, backend_->deallocation_count)
        << "All allocations should be freed on destruction";
}

} // namespace test
} // namespace tenzor
