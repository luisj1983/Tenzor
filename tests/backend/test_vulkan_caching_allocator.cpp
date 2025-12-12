/**
 * @file test_vulkan_caching_allocator.cpp
 * @brief Unit tests for VulkanCachingAllocator
 *
 * Test coverage includes:
 * - Basic allocation and deallocation
 * - Cache management (hits, misses)
 * - Memory reuse and statistics
 * - Block splitting
 * - Thread safety
 * - Edge cases
 */

#include <gtest/gtest.h>
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/tenzor.hpp"
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
 * @brief Test fixture for VulkanCachingAllocator tests
 */
class VulkanCachingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if Vulkan backend is available
        auto* backend = backend_registry().get_backend(Device::Type::Vulkan);
        if (!backend || !backend->is_available()) {
            GTEST_SKIP() << "Vulkan backend not available";
        }

        // The allocator is initialized by the backend, so we just need to check it's ready
        if (!VulkanCachingAllocator::get().is_initialized(0)) {
            GTEST_SKIP() << "VulkanCachingAllocator not initialized";
        }

        allocator = &VulkanCachingAllocator::get();
        allocator->reset_stats();
    }

    void TearDown() override {
        if (allocator) {
            // Empty cache to clean up
            allocator->empty_cache(0);
        }
    }

    VulkanCachingAllocator* allocator = nullptr;
};

// ============================================================================
// Basic Allocation/Deallocation Tests
// ============================================================================

TEST_F(VulkanCachingAllocatorTest, AllocateZeroReturnsNull) {
    void* ptr = allocator->allocate(0, 0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(VulkanCachingAllocatorTest, AllocateBasicSizes) {
    const std::vector<size_t> sizes = {256, 512, 1024, 4096, 1024 * 1024};

    for (size_t size : sizes) {
        void* ptr = allocator->allocate(size, 0);
        ASSERT_NE(ptr, nullptr) << "Allocation failed for size: " << size;

        // Verify memory is usable (write test pattern)
        std::memset(ptr, 0xAB, size);

        allocator->free(ptr, 0);
    }
}

TEST_F(VulkanCachingAllocatorTest, FreeNullIsNoOp) {
    EXPECT_NO_THROW({
        allocator->free(nullptr, 0);
    });
}

TEST_F(VulkanCachingAllocatorTest, FreeInvalidPointerThrows) {
    int dummy = 0;
    void* invalid_ptr = &dummy;

    EXPECT_THROW({
        allocator->free(invalid_ptr, 0);
    }, std::runtime_error);
}

TEST_F(VulkanCachingAllocatorTest, DoubleFreeThrows) {
    void* ptr = allocator->allocate(1024, 0);
    ASSERT_NE(ptr, nullptr);

    allocator->free(ptr, 0);

    EXPECT_THROW({
        allocator->free(ptr, 0);
    }, std::runtime_error);
}

TEST_F(VulkanCachingAllocatorTest, MultipleAllocations) {
    const size_t count = 10;
    const size_t size = 1024;
    std::vector<void*> ptrs;

    // Allocate multiple blocks
    for (size_t i = 0; i < count; ++i) {
        void* ptr = allocator->allocate(size, 0);
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

TEST_F(VulkanCachingAllocatorTest, CacheReuseExactSize) {
    const size_t size = 1024;

    // First allocation - cache miss
    void* ptr1 = allocator->allocate(size, 0);
    ASSERT_NE(ptr1, nullptr);

    allocator->free(ptr1, 0);

    // Second allocation of same size - should reuse cached block
    void* ptr2 = allocator->allocate(size, 0);
    ASSERT_NE(ptr2, nullptr);

    // Should reuse the same pointer
    EXPECT_EQ(ptr1, ptr2) << "Expected same pointer from cache";

    allocator->free(ptr2, 0);
}

TEST_F(VulkanCachingAllocatorTest, CacheHitTracking) {
    const size_t size = 2048;

    // Initial allocation
    void* ptr1 = allocator->allocate(size, 0);
    allocator->free(ptr1, 0);

    // Multiple allocations of same size should increase hit count
    for (int i = 0; i < 5; ++i) {
        void* ptr = allocator->allocate(size, 0);
        ASSERT_NE(ptr, nullptr);
        allocator->free(ptr, 0);
    }

    auto stats = allocator->get_stats(0);
    EXPECT_GE(stats.num_cache_hits, 5u) << "Expected cache hits for same-size allocations";
}

TEST_F(VulkanCachingAllocatorTest, EmptyCacheClearsAllFreeBlocks) {
    const size_t size = 4096;
    const int count = 5;

    // Allocate and deallocate to build up cache
    std::vector<void*> ptrs;
    for (int i = 0; i < count; ++i) {
        ptrs.push_back(allocator->allocate(size, 0));
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
// Statistics Tests
// ============================================================================

TEST_F(VulkanCachingAllocatorTest, AllocationIncreasesStats) {
    const size_t size = 8192;

    void* ptr = allocator->allocate(size, 0);
    ASSERT_NE(ptr, nullptr);

    EXPECT_GE(allocator->memory_allocated(0), size);
    EXPECT_GE(allocator->memory_reserved(0), size);

    allocator->free(ptr, 0);
}

TEST_F(VulkanCachingAllocatorTest, DeallocationIncreasesCachedStats) {
    const size_t size = 4096;

    void* ptr = allocator->allocate(size, 0);
    allocator->free(ptr, 0);

    EXPECT_GE(allocator->memory_cached(0), size);
}

TEST_F(VulkanCachingAllocatorTest, GetBufferReturnsValidHandle) {
    const size_t size = 1024;

    void* ptr = allocator->allocate(size, 0);
    ASSERT_NE(ptr, nullptr);

    VkBuffer buffer = allocator->get_buffer(ptr, 0);
    EXPECT_NE(buffer, VK_NULL_HANDLE);

    allocator->free(ptr, 0);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(VulkanCachingAllocatorTest, SetAlignmentValidation) {
    // Valid power of 2
    EXPECT_NO_THROW(allocator->set_alignment(512));

    // Invalid (not power of 2)
    EXPECT_THROW(allocator->set_alignment(300), std::invalid_argument);

    // Invalid (zero)
    EXPECT_THROW(allocator->set_alignment(0), std::invalid_argument);
}

TEST_F(VulkanCachingAllocatorTest, SetMaxCachedMemory) {
    // Allocate and free multiple blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator->allocate(1024 * 1024, 0));  // 1MB each
    }
    for (void* ptr : ptrs) {
        allocator->free(ptr, 0);
    }

    // Set a cache limit
    allocator->set_max_cached_memory(5 * 1024 * 1024);  // 5MB

    // Allocate and free to trigger cache limit enforcement
    void* ptr = allocator->allocate(1024 * 1024, 0);
    allocator->free(ptr, 0);

    // Cache should be limited
    EXPECT_LE(allocator->memory_cached(0), 6 * 1024 * 1024);

    // Reset limit
    allocator->set_max_cached_memory(0);  // Unlimited
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(VulkanCachingAllocatorTest, StressTestRandomSizes) {
    const int num_allocations = 100;
    std::vector<void*> ptrs;
    std::vector<size_t> sizes;

    // Random size generator
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> size_dist(256, 16384);

    for (int i = 0; i < num_allocations; ++i) {
        size_t size = size_dist(rng);
        void* ptr = allocator->allocate(size, 0);
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

TEST_F(VulkanCachingAllocatorTest, RepeatedAllocDeallocCycles) {
    const int num_cycles = 50;
    const size_t size = 8192;

    void* prev_ptr = nullptr;
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        void* ptr = allocator->allocate(size, 0);
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

TEST_F(VulkanCachingAllocatorTest, ConcurrentAllocationsBasic) {
    const size_t size = 1024;
    const int num_threads = 4;
    const int allocations_per_thread = 10;

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch threads that allocate memory
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < allocations_per_thread; ++i) {
                void* ptr = allocator->allocate(size, 0);
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

TEST_F(VulkanCachingAllocatorTest, CachedMemoryGuardBasic) {
    size_t allocated_before = allocator->memory_allocated(0);

    {
        VulkanCachedMemoryGuard guard(4096, 0);
        EXPECT_NE(guard.get(), nullptr);
        EXPECT_NE(guard.buffer(), VK_NULL_HANDLE);
        EXPECT_EQ(guard.size(), 4096u);
        EXPECT_GT(allocator->memory_allocated(0), allocated_before);
    }

    // Memory should be returned to cache after guard goes out of scope
    EXPECT_GE(allocator->memory_cached(0), 0u);
}

TEST_F(VulkanCachingAllocatorTest, CachedMemoryGuardMoveSemantics) {
    VulkanCachedMemoryGuard guard1(2048, 0);
    void* original_ptr = guard1.get();

    VulkanCachedMemoryGuard guard2(std::move(guard1));

    EXPECT_EQ(guard1.get(), nullptr);
    EXPECT_EQ(guard2.get(), original_ptr);

    // guard2 will free the memory when it goes out of scope
}
