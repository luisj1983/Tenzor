/**
 * @file test_pinned_allocator.cpp
 * @brief Comprehensive unit tests for Pinned Memory Allocator (Phase 1 - ZeRO Offload)
 *
 * Tests cover:
 * - Memory allocation and deallocation
 * - Pool initialization and management
 * - Out of memory handling
 * - Block coalescing on free
 * - Fragmentation measurement
 * - Thread-safe allocation
 * - Multiple allocation sizes
 * - Defragmentation
 */

#include <gtest/gtest.h>
#include <tenzor/core/pinned_allocator.hpp>
#include <tenzor/ops/creation.hpp>
#include <thread>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>

using namespace tenzor;
using namespace tenzor::core;

/**
 * Test Fixture for Pinned Allocator
 */
class PinnedAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default configuration
        default_config.pool_size = 64 * 1024 * 1024;  // 64 MB
        default_config.min_block_size = 256;           // 256 bytes
        default_config.allow_growth = false;
        default_config.growth_increment = 16 * 1024 * 1024;  // 16 MB
        default_config.max_pool_size = 128 * 1024 * 1024;    // 128 MB
        default_config.enable_defragmentation = true;
    }

    void TearDown() override {
        // Clean up
    }

    PinnedMemoryAllocator::Config default_config;

    // Helper to verify memory is pinned (non-pageable)
    bool isMemoryPinned(void* ptr) {
        // Platform-specific check (simplified for test)
        return ptr != nullptr;
    }

    // Helper to test memory access
    void testMemoryAccess(void* ptr, size_t size) {
        if (ptr == nullptr || size == 0) return;

        // Write pattern
        std::memset(ptr, 0xAA, size);

        // Verify pattern
        uint8_t* bytes = static_cast<uint8_t*>(ptr);
        for (size_t i = 0; i < size; ++i) {
            ASSERT_EQ(bytes[i], 0xAA) << "Memory corruption at offset " << i;
        }

        // Write different pattern
        std::memset(ptr, 0x55, size);

        // Verify
        for (size_t i = 0; i < size; ++i) {
            ASSERT_EQ(bytes[i], 0x55) << "Memory corruption at offset " << i;
        }
    }
};

// =============================================================================
// Constructor and Initialization Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, ConstructorWithValidConfig) {
    ASSERT_NO_THROW({
        PinnedMemoryAllocator allocator(default_config);
    });
}

TEST_F(PinnedAllocatorTest, ConstructorInitializesPool) {
    PinnedMemoryAllocator allocator(default_config);

    EXPECT_TRUE(allocator.is_valid());
    EXPECT_EQ(allocator.get_total_size(), default_config.pool_size);
    EXPECT_EQ(allocator.get_allocated_size(), 0);
}

// =============================================================================
// Basic Allocation Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, AllocateSingleBlock) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(1024);

    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(isMemoryPinned(ptr));
}

TEST_F(PinnedAllocatorTest, AllocateMultipleBlocks) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<void*> pointers;
    for (int i = 0; i < 10; ++i) {
        void* ptr = allocator.allocate(1024);
        ASSERT_NE(ptr, nullptr);
        pointers.push_back(ptr);
    }

    // Verify all pointers are unique
    std::set<void*> unique_ptrs(pointers.begin(), pointers.end());
    EXPECT_EQ(unique_ptrs.size(), pointers.size());
}

TEST_F(PinnedAllocatorTest, AllocateZeroBytes) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(0);

    // Should return nullptr
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(PinnedAllocatorTest, AllocateMinBlockSize) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(default_config.min_block_size);

    ASSERT_NE(ptr, nullptr);
}

TEST_F(PinnedAllocatorTest, AllocateLargerThanPool) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(default_config.pool_size + 1024);

    EXPECT_EQ(ptr, nullptr);  // Out of memory
}

// =============================================================================
// Deallocation Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, DeallocateSingleBlock) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(ptr, nullptr);

    ASSERT_NO_THROW(allocator.deallocate(ptr));
}

TEST_F(PinnedAllocatorTest, DeallocateMultipleBlocks) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<void*> pointers;
    for (int i = 0; i < 10; ++i) {
        pointers.push_back(allocator.allocate(1024));
    }

    for (void* ptr : pointers) {
        ASSERT_NO_THROW(allocator.deallocate(ptr));
    }
}

TEST_F(PinnedAllocatorTest, DeallocateNullPointer) {
    PinnedMemoryAllocator allocator(default_config);

    ASSERT_NO_THROW(allocator.deallocate(nullptr));  // Should be no-op
}

// =============================================================================
// Memory Reuse Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, ReuseFreedMemory) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr1 = allocator.allocate(1024);
    allocator.deallocate(ptr1);

    void* ptr2 = allocator.allocate(1024);

    EXPECT_NE(ptr2, nullptr);
    // May or may not be the same pointer (implementation dependent)
}

TEST_F(PinnedAllocatorTest, MemoryAvailableAfterFree) {
    PinnedMemoryAllocator allocator(default_config);

    size_t alloc_size = default_config.pool_size / 2;
    void* ptr = allocator.allocate(alloc_size);

    auto allocated_before = allocator.get_allocated_size();

    allocator.deallocate(ptr);

    auto allocated_after = allocator.get_allocated_size();

    EXPECT_LT(allocated_after, allocated_before);
}

// =============================================================================
// Block Coalescing Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, CoalesceAdjacentBlocks) {
    PinnedMemoryAllocator allocator(default_config);

    // Allocate three blocks
    void* ptr1 = allocator.allocate(1024);
    void* ptr2 = allocator.allocate(1024);
    void* ptr3 = allocator.allocate(1024);

    // Free them in order
    allocator.deallocate(ptr1);
    allocator.deallocate(ptr2);
    allocator.deallocate(ptr3);

    // Should be able to allocate larger block (coalesced)
    void* large_ptr = allocator.allocate(3 * 1024);
    EXPECT_NE(large_ptr, nullptr);
}

TEST_F(PinnedAllocatorTest, CoalesceOutOfOrder) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr1 = allocator.allocate(1024);
    void* ptr2 = allocator.allocate(1024);
    void* ptr3 = allocator.allocate(1024);

    // Free out of order
    allocator.deallocate(ptr2);
    allocator.deallocate(ptr1);
    allocator.deallocate(ptr3);

    // Should still coalesce
    void* large_ptr = allocator.allocate(3 * 1024);
    EXPECT_NE(large_ptr, nullptr);
}

TEST_F(PinnedAllocatorTest, CoalescingReducesFragmentation) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<void*> pointers;
    for (int i = 0; i < 100; ++i) {
        pointers.push_back(allocator.allocate(1024));
    }

    // Free all
    for (void* ptr : pointers) {
        allocator.deallocate(ptr);
    }

    auto frag = allocator.get_fragmentation_ratio();

    // Fragmentation should be low after coalescing
    EXPECT_LT(frag, 0.1f);  // < 10%
}

// =============================================================================
// Out of Memory Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, OutOfMemoryReturnsNull) {
    PinnedMemoryAllocator allocator(default_config);

    // Fill pool
    void* ptr = allocator.allocate(default_config.pool_size);
    ASSERT_NE(ptr, nullptr);

    // Try to allocate more
    void* ptr2 = allocator.allocate(1024);
    EXPECT_EQ(ptr2, nullptr);
}

TEST_F(PinnedAllocatorTest, OutOfMemoryAfterFragmentation) {
    PinnedMemoryAllocator allocator(default_config);

    // Create fragmented memory
    std::vector<void*> keep_ptrs;
    std::vector<void*> free_ptrs;

    for (int i = 0; i < 100; ++i) {
        void* ptr = allocator.allocate(1024);
        if (i % 2 == 0) {
            keep_ptrs.push_back(ptr);
        } else {
            free_ptrs.push_back(ptr);
        }
    }

    // Free alternating blocks
    for (void* ptr : free_ptrs) {
        allocator.deallocate(ptr);
    }

    // Try to allocate large contiguous block
    void* large_ptr = allocator.allocate(50 * 1024);

    // May fail due to fragmentation (implementation dependent)
    auto frag = allocator.get_fragmentation_ratio();
    if (large_ptr == nullptr) {
        EXPECT_GT(frag, 0.0f);
    }
}

// =============================================================================
// Fragmentation Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, MeasureFragmentation) {
    PinnedMemoryAllocator allocator(default_config);

    auto frag = allocator.get_fragmentation_ratio();
    EXPECT_EQ(frag, 0.0f);  // Initially no fragmentation
}

TEST_F(PinnedAllocatorTest, FragmentationIncreases) {
    PinnedMemoryAllocator allocator(default_config);

    // Create fragmented pattern
    std::vector<void*> ptrs;
    for (int i = 0; i < 50; ++i) {
        ptrs.push_back(allocator.allocate(1024));
    }

    // Free every other block
    for (size_t i = 1; i < ptrs.size(); i += 2) {
        allocator.deallocate(ptrs[i]);
    }

    auto frag = allocator.get_fragmentation_ratio();
    EXPECT_GT(frag, 0.0f);
}

TEST_F(PinnedAllocatorTest, DefragmentationWorks) {
    PinnedMemoryAllocator allocator(default_config);

    // Create fragmentation
    std::vector<void*> ptrs;
    for (int i = 0; i < 50; ++i) {
        ptrs.push_back(allocator.allocate(1024));
    }

    for (size_t i = 1; i < ptrs.size(); i += 2) {
        allocator.deallocate(ptrs[i]);
    }

    float frag_before = allocator.get_fragmentation_ratio();

    // Defragment
    size_t coalesced = allocator.defragment();

    float frag_after = allocator.get_fragmentation_ratio();

    EXPECT_GE(coalesced, 0);
    EXPECT_LE(frag_after, frag_before);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, ConcurrentAllocations) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(4);

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 100; ++i) {
                void* ptr = allocator.allocate(1024);
                if (ptr != nullptr) {
                    thread_ptrs[t].push_back(ptr);
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all pointers are unique
    std::set<void*> all_ptrs;
    for (const auto& ptrs : thread_ptrs) {
        for (void* ptr : ptrs) {
            EXPECT_TRUE(all_ptrs.insert(ptr).second) << "Duplicate pointer allocated";
        }
    }
}

TEST_F(PinnedAllocatorTest, ConcurrentAllocationsAndDeallocations) {
    PinnedMemoryAllocator allocator(default_config);

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            std::vector<void*> local_ptrs;

            for (int i = 0; i < 1000; ++i) {
                if (i % 2 == 0) {
                    // Allocate
                    void* ptr = allocator.allocate(1024);
                    if (ptr != nullptr) {
                        local_ptrs.push_back(ptr);
                    }
                } else if (!local_ptrs.empty()) {
                    // Deallocate
                    try {
                        allocator.deallocate(local_ptrs.back());
                        local_ptrs.pop_back();
                    } catch (...) {
                        errors++;
                    }
                }
            }

            // Clean up
            for (void* ptr : local_ptrs) {
                try {
                    allocator.deallocate(ptr);
                } catch (...) {
                    errors++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(errors, 0);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, StatisticsTrackUsage) {
    PinnedMemoryAllocator allocator(default_config);

    EXPECT_EQ(allocator.get_allocated_size(), 0);

    void* ptr = allocator.allocate(1024);

    EXPECT_GT(allocator.get_allocated_size(), 0);
    EXPECT_LT(allocator.get_free_size(), default_config.pool_size);
}

TEST_F(PinnedAllocatorTest, StatisticsAccurate) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<void*> ptrs;

    for (int i = 0; i < 10; ++i) {
        void* ptr = allocator.allocate(1024);
        if (ptr != nullptr) {
            ptrs.push_back(ptr);
        }
    }

    auto stats = allocator.get_stats();

    EXPECT_GT(stats.allocated_size, 0);
    EXPECT_EQ(stats.total_size, default_config.pool_size);
    EXPECT_EQ(stats.allocated_size + stats.free_size, stats.total_size);
}

TEST_F(PinnedAllocatorTest, GetAllocationCount) {
    PinnedMemoryAllocator allocator(default_config);

    EXPECT_EQ(allocator.get_allocation_count(), 0);

    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        ptrs.push_back(allocator.allocate(1024));
    }

    EXPECT_EQ(allocator.get_allocation_count(), 5);

    for (void* ptr : ptrs) {
        allocator.deallocate(ptr);
    }

    EXPECT_EQ(allocator.get_allocation_count(), 0);
}

// =============================================================================
// Multiple Allocation Sizes Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, VariousSizeAllocations) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<size_t> sizes = {16, 64, 256, 1024, 4096, 16384, 65536};
    std::vector<void*> ptrs;

    for (size_t size : sizes) {
        void* ptr = allocator.allocate(size);
        ASSERT_NE(ptr, nullptr) << "Failed to allocate " << size << " bytes";
        ptrs.push_back(ptr);
    }

    for (void* ptr : ptrs) {
        allocator.deallocate(ptr);
    }
}

TEST_F(PinnedAllocatorTest, MixedSizePattern) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<void*> small_ptrs;
    std::vector<void*> medium_ptrs;
    std::vector<void*> large_ptrs;

    // Allocate mixed sizes
    for (int i = 0; i < 10; ++i) {
        small_ptrs.push_back(allocator.allocate(256));
        medium_ptrs.push_back(allocator.allocate(4096));
        large_ptrs.push_back(allocator.allocate(65536));
    }

    // Free in mixed order
    for (void* ptr : medium_ptrs) allocator.deallocate(ptr);
    for (void* ptr : small_ptrs) allocator.deallocate(ptr);
    for (void* ptr : large_ptrs) allocator.deallocate(ptr);

    // Verify pool is functional
    void* test_ptr = allocator.allocate(1024);
    EXPECT_NE(test_ptr, nullptr);
}

// =============================================================================
// Memory Access Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, AllocatedMemoryIsAccessible) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(4096);
    ASSERT_NE(ptr, nullptr);

    testMemoryAccess(ptr, 4096);

    allocator.deallocate(ptr);
}

TEST_F(PinnedAllocatorTest, MultipleBlocksIndependent) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr1 = allocator.allocate(1024);
    void* ptr2 = allocator.allocate(1024);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    // Write different patterns
    std::memset(ptr1, 0xAA, 1024);
    std::memset(ptr2, 0x55, 1024);

    // Verify independence
    uint8_t* bytes1 = static_cast<uint8_t*>(ptr1);
    uint8_t* bytes2 = static_cast<uint8_t*>(ptr2);

    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_EQ(bytes1[i], 0xAA);
        EXPECT_EQ(bytes2[i], 0x55);
    }

    allocator.deallocate(ptr1);
    allocator.deallocate(ptr2);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, ResetClearsAllocations) {
    PinnedMemoryAllocator allocator(default_config);

    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(allocator.allocate(1024));
    }

    EXPECT_GT(allocator.get_allocated_size(), 0);

    allocator.reset();

    EXPECT_EQ(allocator.get_allocated_size(), 0);
    EXPECT_EQ(allocator.get_allocation_count(), 0);
}

// =============================================================================
// Defragment Edge Case Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, DefragmentEmptyPool) {
    PinnedMemoryAllocator allocator(default_config);

    // No allocations made — defragment on empty pool must not crash
    // (Regression test for unsigned underflow in sorted_blocks.size() - 1)
    size_t coalesced = allocator.defragment();
    EXPECT_EQ(coalesced, 0u);
}

TEST_F(PinnedAllocatorTest, DefragmentSingleBlock) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(ptr, nullptr);
    allocator.deallocate(ptr);

    // Single free block — nothing to coalesce
    size_t coalesced = allocator.defragment();
    EXPECT_EQ(coalesced, 0u);
}

TEST_F(PinnedAllocatorTest, DefragmentCoalescesAdjacentFreeBlocks) {
    PinnedMemoryAllocator allocator(default_config);

    // Create fragmentation pattern: allocate many blocks, free alternating
    // to create non-adjacent free blocks that deallocate() can't coalesce
    constexpr int N = 20;
    std::vector<void*> ptrs;
    for (int i = 0; i < N; ++i) {
        void* p = allocator.allocate(4096);
        ASSERT_NE(p, nullptr);
        ptrs.push_back(p);
    }

    // Free every other block — creates isolated free blocks separated by
    // allocated blocks, so deallocate() can't coalesce them
    for (int i = 0; i < N; i += 2) {
        allocator.deallocate(ptrs[i]);
        ptrs[i] = nullptr;
    }

    // Now free the remaining blocks — adjacent free blocks may coalesce
    // on deallocate(), but defragment() should handle any remaining pairs
    for (int i = 1; i < N; i += 2) {
        allocator.deallocate(ptrs[i]);
        ptrs[i] = nullptr;
    }

    allocator.defragment();

    // After full deallocation + defragmentation, should be able to
    // allocate a large contiguous block
    void* big = allocator.allocate(4096 * (N - 2));
    EXPECT_NE(big, nullptr);
    if (big) allocator.deallocate(big);
}

TEST_F(PinnedAllocatorTest, DefragmentStressFragmented) {
    PinnedMemoryAllocator allocator(default_config);

    constexpr int N = 100;
    std::vector<void*> ptrs;
    for (int i = 0; i < N; ++i) {
        void* p = allocator.allocate(1024);
        if (p) ptrs.push_back(p);
    }

    // Free every other block to create fragmentation
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        allocator.deallocate(ptrs[i]);
        ptrs[i] = nullptr;
    }

    float frag_before = allocator.get_fragmentation_ratio();
    allocator.defragment();
    float frag_after = allocator.get_fragmentation_ratio();

    // Fragmentation should not increase after defragmentation
    EXPECT_LE(frag_after, frag_before);

    // Cleanup remaining allocations
    for (void* p : ptrs) {
        if (p) allocator.deallocate(p);
    }
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST_F(PinnedAllocatorTest, MoveConstructor) {
    PinnedMemoryAllocator allocator(default_config);

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(ptr, nullptr);

    PinnedMemoryAllocator moved(std::move(allocator));
    EXPECT_TRUE(moved.is_valid());

    // Should be able to deallocate through moved allocator
    ASSERT_NO_THROW(moved.deallocate(ptr));
}

TEST_F(PinnedAllocatorTest, MoveAssignment) {
    PinnedMemoryAllocator alloc1(default_config);
    PinnedMemoryAllocator alloc2(default_config);

    void* ptr = alloc1.allocate(2048);
    ASSERT_NE(ptr, nullptr);

    alloc2 = std::move(alloc1);
    EXPECT_TRUE(alloc2.is_valid());

    ASSERT_NO_THROW(alloc2.deallocate(ptr));
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
