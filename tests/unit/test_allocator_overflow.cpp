/**
 * @file test_allocator_overflow.cpp
 * @brief Tests for integer overflow protection in memory allocators
 *
 * Verifies that allocating near-SIZE_MAX bytes throws std::overflow_error
 * rather than silently wrapping around and allocating a tiny buffer.
 */

#include <gtest/gtest.h>
#include "tenzor/core/pinned_allocator.hpp"
#include <climits>
#include <stdexcept>

using namespace tenzor::core;

// Test that PinnedMemoryAllocator rejects absurdly large allocations
TEST(AllocatorOverflow, PinnedAllocator_RejectsNearSizeMax) {
    try {
        PinnedMemoryAllocator::Config config;
        config.pool_size = 1024 * 1024;  // 1MB pool
        config.min_block_size = 256;
        PinnedMemoryAllocator allocator(config);

        // This should throw overflow_error or return nullptr, never succeed
        void* ptr = allocator.allocate(SIZE_MAX);
        if (ptr != nullptr) {
            allocator.deallocate(ptr);
            FAIL() << "allocate(SIZE_MAX) should not return a valid pointer";
        }
        // nullptr return is acceptable (out of memory)
    } catch (const std::overflow_error&) {
        // Expected: align_size overflow protection triggered
        SUCCEED();
    } catch (const std::runtime_error&) {
        // Also acceptable: CUDA not available, or pool init failed
        GTEST_SKIP() << "PinnedMemoryAllocator not available (no CUDA pinned memory)";
    } catch (const std::exception& e) {
        // Any exception is fine — the key is no silent wraparound
        SUCCEED() << "Threw: " << e.what();
    }
}

// Test that SIZE_MAX - small_value also triggers protection
TEST(AllocatorOverflow, PinnedAllocator_RejectsNearMaxAlignment) {
    try {
        PinnedMemoryAllocator::Config config;
        config.pool_size = 1024 * 1024;
        config.min_block_size = 256;
        PinnedMemoryAllocator allocator(config);

        // SIZE_MAX - 254 would overflow when align_size adds 255 for 256-byte alignment
        void* ptr = allocator.allocate(SIZE_MAX - 254);
        if (ptr != nullptr) {
            allocator.deallocate(ptr);
            FAIL() << "allocate(SIZE_MAX - 254) should not succeed";
        }
    } catch (const std::overflow_error&) {
        SUCCEED();
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "PinnedMemoryAllocator not available";
    } catch (const std::exception&) {
        SUCCEED();
    }
}
