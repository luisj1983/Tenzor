/**
 * @file test_pinned_allocator_simple.cpp
 * @brief Simple standalone test for PinnedMemoryAllocator
 */

#include "tenzor/core/pinned_allocator.hpp"
#include <iostream>
#include <cstring>
#include <cassert>

using namespace tenzor::core;

void test_basic_allocation() {
    std::cout << "Test: Basic Allocation\n";

    PinnedMemoryAllocator::Config config;
    config.pool_size = 10 * 1024 * 1024;  // 10 MB
    config.min_block_size = 256;

    PinnedMemoryAllocator allocator(config);

    // Test allocation
    void* ptr = allocator.allocate(1024);
    assert(ptr != nullptr);

    // Test memory access
    std::memset(ptr, 0xAB, 1024);
    assert(static_cast<unsigned char*>(ptr)[0] == 0xAB);

    // Test deallocation
    allocator.deallocate(ptr);

    std::cout << "  ✓ Basic allocation test passed\n";
}

void test_multiple_allocations() {
    std::cout << "Test: Multiple Allocations\n";

    PinnedMemoryAllocator::Config config;
    config.pool_size = 10 * 1024 * 1024;

    PinnedMemoryAllocator allocator(config);

    void* ptr1 = allocator.allocate(1024);
    void* ptr2 = allocator.allocate(2048);
    void* ptr3 = allocator.allocate(4096);

    assert(ptr1 != nullptr);
    assert(ptr2 != nullptr);
    assert(ptr3 != nullptr);
    assert(ptr1 != ptr2);
    assert(ptr2 != ptr3);

    allocator.deallocate(ptr1);
    allocator.deallocate(ptr2);
    allocator.deallocate(ptr3);

    std::cout << "  ✓ Multiple allocations test passed\n";
}

void test_statistics() {
    std::cout << "Test: Statistics\n";

    PinnedMemoryAllocator::Config config;
    config.pool_size = 10 * 1024 * 1024;

    PinnedMemoryAllocator allocator(config);

    auto stats_initial = allocator.get_stats();
    assert(stats_initial.total_size == config.pool_size);
    assert(stats_initial.allocated_size == 0);
    assert(stats_initial.num_allocations == 0);

    void* ptr = allocator.allocate(1024);

    auto stats_allocated = allocator.get_stats();
    assert(stats_allocated.num_allocations == 1);
    assert(stats_allocated.allocated_size > 0);

    allocator.deallocate(ptr);

    auto stats_freed = allocator.get_stats();
    assert(stats_freed.num_allocations == 0);
    assert(stats_freed.allocated_size == 0);

    std::cout << "  ✓ Statistics test passed\n";
}

void test_defragmentation() {
    std::cout << "Test: Defragmentation\n";

    PinnedMemoryAllocator::Config config;
    config.pool_size = 10 * 1024 * 1024;

    PinnedMemoryAllocator allocator(config);

    // Allocate multiple blocks
    void* ptrs[10];
    for (int i = 0; i < 10; ++i) {
        ptrs[i] = allocator.allocate(1024);
    }

    // Free alternating blocks to create fragmentation
    for (int i = 1; i < 10; i += 2) {
        allocator.deallocate(ptrs[i]);
    }

    // Defragment
    size_t coalesced = allocator.defragment();

    // Free remaining
    for (int i = 0; i < 10; i += 2) {
        allocator.deallocate(ptrs[i]);
    }

    std::cout << "  ✓ Defragmentation test passed (coalesced " << coalesced << " blocks)\n";
}

void test_out_of_memory() {
    std::cout << "Test: Out of Memory\n";

    PinnedMemoryAllocator::Config config;
    config.pool_size = 1024 * 1024;  // 1 MB
    config.allow_growth = false;

    PinnedMemoryAllocator allocator(config);

    // Try to allocate more than pool size
    void* ptr = allocator.allocate(2 * 1024 * 1024);
    assert(ptr == nullptr);

    std::cout << "  ✓ Out of memory test passed\n";
}

void test_reset() {
    std::cout << "Test: Reset\n";

    PinnedMemoryAllocator::Config config;
    config.pool_size = 10 * 1024 * 1024;

    PinnedMemoryAllocator allocator(config);

    // Allocate some memory
    for (int i = 0; i < 5; ++i) {
        allocator.allocate(1024);
    }

    assert(allocator.get_allocation_count() == 5);

    // Reset
    allocator.reset();

    assert(allocator.get_allocation_count() == 0);
    assert(allocator.get_allocated_size() == 0);

    std::cout << "  ✓ Reset test passed\n";
}

void print_stats(const PinnedMemoryStats& stats) {
    std::cout << "\nMemory Statistics:\n";
    std::cout << "  Total Size:      " << stats.total_size / 1024 << " KB\n";
    std::cout << "  Allocated:       " << stats.allocated_size / 1024 << " KB\n";
    std::cout << "  Free:            " << stats.free_size / 1024 << " KB\n";
    std::cout << "  Allocations:     " << stats.num_allocations << "\n";
    std::cout << "  Blocks:          " << stats.num_blocks << "\n";
    std::cout << "  Free Blocks:     " << stats.num_free_blocks << "\n";
    std::cout << "  Fragmentation:   " << (stats.fragmentation_ratio * 100.0f) << "%\n";
    std::cout << "  Peak Allocated:  " << stats.peak_allocated / 1024 << " KB\n";
    std::cout << "  Defragmentations:" << stats.num_defragmentations << "\n";
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "  PinnedMemoryAllocator Test Suite\n";
    std::cout << "=================================================\n\n";

    try {
        test_basic_allocation();
        test_multiple_allocations();
        test_statistics();
        test_defragmentation();
        test_out_of_memory();
        test_reset();

        // Comprehensive test
        std::cout << "\nTest: Comprehensive Usage\n";
        PinnedMemoryAllocator::Config config;
        config.pool_size = 100 * 1024 * 1024;  // 100 MB
        config.min_block_size = 256;
        config.allow_growth = false;

        PinnedMemoryAllocator allocator(config);

        // Allocate various sizes
        void* ptr1 = allocator.allocate(1024);
        void* ptr2 = allocator.allocate(4096);
        void* ptr3 = allocator.allocate(16384);

        print_stats(allocator.get_stats());

        allocator.deallocate(ptr2);
        allocator.deallocate(ptr1);
        allocator.deallocate(ptr3);

        std::cout << "\nAfter deallocation:\n";
        print_stats(allocator.get_stats());

        std::cout << "\n=================================================\n";
        std::cout << "  ✓ ALL TESTS PASSED\n";
        std::cout << "=================================================\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
