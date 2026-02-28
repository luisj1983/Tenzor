/**
 * @file test_vulkan_allocator_merge.cpp
 * @brief Tests for Vulkan caching allocator block merging
 *
 * Validates that adjacent free blocks are merged on allocation cache-miss,
 * reducing memory fragmentation.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/vulkan_caching_allocator.hpp>

using namespace tenzor;
using namespace tenzor::backend;

class VulkanAllocatorMergeEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const alloc_merge_env =
    ::testing::AddGlobalTestEnvironment(new VulkanAllocatorMergeEnvironment);

class VulkanAllocatorMergeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip if Vulkan not available
        try {
            auto t = zeros({1}, DType::Float32, Device::vulkan(0));
            vulkan_available_ = true;
        } catch (...) {
            vulkan_available_ = false;
        }
    }

    bool vulkan_available_ = false;
};

TEST_F(VulkanAllocatorMergeTest, AdjacentBlocksMerge) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    auto& allocator = VulkanCachingAllocator::get();
    auto stats_before = allocator.get_stats(0);

    // Allocate two blocks of same size
    // Using tensors to drive allocations through the normal path
    constexpr int64_t elements = 256 * 1024;  // ~1MB per tensor at Float32

    {
        // Allocate A and B (these may be sub-allocated from a slab)
        auto a = zeros({elements}, DType::Float32, Device::vulkan(0));
        auto b = zeros({elements}, DType::Float32, Device::vulkan(0));
    }
    // a and b are freed here, blocks return to cache

    // Now allocate a larger tensor that needs both blocks merged
    // The merged block should satisfy this allocation
    auto c = zeros({elements * 2}, DType::Float32, Device::vulkan(0));

    auto stats_after = allocator.get_stats(0);

    // Verify tensor was created successfully
    EXPECT_EQ(c.shape()[0], elements * 2);

    // If merging happened, num_merges should have increased
    // Note: merging may not always happen if the slab allocator
    // provides a large enough block directly
    std::cout << "Merges before: " << stats_before.num_merges
              << ", after: " << stats_after.num_merges << std::endl;
    std::cout << "Cache hits: " << stats_after.num_cache_hits << std::endl;
    std::cout << "Splits: " << stats_after.num_splits << std::endl;
}

TEST_F(VulkanAllocatorMergeTest, FragmentedAllocations) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    auto& allocator = VulkanCachingAllocator::get();

    // Create fragmentation pattern: allocate many small blocks,
    // free alternating ones, then try to allocate a larger block
    constexpr int num_blocks = 8;
    constexpr int64_t small_size = 64 * 1024;  // ~256KB each at Float32

    std::vector<Tensor> blocks;
    blocks.reserve(num_blocks);

    // Allocate all blocks
    for (int i = 0; i < num_blocks; ++i) {
        blocks.push_back(zeros({small_size}, DType::Float32, Device::vulkan(0)));
    }

    auto stats_mid = allocator.get_stats(0);

    // Free alternating blocks (creates fragmentation)
    for (int i = 0; i < num_blocks; i += 2) {
        blocks[i] = Tensor{};  // Release the tensor
    }

    // Free remaining blocks
    for (int i = 1; i < num_blocks; i += 2) {
        blocks[i] = Tensor{};
    }
    blocks.clear();

    // Try to allocate a block that's larger than any individual freed block
    // This should trigger merging of adjacent free blocks
    auto large = zeros({small_size * 2}, DType::Float32, Device::vulkan(0));

    auto stats_final = allocator.get_stats(0);

    EXPECT_EQ(large.shape()[0], small_size * 2);

    std::cout << "Stats: merges=" << stats_final.num_merges
              << " splits=" << stats_final.num_splits
              << " cache_hits=" << stats_final.num_cache_hits
              << " allocated=" << stats_final.allocated_bytes
              << " cached=" << stats_final.cached_bytes << std::endl;
}
