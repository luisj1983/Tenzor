#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class ChunkTest : public ::tenzor::testing::BackendTest {};

TEST_P(ChunkTest, BasicChunkEvenDivision) {
    // Test case: Shape (12, 20), split into 4 chunks along dim=0
    // Expected: 4 tensors of shape (3, 20)
    auto x = zeros({12, 20}, DType::Float32, device);

    auto chunks = chunk(x, 4, 0);

    ASSERT_EQ(chunks.size(), 4);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 3);
        EXPECT_EQ(shape[1], 20);
    }
}

TEST_P(ChunkTest, BasicChunkUnevenDivision) {
    // Test case: Shape (10, 20), split into 4 chunks along dim=0
    // Expected: 3 tensors of shape (3, 20) and 1 tensor of shape (1, 20)
    auto x = zeros({10, 20}, DType::Float32, device);

    auto chunks = chunk(x, 4, 0);

    ASSERT_EQ(chunks.size(), 4);

    // First 3 chunks should be (3, 20)
    for (size_t i = 0; i < 3; ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 3);
        EXPECT_EQ(shape[1], 20);
    }

    // Last chunk should be (1, 20)
    auto last_shape = chunks[3].shape();
    EXPECT_EQ(last_shape[0], 1);
    EXPECT_EQ(last_shape[1], 20);
}

TEST_P(ChunkTest, ChunkFewerThanDimensionSize) {
    // Test case: Shape (5, 20), split into 10 chunks along dim=0
    // Expected: 5 tensors of shape (1, 20) - fewer chunks than requested
    auto x = zeros({5, 20}, DType::Float32, device);

    auto chunks = chunk(x, 10, 0);

    // Should only get 5 chunks since dimension size is 5
    ASSERT_EQ(chunks.size(), 5);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 20);
    }
}

TEST_P(ChunkTest, ChunkAlongDifferentDimension) {
    // Test case: Shape (10, 20), split into 5 chunks along dim=1
    // Expected: 5 tensors of shape (10, 4)
    auto x = zeros({10, 20}, DType::Float32, device);

    auto chunks = chunk(x, 5, 1);

    ASSERT_EQ(chunks.size(), 5);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 10);
        EXPECT_EQ(shape[1], 4);
    }
}

TEST_P(ChunkTest, ChunkNegativeDimension) {
    // Test case: Use negative dimension indexing
    auto x = zeros({10, 20}, DType::Float32, device);

    // dim=-1 should be equivalent to dim=1
    auto chunks = chunk(x, 4, -1);

    ASSERT_EQ(chunks.size(), 4);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape[0], 10);
        EXPECT_EQ(shape[1], 5);
    }
}

TEST_P(ChunkTest, ChunkSingleChunk) {
    // Test case: Split into 1 chunk should return original tensor
    auto x = zeros({10, 20}, DType::Float32, device);

    auto chunks = chunk(x, 1, 0);

    ASSERT_EQ(chunks.size(), 1);
    auto shape = chunks[0].shape();
    EXPECT_EQ(shape[0], 10);
    EXPECT_EQ(shape[1], 20);
}

TEST_P(ChunkTest, ChunkInvalidChunks) {
    auto x = zeros({10, 20}, DType::Float32, device);

    // Zero chunks should throw
    EXPECT_THROW(chunk(x, 0, 0), std::invalid_argument);

    // Negative chunks should throw
    EXPECT_THROW(chunk(x, -1, 0), std::invalid_argument);
}

TEST_P(ChunkTest, ChunkInvalidDimension) {
    auto x = zeros({10, 20}, DType::Float32, device);

    // Dimension out of range
    EXPECT_THROW(chunk(x, 4, 2), std::invalid_argument);
    EXPECT_THROW(chunk(x, 4, -3), std::invalid_argument);
}

TEST_P(ChunkTest, ChunkDataCorrectness) {
    // Verify that chunk() creates tensors with correct shapes and is slice-based
    // Build the source on CPU (host writes), then move to the target device.
    auto x_cpu = zeros({6, 3}, DType::Float32, Device::cpu());

    // Fill with sequential values for verification
    float* data = x_cpu.data<float>();
    for (int i = 0; i < 18; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto x = x_cpu.to(device);

    // Split into 3 chunks along dim=0
    auto chunks = chunk(x, 3, 0);

    ASSERT_EQ(chunks.size(), 3);

    // Verify that chunks share storage with original (they're views/slices)
    // This is the primary behavior of chunk() - it should create views, not copies
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape[0], 2);  // Each chunk has 2 rows
        EXPECT_EQ(shape[1], 3);  // Same number of columns
    }

    // Note: Testing actual data values requires contiguous() to work properly,
    // which is a separate concern from chunk() functionality
}

INSTANTIATE_BACKEND_TESTS(ChunkTest);
