/**
 * @file test_split_operation.cpp
 * @brief Tests for the split() operation
 */

#include "backend_test_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include <vector>
#include <cmath>

using namespace tenzor;

class SplitOperationTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        // Basic test shapes
        shape_1d = {12};
        shape_2d = {6, 8};
        shape_3d = {4, 6, 8};
    }

    std::vector<int64_t> shape_1d;
    std::vector<int64_t> shape_2d;
    std::vector<int64_t> shape_3d;
};

// Test 1D tensor split
TEST_P(SplitOperationTest, Split1DTensorEvenDivision) {
    auto tensor = arange(0.0f, 12.0f, 1.0f, DType::Float32, device);

    // Split into chunks of size 3
    auto chunks = split(tensor, 3, 0);

    ASSERT_EQ(chunks.size(), 4);  // 12 / 3 = 4 chunks

    // Verify each chunk has the correct shape and values
    for (size_t i = 0; i < chunks.size(); ++i) {
        EXPECT_EQ(chunks[i].shape()[0], 3);
        EXPECT_EQ(chunks[i].numel(), 3);

        // Verify values
        auto chunk_cpu = chunks[i].cpu();
        auto data = chunk_cpu.data<float>();
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(data[j], i * 3 + j);
        }
    }
}

// Test 1D tensor split with uneven division
TEST_P(SplitOperationTest, Split1DTensorUnevenDivision) {
    auto tensor = arange(0.0f, 10.0f, 1.0f, DType::Float32, device);

    // Split into chunks of size 3 (will have 4 chunks: 3, 3, 3, 1)
    auto chunks = split(tensor, 3, 0);

    ASSERT_EQ(chunks.size(), 4);  // Ceiling(10 / 3) = 4 chunks

    // First three chunks should have size 3
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(chunks[i].shape()[0], 3);
        EXPECT_EQ(chunks[i].numel(), 3);
    }

    // Last chunk should have size 1
    EXPECT_EQ(chunks[3].shape()[0], 1);
    EXPECT_EQ(chunks[3].numel(), 1);

    // Verify last chunk value
    auto last_cpu = chunks[3].cpu();
    auto last_data = last_cpu.data<float>();
    EXPECT_FLOAT_EQ(last_data[0], 9.0f);
}

// Test 2D tensor split along dim 0
TEST_P(SplitOperationTest, Split2DTensorDim0) {
    auto tensor = ones(shape_2d, DType::Float32, device);

    // Split into chunks of size 2 along dimension 0
    auto chunks = split(tensor, 2, 0);

    ASSERT_EQ(chunks.size(), 3);  // 6 / 2 = 3 chunks

    // Verify each chunk shape
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto chunk_shape = chunks[i].shape();
        EXPECT_EQ(chunk_shape[0], 2);  // Split dimension
        EXPECT_EQ(chunk_shape[1], 8);  // Unchanged dimension
        EXPECT_EQ(chunks[i].numel(), 2 * 8);
    }
}

// Test 2D tensor split along dim 1
TEST_P(SplitOperationTest, Split2DTensorDim1) {
    auto tensor = ones(shape_2d, DType::Float32, device);

    // Split into chunks of size 3 along dimension 1
    auto chunks = split(tensor, 3, 1);

    ASSERT_EQ(chunks.size(), 3);  // Ceiling(8 / 3) = 3 chunks

    // Verify shapes: first two chunks size 3, last chunk size 2
    EXPECT_EQ(chunks[0].shape()[0], 6);
    EXPECT_EQ(chunks[0].shape()[1], 3);

    EXPECT_EQ(chunks[1].shape()[0], 6);
    EXPECT_EQ(chunks[1].shape()[1], 3);

    EXPECT_EQ(chunks[2].shape()[0], 6);
    EXPECT_EQ(chunks[2].shape()[1], 2);  // Last chunk is smaller
}

// Test 3D tensor split along different dimensions
TEST_P(SplitOperationTest, Split3DTensorMultipleDims) {
    auto tensor = zeros(shape_3d, DType::Float32, device);

    // Split along dimension 0
    auto chunks_dim0 = split(tensor, 2, 0);
    ASSERT_EQ(chunks_dim0.size(), 2);  // 4 / 2 = 2 chunks
    EXPECT_EQ(chunks_dim0[0].shape()[0], 2);
    EXPECT_EQ(chunks_dim0[0].shape()[1], 6);
    EXPECT_EQ(chunks_dim0[0].shape()[2], 8);

    // Split along dimension 1
    auto chunks_dim1 = split(tensor, 2, 1);
    ASSERT_EQ(chunks_dim1.size(), 3);  // 6 / 2 = 3 chunks
    EXPECT_EQ(chunks_dim1[0].shape()[0], 4);
    EXPECT_EQ(chunks_dim1[0].shape()[1], 2);
    EXPECT_EQ(chunks_dim1[0].shape()[2], 8);

    // Split along dimension 2
    auto chunks_dim2 = split(tensor, 5, 2);
    ASSERT_EQ(chunks_dim2.size(), 2);  // Ceiling(8 / 5) = 2 chunks
    EXPECT_EQ(chunks_dim2[0].shape()[0], 4);
    EXPECT_EQ(chunks_dim2[0].shape()[1], 6);
    EXPECT_EQ(chunks_dim2[0].shape()[2], 5);  // First chunk size 5

    EXPECT_EQ(chunks_dim2[1].shape()[0], 4);
    EXPECT_EQ(chunks_dim2[1].shape()[1], 6);
    EXPECT_EQ(chunks_dim2[1].shape()[2], 3);  // Last chunk size 3
}

// Test negative dimension indexing
TEST_P(SplitOperationTest, SplitNegativeDimension) {
    auto tensor = ones({4, 6, 8}, DType::Float32, device);

    // Split along dimension -1 (same as dimension 2)
    auto chunks = split(tensor, 4, -1);

    ASSERT_EQ(chunks.size(), 2);  // 8 / 4 = 2 chunks
    EXPECT_EQ(chunks[0].shape()[2], 4);
    EXPECT_EQ(chunks[1].shape()[2], 4);
}

// Test that split creates views (zero-copy)
TEST_P(SplitOperationTest, SplitCreatesViews) {
    auto tensor = arange(0.0f, 10.0f, 1.0f, DType::Float32, device);

    // Split into chunks
    auto chunks = split(tensor, 3, 0);

    // Modify first chunk in-place and verify it affects original tensor.
    // fill_() is device-aware and mutates shared storage, so this exercises
    // the view (zero-copy) property on every backend.
    auto first_chunk = chunks[0];
    first_chunk.fill_(999.0f);

    // Check that original tensor is modified (because slice creates views)
    auto orig_cpu = tensor.cpu();
    float* orig_data = orig_cpu.data<float>();
    EXPECT_FLOAT_EQ(orig_data[0], 999.0f);
    EXPECT_FLOAT_EQ(orig_data[1], 999.0f);
    EXPECT_FLOAT_EQ(orig_data[2], 999.0f);
}

// Test split with size equal to dimension size
TEST_P(SplitOperationTest, SplitSizeEqualsDimSize) {
    auto tensor = ones({6, 8}, DType::Float32, device);

    // Split with split_size == dim_size (should return single chunk)
    auto chunks = split(tensor, 6, 0);

    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0].shape()[0], 6);
    EXPECT_EQ(chunks[0].shape()[1], 8);
}

// Test split with size larger than dimension size
TEST_P(SplitOperationTest, SplitSizeLargerThanDimSize) {
    auto tensor = ones({4, 8}, DType::Float32, device);

    // Split with split_size > dim_size (should return single chunk)
    auto chunks = split(tensor, 10, 0);

    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0].shape()[0], 4);
    EXPECT_EQ(chunks[0].shape()[1], 8);
}

// Test split with split_size = 1
TEST_P(SplitOperationTest, SplitSizeOne) {
    auto tensor = arange(0.0f, 5.0f, 1.0f, DType::Float32, device);

    // Split into individual elements
    auto chunks = split(tensor, 1, 0);

    ASSERT_EQ(chunks.size(), 5);
    for (size_t i = 0; i < chunks.size(); ++i) {
        EXPECT_EQ(chunks[i].shape()[0], 1);
        EXPECT_EQ(chunks[i].numel(), 1);

        auto chunk_cpu = chunks[i].cpu();
        float value = chunk_cpu.data<float>()[0];
        EXPECT_FLOAT_EQ(value, static_cast<float>(i));
    }
}

// Test error handling: invalid split size
TEST_P(SplitOperationTest, ErrorInvalidSplitSize) {
    auto tensor = ones({6, 8}, DType::Float32, device);

    EXPECT_THROW(split(tensor, 0, 0), std::invalid_argument);
    EXPECT_THROW(split(tensor, -1, 0), std::invalid_argument);
}

// Test error handling: dimension out of range
TEST_P(SplitOperationTest, ErrorDimensionOutOfRange) {
    auto tensor = ones({6, 8}, DType::Float32, device);

    EXPECT_THROW(split(tensor, 2, 2), std::invalid_argument);
    EXPECT_THROW(split(tensor, 2, -3), std::invalid_argument);
}

// Test split use case: multi-head attention
TEST_P(SplitOperationTest, MultiHeadAttentionUseCase) {
    // Simulate splitting attention heads
    // Input: [batch_size, seq_len, hidden_dim] = [2, 10, 64]
    // Split hidden_dim into 8 heads of size 8 each

    auto tensor = ones({2, 10, 64}, DType::Float32, device);
    auto heads = split(tensor, 8, 2);  // Split along hidden dimension

    ASSERT_EQ(heads.size(), 8);  // 8 attention heads

    for (size_t i = 0; i < heads.size(); ++i) {
        auto head_shape = heads[i].shape();
        EXPECT_EQ(head_shape[0], 2);   // batch_size unchanged
        EXPECT_EQ(head_shape[1], 10);  // seq_len unchanged
        EXPECT_EQ(head_shape[2], 8);   // head_dim = 64 / 8
    }
}

// Test split use case: batch processing
TEST_P(SplitOperationTest, BatchProcessingUseCase) {
    // Split large batch into mini-batches
    // Input: [batch_size, features] = [100, 32]
    // Split into mini-batches of size 10

    auto tensor = ones({100, 32}, DType::Float32, device);
    auto mini_batches = split(tensor, 10, 0);

    ASSERT_EQ(mini_batches.size(), 10);  // 100 / 10 = 10 mini-batches

    for (size_t i = 0; i < mini_batches.size(); ++i) {
        auto batch_shape = mini_batches[i].shape();
        EXPECT_EQ(batch_shape[0], 10);  // mini_batch_size
        EXPECT_EQ(batch_shape[1], 32);  // features unchanged
    }
}

// Test split on contiguous vs non-contiguous tensors
TEST_P(SplitOperationTest, SplitNonContiguousTensor) {
    // Create a non-contiguous tensor via transpose
    auto tensor = ones({4, 8}, DType::Float32, device);
    auto transposed = tensor.transpose(0, 1);  // Non-contiguous

    EXPECT_FALSE(transposed.is_contiguous());

    // Split should still work on non-contiguous tensors
    auto chunks = split(transposed, 2, 0);

    ASSERT_EQ(chunks.size(), 4);  // 8 / 2 = 4 chunks
    for (size_t i = 0; i < chunks.size(); ++i) {
        EXPECT_EQ(chunks[i].shape()[0], 2);
        EXPECT_EQ(chunks[i].shape()[1], 4);
    }
}

INSTANTIATE_BACKEND_TESTS(SplitOperationTest);
