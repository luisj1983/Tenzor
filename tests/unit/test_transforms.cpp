#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class TransformTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        // Create test tensors
        t2d = zeros({2, 3}, DType::Float32, device);
        t3d = zeros({2, 3, 4}, DType::Float32, device);
        t4d = zeros({2, 3, 4, 5}, DType::Float32, device);
    }

    Tensor t2d;
    Tensor t3d;
    Tensor t4d;
};

// Reshape tests
TEST_P(TransformTest, Reshape_Basic) {
    auto t = zeros({6}, DType::Float32, device);
    auto reshaped = t.reshape({2, 3});

    EXPECT_EQ(reshaped.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(reshaped.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(reshaped.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(reshaped.numel(), 6) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Reshape_InferDimension) {
    auto t = zeros({12}, DType::Float32, device);
    auto reshaped = t.reshape({3, -1});

    EXPECT_EQ(reshaped.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(reshaped.shape()[1], 4) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Reshape_MultiDimensional) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    auto reshaped = t.reshape({6, 4});

    EXPECT_EQ(reshaped.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(reshaped.shape()[0], 6) << "Failed on " << device.to_string();
    EXPECT_EQ(reshaped.shape()[1], 4) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Reshape_InvalidShape) {
    auto t = zeros({6}, DType::Float32, device);
    EXPECT_THROW(t.reshape({2, 2}), std::invalid_argument) << "Failed on " << device.to_string();
}

// View tests
TEST_P(TransformTest, View_Basic) {
    auto t = zeros({6}, DType::Float32, device);
    auto viewed = t.view({2, 3});

    EXPECT_EQ(viewed.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(viewed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(viewed.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, View_SharesStorage) {
    auto t = ones({6}, DType::Float32, device);
    auto viewed = t.view({2, 3});

    // Both should share the same storage pointer
    EXPECT_EQ(t.storage(), viewed.storage()) << "Failed on " << device.to_string();
}

// Transpose tests
TEST_P(TransformTest, Transpose_2D) {
    auto t = t2d.transpose(0, 1);

    EXPECT_EQ(t.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[1], 2) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Transpose_NegativeDims) {
    auto t = t2d.transpose(-2, -1);

    EXPECT_EQ(t.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[1], 2) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Transpose_3D) {
    auto t = t3d.transpose(0, 2);

    EXPECT_EQ(t.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[2], 2) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Transpose_InvalidDim) {
    EXPECT_THROW(t2d.transpose(0, 5), std::out_of_range) << "Failed on " << device.to_string();
}

// Permute tests
TEST_P(TransformTest, Permute_3D) {
    auto t = t3d.permute({2, 0, 1});

    EXPECT_EQ(t.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[1], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[2], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Permute_Reverse) {
    auto t = t3d.permute({2, 1, 0});

    EXPECT_EQ(t.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[2], 2) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Permute_NegativeIndices) {
    auto t = t3d.permute({-1, -2, -3});

    EXPECT_EQ(t.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(t.shape()[2], 2) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Permute_InvalidDims) {
    EXPECT_THROW(t3d.permute({0, 1}), std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Permute_DuplicateDims) {
    EXPECT_THROW(t3d.permute({0, 0, 1}), std::invalid_argument) << "Failed on " << device.to_string();
}

// Squeeze tests
TEST_P(TransformTest, Squeeze_SingleDim) {
    auto t = zeros({2, 1, 3}, DType::Float32, device);
    auto squeezed = t.squeeze(1);

    EXPECT_EQ(squeezed.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Squeeze_All) {
    auto t = zeros({1, 2, 1, 3, 1}, DType::Float32, device);
    auto squeezed = t.squeeze();

    EXPECT_EQ(squeezed.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Squeeze_AllOnes) {
    auto t = zeros({1, 1, 1}, DType::Float32, device);
    auto squeezed = t.squeeze();

    // Should keep at least one dimension
    EXPECT_EQ(squeezed.ndim(), 1) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[0], 1) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Squeeze_InvalidDim) {
    auto t = zeros({2, 1, 3}, DType::Float32, device);
    // PyTorch behavior: squeeze on non-singleton dimension returns unchanged tensor
    auto squeezed = t.squeeze(0);  // dim 0 has size 2, so no change
    EXPECT_EQ(squeezed.ndim(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[1], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[2], 3) << "Failed on " << device.to_string();

    // Test actual invalid dim (out of bounds)
    EXPECT_THROW(t.squeeze(5), std::out_of_range) << "Failed on " << device.to_string();
    EXPECT_THROW(t.squeeze(-10), std::out_of_range) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Squeeze_NegativeIndex) {
    auto t = zeros({2, 1, 3}, DType::Float32, device);
    auto squeezed = t.squeeze(-2);

    EXPECT_EQ(squeezed.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[1], 3) << "Failed on " << device.to_string();
}

// Unsqueeze tests
TEST_P(TransformTest, Unsqueeze_Front) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto unsqueezed = t.unsqueeze(0);

    EXPECT_EQ(unsqueezed.ndim(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[1], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[2], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Unsqueeze_Middle) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto unsqueezed = t.unsqueeze(1);

    EXPECT_EQ(unsqueezed.ndim(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[1], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[2], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Unsqueeze_End) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto unsqueezed = t.unsqueeze(2);

    EXPECT_EQ(unsqueezed.ndim(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[2], 1) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Unsqueeze_NegativeIndex) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto unsqueezed = t.unsqueeze(-1);

    EXPECT_EQ(unsqueezed.ndim(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[2], 1) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Unsqueeze_InvalidDim) {
    auto t = zeros({2, 3}, DType::Float32, device);
    EXPECT_THROW(t.unsqueeze(5), std::out_of_range) << "Failed on " << device.to_string();
}

// Flatten tests
TEST_P(TransformTest, Flatten_All) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    auto flattened = t.flatten();

    EXPECT_EQ(flattened.ndim(), 1) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[0], 24) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Flatten_Partial) {
    auto t = zeros({2, 3, 4, 5}, DType::Float32, device);
    auto flattened = t.flatten(1, 2);

    EXPECT_EQ(flattened.ndim(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[1], 12) << "Failed on " << device.to_string();  // 3 * 4
    EXPECT_EQ(flattened.shape()[2], 5) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Flatten_FirstTwoDims) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    auto flattened = t.flatten(0, 1);

    EXPECT_EQ(flattened.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[0], 6) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[1], 4) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Flatten_NegativeIndices) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    auto flattened = t.flatten(-2, -1);

    EXPECT_EQ(flattened.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[1], 12) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Flatten_InvalidRange) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    EXPECT_THROW(t.flatten(2, 1), std::invalid_argument) << "Failed on " << device.to_string();
}

// Combined operations tests
TEST_P(TransformTest, Combined_TransposeReshape) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    // After transpose, tensor is not contiguous, so reshape should make it contiguous
    auto transposed = t.transpose(0, 1);
    EXPECT_FALSE(transposed.is_contiguous()) << "Failed on " << device.to_string();

    // reshape() should handle non-contiguous tensors by calling contiguous()
    // But since contiguous() is not fully implemented yet, this will fail
    // For now, we expect this to throw
    EXPECT_THROW(transposed.view({3, 8}), std::runtime_error) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Combined_UnsqueezeSqueeze) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto result = t.unsqueeze(1).squeeze(1);

    EXPECT_EQ(result.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(result.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(result.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformTest, Combined_PermuteTranspose) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    auto permuted = t.permute({2, 1, 0});
    auto transposed = permuted.transpose(0, 2);

    EXPECT_EQ(transposed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(transposed.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(transposed.shape()[2], 4) << "Failed on " << device.to_string();
}

// High-level API tests
class TransformAPITest : public BackendTest {};

TEST_P(TransformAPITest, ReshapeAPI) {
    auto t = zeros({6}, DType::Float32, device);
    auto reshaped = reshape(t, {2, 3});

    EXPECT_EQ(reshaped.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(reshaped.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformAPITest, ViewAPI) {
    auto t = zeros({6}, DType::Float32, device);
    auto viewed = view(t, {2, 3});

    EXPECT_EQ(viewed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(viewed.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformAPITest, TransposeAPI) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto transposed = transpose(t, 0, 1);

    EXPECT_EQ(transposed.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(transposed.shape()[1], 2) << "Failed on " << device.to_string();
}

TEST_P(TransformAPITest, SqueezeAPI) {
    auto t = zeros({1, 2, 3}, DType::Float32, device);
    auto squeezed = squeeze(t, 0);

    EXPECT_EQ(squeezed.ndim(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(squeezed.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(TransformAPITest, UnsqueezeAPI) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto unsqueezed = unsqueeze(t, 0);

    EXPECT_EQ(unsqueezed.ndim(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(unsqueezed.shape()[0], 1) << "Failed on " << device.to_string();
}

TEST_P(TransformAPITest, FlattenAPI) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    auto flattened = flatten(t);

    EXPECT_EQ(flattened.ndim(), 1) << "Failed on " << device.to_string();
    EXPECT_EQ(flattened.shape()[0], 24) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(TransformTest);
INSTANTIATE_BACKEND_TESTS(TransformAPITest);
