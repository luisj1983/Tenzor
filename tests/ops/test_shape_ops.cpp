#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;

/**
 * @file test_shape_ops.cpp
 * @brief Backend-agnostic tests for shape manipulation operations
 *
 * Tests: squeeze, unsqueeze
 * All backends: CPU, CUDA, Vulkan, OneAPI, ROCm
 */

class ShapeOpsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const shape_env =
    ::testing::AddGlobalTestEnvironment(new ShapeOpsTestEnvironment);

class ShapeOpsTest : public tenzor::testing::BackendTest {};

// ============================================================================
// Squeeze Tests
// ============================================================================

TEST_P(ShapeOpsTest, SqueezeAll) {
    // Remove all dimensions of size 1
    auto input = ones({1, 3, 1, 5, 1}, DType::Float32, device);
    auto output = squeeze(input);

    // Should become [3, 5]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(ShapeOpsTest, SqueezeSpecificDim) {
    // Remove specific dimension
    auto input = ones({1, 3, 1, 5}, DType::Float32, device);
    auto output = squeeze(input, 2);  // Remove dim 2 (size 1)

    // Should become [1, 3, 5]
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 5);
}

TEST_P(ShapeOpsTest, SqueezeNonSingletonDim) {
    // Squeezing a non-size-1 dimension should not change shape
    auto input = ones({2, 3, 4}, DType::Float32, device);
    auto output = squeeze(input, 1);  // dim 1 has size 3

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
}

TEST_P(ShapeOpsTest, SqueezeNoSingletonDims) {
    // Tensor with no size-1 dimensions
    auto input = ones({2, 3, 4, 5}, DType::Float32, device);
    auto output = squeeze(input);

    // Should remain unchanged
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
}

TEST_P(ShapeOpsTest, SqueezeScalar) {
    // Edge case: all dimensions are 1 -> becomes scalar
    auto input = ones({1, 1, 1}, DType::Float32, device);
    auto output = squeeze(input);

    EXPECT_EQ(output.shape().size(), 0);  // Scalar
    EXPECT_EQ(output.numel(), 1);
}

// ============================================================================
// Unsqueeze Tests
// ============================================================================

TEST_P(ShapeOpsTest, UnsqueezeBeginning) {
    // Add dimension at the start
    auto input = ones({3, 4, 5}, DType::Float32, device);
    auto output = unsqueeze(input, 0);

    // Should become [1, 3, 4, 5]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(ShapeOpsTest, UnsqueezeMiddle) {
    // Add dimension in the middle
    auto input = ones({2, 3, 4}, DType::Float32, device);
    auto output = unsqueeze(input, 2);

    // Should become [2, 3, 1, 4]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 4);
}

TEST_P(ShapeOpsTest, UnsqueezeEnd) {
    // Add dimension at the end
    auto input = ones({2, 3}, DType::Float32, device);
    auto output = unsqueeze(input, 2);

    // Should become [2, 3, 1]
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 1);
}

TEST_P(ShapeOpsTest, UnsqueezeNegativeDim) {
    // Negative dimension index
    auto input = ones({2, 3, 4}, DType::Float32, device);
    auto output = unsqueeze(input, -1);  // Add at end

    // Should become [2, 3, 4, 1]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST_P(ShapeOpsTest, UnsqueezeScalar) {
    // Unsqueeze a scalar tensor
    auto input = ones({}, DType::Float32, device);  // Scalar
    auto output = unsqueeze(input, 0);

    EXPECT_EQ(output.shape().size(), 1);
    EXPECT_EQ(output.shape()[0], 1);
}

// ============================================================================
// Squeeze/Unsqueeze Round-trip Tests
// ============================================================================

TEST_P(ShapeOpsTest, SqueezeUnsqueezeRoundTrip) {
    // Original -> Unsqueeze -> Squeeze -> Should match original
    auto original = ones({2, 3, 4}, DType::Float32, device);
    auto unsqueezed = unsqueeze(original, 1);  // [2, 1, 3, 4]
    auto squeezed = squeeze(unsqueezed, 1);    // [2, 3, 4]

    EXPECT_EQ(squeezed.shape().size(), 3);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);
    EXPECT_EQ(squeezed.shape()[2], 4);
}

TEST_P(ShapeOpsTest, MultipleUnsqueezes) {
    // Add multiple dimensions
    auto input = ones({2, 3}, DType::Float32, device);
    auto out1 = unsqueeze(input, 0);      // [1, 2, 3]
    auto out2 = unsqueeze(out1, 3);        // [1, 2, 3, 1]
    auto out3 = unsqueeze(out2, 2);        // [1, 2, 1, 3, 1]

    EXPECT_EQ(out3.shape().size(), 5);
    EXPECT_EQ(out3.shape()[0], 1);
    EXPECT_EQ(out3.shape()[1], 2);
    EXPECT_EQ(out3.shape()[2], 1);
    EXPECT_EQ(out3.shape()[3], 3);
    EXPECT_EQ(out3.shape()[4], 1);
}

// ============================================================================
// Data Preservation Tests
// ============================================================================

TEST_P(ShapeOpsTest, SqueezePreservesData) {
    // Ensure data is unchanged, only shape changes
    auto input_cpu = arange(0.0f, 12.0f, 1.0f, DType::Float32, Device::cpu()).reshape({1, 3, 4, 1});
    auto input = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);

    auto squeezed = squeeze(input);  // [3, 4]

    auto squeezed_cpu = squeezed.to(Device::cpu());
    const float* data = squeezed_cpu.data<float>();

    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

TEST_P(ShapeOpsTest, UnsqueezePreservesData) {
    auto input_cpu = arange(0.0f, 6.0f, 1.0f, DType::Float32, Device::cpu()).reshape({2, 3});
    auto input = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);

    auto unsqueezed = unsqueeze(input, 1);  // [2, 1, 3]

    auto unsqueezed_cpu = unsqueezed.to(Device::cpu());
    const float* data = unsqueezed_cpu.data<float>();

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_P(ShapeOpsTest, BatchDimensionHandling) {
    // Common pattern: add batch dimension
    auto input = ones({3, 224, 224}, DType::Float32, device);  // Single image
    auto batched = unsqueeze(input, 0);  // [1, 3, 224, 224]

    EXPECT_EQ(batched.shape()[0], 1);
    EXPECT_EQ(batched.shape()[1], 3);
    EXPECT_EQ(batched.shape()[2], 224);
    EXPECT_EQ(batched.shape()[3], 224);

    // Remove batch dimension
    auto unbatched = squeeze(batched, 0);
    EXPECT_EQ(unbatched.shape().size(), 3);
    EXPECT_EQ(unbatched.shape()[0], 3);
    EXPECT_EQ(unbatched.shape()[1], 224);
    EXPECT_EQ(unbatched.shape()[2], 224);
}

INSTANTIATE_BACKEND_TESTS(ShapeOpsTest);
