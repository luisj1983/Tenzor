#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @file test_flatten.cpp
 * @brief Comprehensive backend-agnostic tests for Flatten layer
 *
 * Tests cover:
 * - Forward pass shape transformations
 * - Backward pass gradients
 * - Edge cases (negative dimensions, boundary conditions)
 * - All backends (CPU, CUDA, Vulkan, OneAPI, ROCm)
 */

// ============================================================================
// Test Environment Setup
// ============================================================================

class FlattenTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const flatten_env =
    ::testing::AddGlobalTestEnvironment(new FlattenTestEnvironment);

// ============================================================================
// Helper Functions
// ============================================================================

bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-5f, float atol = 1e-7f) {
    if (a.numel() != b.numel()) return false;

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());

    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    size_t numel = a_cpu.numel();

    for (size_t i = 0; i < numel; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Backend-Agnostic Parameterized Tests
// ============================================================================

class FlattenTest : public tenzor::testing::BackendTest {};

// Test basic flattening from default start_dim=1
TEST_P(FlattenTest, BasicFlattenFromDim1) {
    auto flatten = Flatten(1);

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Expected shape: [2, 3*4*5] = [2, 60]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 60);
    EXPECT_EQ(output.tensor().device().type, device.type);
}

// Test flattening from dimension 0 (full flatten)
TEST_P(FlattenTest, FlattenFromDim0) {
    auto flatten = Flatten(0);

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Expected shape: [2*3*4*5] = [120]
    EXPECT_EQ(output.shape().size(), 1);
    EXPECT_EQ(output.shape()[0], 120);
}

// Test partial flattening with custom range
TEST_P(FlattenTest, PartialFlattenCustomRange) {
    auto flatten = Flatten(1, 2);  // Flatten dims 1-2, keep 0 and 3

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Expected shape: [2, 3*4, 5] = [2, 12, 5]
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 12);  // 3*4
    EXPECT_EQ(output.shape()[2], 5);
}

// Test negative dimension indices
TEST_P(FlattenTest, NegativeDimensions) {
    auto flatten = Flatten(-3, -1);  // Last 3 dimensions

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Flatten dims -3,-2,-1 (which are 1,2,3)
    // Expected shape: [2, 3*4*5] = [2, 60]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 60);
}

// Test gradient flow through flatten (backward pass)
TEST_P(FlattenTest, GradientFlowBackward) {
    auto flatten = Flatten(1);

    // Input: [2, 3, 4]
    auto input = Variable(randn({2, 3, 4}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Backward pass
    auto grad_output = ones(output.shape(), DType::Float32, device);
    output.backward(grad_output);

    // Check gradient exists and has correct shape
    ASSERT_TRUE(input.grad().has_value());
    auto grad_input = input.grad().value();

    // Gradient should have same shape as input
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], 3);
    EXPECT_EQ(grad_input.shape()[2], 4);

    // Gradient values should match (flatten backward just reshapes)
    EXPECT_TRUE(tensors_close(grad_input, grad_output.reshape({2, 3, 4})));
}

// Test reshape correctness (data ordering)
TEST_P(FlattenTest, DataOrdering) {
    auto flatten = Flatten(0);

    // Create input with known values
    auto input_cpu = arange(0.0f, 24.0f, 1.0f, DType::Float32, Device::cpu()).reshape({2, 3, 4});
    auto input_data = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);
    auto input = Variable(input_data, true);

    auto output = flatten.forward(input);

    // Check that flattened data matches original ordering
    auto output_cpu = output.tensor().to(Device::cpu());
    const float* out_data = output_cpu.data<float>();

    for (int i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], static_cast<float>(i));
    }
}

// Test single element tensor
TEST_P(FlattenTest, SingleElement) {
    auto flatten = Flatten(0);

    auto input = Variable(ones({1, 1, 1, 1}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    EXPECT_EQ(output.shape().size(), 1);
    EXPECT_EQ(output.shape()[0], 1);
}

// Test 2D tensor (already flat in some sense)
TEST_P(FlattenTest, TwoDimensionalInput) {
    auto flatten = Flatten(1);

    auto input = Variable(randn({5, 10}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Should be unchanged
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 10);
}

// Test large tensor
TEST_P(FlattenTest, LargeTensor) {
    auto flatten = Flatten(1);

    auto input = Variable(randn({32, 64, 28, 28}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Expected: [32, 64*28*28] = [32, 50176]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 50176);
}

// Test flatten in a mini neural network (integration test)
TEST_P(FlattenTest, IntegrationWithLinear) {
    // Simulate CNN -> Flatten -> FC pattern
    auto conv_output = Variable(randn({4, 32, 7, 7}, DType::Float32, device), true);

    auto flatten = Flatten(1);
    auto flattened = flatten.forward(conv_output);

    // Should be [4, 32*7*7] = [4, 1568]
    EXPECT_EQ(flattened.shape()[0], 4);
    EXPECT_EQ(flattened.shape()[1], 1568);

    // Gradient flow test
    auto grad = ones(flattened.shape(), DType::Float32, device);
    flattened.backward(grad);

    ASSERT_TRUE(conv_output.grad().has_value());
    auto grad_conv = conv_output.grad().value();
    EXPECT_EQ(grad_conv.shape()[0], 4);
    EXPECT_EQ(grad_conv.shape()[1], 32);
    EXPECT_EQ(grad_conv.shape()[2], 7);
    EXPECT_EQ(grad_conv.shape()[3], 7);
}

// Test end_dim equal to start_dim
TEST_P(FlattenTest, SingleDimRange) {
    auto flatten = Flatten(2, 2);  // Only flatten dim 2 (no-op in some sense)

    auto input = Variable(randn({2, 3, 4, 5}, DType::Float32, device), true);
    auto output = flatten.forward(input);

    // Shape should be [2, 3, 4, 5] still
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(FlattenErrorTest, InvalidDimensionRange) {
    auto flatten = Flatten(2, 1);  // start > end

    auto input = Variable(randn({2, 3, 4, 5}), true);

    EXPECT_THROW(flatten.forward(input), std::invalid_argument);
}

TEST(FlattenErrorTest, OutOfRangeDimension) {
    auto flatten = Flatten(0, 10);  // end_dim out of range

    auto input = Variable(randn({2, 3, 4}), true);

    EXPECT_THROW(flatten.forward(input), std::invalid_argument);
}

TEST(FlattenErrorTest, NegativeOutOfRange) {
    auto flatten = Flatten(-10, -1);  // start_dim out of range

    auto input = Variable(randn({2, 3, 4}), true);

    EXPECT_THROW(flatten.forward(input), std::invalid_argument);
}

// ============================================================================
// Instantiate Tests for All Backends
// ============================================================================

INSTANTIATE_BACKEND_TESTS(FlattenTest);
