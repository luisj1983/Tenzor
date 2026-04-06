/**
 * @file test_pooling_multidtype.cpp
 * @brief Multi-dtype tests for pooling layers (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d)
 *
 * Tests pooling operations with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct output shapes and values
 * - Proper max value selection for MaxPool2d
 * - Accurate average computation for AvgPool2d
 * - Flexible output sizing for AdaptiveAvgPool2d
 * - Gradient flow through pooling layers
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Pooling Multi-Backend Multi-DType Test Fixture
// ============================================================================

class PoolingMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// MaxPool2d Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, MaxPool2dForwardBasic) {
    auto pool = nn::MaxPool2d(2, 2, 0);

    Variable input = createInput({2, 3, 32, 32}, true);
    auto output = pool.forward(input);

    // Check shape
    expectShape(output.tensor(), {2, 3, 16, 16});
    expectDType(output.tensor());
}

TEST_P(PoolingMultiDTypeTest, MaxPool2dMaxValueSelection) {
    auto pool = nn::MaxPool2d(2, 2, 0);

    // Create known input on CPU first
    auto input_tensor = tenzor::zeros({1, 1, 4, 4}, DType::Float32, Device::cpu());
    float* data = input_tensor.data<float>();

    data[0] = 1.0f;  data[1] = 2.0f;  data[2] = 3.0f;  data[3] = 4.0f;
    data[4] = 5.0f;  data[5] = 6.0f;  data[6] = 7.0f;  data[7] = 8.0f;
    data[8] = 9.0f;  data[9] = 10.0f; data[10] = 11.0f; data[11] = 12.0f;
    data[12] = 13.0f; data[13] = 14.0f; data[14] = 15.0f; data[15] = 16.0f;

    // Convert to test dtype and device
    if (dtype() != DType::Float32) {
        input_tensor = input_tensor.to(dtype());
    }
    if (device() != Device::cpu()) {
        input_tensor = input_tensor.to(device());
    }

    Variable input(input_tensor, true);
    auto output = pool.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* out_data = output_cpu.data<float>();

    EXPECT_NEAR(out_data[0], 6.0f, atol());
    EXPECT_NEAR(out_data[1], 8.0f, atol());
    EXPECT_NEAR(out_data[2], 14.0f, atol());
    EXPECT_NEAR(out_data[3], 16.0f, atol());
}

TEST_P(PoolingMultiDTypeTest, MaxPool2dGradientFlow) {
    auto pool = nn::MaxPool2d(2);

    Variable input = createInput({2, 3, 16, 16}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// AvgPool2d Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, AvgPool2dForwardBasic) {
    auto pool = nn::AvgPool2d(2, 2, 0);

    Variable input = createInput({2, 3, 32, 32}, true);
    auto output = pool.forward(input);

    expectShape(output.tensor(), {2, 3, 16, 16});
    expectDType(output.tensor());
}

TEST_P(PoolingMultiDTypeTest, AvgPool2dAverageComputation) {
    auto pool = nn::AvgPool2d(2, 2, 0);

    // Create known input on CPU first
    auto input_tensor = tenzor::zeros({1, 1, 4, 4}, DType::Float32, Device::cpu());
    float* data = input_tensor.data<float>();

    for (int i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    // Convert to test dtype and device
    if (dtype() != DType::Float32) {
        input_tensor = input_tensor.to(dtype());
    }
    if (device() != Device::cpu()) {
        input_tensor = input_tensor.to(device());
    }

    Variable input(input_tensor, true);
    auto output = pool.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* out_data = output_cpu.data<float>();

    EXPECT_NEAR(out_data[0], 3.5f, atol());   // avg of [1,2,5,6]
    EXPECT_NEAR(out_data[1], 5.5f, atol());   // avg of [3,4,7,8]
    EXPECT_NEAR(out_data[2], 11.5f, atol());  // avg of [9,10,13,14]
    EXPECT_NEAR(out_data[3], 13.5f, atol());  // avg of [11,12,15,16]
}

TEST_P(PoolingMultiDTypeTest, AvgPool2dGradientFlow) {
    auto pool = nn::AvgPool2d(2);

    Variable input = createInput({2, 3, 16, 16}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// AdaptiveAvgPool2d Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, AdaptiveAvgPool2dForwardBasic) {
    auto pool = nn::AdaptiveAvgPool2d(7, 7);

    Variable input = createInput({2, 3, 32, 32}, true);
    auto output = pool.forward(input);

    expectShape(output.tensor(), {2, 3, 7, 7});
    expectDType(output.tensor());
}

TEST_P(PoolingMultiDTypeTest, AdaptiveAvgPool2dGlobalPooling) {
    auto pool = nn::AdaptiveAvgPool2d(1, 1);

    Variable input = createInput({2, 64, 14, 14}, true);
    auto output = pool.forward(input);

    expectShape(output.tensor(), {2, 64, 1, 1});
    expectDType(output.tensor());
}

TEST_P(PoolingMultiDTypeTest, AdaptiveAvgPool2dGradientFlow) {
    auto pool = nn::AdaptiveAvgPool2d(7);

    Variable input = createInput({2, 3, 32, 32}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// Mixed Precision Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, SequentialPoolingPreservesType) {
    auto pool1 = nn::MaxPool2d(2);
    auto pool2 = nn::MaxPool2d(2);

    Variable input = createInput({2, 3, 64, 64}, true);
    auto x = pool1.forward(input);
    auto output = pool2.forward(x);

    expectShape(output.tensor(), {2, 3, 16, 16});
    expectDType(output.tensor());
}

// ============================================================================
// AdaptiveMaxPool2d Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, AdaptiveMaxPool2dOutputShape) {
    auto pool = nn::AdaptiveMaxPool2d(4, 4);

    Variable input = createInput({2, 3, 16, 16}, true);
    auto output = pool.forward(input);

    expectShape(output.tensor(), {2, 3, 4, 4});
    expectDType(output.tensor());
}

TEST_P(PoolingMultiDTypeTest, AdaptiveMaxPool2dSquareOutput) {
    auto pool = nn::AdaptiveMaxPool2d(1);  // global max pool

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pool.forward(input);

    expectShape(output.tensor(), {2, 3, 1, 1});
    expectDType(output.tensor());
}

TEST_P(PoolingMultiDTypeTest, AdaptiveMaxPool2dNonSquareInput) {
    auto pool = nn::AdaptiveMaxPool2d(3, 5);

    Variable input = createInput({1, 4, 12, 20}, true);
    auto output = pool.forward(input);

    expectShape(output.tensor(), {1, 4, 3, 5});
    expectDType(output.tensor());
}

TEST_P(PoolingMultiDTypeTest, AdaptiveMaxPool2dGradientFlow) {
    auto pool = nn::AdaptiveMaxPool2d(2, 2);

    Variable input = createInput({1, 2, 8, 8}, true);
    auto output = pool.forward(input);

    auto loss = tenzor::sum(output.tensor());
    auto loss_var = Variable(loss, true);
    loss_var.backward();

    ASSERT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {1, 2, 8, 8});
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(PoolingMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 11
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 11 tests × 3 dtypes × 3 backends = 99 test scenarios
 *
 * Coverage:
 * - MaxPool2d: forward, value selection, gradient flow
 * - AvgPool2d: forward, average computation, gradient flow
 * - AdaptiveAvgPool2d: forward, global pooling, gradient flow
 * - Mixed precision: sequential pooling type preservation
 */
