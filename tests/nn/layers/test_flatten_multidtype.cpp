/**
 * @file test_flatten_multidtype.cpp
 * @brief Multi-dtype tests for Flatten layer
 *
 * Tests flatten operations with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct shape transformations
 * - Data ordering preservation
 * - Gradient flow through flatten operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Flatten Multi-Backend Multi-DType Test Fixture
// ============================================================================

class FlattenMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Basic Flatten Tests
// ============================================================================

TEST_P(FlattenMultiDTypeTest, BasicFlattenFromDim1) {
    auto flatten = Flatten(1);

    Variable input = createInput({2, 3, 4, 5}, true);
    auto output = flatten.forward(input);

    // Expected shape: [2, 3*4*5] = [2, 60]
    expectShape(output.tensor(), {2, 60});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, FlattenFromDim0) {
    auto flatten = Flatten(0);

    Variable input = createInput({2, 3, 4, 5}, true);
    auto output = flatten.forward(input);

    // Expected shape: [2*3*4*5] = [120]
    expectShape(output.tensor(), {120});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, PartialFlattenCustomRange) {
    auto flatten = Flatten(1, 2);  // Flatten dims 1-2, keep 0 and 3

    Variable input = createInput({2, 3, 4, 5}, true);
    auto output = flatten.forward(input);

    // Expected shape: [2, 3*4, 5] = [2, 12, 5]
    expectShape(output.tensor(), {2, 12, 5});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, NegativeDimensions) {
    auto flatten = Flatten(-3, -1);  // Last 3 dimensions

    Variable input = createInput({2, 3, 4, 5}, true);
    auto output = flatten.forward(input);

    // Flatten dims -3,-2,-1 (which are 1,2,3)
    // Expected shape: [2, 3*4*5] = [2, 60]
    expectShape(output.tensor(), {2, 60});
    expectDType(output.tensor());
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_P(FlattenMultiDTypeTest, GradientFlowBackward) {
    auto flatten = Flatten(1);

    Variable input = createInput({2, 3, 4}, true);
    auto output = flatten.forward(input);

    // Backward pass
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);

    // Check gradient exists, is non-zero, and has correct shape
    EXPECT_GRAD_FLOWS(input);
    auto grad_input = input.grad().value();

    // Gradient should have same shape as input
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], 3);
    EXPECT_EQ(grad_input.shape()[2], 4);
    EXPECT_EQ(grad_input.dtype(), dtype());
}

// ============================================================================
// Data Ordering Tests
// ============================================================================

TEST_P(FlattenMultiDTypeTest, DataOrdering) {
    auto flatten = Flatten(0);

    // Create input with known values
    Tensor input_cpu = tenzor::arange(0.0f, 24.0f, 1.0f, DType::Float32, Device::cpu()).reshape({2, 3, 4});
    if (dtype() != DType::Float32) {
        input_cpu = input_cpu.to(dtype());
    }

    auto input_data = (device() == Device::cpu()) ? input_cpu : input_cpu.to(device());
    auto input = Variable(input_data, true);

    auto output = flatten.forward(input);

    // Check that flattened data matches original ordering
    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* out_data = output_cpu.data<float>();

    for (int i = 0; i < 24; ++i) {
        EXPECT_NEAR(out_data[i], static_cast<float>(i), atol());
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(FlattenMultiDTypeTest, SingleElement) {
    auto flatten = Flatten(0);

    auto input_tensor = createOnes({1, 1, 1, 1});
    auto input = Variable(input_tensor, true);
    auto output = flatten.forward(input);

    expectShape(output.tensor(), {1});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, TwoDimensionalInput) {
    auto flatten = Flatten(1);

    Variable input = createInput({5, 10}, true);
    auto output = flatten.forward(input);

    // Should be unchanged
    expectShape(output.tensor(), {5, 10});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, LargeTensor) {
    auto flatten = Flatten(1);

    Variable input = createInput({32, 64, 28, 28}, false);
    auto output = flatten.forward(input);

    // Expected: [32, 64*28*28] = [32, 50176]
    expectShape(output.tensor(), {32, 50176});
    expectDType(output.tensor());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_P(FlattenMultiDTypeTest, IntegrationWithLinear) {
    // Simulate CNN -> Flatten -> FC pattern
    Variable conv_output = createInput({4, 32, 7, 7}, true);

    auto flatten = Flatten(1);
    auto flattened = flatten.forward(conv_output);

    // Should be [4, 32*7*7] = [4, 1568]
    expectShape(flattened.tensor(), {4, 1568});
    expectDType(flattened.tensor());

    // Gradient flow test
    auto out_shape = flattened.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad = tenzor::ones(out_shape_vec, dtype(), device());

    flattened.backward(grad);

    EXPECT_GRAD_FLOWS(conv_output);
    auto grad_conv = conv_output.grad().value();
    EXPECT_EQ(grad_conv.shape()[0], 4);
    EXPECT_EQ(grad_conv.shape()[1], 32);
    EXPECT_EQ(grad_conv.shape()[2], 7);
    EXPECT_EQ(grad_conv.shape()[3], 7);
    EXPECT_EQ(grad_conv.dtype(), dtype());
}

TEST_P(FlattenMultiDTypeTest, SingleDimRange) {
    auto flatten = Flatten(2, 2);  // Only flatten dim 2 (no-op in some sense)

    Variable input = createInput({2, 3, 4, 5}, true);
    auto output = flatten.forward(input);

    // Shape should be [2, 3, 4, 5] still
    expectShape(output.tensor(), {2, 3, 4, 5});
    expectDType(output.tensor());
}

// ============================================================================
// Mixed Precision Tests
// ============================================================================

TEST_P(FlattenMultiDTypeTest, SequentialFlattenPreservesType) {
    auto flatten1 = Flatten(2, 3);
    auto flatten2 = Flatten(1, 2);

    Variable input = createInput({2, 3, 4, 5, 6}, true);
    auto x = flatten1.forward(input);  // [2, 3, 20, 6]
    auto output = flatten2.forward(x);  // [2, 60, 6]

    EXPECT_EQ(output.shape().size(), 3);
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, FlattenWithBatchNorm) {
    // BatchNorm -> Flatten pattern
    BatchNorm2d bn(16);
    convert_model(bn);
    bn.eval();

    Variable input = createInput({4, 16, 8, 8}, false);
    auto bn_out = bn.forward(input);

    auto flatten = Flatten(1);
    auto output = flatten.forward(bn_out);

    // Should be [4, 16*8*8] = [4, 1024]
    expectShape(output.tensor(), {4, 1024});
    expectDType(output.tensor());
}

// ============================================================================
// Different Tensor Shapes
// ============================================================================

TEST_P(FlattenMultiDTypeTest, HighDimensionalInput) {
    auto flatten = Flatten(1);

    Variable input = createInput({2, 2, 2, 2, 2, 2}, false);
    auto output = flatten.forward(input);

    // [2, 2*2*2*2*2] = [2, 32]
    expectShape(output.tensor(), {2, 32});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, VariousFlattenDims) {
    // Test different start_dim and end_dim combinations
    std::vector<std::tuple<int64_t, int64_t, std::vector<int64_t>>> test_cases = {
        {0, -1, {120}},           // Flatten all: [2,3,4,5] -> [120]
        {1, -1, {2, 60}},         // Keep batch: [2,3,4,5] -> [2, 60]
        {2, -1, {2, 3, 20}},      // Keep batch and channels: [2,3,4,5] -> [2,3,20]
        {1, 2, {2, 12, 5}},       // Flatten middle dims: [2,3,4,5] -> [2,12,5]
    };

    for (const auto& [start_dim, end_dim, expected_shape] : test_cases) {
        auto flatten = Flatten(start_dim, end_dim);
        Variable input = createInput({2, 3, 4, 5}, false);
        auto output = flatten.forward(input);

        expectShape(output.tensor(), expected_shape);
        expectDType(output.tensor());
    }
}

// ============================================================================
// PixelShuffle Tests
// ============================================================================

TEST_P(FlattenMultiDTypeTest, PixelShuffleOutputShape) {
    auto ps = PixelShuffle(2);

    // Input: (1, C*r^2, H, W) = (1, 8, 4, 4) -> Output: (1, 2, 8, 8)
    Variable input = createInput({1, 8, 4, 4}, true);
    auto output = ps.forward(input);

    expectShape(output.tensor(), {1, 2, 8, 8});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, PixelShuffleGradientFlow) {
    auto ps = PixelShuffle(2);

    Variable input = createInput({1, 8, 4, 4}, true);
    auto output = ps.forward(input);

    // Use the autograd-aware sum so backward reaches `input`.
    auto loss_var = tenzor::sum(output);
    loss_var.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {1, 8, 4, 4});
}

// ============================================================================
// PixelUnshuffle Tests
// ============================================================================

TEST_P(FlattenMultiDTypeTest, PixelUnshuffleOutputShape) {
    auto pus = PixelUnshuffle(2);

    // Input: (1, C, H*r, W*r) = (1, 2, 8, 8) -> Output: (1, 8, 4, 4)
    Variable input = createInput({1, 2, 8, 8}, true);
    auto output = pus.forward(input);

    expectShape(output.tensor(), {1, 8, 4, 4});
    expectDType(output.tensor());
}

TEST_P(FlattenMultiDTypeTest, PixelUnshuffleGradientFlow) {
    auto pus = PixelUnshuffle(2);

    Variable input = createInput({1, 2, 8, 8}, true);
    auto output = pus.forward(input);

    // Use the autograd-aware sum so backward reaches `input`.
    auto loss_var = tenzor::sum(output);
    loss_var.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {1, 2, 8, 8});
}

TEST_P(FlattenMultiDTypeTest, PixelShuffleUnshuffleRoundtrip) {
    auto ps = PixelShuffle(2);
    auto pus = PixelUnshuffle(2);

    // Start with (1, 8, 4, 4), shuffle to (1, 2, 8, 8), unshuffle back to (1, 8, 4, 4)
    Variable input = createInput({1, 8, 4, 4});
    auto shuffled = ps.forward(input);
    auto roundtrip = pus.forward(shuffled);

    expectShape(roundtrip.tensor(), {1, 8, 4, 4});

    // Verify values are preserved
    expectTensorNear(roundtrip.tensor(), input.tensor());
}

TEST_P(FlattenMultiDTypeTest, PixelUnshuffleFactor3) {
    auto pus = PixelUnshuffle(3);

    // Input: (1, 1, 9, 9) -> Output: (1, 9, 3, 3)
    Variable input = createInput({1, 1, 9, 9});
    auto output = pus.forward(input);

    expectShape(output.tensor(), {1, 9, 3, 3});
    expectDType(output.tensor());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FlattenMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 16
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 16 tests × 3 dtypes × 3 backends = 144 test scenarios
 *
 * Coverage:
 * - Basic flatten: from dim 0, dim 1, custom range, negative dims
 * - Gradient flow: backward pass, gradient shape verification
 * - Data ordering: value preservation after flatten
 * - Edge cases: single element, 2D input, large tensors
 * - Integration: with linear layers, with batch norm
 * - Various dimensions: high-dimensional input, various flatten ranges
 */
