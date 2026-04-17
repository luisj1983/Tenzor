/**
 * @file test_mask_head_multidtype.cpp
 * @brief Multi-dtype tests for MaskHead (Mask R-CNN mask prediction head)
 *
 * Tests MaskHead with Float32, Float64, and Float16 dtypes across CPU, CUDA,
 * OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct construction with default and custom parameters
 * - Output shape correctness for various ROI counts
 * - Proper dtype propagation through forward pass
 * - Finite output values (no NaN/Inf)
 * - Gradient flow through all layers
 * - Trainable parameters are present
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/mask_head.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::detection;
using namespace tenzor::testing;

// ============================================================================
// MaskHead Multi-Backend Multi-DType Tests
// ============================================================================

class MaskHeadMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(MaskHeadMultiDTypeTest, Construction) {
    auto head = MaskHead(256, 80);
    EXPECT_EQ(head.mask_size(), 28);
    EXPECT_EQ(head.num_classes(), 80);
}

TEST_P(MaskHeadMultiDTypeTest, ConstructionCustom) {
    auto head = MaskHead(128, 20, 128, 2, 14);
    EXPECT_EQ(head.mask_size(), 14);
    EXPECT_EQ(head.num_classes(), 20);
}

TEST_P(MaskHeadMultiDTypeTest, ForwardOutputShape) {
    auto head = MaskHead(256, 80);
    convert_model(head);

    Variable input = createInput({10, 256, 14, 14}, false);
    auto output = head.forward(input);

    expectShape(output.tensor(), {10, 80, 28, 28});
}

TEST_P(MaskHeadMultiDTypeTest, ForwardOutputDType) {
    auto head = MaskHead(256, 80);
    convert_model(head);

    Variable input = createInput({2, 256, 14, 14}, false);
    auto output = head.forward(input);

    expectDType(output.tensor());
}

TEST_P(MaskHeadMultiDTypeTest, ForwardSingleROI) {
    auto head = MaskHead(256, 80);
    convert_model(head);

    Variable input = createInput({1, 256, 14, 14}, false);
    auto output = head.forward(input);

    expectShape(output.tensor(), {1, 80, 28, 28});
}

TEST_P(MaskHeadMultiDTypeTest, ForwardOutputFinite) {
    auto head = MaskHead(256, 80);
    convert_model(head);

    Variable input = createInput({2, 256, 14, 14}, false);
    auto output = head.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = output_cpu.data<float>();

    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite value at index " << i << ": " << data[i];
    }
}

TEST_P(MaskHeadMultiDTypeTest, BackwardGradientFlow) {
    auto head = MaskHead(256, 80);
    convert_model(head);

    Variable input = createInput({2, 256, 14, 14}, true);
    auto output = head.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(shape_vec, dtype(), device());
    output.backward(grad_output);

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();

    // Check gradient shape matches input
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 256);
    EXPECT_EQ(grad.shape()[2], 14);
    EXPECT_EQ(grad.shape()[3], 14);
    EXPECT_EQ(grad.dtype(), dtype());

    // Check gradient values are finite
    auto grad_cpu = grad.to(Device::cpu()).to(DType::Float32);
    const float* grad_data = grad_cpu.data<float>();
    for (int64_t i = 0; i < std::min<int64_t>(1000, grad_cpu.numel()); ++i) {
        EXPECT_TRUE(std::isfinite(grad_data[i]))
            << "Non-finite gradient at index " << i << ": " << grad_data[i];
    }
}

TEST_P(MaskHeadMultiDTypeTest, ParameterCount) {
    auto head = MaskHead(256, 80);
    convert_model(head);

    auto params = head.parameters();
    EXPECT_FALSE(params.empty()) << "MaskHead should have trainable parameters";

    // Verify total parameter count is reasonable
    size_t total = countParameters(params);
    EXPECT_GT(total, 0u) << "Total parameter count should be positive";
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MaskHeadMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Classes: 1
 * - MaskHeadMultiDTypeTest: 8 tests
 *
 * Total Test Cases: 8
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI, Vulkan, ROCm
 * Total Scenarios: 8 tests x 3 dtypes x 5 backends = 120 test scenarios
 *
 * Coverage:
 * - Construction: default and custom parameters, accessor verification
 * - Forward pass: output shape, dtype propagation, single/multi ROI
 * - Numerical stability: finite output check (no NaN/Inf)
 * - Gradient flow: backward pass, gradient shape and finiteness
 * - Parameters: non-empty trainable parameter list
 */
