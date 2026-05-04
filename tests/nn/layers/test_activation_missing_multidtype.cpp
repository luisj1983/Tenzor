/**
 * @file test_activation_missing_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Hardsigmoid, Softmin, Softshrink,
 *        and Tanhshrink activation layers.
 *
 * These layers previously had no unit tests. Each test verifies:
 * - Forward shape preservation
 * - Forward value correctness against known formulas
 * - Backward gradient flow
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class ActivationMissingMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Hardsigmoid: clamp(x/6 + 0.5, 0, 1)
// ============================================================================

TEST_P(ActivationMissingMultiDTypeTest, Hardsigmoid_ForwardShape) {
    Hardsigmoid act;
    auto input = Variable(createRandn({4, 8}), false);
    auto output = act.forward(input);
    ASSERT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    expectDType(output.tensor());
}

TEST_P(ActivationMissingMultiDTypeTest, Hardsigmoid_ForwardValues) {
    Hardsigmoid act;
    // Create a known-value tensor on CPU Float32
    auto input_cpu = zeros({5}, DType::Float32, Device::cpu());
    auto* in_ptr = input_cpu.data<float>();
    in_ptr[0] = -4.0f;
    in_ptr[1] = -3.0f;
    in_ptr[2] = 0.0f;
    in_ptr[3] = 3.0f;
    in_ptr[4] = 4.0f;

    auto input_var = Variable(input_cpu, false);
    auto output = act.forward(input_var).tensor();
    auto* out = output.data<float>();

    // clamp(x/6 + 0.5, 0, 1)
    EXPECT_NEAR(out[0], 0.0f, 1e-5f);       // -4/6 + 0.5 = -0.167 -> 0
    EXPECT_NEAR(out[1], 0.0f, 1e-5f);       // -3/6 + 0.5 = 0.0
    EXPECT_NEAR(out[2], 0.5f, 1e-5f);       //  0/6 + 0.5 = 0.5
    EXPECT_NEAR(out[3], 1.0f, 1e-5f);       //  3/6 + 0.5 = 1.0
    EXPECT_NEAR(out[4], 1.0f, 1e-5f);       //  4/6 + 0.5 = 1.167 -> 1
}

TEST_P(ActivationMissingMultiDTypeTest, Hardsigmoid_Backward) {
    Hardsigmoid act;
    // Use small-magnitude inputs (default randn ~ N(0,1)) to keep most
    // elements inside the active band [-3, 3] where the slope is non-zero.
    auto input = Variable(createRandn({4, 8}), true);
    auto output = act.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape().size(), 2u);
    EXPECT_EQ(input.grad().value().shape()[0], 4);
    EXPECT_EQ(input.grad().value().shape()[1], 8);
}

// ============================================================================
// Softmin: softmax(-x, dim)
// ============================================================================

TEST_P(ActivationMissingMultiDTypeTest, Softmin_ForwardShape) {
    Softmin act(/*dim=*/-1);
    auto input = Variable(createRandn({4, 8}), false);
    auto output = act.forward(input);
    ASSERT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    expectDType(output.tensor());
}

TEST_P(ActivationMissingMultiDTypeTest, Softmin_SumsToOne) {
    Softmin act(/*dim=*/-1);
    auto input = Variable(createRandn({2, 5}), false);
    auto output = act.forward(input).tensor();
    // Each row should sum to 1. Half-precision dtypes have larger
    // accumulation drift (~5e-3 across 5 values), so tolerance is dtype-aware
    // rather than skipping the assertion (per the no-skip testing rule).
    auto sums = tenzor::sum(output, /*dim=*/-1).to(Device::cpu()).to(DType::Float32);
    auto* s = sums.data<float>();
    const float tol = (dtype() == DType::Float16 || dtype() == DType::BFloat16)
                          ? 5e-3f
                          : 1e-4f;
    for (int64_t i = 0; i < sums.numel(); ++i) {
        EXPECT_NEAR(s[i], 1.0f, tol) << "Row " << i << " does not sum to 1";
    }
}

TEST_P(ActivationMissingMultiDTypeTest, Softmin_Backward) {
    Softmin act(/*dim=*/-1);
    auto input = Variable(createRandn({4, 8}), true);
    auto output = act.forward(input);
    // sum(softmax(...)) along the softmax axis is identically 1, so
    // d sum/dx is zero — use sum(output*output) to produce a non-trivial
    // gradient that EXPECT_GRAD_FLOWS can verify.
    auto loss = tenzor::sum(output * output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape().size(), 2u);
}

// ============================================================================
// Tanhshrink: x - tanh(x)
// ============================================================================

TEST_P(ActivationMissingMultiDTypeTest, Tanhshrink_ForwardShape) {
    Tanhshrink act;
    auto input = Variable(createRandn({4, 8}), false);
    auto output = act.forward(input);
    ASSERT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    expectDType(output.tensor());
}

TEST_P(ActivationMissingMultiDTypeTest, Tanhshrink_ForwardValues) {
    Tanhshrink act;
    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* in_ptr = input_cpu.data<float>();
    in_ptr[0] = 0.0f;
    in_ptr[1] = 1.0f;
    in_ptr[2] = -1.0f;
    in_ptr[3] = 5.0f;

    auto input_var = Variable(input_cpu, false);
    auto output = act.forward(input_var).tensor();
    auto* out = output.data<float>();

    // x - tanh(x)
    EXPECT_NEAR(out[0], 0.0f - std::tanh(0.0f), 1e-5f);
    EXPECT_NEAR(out[1], 1.0f - std::tanh(1.0f), 1e-5f);
    EXPECT_NEAR(out[2], -1.0f - std::tanh(-1.0f), 1e-5f);
    EXPECT_NEAR(out[3], 5.0f - std::tanh(5.0f), 1e-4f);
}

TEST_P(ActivationMissingMultiDTypeTest, Tanhshrink_Backward) {
    Tanhshrink act;
    auto input = Variable(createRandn({4, 8}), true);
    auto output = act.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape().size(), 2u);
}

// ============================================================================
// Softshrink: sign(x) * max(|x| - lambda, 0)
// ============================================================================

TEST_P(ActivationMissingMultiDTypeTest, Softshrink_ForwardShape) {
    Softshrink act(0.5);
    auto input = Variable(createRandn({4, 8}), false);
    auto output = act.forward(input);
    ASSERT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    expectDType(output.tensor());
}

TEST_P(ActivationMissingMultiDTypeTest, Softshrink_ForwardValues) {
    Softshrink act(0.5);
    auto input_cpu = zeros({5}, DType::Float32, Device::cpu());
    auto* in_ptr = input_cpu.data<float>();
    in_ptr[0] = 0.0f;
    in_ptr[1] = 0.3f;
    in_ptr[2] = -0.3f;
    in_ptr[3] = 1.0f;
    in_ptr[4] = -1.0f;

    auto input_var = Variable(input_cpu, false);
    auto output = act.forward(input_var).tensor();
    auto* out = output.data<float>();

    // sign(x) * max(|x| - 0.5, 0)
    EXPECT_NEAR(out[0], 0.0f, 1e-5f);       // |0| < 0.5 -> 0
    EXPECT_NEAR(out[1], 0.0f, 1e-5f);       // |0.3| < 0.5 -> 0
    EXPECT_NEAR(out[2], 0.0f, 1e-5f);       // |-0.3| < 0.5 -> 0
    EXPECT_NEAR(out[3], 0.5f, 1e-5f);       // 1.0 - 0.5 = 0.5
    EXPECT_NEAR(out[4], -0.5f, 1e-5f);      // -(1.0 - 0.5) = -0.5
}

TEST_P(ActivationMissingMultiDTypeTest, Softshrink_Backward) {
    Softshrink act(0.5);
    // Use values outside the dead zone to get non-zero gradients
    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* in_ptr = input_cpu.data<float>();
    in_ptr[0] = 1.0f;
    in_ptr[1] = -1.0f;
    in_ptr[2] = 2.0f;
    in_ptr[3] = -2.0f;

    auto input = Variable(input_cpu.to(device()).to(dtype()), true);
    auto output = act.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape().size(), 1u);
    // Gradient should be 1 outside dead zone
    auto grad_cpu = input.grad().value().to(Device::cpu()).to(DType::Float32);
    auto* g = grad_cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(g[i], 1.0f, 0.1f) << "Gradient at index " << i;
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ActivationMissingMultiDTypeTest);
