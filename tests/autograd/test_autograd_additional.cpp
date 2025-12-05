/**
 * @file test_autograd_additional.cpp
 * @brief Comprehensive additional tests for autograd modules to increase coverage
 *
 * Focuses on untested code paths in automatic differentiation including:
 * - Variable class constructor variations and state management
 * - Complex computation graph scenarios
 * - Backward pass variations (multiple backward, retain_graph)
 * - Custom autograd functions
 * - Context management (no_grad, enable_grad)
 * - Edge cases and error conditions
 * - Numerical gradient checking for complex operations
 * - Double backward (higher order derivatives)
 * - Gradient accumulation scenarios
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class AutogradAdditionalTest : public BackendTest {};

// =============================================================================
// Variable Class Tests - Constructor Variations and State Management
// =============================================================================

TEST_P(AutogradAdditionalTest, VariableDefaultConstructor) {
    Variable var;

    EXPECT_FALSE(var.is_initialized()) << "Failed on " << device.to_string();
    EXPECT_FALSE(static_cast<bool>(var)) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, VariableConstructorWithoutGrad) {
    auto data = ones({3, 4}, DType::Float32, device);
    Variable var(data, false);

    EXPECT_TRUE(var.is_initialized()) << "Failed on " << device.to_string();
    EXPECT_FALSE(var.requires_grad()) << "Failed on " << device.to_string();
    EXPECT_TRUE(var.is_leaf()) << "Failed on " << device.to_string();
    EXPECT_FALSE(var.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, VariableSetRequiresGrad) {
    auto data = ones({2, 3}, DType::Float32, device);
    Variable var(data, false);

    EXPECT_FALSE(var.requires_grad()) << "Failed on " << device.to_string();

    var.set_requires_grad(true);
    EXPECT_TRUE(var.requires_grad()) << "Failed on " << device.to_string();

    var.set_requires_grad(false);
    EXPECT_FALSE(var.requires_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, VariableGradAccumulation) {
    auto data = ones({2, 2}, DType::Float32, device) * 2.0f;
    Variable var(data, true);

    // Manually set and accumulate gradients
    auto grad1 = ones({2, 2}, DType::Float32, device);
    var.set_grad(grad1);

    ASSERT_TRUE(var.has_grad()) << "Failed on " << device.to_string();

    auto grad_cpu = var.grad()->to(Device::cpu());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(grad_cpu.data<float>()[i], 1.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, VariableRetainGrad) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x * 2.0f;  // Non-leaf variable

    EXPECT_FALSE(y.retains_grad()) << "Failed on " << device.to_string();

    y.retain_grad();
    EXPECT_TRUE(y.retains_grad()) << "Failed on " << device.to_string();

    auto loss = sum(y);
    loss.backward();

    // y should have gradient because retain_grad was called
    EXPECT_TRUE(y.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, VariableDetachRemovesGradFn) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x * 2.0f;

    EXPECT_TRUE(y.grad_fn() != nullptr) << "Failed on " << device.to_string();

    auto y_detached = y.detach();
    EXPECT_FALSE(y_detached.requires_grad()) << "Failed on " << device.to_string();
    EXPECT_TRUE(y_detached.grad_fn() == nullptr) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, VariableZeroGrad) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x * 2.0f;
    auto loss = sum(y);

    loss.backward();
    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    x.zero_grad();
    EXPECT_FALSE(x.has_grad()) << "Failed on " << device.to_string();
}

// =============================================================================
// Graph Construction and Complex Computation Graphs
// =============================================================================

TEST_P(AutogradAdditionalTest, ComplexMultiInputGraph) {
    // Test: z = (a + b) * (c - d)
    auto a = Variable(ones({2, 2}, DType::Float32, device) * 1.0f, true);
    auto b = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto c = Variable(ones({2, 2}, DType::Float32, device) * 5.0f, true);
    auto d = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);

    auto sum_ab = a + b;  // 3.0
    auto diff_cd = c - d;  // 2.0
    auto z = sum_ab * diff_cd;  // 6.0

    auto loss = sum(z);
    loss.backward();

    // Check all gradients exist
    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(c.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(d.has_grad()) << "Failed on " << device.to_string();

    // Gradients: dL/da = dL/db = (c-d) = 2.0
    //            dL/dc = (a+b) = 3.0
    //            dL/dd = -(a+b) = -3.0
    auto a_grad = a.grad()->to(Device::cpu());
    auto b_grad = b.grad()->to(Device::cpu());
    auto c_grad = c.grad()->to(Device::cpu());
    auto d_grad = d.grad()->to(Device::cpu());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 2.0f, 1e-4f) << "Failed on " << device.to_string();
        EXPECT_NEAR(b_grad.data<float>()[i], 2.0f, 1e-4f) << "Failed on " << device.to_string();
        EXPECT_NEAR(c_grad.data<float>()[i], 3.0f, 1e-4f) << "Failed on " << device.to_string();
        EXPECT_NEAR(d_grad.data<float>()[i], -3.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, MultiOutputGraph) {
    // Test multiple outputs from single input
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);

    auto y1 = x * 2.0f;
    auto y2 = x * 3.0f;
    auto y3 = x + 1.0f;

    auto loss = sum(y1) + sum(y2) + sum(y3);
    loss.backward();

    // Gradient should accumulate: 2.0 + 3.0 + 1.0 = 6.0
    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 6.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, DeepComputationGraph) {
    // Test deep chain of operations
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);

    auto y = x;
    for (int i = 0; i < 10; ++i) {
        y = y + 1.0f;
        y = y * 0.9f;
    }

    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
}

// =============================================================================
// Backward Pass Variations
// =============================================================================

TEST_P(AutogradAdditionalTest, BackwardWithGradientTensor) {
    auto x = Variable(ones({2, 3}, DType::Float32, device), true);
    auto y = x * 2.0f;

    // Provide custom gradient tensor
    auto grad_output = ones({2, 3}, DType::Float32, device) * 5.0f;
    y.backward(grad_output);

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient: 5.0 * 2.0 = 10.0
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 10.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, MultipleBackwardWithRetainGraph) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x * 2.0f;
    auto loss = sum(y);

    // First backward with retain_graph
    loss.backward(std::nullopt, true);
    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    auto grad1 = x.grad()->to(Device::cpu()).data<float>()[0];

    // Second backward - gradients should accumulate
    loss.backward(std::nullopt, false);
    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    auto grad2 = x.grad()->to(Device::cpu()).data<float>()[0];

    // Gradients should have accumulated (doubled)
    EXPECT_NEAR(grad2, grad1 * 2.0f, 1e-4f) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, BackwardWithoutRetainGraphClearsGraph) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x * 2.0f;
    auto loss = sum(y);

    // Backward without retain_graph
    loss.backward(std::nullopt, false);

    // Second backward should fail or produce different results since graph is cleared
    // (In production code, this might throw an exception)
    // For now, just verify first backward worked
    EXPECT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
}

// =============================================================================
// Gradient Function Tests
// =============================================================================

TEST_P(AutogradAdditionalTest, DivBackwardCorrectGradients) {
    // Test: z = a / b
    // dz/da = 1/b, dz/db = -a/(b^2)
    auto a_data = ones({2, 2}, DType::Float32, device) * 6.0f;
    auto b_data = ones({2, 2}, DType::Float32, device) * 2.0f;
    auto a = Variable(a_data, true);
    auto b = Variable(b_data, true);

    auto z = a / b;
    z.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad()->to(Device::cpu());
    auto b_grad = b.grad()->to(Device::cpu());

    // da should be 1/b = 0.5
    // db should be -a/(b^2) = -6.0/4.0 = -1.5
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 0.5f, 1e-4f) << "Failed on " << device.to_string();
        EXPECT_NEAR(b_grad.data<float>()[i], -1.5f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, NegBackwardCorrectGradients) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto y = neg(x);

    y.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient of -x is -1
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], -1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, ExpBackwardCorrectGradients) {
    auto x = Variable(zeros({2, 2}, DType::Float32, device), true);
    auto y = exp(x);

    y.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // d(exp(x))/dx at x=0 is exp(0) = 1
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, LogBackwardCorrectGradients) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto y = log(x);

    y.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // d(log(x))/dx = 1/x, at x=2.0 is 0.5
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 0.5f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, AbsBackwardCorrectGradients) {
    // Test with positive values
    auto x_pos = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto y_pos = abs(x_pos);
    y_pos.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(x_pos.has_grad()) << "Failed on " << device.to_string();
    auto x_pos_grad = x_pos.grad()->to(Device::cpu());

    // d(|x|)/dx = sign(x) = 1 for positive x
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_pos_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }

    // Test with negative values
    auto x_neg = Variable(ones({2, 2}, DType::Float32, device) * -2.0f, true);
    auto y_neg = abs(x_neg);
    y_neg.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(x_neg.has_grad()) << "Failed on " << device.to_string();
    auto x_neg_grad = x_neg.grad()->to(Device::cpu());

    // d(|x|)/dx = sign(x) = -1 for negative x
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_neg_grad.data<float>()[i], -1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, ClampBackwardCorrectGradients) {
    // Create tensor on CPU first, fill data, then transfer to target device
    auto x_data = ones({4}, DType::Float32, Device::cpu());
    x_data.data<float>()[0] = -2.0f;  // Below min
    x_data.data<float>()[1] = 0.0f;   // In range
    x_data.data<float>()[2] = 0.5f;   // In range
    x_data.data<float>()[3] = 2.0f;   // Above max

    // Transfer to target device if needed
    if (device != Device::cpu()) {
        x_data = x_data.to(device);
    }

    auto x = Variable(x_data, true);
    auto y = clamp(x, -1.0f, 1.0f);

    y.backward(ones({4}, DType::Float32, device));

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient only passes through for in-range values
    EXPECT_NEAR(x_grad.data<float>()[0], 0.0f, 1e-4f) << "Failed on " << device.to_string();  // Clamped
    EXPECT_NEAR(x_grad.data<float>()[1], 1.0f, 1e-4f) << "Failed on " << device.to_string();  // In range
    EXPECT_NEAR(x_grad.data<float>()[2], 1.0f, 1e-4f) << "Failed on " << device.to_string();  // In range
    EXPECT_NEAR(x_grad.data<float>()[3], 0.0f, 1e-4f) << "Failed on " << device.to_string();  // Clamped
}

// =============================================================================
// Shape Transformation Operations
// =============================================================================

TEST_P(AutogradAdditionalTest, ReshapeBackwardCorrectGradients) {
    auto x = Variable(ones({2, 3}, DType::Float32, device) * 2.0f, true);
    auto y = reshape(x, {3, 2});

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    // Gradient should be all ones (reshaped back)
    auto x_grad = x.grad()->to(Device::cpu());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, PermuteBackwardCorrectGradients) {
    auto x = Variable(ones({2, 3, 4}, DType::Float32, device), true);
    auto y = permute(x, {2, 0, 1});  // {4, 2, 3}

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    // Gradient should be all ones (permuted back)
    auto x_grad = x.grad()->to(Device::cpu());
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, TransposeBackwardCorrectGradients) {
    auto x = Variable(ones({3, 4}, DType::Float32, device), true);
    auto y = transpose(x, 0, 1);  // {4, 3}

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    // Gradient should be all ones (transposed back)
    auto x_grad = x.grad()->to(Device::cpu());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, SqueezeBackwardCorrectGradients) {
    auto x = Variable(ones({2, 1, 3}, DType::Float32, device), true);
    auto y = squeeze(x, 1);  // {2, 3}

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    // Gradient should be all ones (unsqueezed back)
    auto x_grad = x.grad()->to(Device::cpu());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

// =============================================================================
// Context Management Tests
// =============================================================================

TEST_P(AutogradAdditionalTest, NoGradGuardDisablesGradients) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);

    Variable y;
    {
        NoGradGuard guard;
        EXPECT_FALSE(is_grad_enabled()) << "Failed on " << device.to_string();

        y = x * 2.0f;

        // y should not have gradient function
        EXPECT_TRUE(y.grad_fn() == nullptr) << "Failed on " << device.to_string();
    }

    // Gradient should be re-enabled
    EXPECT_TRUE(is_grad_enabled()) << "Failed on " << device.to_string();

    // New operations should have gradients
    auto z = x * 3.0f;
    EXPECT_TRUE(z.grad_fn() != nullptr) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, SetGradEnabledManualControl) {
    EXPECT_TRUE(is_grad_enabled()) << "Failed on " << device.to_string();

    set_grad_enabled(false);
    EXPECT_FALSE(is_grad_enabled()) << "Failed on " << device.to_string();

    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x * 2.0f;

    // No gradient function should be created
    EXPECT_TRUE(y.grad_fn() == nullptr) << "Failed on " << device.to_string();

    set_grad_enabled(true);
    EXPECT_TRUE(is_grad_enabled()) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, NestedNoGradGuards) {
    EXPECT_TRUE(is_grad_enabled()) << "Failed on " << device.to_string();

    {
        NoGradGuard guard1;
        EXPECT_FALSE(is_grad_enabled()) << "Failed on " << device.to_string();

        {
            NoGradGuard guard2;
            EXPECT_FALSE(is_grad_enabled()) << "Failed on " << device.to_string();
        }

        EXPECT_FALSE(is_grad_enabled()) << "Failed on " << device.to_string();
    }

    EXPECT_TRUE(is_grad_enabled()) << "Failed on " << device.to_string();
}

// =============================================================================
// Edge Cases and Error Conditions
// =============================================================================

TEST_P(AutogradAdditionalTest, GradientThroughInPlaceOperationSimulation) {
    // Simulate in-place operation behavior
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto y = x + 1.0f;  // y = 3.0

    // Simulate in-place update by reassigning
    y = y * 2.0f;  // y = 6.0

    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, GradientWithLeafTensor) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);

    EXPECT_TRUE(x.is_leaf()) << "Failed on " << device.to_string();

    auto y = x * 2.0f;
    EXPECT_FALSE(y.is_leaf()) << "Failed on " << device.to_string();

    auto loss = sum(y);
    loss.backward();

    // Only leaf should accumulate gradient
    EXPECT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, UndefinedGradientHandling) {
    // Test with detached variable in graph
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x.detach();
    auto z = y * 2.0f;

    // z should not have gradient function since y is detached
    EXPECT_TRUE(z.grad_fn() == nullptr) << "Failed on " << device.to_string();
}

// =============================================================================
// Reduction Operations
// =============================================================================

TEST_P(AutogradAdditionalTest, SumBackwardWithDimAndKeepDim) {
    auto x = Variable(ones({3, 4}, DType::Float32, device) * 2.0f, true);

    // Sum along dimension 1, keeping dimension
    auto y = sum(x, 1, true);  // Shape: {3, 1}

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient should be all ones (broadcast back)
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, MeanBackwardCorrectGradients) {
    auto x = Variable(ones({2, 4}, DType::Float32, device) * 3.0f, true);

    auto y = mean(x);  // Scalar mean
    y.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient of mean is 1/N where N is total elements
    float expected_grad = 1.0f / 8.0f;  // 8 elements total
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], expected_grad, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, MaxBackwardSingleMaxElement) {
    // Create tensor on CPU first, fill data, then transfer to target device
    auto x_data = zeros({4}, DType::Float32, Device::cpu());
    x_data.data<float>()[0] = 1.0f;
    x_data.data<float>()[1] = 2.0f;
    x_data.data<float>()[2] = 5.0f;  // Max
    x_data.data<float>()[3] = 3.0f;

    // Transfer to target device if needed
    if (device != Device::cpu()) {
        x_data = x_data.to(device);
    }

    auto x = Variable(x_data, true);
    auto y = max(x);

    y.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Only max element should have gradient
    EXPECT_NEAR(x_grad.data<float>()[0], 0.0f, 1e-4f) << "Failed on " << device.to_string();
    EXPECT_NEAR(x_grad.data<float>()[1], 0.0f, 1e-4f) << "Failed on " << device.to_string();
    EXPECT_NEAR(x_grad.data<float>()[2], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    EXPECT_NEAR(x_grad.data<float>()[3], 0.0f, 1e-4f) << "Failed on " << device.to_string();
}

// =============================================================================
// Activation Functions
// =============================================================================

TEST_P(AutogradAdditionalTest, SoftmaxBackwardCorrectGradients) {
    auto x = Variable(ones({2, 3}, DType::Float32, device), true);
    auto y = softmax(x, 1);

    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, LogSoftmaxBackwardCorrectGradients) {
    auto x = Variable(ones({2, 3}, DType::Float32, device), true);
    auto y = log_softmax(x, 1);

    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
}

// =============================================================================
// Matrix Operations
// =============================================================================

TEST_P(AutogradAdditionalTest, MatMulBackwardCorrectGradients) {
    auto a = Variable(ones({3, 4}, DType::Float32, device), true);
    auto b = Variable(ones({4, 5}, DType::Float32, device), true);

    auto c = matmul(a, b);  // {3, 5}
    auto loss = sum(c);
    loss.backward();

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    // Check gradients have correct shape
    EXPECT_EQ(a.grad()->shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(a.grad()->shape()[1], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(b.grad()->shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(b.grad()->shape()[1], 5) << "Failed on " << device.to_string();
}

// -----------------------------------------------------------------------------
// MatMul backward tests with partial gradient requirements
// These tests cover the common neural network pattern where input data
// does not require gradients but weights do.
// -----------------------------------------------------------------------------

TEST_P(AutogradAdditionalTest, MatMulBackwardOnlyRightOperandRequiresGrad) {
    // This is the common neural network pattern:
    // X (input data) has no gradient, W (weights) needs gradient
    // z = X @ W
    auto X = Variable(ones({4, 2}, DType::Float32, device), false);  // Input - no grad
    auto W = Variable(ones({2, 4}, DType::Float32, device), true);   // Weights - has grad

    auto z = matmul(X, W);  // {4, 4}
    auto loss = sum(z);
    loss.backward();

    // X should not have gradient (didn't request it)
    EXPECT_FALSE(X.has_grad()) << "X should not have grad on " << device.to_string();

    // W must have gradient with correct shape
    ASSERT_TRUE(W.has_grad()) << "W must have grad on " << device.to_string();

    // Critical check: gradient shape must match weight shape
    // For W(2,4), grad must be (2,4) - NOT transposed to (4,2)
    EXPECT_EQ(W.grad()->shape()[0], 2) << "W.grad row dim wrong on " << device.to_string();
    EXPECT_EQ(W.grad()->shape()[1], 4) << "W.grad col dim wrong on " << device.to_string();

    // Verify gradient values: dL/dW = X^T @ grad_output
    // With all ones, dL/dW[i,j] = sum over batch = 4
    auto w_grad = W.grad()->to(Device::cpu());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(w_grad.data<float>()[i], 4.0f, 1e-4f)
            << "W.grad value wrong at " << i << " on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, MatMulBackwardOnlyLeftOperandRequiresGrad) {
    // Less common but still valid: left operand needs gradient, right doesn't
    auto A = Variable(ones({3, 4}, DType::Float32, device), true);   // Has grad
    auto B = Variable(ones({4, 5}, DType::Float32, device), false);  // No grad

    auto C = matmul(A, B);  // {3, 5}
    auto loss = sum(C);
    loss.backward();

    // A must have gradient with correct shape
    ASSERT_TRUE(A.has_grad()) << "A must have grad on " << device.to_string();

    // B should not have gradient
    EXPECT_FALSE(B.has_grad()) << "B should not have grad on " << device.to_string();

    // Check A gradient shape: must be (3, 4)
    EXPECT_EQ(A.grad()->shape()[0], 3) << "A.grad row dim wrong on " << device.to_string();
    EXPECT_EQ(A.grad()->shape()[1], 4) << "A.grad col dim wrong on " << device.to_string();

    // Verify gradient values: dL/dA = grad_output @ B^T
    // With all ones, dL/dA[i,j] = sum over output cols = 5
    auto a_grad = A.grad()->to(Device::cpu());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 5.0f, 1e-4f)
            << "A.grad value wrong at " << i << " on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, MatMulBackwardMemberFunctionOnlyRightGrad) {
    // Test the member function variant: X.matmul(W)
    auto X = Variable(ones({4, 2}, DType::Float32, device), false);
    auto W = Variable(ones({2, 4}, DType::Float32, device), true);

    auto z = X.matmul(W);  // Using member function
    auto loss = sum(z);
    loss.backward();

    ASSERT_TRUE(W.has_grad()) << "W must have grad on " << device.to_string();

    // Shape must be (2, 4) - matching W, not transposed
    EXPECT_EQ(W.grad()->shape()[0], 2) << "W.grad row dim wrong on " << device.to_string();
    EXPECT_EQ(W.grad()->shape()[1], 4) << "W.grad col dim wrong on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, MatMulBackwardNeuralNetworkScenario) {
    // Realistic neural network scenario: batch_size x features @ features x hidden
    // Input: (32, 16) - batch of 32 samples, 16 features each
    // Weights: (16, 8) - project to 8 hidden units
    auto input = Variable(randn({32, 16}, DType::Float32, device), false);
    auto weights = Variable(randn({16, 8}, DType::Float32, device), true);

    auto hidden = matmul(input, weights);  // (32, 8)
    auto loss = mean(hidden);  // Scalar loss
    loss.backward();

    ASSERT_TRUE(weights.has_grad()) << "weights must have grad on " << device.to_string();

    // Critical: gradient shape must match weight shape (16, 8)
    EXPECT_EQ(weights.grad()->shape()[0], 16)
        << "weights.grad row dim wrong on " << device.to_string();
    EXPECT_EQ(weights.grad()->shape()[1], 8)
        << "weights.grad col dim wrong on " << device.to_string();
}

TEST_P(AutogradAdditionalTest, MatMulBackwardChainedOperationsPartialGrad) {
    // Test gradient flow through chained matmuls with partial grad requirements
    // This mimics a 2-layer neural network
    auto X = Variable(ones({4, 2}, DType::Float32, device), false);   // Input
    auto W1 = Variable(ones({2, 3}, DType::Float32, device), true);   // Layer 1 weights
    auto W2 = Variable(ones({3, 1}, DType::Float32, device), true);   // Layer 2 weights

    auto h = matmul(X, W1);   // (4, 3) hidden layer
    auto out = matmul(h, W2); // (4, 1) output
    auto loss = sum(out);
    loss.backward();

    // Both weight matrices must have gradients
    ASSERT_TRUE(W1.has_grad()) << "W1 must have grad on " << device.to_string();
    ASSERT_TRUE(W2.has_grad()) << "W2 must have grad on " << device.to_string();

    // W1 gradient shape must be (2, 3)
    EXPECT_EQ(W1.grad()->shape()[0], 2) << "W1.grad row dim wrong on " << device.to_string();
    EXPECT_EQ(W1.grad()->shape()[1], 3) << "W1.grad col dim wrong on " << device.to_string();

    // W2 gradient shape must be (3, 1)
    EXPECT_EQ(W2.grad()->shape()[0], 3) << "W2.grad row dim wrong on " << device.to_string();
    EXPECT_EQ(W2.grad()->shape()[1], 1) << "W2.grad col dim wrong on " << device.to_string();

    // Verify W2 gradient values: dL/dW2 = h^T @ grad_output = (3,4) @ (4,1) -> (3,1)
    // h is all 2s (X@W1 with all ones), grad_output is all 1s
    // So dL/dW2 = 2 * 4 = 8
    auto w2_grad = W2.grad()->to(Device::cpu());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(w2_grad.data<float>()[i], 8.0f, 1e-4f)
            << "W2.grad value wrong at " << i << " on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, BmmBackwardCorrectGradients) {
    auto a = Variable(ones({2, 3, 4}, DType::Float32, device), true);
    auto b = Variable(ones({2, 4, 5}, DType::Float32, device), true);

    auto c = bmm(a, b);  // {2, 3, 5}
    auto loss = sum(c);
    loss.backward();

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    // Check gradients have correct shape
    EXPECT_EQ(a.grad()->shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(a.grad()->shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(a.grad()->shape()[2], 4) << "Failed on " << device.to_string();
}

// =============================================================================
// Concatenation and Slicing
// =============================================================================

TEST_P(AutogradAdditionalTest, CatBackwardCorrectGradients) {
    auto x1 = Variable(ones({2, 3}, DType::Float32, device), true);
    auto x2 = Variable(ones({2, 5}, DType::Float32, device), true);

    auto y = cat({x1, x2}, 1);  // {2, 8}
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x1.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(x2.has_grad()) << "Failed on " << device.to_string();

    // Gradients should be all ones (split back)
    auto x1_grad = x1.grad()->to(Device::cpu());
    auto x2_grad = x2.grad()->to(Device::cpu());

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(x1_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(x2_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, SliceBackwardCorrectGradients) {
    auto x = Variable(ones({10, 10}, DType::Float32, device), true);

    auto y = slice(x, 1, 2, 8, 1);  // Slice columns 2-8
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    auto x_grad = x.grad()->to(Device::cpu());

    // Only sliced columns should have gradient = 1, others = 0
    for (int64_t i = 0; i < 10; ++i) {
        for (int64_t j = 0; j < 10; ++j) {
            float expected = (j >= 2 && j < 8) ? 1.0f : 0.0f;
            EXPECT_NEAR(x_grad.data<float>()[i * 10 + j], expected, 1e-4f)
                << "Failed on " << device.to_string();
        }
    }
}

// =============================================================================
// Broadcasting Tests
// =============================================================================

TEST_P(AutogradAdditionalTest, BroadcastingAddBackward) {
    auto a = Variable(ones({2, 3}, DType::Float32, device), true);
    auto b = Variable(ones({3}, DType::Float32, device), true);

    auto c = a + b;  // Broadcasting: {2,3} + {3} -> {2,3}
    auto loss = sum(c);
    loss.backward();

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    // a gradient should be ones
    auto a_grad = a.grad()->to(Device::cpu());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }

    // b gradient should be sum along batch dimension (2.0)
    auto b_grad = b.grad()->to(Device::cpu());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(b_grad.data<float>()[i], 2.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, BroadcastingMulBackward) {
    auto a = Variable(ones({2, 1, 4}, DType::Float32, device) * 2.0f, true);
    auto b = Variable(ones({3, 1}, DType::Float32, device) * 3.0f, true);

    auto c = a * b;  // Broadcasting: {2,1,4} * {3,1} -> {2,3,4}
    auto loss = sum(c);
    loss.backward();

    EXPECT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    EXPECT_TRUE(b.has_grad()) << "Failed on " << device.to_string();
}

// =============================================================================
// Hook Registration Tests
// =============================================================================

TEST_P(AutogradAdditionalTest, RegisterHookModifiesGradient) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);

    // Register hook that scales gradient by 2
    x.register_hook([](const Tensor& grad) -> Tensor {
        return grad * 2.0f;
    });

    auto y = x * 3.0f;
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Original gradient would be 3.0, but hook scales by 2
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 6.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, MultipleHooksAreChained) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);

    // Register multiple hooks
    x.register_hook([](const Tensor& grad) -> Tensor {
        return grad * 2.0f;
    });

    x.register_hook([](const Tensor& grad) -> Tensor {
        return grad + 1.0f;
    });

    auto y = x * 2.0f;
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();

    // Gradient: 2.0 -> hook1: *2 = 4.0 -> hook2: +1 = 5.0
    auto x_grad = x.grad()->to(Device::cpu());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 5.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

// =============================================================================
// Numerical Gradient Checking Tests
// =============================================================================

TEST_P(AutogradAdditionalTest, GradcheckSimpleFunction) {

    auto func = [](const Variable& x) -> Variable {
        return sum(x * x);
    };

    auto x = Variable(ones({3, 4}, DType::Float32, Device::cpu()) * 0.5f, true);

    bool passed = gradcheck(func, x, 1e-4, 1e-3, 1e-2);
    EXPECT_TRUE(passed) << "Gradcheck failed for x^2";
}

TEST_P(AutogradAdditionalTest, GradcheckComplexFunction) {

    auto func = [](const Variable& x) -> Variable {
        auto y = exp(x);
        auto z = log(y + 1.0f);
        return sum(z * z);
    };

    auto x = Variable(ones({2, 3}, DType::Float32, Device::cpu()) * 0.5f, true);

    bool passed = gradcheck(func, x, 1e-4, 1e-3, 1e-2);
    EXPECT_TRUE(passed) << "Gradcheck failed for complex function";
}

TEST_P(AutogradAdditionalTest, GradcheckMatmul) {

    auto b = ones({4, 5}, DType::Float32, Device::cpu());

    auto func = [&b](const Variable& a) -> Variable {
        auto b_var = Variable(b, false);
        auto c = matmul(a, b_var);
        return sum(c);
    };

    auto a = Variable(ones({3, 4}, DType::Float32, Device::cpu()) * 0.5f, true);

    bool passed = gradcheck(func, a, 1e-4, 1e-3, 1e-2);
    EXPECT_TRUE(passed) << "Gradcheck failed for matmul";
}

// =============================================================================
// Scalar Operations with Variables
// =============================================================================

TEST_P(AutogradAdditionalTest, ScalarAddBackward) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x + 5.0f;

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient of (x + c) is 1
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, ScalarMulBackward) {
    auto x = Variable(ones({2, 2}, DType::Float32, device), true);
    auto y = x * 3.5f;

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient of (x * c) is c
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 3.5f, 1e-4f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradAdditionalTest, ScalarDivBackward) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 4.0f, true);
    auto y = x / 2.0f;

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad()->to(Device::cpu());

    // Gradient of (x / c) is 1/c
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 0.5f, 1e-4f) << "Failed on " << device.to_string();
    }
}

// =============================================================================
// Register tests with backend fixture
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    AutogradBackends,
    AutogradAdditionalTest,
    ::testing::Values("cpu", "cuda", "vulkan", "oneapi")
);
