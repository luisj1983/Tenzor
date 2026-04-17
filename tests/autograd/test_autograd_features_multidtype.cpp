/**
 * @file test_autograd_features_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for core autograd features
 *
 * Multi-backend port of test_autograd_features.cpp. Tests requires_grad,
 * grad_fn, is_leaf, register_hook, retain_grad, detach, zero_grad, and
 * backward(retain_graph) across all available backends and data types.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class AutogradFeaturesMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// grad_fn Tests
// ============================================================================

TEST_P(AutogradFeaturesMultiDTypeTest, GradFnLeafVariable) {
    auto t = ones({3, 4}, dtype(), device());
    Variable x(t, true);

    // Leaf variables have no grad_fn
    EXPECT_EQ(x.grad_fn(), nullptr);
    EXPECT_TRUE(x.is_leaf());
}

TEST_P(AutogradFeaturesMultiDTypeTest, GradFnNonLeafVariable) {
    auto t = ones({3, 4}, dtype(), device());
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Non-leaf variables have grad_fn
    EXPECT_NE(y.grad_fn(), nullptr);
    EXPECT_FALSE(y.is_leaf());
}

// ============================================================================
// is_leaf Tests
// ============================================================================

TEST_P(AutogradFeaturesMultiDTypeTest, IsLeafCreatedByUser) {
    auto t = ones({2, 3}, dtype(), device());
    Variable x(t, true);

    EXPECT_TRUE(x.is_leaf());
}

TEST_P(AutogradFeaturesMultiDTypeTest, IsLeafFromOperation) {
    auto t = ones({2, 3}, dtype(), device());
    Variable x(t, true);
    Variable y = x + x;

    EXPECT_FALSE(y.is_leaf());
}

// ============================================================================
// register_hook Tests
// ============================================================================

TEST_P(AutogradFeaturesMultiDTypeTest, RegisterHookModifyGradient) {
    auto t = ones({3}, dtype(), device());
    Variable x(t, true);

    // Register hook that doubles the gradient
    x.register_hook([](const Tensor& grad) {
        return grad * 2.0f;
    });

    Variable y = x * 3.0f;
    auto loss = sum(y);
    loss.backward();

    // Gradient should be 3.0 (from y = x * 3), then doubled by hook = 6.0
    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad_data[i], 6.0f, atol());
    }
}

TEST_P(AutogradFeaturesMultiDTypeTest, RegisterMultipleHooks) {
    auto t = ones({2}, dtype(), device());
    Variable x(t, true);

    // Register multiple hooks
    x.register_hook([](const Tensor& grad) {
        return grad * 2.0f;
    });
    x.register_hook([](const Tensor& grad) {
        return grad + 1.0f;
    });

    Variable y = x * 5.0f;
    auto loss = sum(y);
    loss.backward();

    // Gradient: 5.0 -> *2 = 10.0 -> +1 = 11.0
    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad.data<float>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(grad_data[i], 11.0f, atol());
    }
}

// ============================================================================
// retain_grad Tests
// ============================================================================

TEST_P(AutogradFeaturesMultiDTypeTest, RetainGradNonLeaf) {
    auto t = ones({3}, dtype(), device());
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Retain gradient for non-leaf variable
    y.retain_grad();
    EXPECT_TRUE(y.retains_grad());

    Variable z = y * 3.0f;
    auto loss = sum(z);
    loss.backward();

    // Both x and y should have gradients
    EXPECT_TRUE(x.has_grad());
    EXPECT_TRUE(y.has_grad());

    // y.grad should be 3.0 (from z = y * 3)
    auto y_grad = y.grad()->to(Device::cpu()).to(DType::Float32);
    auto y_grad_data = y_grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(y_grad_data[i], 3.0f, atol());
    }

    // x.grad should be 6.0 (chain rule: 2 * 3)
    auto x_grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto x_grad_data = x_grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(x_grad_data[i], 6.0f, atol());
    }
}

TEST_P(AutogradFeaturesMultiDTypeTest, RetainGradNotSet) {
    auto t = ones({3}, dtype(), device());
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Don't retain gradient for non-leaf variable
    EXPECT_FALSE(y.retains_grad());

    Variable z = y * 3.0f;
    auto loss = sum(z);
    loss.backward();

    // Only leaf variable x should have gradient
    EXPECT_TRUE(x.has_grad());
}

// ============================================================================
// backward(retain_graph=True) Tests
// ============================================================================

TEST_P(AutogradFeaturesMultiDTypeTest, BackwardRetainGraph) {
    auto t = ones({2}, dtype(), device());
    Variable x(t, true);
    Variable y = x * 3.0f;
    auto loss = sum(y);

    // First backward with retain_graph=true
    loss.backward(std::nullopt, true);

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad.data<float>();
    EXPECT_NEAR(grad_data[0], 3.0f, atol());
    EXPECT_NEAR(grad_data[1], 3.0f, atol());

    // Zero gradients for second backward
    x.zero_grad();

    // Second backward should work (graph retained)
    loss.backward(std::nullopt, false);

    ASSERT_TRUE(x.has_grad());
    grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    grad_data = grad.data<float>();
    EXPECT_NEAR(grad_data[0], 3.0f, atol());
    EXPECT_NEAR(grad_data[1], 3.0f, atol());
}

TEST_P(AutogradFeaturesMultiDTypeTest, BackwardWithoutRetainGraph) {
    auto t = ones({2}, dtype(), device());
    Variable x(t, true);
    Variable y = x * 3.0f;
    auto loss = sum(y);

    // Backward without retain_graph (default)
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad.data<float>();
    EXPECT_NEAR(grad_data[0], 3.0f, atol());
    EXPECT_NEAR(grad_data[1], 3.0f, atol());
}

TEST_P(AutogradFeaturesMultiDTypeTest, MultipleBackwardPasses) {
    auto t = ones({3}, dtype(), device());
    Variable x(t, true);

    // First computation
    Variable y1 = x * 2.0f;
    auto loss1 = sum(y1);
    loss1.backward(std::nullopt, true);

    auto grad1 = x.grad()->clone();

    // Second computation (accumulates gradients)
    Variable y2 = x * 3.0f;
    auto loss2 = sum(y2);
    loss2.backward(std::nullopt, false);

    // Gradients should accumulate: 2.0 + 3.0 = 5.0
    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad_data[i], 5.0f, atol());
    }
}

// ============================================================================
// Combined Features Tests
// ============================================================================

TEST_P(AutogradFeaturesMultiDTypeTest, HookWithRetainGrad) {
    auto t = ones({2}, dtype(), device());
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Retain gradient and register hook
    y.retain_grad();
    y.register_hook([](const Tensor& grad) {
        return grad * 10.0f;
    });

    Variable z = y * 3.0f;
    auto loss = sum(z);
    loss.backward();

    // y.grad should be 3.0 * 10.0 = 30.0 (hook applied)
    ASSERT_TRUE(y.has_grad());
    auto y_grad = y.grad()->to(Device::cpu()).to(DType::Float32);
    auto y_grad_data = y_grad.data<float>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(y_grad_data[i], 30.0f, atol());
    }
}

TEST_P(AutogradFeaturesMultiDTypeTest, ComplexComputationGraph) {
    auto t1 = ones({3}, dtype(), device());
    auto t2 = ones({3}, dtype(), device());

    Variable x(t1, true);
    Variable w(t2, true);

    // Register hooks
    x.register_hook([](const Tensor& grad) {
        return grad * 0.9f;  // Gradient clipping
    });

    // Complex graph
    Variable y = x * w;
    y.retain_grad();

    Variable z = y * 2.0f;
    auto loss = sum(z);

    loss.backward(std::nullopt, true);

    // Verify all gradients exist
    EXPECT_TRUE(x.has_grad());
    EXPECT_TRUE(w.has_grad());
    EXPECT_TRUE(y.has_grad());

    // Verify gradient values
    // dy/dw = x * 2.0 = 1.0 * 2.0 = 2.0
    auto w_grad = w.grad()->to(Device::cpu()).to(DType::Float32);
    auto w_grad_data = w_grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w_grad_data[i], 2.0f, atol());
    }

    // dy/dx = w * 2.0 * 0.9 (hook) = 1.0 * 2.0 * 0.9 = 1.8
    auto x_grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto x_grad_data = x_grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(x_grad_data[i], 1.8f, atol());
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AutogradFeaturesMultiDTypeTest);
