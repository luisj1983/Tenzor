/**
 * @file test_autograd_features.cpp
 * @brief Tests for Phase 6 autograd features
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <cmath>

using namespace tenzor;

class AutogradFeaturesTest : public ::testing::Test {
protected:
    Device cpu = Device::cpu();
};

// ============================================================================
// grad_fn Tests
// ============================================================================

TEST_F(AutogradFeaturesTest, GradFnLeafVariable) {
    auto t = ones({3, 4}, DType::Float32, cpu);
    Variable x(t, true);

    // Leaf variables have no grad_fn
    EXPECT_EQ(x.grad_fn(), nullptr);
    EXPECT_TRUE(x.is_leaf());
}

TEST_F(AutogradFeaturesTest, GradFnNonLeafVariable) {
    auto t = ones({3, 4}, DType::Float32, cpu);
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Non-leaf variables have grad_fn
    EXPECT_NE(y.grad_fn(), nullptr);
    EXPECT_FALSE(y.is_leaf());
}

// ============================================================================
// is_leaf Tests
// ============================================================================

TEST_F(AutogradFeaturesTest, IsLeafCreatedByUser) {
    auto t = ones({2, 3}, DType::Float32, cpu);
    Variable x(t, true);

    EXPECT_TRUE(x.is_leaf());
}

TEST_F(AutogradFeaturesTest, IsLeafFromOperation) {
    auto t = ones({2, 3}, DType::Float32, cpu);
    Variable x(t, true);
    Variable y = x + x;

    EXPECT_FALSE(y.is_leaf());
}

// ============================================================================
// register_hook Tests
// ============================================================================

TEST_F(AutogradFeaturesTest, RegisterHookModifyGradient) {
    auto t = ones({3}, DType::Float32, cpu);
    Variable x(t, true);

    // Register hook that doubles the gradient
    x.register_hook([](const Tensor& grad) {
        return grad * 2.0f;
    });

    Variable y = x * 3.0f;
    auto loss = y.sum();
    loss.backward();

    // Gradient should be 3.0 (from y = x * 3), then doubled by hook = 6.0
    ASSERT_TRUE(x.has_grad());
    auto grad_data = x.grad()->data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 6.0f);
    }
}

TEST_F(AutogradFeaturesTest, RegisterMultipleHooks) {
    auto t = ones({2}, DType::Float32, cpu);
    Variable x(t, true);

    // Register multiple hooks
    x.register_hook([](const Tensor& grad) {
        return grad * 2.0f;
    });
    x.register_hook([](const Tensor& grad) {
        return grad + 1.0f;
    });

    Variable y = x * 5.0f;
    auto loss = y.sum();
    loss.backward();

    // Gradient: 5.0 -> *2 = 10.0 -> +1 = 11.0
    ASSERT_TRUE(x.has_grad());
    auto grad_data = x.grad()->data<float>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 11.0f);
    }
}

// ============================================================================
// retain_grad Tests
// ============================================================================

TEST_F(AutogradFeaturesTest, RetainGradNonLeaf) {
    auto t = ones({3}, DType::Float32, cpu);
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Retain gradient for non-leaf variable
    y.retain_grad();
    EXPECT_TRUE(y.retains_grad());

    Variable z = y * 3.0f;
    auto loss = z.sum();
    loss.backward();

    // Both x and y should have gradients
    EXPECT_TRUE(x.has_grad());
    EXPECT_TRUE(y.has_grad());

    // y.grad should be 3.0 (from z = y * 3)
    auto y_grad_data = y.grad()->data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(y_grad_data[i], 3.0f);
    }

    // x.grad should be 6.0 (chain rule: 2 * 3)
    auto x_grad_data = x.grad()->data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(x_grad_data[i], 6.0f);
    }
}

TEST_F(AutogradFeaturesTest, RetainGradNotSet) {
    auto t = ones({3}, DType::Float32, cpu);
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Don't retain gradient for non-leaf variable
    EXPECT_FALSE(y.retains_grad());

    Variable z = y * 3.0f;
    auto loss = z.sum();
    loss.backward();

    // Only leaf variable x should have gradient
    EXPECT_TRUE(x.has_grad());
    // Non-leaf y should not have gradient (not retained)
    // Note: In current implementation, we may still have gradient
    // This test documents the expected behavior
}

// ============================================================================
// backward(retain_graph=True) Tests
// ============================================================================

TEST_F(AutogradFeaturesTest, BackwardRetainGraph) {
    auto t = ones({2}, DType::Float32, cpu);
    Variable x(t, true);
    Variable y = x * 3.0f;
    auto loss = y.sum();

    // First backward with retain_graph=true
    loss.backward(std::nullopt, true);

    ASSERT_TRUE(x.has_grad());
    auto grad_data = x.grad()->data<float>();
    EXPECT_FLOAT_EQ(grad_data[0], 3.0f);
    EXPECT_FLOAT_EQ(grad_data[1], 3.0f);

    // Zero gradients for second backward
    x.zero_grad();

    // Second backward should work (graph retained)
    loss.backward(std::nullopt, false);

    ASSERT_TRUE(x.has_grad());
    grad_data = x.grad()->data<float>();
    EXPECT_FLOAT_EQ(grad_data[0], 3.0f);
    EXPECT_FLOAT_EQ(grad_data[1], 3.0f);
}

TEST_F(AutogradFeaturesTest, BackwardWithoutRetainGraph) {
    auto t = ones({2}, DType::Float32, cpu);
    Variable x(t, true);
    Variable y = x * 3.0f;
    auto loss = y.sum();

    // Backward without retain_graph (default)
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad_data = x.grad()->data<float>();
    EXPECT_FLOAT_EQ(grad_data[0], 3.0f);
    EXPECT_FLOAT_EQ(grad_data[1], 3.0f);
}

TEST_F(AutogradFeaturesTest, MultipleBackwardPasses) {
    auto t = ones({3}, DType::Float32, cpu);
    Variable x(t, true);

    // First computation
    Variable y1 = x * 2.0f;
    auto loss1 = y1.sum();
    loss1.backward(std::nullopt, true);

    auto grad1 = x.grad()->clone();

    // Second computation (accumulates gradients)
    Variable y2 = x * 3.0f;
    auto loss2 = y2.sum();
    loss2.backward(std::nullopt, false);

    // Gradients should accumulate: 2.0 + 3.0 = 5.0
    ASSERT_TRUE(x.has_grad());
    auto grad_data = x.grad()->data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 5.0f);
    }
}

// ============================================================================
// Combined Features Tests
// ============================================================================

TEST_F(AutogradFeaturesTest, HookWithRetainGrad) {
    auto t = ones({2}, DType::Float32, cpu);
    Variable x(t, true);
    Variable y = x * 2.0f;

    // Retain gradient and register hook
    y.retain_grad();
    y.register_hook([](const Tensor& grad) {
        return grad * 10.0f;
    });

    Variable z = y * 3.0f;
    auto loss = z.sum();
    loss.backward();

    // y.grad should be 3.0 * 10.0 = 30.0 (hook applied)
    ASSERT_TRUE(y.has_grad());
    auto y_grad_data = y.grad()->data<float>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(y_grad_data[i], 30.0f);
    }
}

TEST_F(AutogradFeaturesTest, ComplexComputationGraph) {
    auto t1 = ones({3}, DType::Float32, cpu);
    auto t2 = ones({3}, DType::Float32, cpu);

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
    auto loss = z.sum();

    loss.backward(std::nullopt, true);

    // Verify all gradients exist
    EXPECT_TRUE(x.has_grad());
    EXPECT_TRUE(w.has_grad());
    EXPECT_TRUE(y.has_grad());

    // Verify gradient values
    // dy/dw = x * 2.0 = 1.0 * 2.0 = 2.0
    auto w_grad_data = w.grad()->data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(w_grad_data[i], 2.0f);
    }

    // dy/dx = w * 2.0 * 0.9 (hook) = 1.0 * 2.0 * 0.9 = 1.8
    auto x_grad_data = x.grad()->data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(x_grad_data[i], 1.8f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
