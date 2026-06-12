/**
 * @file test_autograd_edge_cases.cpp
 * @brief Tests for autograd edge cases
 *
 * Covers:
 * - x*x gradient (should be 2*x)
 * - Detach mid-graph
 * - Gradient accumulation
 * - retain_graph=true multi-backward
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"
#include <cmath>

using namespace tenzor;

class AutogradEdgeCaseTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// 1. x*x Gradient (Should Be 2*x)
// ============================================================================

TEST_P(AutogradEdgeCaseTest, SquareGradient) {
    // f(x) = x*x, df/dx = 2*x
    auto x_host = zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = x_host.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    ptr[3] = -1.0f;
    auto x_data = x_host.to(device);

    Variable x(x_data, true);
    auto y = x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    auto* g = grad.data<float>();

    // Gradient of sum(x*x) w.r.t. x should be 2*x
    EXPECT_NEAR(g[0], 2.0f, 1e-5f);   // 2*1 = 2
    EXPECT_NEAR(g[1], 4.0f, 1e-5f);   // 2*2 = 4
    EXPECT_NEAR(g[2], 6.0f, 1e-5f);   // 2*3 = 6
    EXPECT_NEAR(g[3], -2.0f, 1e-5f);  // 2*(-1) = -2
}

TEST_P(AutogradEdgeCaseTest, SquareGradientScalar) {
    // Single element: f(x) = x^2, df/dx = 2x at x=5
    auto x_data = full({1}, 5.0f, DType::Float32, device);
    Variable x(x_data, true);

    auto y = x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    EXPECT_NEAR(grad.data<float>()[0], 10.0f, 1e-5f);  // 2*5 = 10
}

TEST_P(AutogradEdgeCaseTest, CubeGradient) {
    // f(x) = x*x*x, df/dx = 3*x^2
    auto x_host = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = x_host.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = -2.0f;
    auto x_data = x_host.to(device);

    Variable x(x_data, true);
    auto y = x * x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    auto* g = grad.data<float>();

    // For x*x*x: gradient is 3*x^2 (by chain rule through the product)
    EXPECT_NEAR(g[0], 3.0f, 1e-4f);    // 3*1^2 = 3
    EXPECT_NEAR(g[1], 12.0f, 1e-4f);   // 3*2^2 = 12
    EXPECT_NEAR(g[2], 12.0f, 1e-4f);   // 3*(-2)^2 = 12
}

// ============================================================================
// 2. Detach Mid-Graph
// ============================================================================

TEST_P(AutogradEdgeCaseTest, DetachMidGraph) {
    // x -> y = 2*x -> y_detached (no grad) -> z = y_detached + 1 -> sum
    // Gradient should NOT flow back to x because y is detached
    auto x_data = ones({3}, DType::Float32, device);
    Variable x(x_data, true);

    auto y = x * 2.0f;
    auto y_detached = y.detach();

    // y_detached should not require grad
    EXPECT_FALSE(y_detached.requires_grad());
    EXPECT_TRUE(y_detached.grad_fn() == nullptr);

    // Create a new computation from detached variable
    Variable y_new(y_detached.tensor(), true);
    auto z = y_new + 1.0f;
    auto loss = tenzor::sum(z);
    loss.backward();

    // x should NOT have gradient (graph was broken by detach)
    EXPECT_FALSE(x.has_grad());

    // y_new should have gradient (it's a leaf in the new graph)
    ASSERT_TRUE(y_new.has_grad());
    auto y_new_grad = y_new.grad()->to(Device::cpu());
    auto* g = y_new_grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g[i], 1.0f, 1e-5f);
    }
}

TEST_P(AutogradEdgeCaseTest, DetachPreservesData) {
    auto x_host = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = x_host.data<float>();
    ptr[0] = 1.0f; ptr[1] = 2.0f; ptr[2] = 3.0f;
    auto x_data = x_host.to(device);

    Variable x(x_data, true);
    auto y = x * 3.0f;
    auto y_detached = y.detach();

    // Data should be preserved
    auto det_data = y_detached.tensor().to(Device::cpu());
    auto* d = det_data.data<float>();
    EXPECT_NEAR(d[0], 3.0f, 1e-5f);
    EXPECT_NEAR(d[1], 6.0f, 1e-5f);
    EXPECT_NEAR(d[2], 9.0f, 1e-5f);
}

// ============================================================================
// 3. Gradient Accumulation
// ============================================================================

TEST_P(AutogradEdgeCaseTest, GradientAccumulation) {
    // Accumulate gradients across multiple backward passes
    auto x_data = ones({3}, DType::Float32, device);
    Variable x(x_data, true);

    // First backward pass: loss = sum(x * 2)
    auto y1 = x * 2.0f;
    auto loss1 = tenzor::sum(y1);
    loss1.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad1 = x.grad()->to(Device::cpu());
    auto* g1 = grad1.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g1[i], 2.0f, 1e-5f);
    }

    // Second backward pass WITHOUT zeroing grad: loss = sum(x * 3)
    // Gradients should accumulate: 2 + 3 = 5
    auto y2 = x * 3.0f;
    auto loss2 = tenzor::sum(y2);
    loss2.backward();

    auto grad2 = x.grad()->to(Device::cpu());
    auto* g2 = grad2.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g2[i], 5.0f, 1e-5f) << "Gradient accumulation failed at index " << i;
    }
}

TEST_P(AutogradEdgeCaseTest, ZeroGradThenBackward) {
    auto x_data = ones({3}, DType::Float32, device);
    Variable x(x_data, true);

    // First backward
    auto y1 = x * 2.0f;
    auto loss1 = tenzor::sum(y1);
    loss1.backward();

    EXPECT_GRAD_FLOWS(x);

    // Zero grad
    x.zero_grad();
    EXPECT_FALSE(x.has_grad());

    // Second backward
    auto y2 = x * 5.0f;
    auto loss2 = tenzor::sum(y2);
    loss2.backward();

    // Gradient should be only from second pass: 5.0
    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    auto* g = grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g[i], 5.0f, 1e-5f);
    }
}

// ============================================================================
// 4. retain_graph=true Multi-Backward
// ============================================================================

TEST_P(AutogradEdgeCaseTest, RetainGraphMultiBackward) {
    auto x_host = ones({3}, DType::Float32, Device::cpu());
    auto* ptr = x_host.data<float>();
    ptr[0] = 1.0f; ptr[1] = 2.0f; ptr[2] = 3.0f;
    auto x_data = x_host.to(device);

    Variable x(x_data, true);
    auto y = x * x;  // y = x^2
    auto loss = tenzor::sum(y);

    // First backward with retain_graph=true
    loss.backward(std::nullopt, /*retain_graph=*/true);

    ASSERT_TRUE(x.has_grad());
    auto grad1 = x.grad()->to(Device::cpu());
    auto* g1 = grad1.data<float>();
    EXPECT_NEAR(g1[0], 2.0f, 1e-5f);
    EXPECT_NEAR(g1[1], 4.0f, 1e-5f);
    EXPECT_NEAR(g1[2], 6.0f, 1e-5f);

    // Second backward (gradients accumulate on top of first)
    loss.backward(std::nullopt, /*retain_graph=*/false);

    auto grad2 = x.grad()->to(Device::cpu());
    auto* g2 = grad2.data<float>();
    // Accumulated: 2*x + 2*x = 4*x
    EXPECT_NEAR(g2[0], 4.0f, 1e-5f);   // 2*1 + 2*1
    EXPECT_NEAR(g2[1], 8.0f, 1e-5f);   // 2*2 + 2*2
    EXPECT_NEAR(g2[2], 12.0f, 1e-5f);  // 2*3 + 2*3
}

TEST_P(AutogradEdgeCaseTest, RetainGraphWithZeroGrad) {
    auto x_data = full({2}, 3.0f, DType::Float32, device);
    Variable x(x_data, true);

    auto y = x * 2.0f;
    auto loss = tenzor::sum(y);

    // First backward with retain_graph
    loss.backward(std::nullopt, /*retain_graph=*/true);
    ASSERT_TRUE(x.has_grad());
    auto* g1 = x.grad()->to(Device::cpu()).data<float>();
    EXPECT_NEAR(g1[0], 2.0f, 1e-5f);

    // Zero grad, then backward again
    x.zero_grad();
    EXPECT_FALSE(x.has_grad());

    loss.backward(std::nullopt, /*retain_graph=*/false);
    ASSERT_TRUE(x.has_grad());
    auto* g2 = x.grad()->to(Device::cpu()).data<float>();
    // Should be fresh gradient: just 2.0 (not accumulated)
    EXPECT_NEAR(g2[0], 2.0f, 1e-5f);
}

// ============================================================================
// 5. Complex Graph Patterns
// ============================================================================

TEST_P(AutogradEdgeCaseTest, DiamondGraph) {
    // x -> y = x + 1
    //   -> z = x * 2
    // loss = sum(y + z) = sum(x + 1 + 2*x) = sum(3*x + 1)
    // dL/dx = 3
    auto x_data = ones({4}, DType::Float32, device);
    Variable x(x_data, true);

    auto y = x + 1.0f;
    auto z = x * 2.0f;
    auto combined = y + z;
    auto loss = tenzor::sum(combined);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    auto* g = grad.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(g[i], 3.0f, 1e-5f) << "Diamond graph gradient incorrect at index " << i;
    }
}

TEST_P(AutogradEdgeCaseTest, VariableReuse) {
    // Use same variable twice in an expression: y = x + x
    // dL/dx should be 2 (sum of both paths)
    auto x_data = ones({3}, DType::Float32, device);
    Variable x(x_data, true);

    auto y = x + x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    auto* g = grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g[i], 2.0f, 1e-5f);
    }
}

// ============================================================================
// 6. Gradient with Zero Input
// ============================================================================

TEST_P(AutogradEdgeCaseTest, GradientAtZero) {
    // f(x) = x*x at x=0 -> gradient should be 0
    auto x_data = zeros({3}, DType::Float32, device);
    Variable x(x_data, true);

    auto y = x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    auto* g = grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g[i], 0.0f, 1e-5f) << "Gradient at x=0 should be 0";
    }
}

// ============================================================================
// 7. No Grad Context
// ============================================================================

TEST_P(AutogradEdgeCaseTest, NoGradGuardPreventsGradient) {
    auto x_data = ones({3}, DType::Float32, device);
    Variable x(x_data, true);

    Variable y;
    {
        NoGradGuard guard;
        y = x * 2.0f;
    }

    // y was created under no_grad, so it should NOT have a grad_fn
    EXPECT_TRUE(y.grad_fn() == nullptr);
}

// ============================================================================
// 8. Leaf Variable Properties
// ============================================================================

TEST_P(AutogradEdgeCaseTest, LeafVariableIsLeaf) {
    auto data = ones({3}, DType::Float32, device);
    Variable x(data, true);
    EXPECT_TRUE(x.is_leaf());
}

TEST_P(AutogradEdgeCaseTest, NonLeafVariableIsNotLeaf) {
    auto data = ones({3}, DType::Float32, device);
    Variable x(data, true);
    auto y = x * 2.0f;
    EXPECT_FALSE(y.is_leaf());
}

TEST_P(AutogradEdgeCaseTest, RetainGradOnNonLeaf) {
    auto data = ones({3}, DType::Float32, device);
    Variable x(data, true);
    auto y = x * 2.0f;

    y.retain_grad();
    auto loss = tenzor::sum(y);
    loss.backward();

    // y should have gradient because of retain_grad
    ASSERT_TRUE(y.has_grad());
    auto y_grad = y.grad()->to(Device::cpu());
    auto* g = y_grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g[i], 1.0f, 1e-5f);
    }

    // x should also have gradient
    ASSERT_TRUE(x.has_grad());
    auto x_grad = x.grad()->to(Device::cpu());
    auto* xg = x_grad.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(xg[i], 2.0f, 1e-5f);
    }
}

// ============================================================================
// 9. Matmul Gradient
// ============================================================================

TEST_P(AutogradEdgeCaseTest, MatmulGradient) {
    // y = matmul(x, w), loss = sum(y)
    // dy/dx = w^T, dy/dw = x^T
    auto x_data = ones({2, 3}, DType::Float32, device);
    auto w_host = ones({3, 4}, DType::Float32, Device::cpu());
    auto* wp = w_host.data<float>();
    for (int i = 0; i < 12; ++i) {
        wp[i] = static_cast<float>(i + 1) * 0.1f;
    }
    auto w_data = w_host.to(device);

    Variable x(x_data, true);
    Variable w(w_data, true);

    auto y = tenzor::matmul(x, w);
    auto loss = tenzor::sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
    EXPECT_GRAD_FLOWS(w);

    // Check gradient shapes
    auto x_grad = x.grad()->to(Device::cpu());
    auto w_grad = w.grad()->to(Device::cpu());
    EXPECT_EQ(x_grad.shape()[0], 2);
    EXPECT_EQ(x_grad.shape()[1], 3);
    EXPECT_EQ(w_grad.shape()[0], 3);
    EXPECT_EQ(w_grad.shape()[1], 4);

    // dL/dx = ones(2,4) @ w^T should give each row = sum of columns of w
    // dL/dw = x^T @ ones(2,4) should give each col = sum of rows of x
    auto* xg = x_grad.data<float>();
    auto* wg = w_grad.data<float>();

    // Verify gradients are finite
    for (int i = 0; i < 6; ++i) {
        EXPECT_FALSE(std::isnan(xg[i])) << "x gradient NaN at index " << i;
        EXPECT_FALSE(std::isinf(xg[i])) << "x gradient Inf at index " << i;
    }
    for (int i = 0; i < 12; ++i) {
        EXPECT_FALSE(std::isnan(wg[i])) << "w gradient NaN at index " << i;
        EXPECT_FALSE(std::isinf(wg[i])) << "w gradient Inf at index " << i;
    }
}

INSTANTIATE_BACKEND_TESTS(AutogradEdgeCaseTest);
