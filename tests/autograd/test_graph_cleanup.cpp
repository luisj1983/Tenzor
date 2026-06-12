/**
 * @file test_graph_cleanup.cpp
 * @brief Tests that backward() properly cleans up the computation graph.
 *
 * Validates that:
 * 1. Intermediate variables have null grad_fn after backward (without retain_graph)
 * 2. Graph cleanup doesn't break retain_graph mode
 * 3. Deep computation graphs are fully cleaned up
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;

class GraphCleanupTest : public ::testing::Test {
protected:
    static bool initialized_;

    void SetUp() override {
        if (!initialized_) {
            tenzor::initialize();
            initialized_ = true;
        }
    }
};

bool GraphCleanupTest::initialized_ = false;

// After backward without retain_graph, root variable's grad_fn should be null
TEST_F(GraphCleanupTest, RootGradFnClearedAfterBackward) {
    auto x = Variable(randn({3, 4}, DType::Float32, Device::cpu()), true);
    auto y = x * x;
    auto loss = tenzor::sum(y);

    ASSERT_NE(loss.grad_fn(), nullptr) << "loss should have grad_fn before backward";

    loss.backward();

    // After backward without retain_graph, root's grad_fn should be cleared
    EXPECT_EQ(loss.grad_fn(), nullptr)
        << "Root variable grad_fn should be null after backward";
}

// After backward, gradients should still be accessible on leaf variables
TEST_F(GraphCleanupTest, LeafGradientsPreservedAfterCleanup) {
    auto x = Variable(randn({2, 3}, DType::Float32, Device::cpu()), true);
    auto y = x * x + x;
    auto loss = tenzor::sum(y);

    loss.backward();

    EXPECT_GRAD_FLOWS(x);
    // dy/dx = 2x + 1
    auto grad = x.grad().value();
    EXPECT_EQ(grad.numel(), 6);
}

// Deep computation graph should be fully cleaned up
TEST_F(GraphCleanupTest, DeepGraphCleanup) {
    auto x = Variable(randn({4}, DType::Float32, Device::cpu()), true);

    // Build a deep chain: x -> y1 -> y2 -> ... -> y10 -> loss
    Variable result = x;
    for (int i = 0; i < 10; ++i) {
        result = result * result;  // Each creates a MulBackward node
    }
    auto loss = tenzor::sum(result);

    // Before backward, loss should have a grad_fn
    ASSERT_NE(loss.grad_fn(), nullptr);

    loss.backward();

    // After backward, grad_fn should be cleaned up
    EXPECT_EQ(loss.grad_fn(), nullptr);

    // Leaf variable should have gradient
    EXPECT_GRAD_FLOWS(x);
}

// With retain_graph=true, grad_fn should NOT be cleared
TEST_F(GraphCleanupTest, RetainGraphPreservesGradFn) {
    auto x = Variable(randn({3, 4}, DType::Float32, Device::cpu()), true);
    auto y = x * x;
    auto loss = tenzor::sum(y);

    auto grad_fn_before = loss.grad_fn();
    ASSERT_NE(grad_fn_before, nullptr);

    loss.backward(std::nullopt, /*retain_graph=*/true);

    // With retain_graph, grad_fn should still be present
    EXPECT_NE(loss.grad_fn(), nullptr)
        << "grad_fn should be preserved with retain_graph=true";

    // Can backward again
    x.zero_grad();
    loss.backward(std::nullopt, /*retain_graph=*/false);

    EXPECT_GRAD_FLOWS(x);
}

// Diamond graph: x feeds into two paths that merge
TEST_F(GraphCleanupTest, DiamondGraphCleanup) {
    auto x = Variable(randn({2, 3}, DType::Float32, Device::cpu()), true);
    auto a = x * x;     // Path A
    auto b = x + x;     // Path B
    auto c = a + b;     // Merge
    auto loss = tenzor::sum(c);

    loss.backward();

    EXPECT_EQ(loss.grad_fn(), nullptr);
    EXPECT_GRAD_FLOWS(x);

    // Gradient should be dy/dx = 2x + 2 (from x^2 and 2x)
    auto grad = x.grad().value();
    EXPECT_EQ(grad.numel(), 6);
}

// Multiple outputs: backward on one should not break the other
TEST_F(GraphCleanupTest, MultipleBackwardCalls) {
    auto x = Variable(randn({2, 3}, DType::Float32, Device::cpu()), true);

    auto y1 = tenzor::sum(x * x);
    y1.backward();

    // x.grad() should have the gradient from y1
    EXPECT_GRAD_FLOWS(x);
    auto grad1 = x.grad().value();

    // Build a new graph and backward again (gradient accumulates)
    auto y2 = tenzor::sum(x + x);
    y2.backward();

    // Gradient should have accumulated
    EXPECT_GRAD_FLOWS(x);
}
