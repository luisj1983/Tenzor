/**
 * @file test_graph_cleanup_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for computation graph cleanup after backward
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GraphCleanupMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(GraphCleanupMultiDTypeTest, RootGradFnClearedAfterBackward) {
    auto x = createInput({3, 4}, true);
    auto y = x * x;
    auto loss = tenzor::sum(y);

    ASSERT_NE(loss.grad_fn(), nullptr);
    loss.backward();
    EXPECT_EQ(loss.grad_fn(), nullptr);
}

TEST_P(GraphCleanupMultiDTypeTest, LeafGradientsPreservedAfterCleanup) {
    auto x = createInput({2, 3}, true);
    auto y = x * x + x;
    auto loss = tenzor::sum(y);

    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.numel(), 6);
    expectDevice(grad);
}

TEST_P(GraphCleanupMultiDTypeTest, RetainGraphPreservesGradFn) {
    auto x = createInput({2, 2}, true);
    auto y = x * x;
    auto loss = tenzor::sum(y);

    loss.backward(std::nullopt, /*retain_graph=*/true);
    EXPECT_NE(loss.grad_fn(), nullptr) << "grad_fn should be preserved with retain_graph";
    ASSERT_TRUE(x.has_grad());
}

TEST_P(GraphCleanupMultiDTypeTest, DeepGraphCleanup) {
    auto x = createInput({4}, true);
    Variable current = x;
    auto one = Variable(createOnes({4}), false);
    for (int i = 0; i < 10; ++i) {
        current = current * current + one;
    }
    auto loss = tenzor::sum(current);

    ASSERT_NE(loss.grad_fn(), nullptr);
    loss.backward();
    EXPECT_EQ(loss.grad_fn(), nullptr);
    ASSERT_TRUE(x.has_grad());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GraphCleanupMultiDTypeTest);
