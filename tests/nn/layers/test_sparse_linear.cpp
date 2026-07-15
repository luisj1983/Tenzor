/**
 * @file test_sparse_linear.cpp
 * @brief Tests for SparseLinear layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sparse_linear.hpp>
#include "../../grad_flow_helpers.hpp"
#include "../../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class SparseLinearTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(SparseLinearTest, ForwardShape) {
    SparseLinear sl(16, 8, /*density=*/0.5, /*bias=*/true);
    sl.to(device);
    auto input = Variable(randn({4, 16}, DType::Float32, device), false);
    auto output = sl.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 4);
    ASSERT_EQ(output.tensor().shape()[1], 8);
}

TEST_P(SparseLinearTest, ForwardShapeNoBias) {
    SparseLinear sl(16, 8, /*density=*/0.5, /*bias=*/false);
    sl.to(device);
    auto input = Variable(randn({4, 16}, DType::Float32, device), false);
    auto output = sl.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 4);
    ASSERT_EQ(output.tensor().shape()[1], 8);
}

TEST_P(SparseLinearTest, Backward) {
    SparseLinear sl(8, 4, 0.5, true);
    sl.to(device);
    auto input = Variable(randn({2, 8}, DType::Float32, device), true);
    auto output = sl.forward(input);
    auto loss = sum(output);
    loss.backward();
    // EXPECT_GRAD_FLOWS asserts non-zero grad — pre-fix, a severed grad_fn
    // chain would still produce a zero-filled grad tensor that passes the
    // shape and has_value() checks but not this magnitude check.
    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape()[0], 2);
    ASSERT_EQ(input.grad().value().shape()[1], 8);
}

TEST_P(SparseLinearTest, DoubleBackward_GradInputHasLiveGraph) {
    // M30 regression: SparseLinearBackward::backward_with_variables() used to
    // unconditionally rewrap backward()'s raw Tensor results as fresh,
    // grad_fn-less Variables — severing grad_outputs[0]'s own graph — while
    // supports_higher_order() claimed full support. grad_input_t = S^T @
    // grad_result_t is exactly linear in grad_result_t (S held as a fixed
    // snapshot), so it can and must stay graph-connected to grad_outputs[0]
    // under create_graph=true. Force grad_outputs[0] (=dL/dy) to itself
    // depend on x (loss nonlinear in y=S@x) so that re-differentiating
    // grad_x = S^T @ dL/dy w.r.t. x a second time is a genuine, nonzero
    // Hessian-vector product — a severed graph collapses this to an exact
    // zero, which EXPECT_GRAD_FLOWS below catches.
    SparseLinear sl(6, 4, 0.6, /*bias=*/false);
    sl.to(device);
    auto x = Variable(randn({3, 6}, DType::Float32, device), true);
    Variable grad_x_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = sl.forward(x);
        auto loss = tenzor::sum(y * y);  // nonlinear in y => dL/dy depends on x
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        ASSERT_TRUE(x.grad_variable().has_value())
            << "create_graph=true must populate grad_variable() for x";
        grad_x_var = x.grad_variable().value();
    }
    x.zero_grad();
    auto grad_norm = tenzor::sum(grad_x_var * grad_x_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(x);
}

INSTANTIATE_BACKEND_TESTS(SparseLinearTest);
