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

INSTANTIATE_BACKEND_TESTS(SparseLinearTest);
