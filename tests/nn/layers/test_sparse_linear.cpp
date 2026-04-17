/**
 * @file test_sparse_linear.cpp
 * @brief Tests for SparseLinear layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sparse_linear.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class SparseLinearTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(SparseLinearTest, ForwardShape) {
    SparseLinear sl(16, 8, /*density=*/0.5, /*bias=*/true);
    auto input = Variable(randn({4, 16}, DType::Float32, Device::cpu()), false);
    auto output = sl.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 4);
    ASSERT_EQ(output.tensor().shape()[1], 8);
}

TEST_F(SparseLinearTest, ForwardShapeNoBias) {
    SparseLinear sl(16, 8, /*density=*/0.5, /*bias=*/false);
    auto input = Variable(randn({4, 16}, DType::Float32, Device::cpu()), false);
    auto output = sl.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 4);
    ASSERT_EQ(output.tensor().shape()[1], 8);
}

TEST_F(SparseLinearTest, Backward) {
    SparseLinear sl(8, 4, 0.5, true);
    auto input = Variable(randn({2, 8}, DType::Float32, Device::cpu()), true);
    auto output = sl.forward(input);
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
    ASSERT_EQ(input.grad().value().shape()[0], 2);
    ASSERT_EQ(input.grad().value().shape()[1], 8);
}
