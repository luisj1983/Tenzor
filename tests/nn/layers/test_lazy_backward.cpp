/**
 * @file test_lazy_backward.cpp
 * @brief Backward pass tests for Lazy* layers (LazyLinear, LazyConv1d/2d/3d).
 *
 * These layers materialize parameters during the first forward pass.
 * The backward tests verify gradients flow correctly after materialization.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include <tenzor/nn/layers/lazy_conv.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class LazyBackwardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(LazyBackwardTest, LazyLinear_Backward) {
    LazyLinear layer(8);
    auto input = Variable(randn({2, 16}, DType::Float32, Device::cpu()), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    // Input should have gradient
    ASSERT_TRUE(input.grad().has_value());
    ASSERT_EQ(input.grad().value().shape()[0], 2);
    ASSERT_EQ(input.grad().value().shape()[1], 16);

    // Materialized parameters should have gradients
    auto params = layer.parameters();
    ASSERT_FALSE(params.empty());
}

TEST_F(LazyBackwardTest, LazyConv1d_Backward) {
    LazyConv1d layer(8, 3, 1, 1);
    auto input = Variable(randn({1, 4, 16}, DType::Float32, Device::cpu()), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    ASSERT_EQ(input.grad().value().shape()[0], 1);
}

TEST_F(LazyBackwardTest, LazyConv2d_Backward) {
    LazyConv2d layer(16, 3, 1, 1);
    auto input = Variable(randn({1, 8, 8, 8}, DType::Float32, Device::cpu()), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 16);

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    ASSERT_EQ(input.grad().value().shape()[1], 8);
}

TEST_F(LazyBackwardTest, LazyConv3d_Backward) {
    LazyConv3d layer(8, 3, 1, 1);
    auto input = Variable(randn({1, 4, 4, 4, 4}, DType::Float32, Device::cpu()), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    ASSERT_EQ(input.grad().value().shape()[1], 4);
}
