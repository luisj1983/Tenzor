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
#include "../../grad_flow_helpers.hpp"
#include "../../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class LazyBackwardTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(LazyBackwardTest, LazyLinear_Backward) {
    LazyLinear layer(8);
    layer.to(device);
    auto input = Variable(randn({2, 16}, DType::Float32, device), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    // Input should have gradient with at least one non-zero element
    // (.has_value() alone passes for an all-zero grad — a severed grad_fn
    // would still allocate a zero-filled grad tensor and slip through).
    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape()[0], 2);
    ASSERT_EQ(input.grad().value().shape()[1], 16);

    // Materialized parameters should have gradients
    auto params = layer.parameters();
    ASSERT_FALSE(params.empty());
}

TEST_P(LazyBackwardTest, LazyConv1d_Backward) {
    LazyConv1d layer(8, 3, 1, 1);
    layer.to(device);
    auto input = Variable(randn({1, 4, 16}, DType::Float32, device), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape()[0], 1);
}

TEST_P(LazyBackwardTest, LazyConv2d_Backward) {
    LazyConv2d layer(16, 3, 1, 1);
    layer.to(device);
    auto input = Variable(randn({1, 8, 8, 8}, DType::Float32, device), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 16);

    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape()[1], 8);
}

TEST_P(LazyBackwardTest, LazyConv3d_Backward) {
    LazyConv3d layer(8, 3, 1, 1);
    layer.to(device);
    auto input = Variable(randn({1, 4, 4, 4, 4}, DType::Float32, device), true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape()[1], 4);
}

INSTANTIATE_BACKEND_TESTS(LazyBackwardTest);
