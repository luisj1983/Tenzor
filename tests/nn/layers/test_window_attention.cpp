/**
 * @file test_window_attention.cpp
 * @brief Tests for WindowAttention layer (Swin Transformer).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/vision.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class WindowAttentionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(WindowAttentionTest, ForwardShape) {
    // dim=32, window_size=7, num_heads=4
    WindowAttention wa(32, 7, 4);
    // Input: (batch * num_windows, window_size*window_size, dim)
    auto input = Variable(randn({4, 49, 32}, DType::Float32, Device::cpu()), false);
    auto output = wa.forward(input, Tensor{});
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 3);
    ASSERT_EQ(shape[0], 4);
    ASSERT_EQ(shape[1], 49);
    ASSERT_EQ(shape[2], 32);
}

TEST_F(WindowAttentionTest, Backward) {
    WindowAttention wa(16, 4, 2);
    auto input = Variable(randn({2, 16, 16}, DType::Float32, Device::cpu()), true);
    auto output = wa.forward(input, Tensor{});
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
    ASSERT_EQ(input.grad().value().shape()[0], 2);
}
