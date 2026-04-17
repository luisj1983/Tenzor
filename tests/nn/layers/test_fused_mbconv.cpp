/**
 * @file test_fused_mbconv.cpp
 * @brief Tests for FusedMBConv (Fused Mobile Inverted Bottleneck) layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/mobilenet.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class FusedMBConvTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(FusedMBConvTest, ForwardShape) {
    FusedMBConv block(16, 32, /*expand_ratio=*/4, /*stride=*/1);
    auto input = Variable(randn({1, 16, 8, 8}, DType::Float32, Device::cpu()), false);
    auto output = block.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 32);
    ASSERT_EQ(shape[2], 8);
    ASSERT_EQ(shape[3], 8);
}

TEST_F(FusedMBConvTest, ForwardWithStride2) {
    FusedMBConv block(16, 32, 4, /*stride=*/2);
    auto input = Variable(randn({1, 16, 8, 8}, DType::Float32, Device::cpu()), false);
    auto output = block.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 32);
    ASSERT_EQ(shape[2], 4);  // 8/2
    ASSERT_EQ(shape[3], 4);
}

TEST_F(FusedMBConvTest, Backward) {
    FusedMBConv block(8, 8, 2, 1);
    auto input = Variable(randn({1, 8, 4, 4}, DType::Float32, Device::cpu()), true);
    auto output = block.forward(input);
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
}
