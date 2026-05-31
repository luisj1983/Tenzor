/**
 * @file test_fused_mbconv.cpp
 * @brief Tests for FusedMBConv (Fused Mobile Inverted Bottleneck) layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/mobilenet.hpp>

#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class FusedMBConvTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(FusedMBConvTest, ForwardShape) {
    FusedMBConv block(16, 32, /*expand_ratio=*/4, /*stride=*/1);
    block.to(device);
    auto input = Variable(randn({1, 16, 8, 8}, DType::Float32, device), false);
    auto output = block.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 32);
    ASSERT_EQ(shape[2], 8);
    ASSERT_EQ(shape[3], 8);
}

TEST_P(FusedMBConvTest, ForwardWithStride2) {
    FusedMBConv block(16, 32, 4, /*stride=*/2);
    block.to(device);
    auto input = Variable(randn({1, 16, 8, 8}, DType::Float32, device), false);
    auto output = block.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 32);
    ASSERT_EQ(shape[2], 4);  // 8/2
    ASSERT_EQ(shape[3], 4);
}

TEST_P(FusedMBConvTest, Backward) {
    FusedMBConv block(8, 8, 2, 1);
    block.to(device);
    auto input = Variable(randn({1, 8, 4, 4}, DType::Float32, device), true);
    auto output = block.forward(input);
    auto loss = sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

INSTANTIATE_BACKEND_TESTS(FusedMBConvTest);
