/**
 * @file test_drop_path.cpp
 * @brief Tests for DropPath (stochastic depth) layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/drop_path.hpp>

#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class DropPathTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(DropPathTest, EvalModePassthrough) {
    DropPath dp(0.5);
    dp.eval();
    auto input = Variable(randn({4, 8}, DType::Float32, Device::cpu()), false);
    auto output = dp.forward(input);
    // In eval mode, output == input
    auto in_data = input.tensor().data<float>();
    auto out_data = output.tensor().data<float>();
    for (int64_t i = 0; i < input.tensor().numel(); ++i) {
        EXPECT_EQ(in_data[i], out_data[i]);
    }
}

TEST_F(DropPathTest, ZeroDropRatePassthrough) {
    DropPath dp(0.0);
    dp.train();
    auto input = Variable(randn({4, 8}, DType::Float32, Device::cpu()), false);
    auto output = dp.forward(input);
    auto in_data = input.tensor().data<float>();
    auto out_data = output.tensor().data<float>();
    for (int64_t i = 0; i < input.tensor().numel(); ++i) {
        EXPECT_EQ(in_data[i], out_data[i]);
    }
}

TEST_F(DropPathTest, TrainingModeOutputShape) {
    DropPath dp(0.3);
    dp.train();
    auto input = Variable(randn({8, 16}, DType::Float32, Device::cpu()), false);
    auto output = dp.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 8);
    ASSERT_EQ(output.tensor().shape()[1], 16);
}

TEST_F(DropPathTest, Backward) {
    // Use a larger batch so the probability of every sample being dropped is
    // negligible — keeps EXPECT_GRAD_FLOWS deterministic with drop_rate=0.3.
    DropPath dp(0.3);
    dp.train();
    auto input = Variable(randn({16, 8}, DType::Float32, Device::cpu()), true);
    auto output = dp.forward(input);
    auto loss = sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}
