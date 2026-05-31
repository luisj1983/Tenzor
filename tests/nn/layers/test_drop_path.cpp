/**
 * @file test_drop_path.cpp
 * @brief Tests for DropPath (stochastic depth) layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/drop_path.hpp>

#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class DropPathTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(DropPathTest, EvalModePassthrough) {
    DropPath dp(0.5);
    dp.to(device);
    dp.eval();
    auto input = Variable(randn({4, 8}, DType::Float32, device), false);
    auto output = dp.forward(input);
    // In eval mode, output == input
    auto in_cpu = input.tensor().cpu();
    auto out_cpu = output.tensor().cpu();
    auto in_data = in_cpu.data<float>();
    auto out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_EQ(in_data[i], out_data[i]);
    }
}

TEST_P(DropPathTest, ZeroDropRatePassthrough) {
    DropPath dp(0.0);
    dp.to(device);
    dp.train();
    auto input = Variable(randn({4, 8}, DType::Float32, device), false);
    auto output = dp.forward(input);
    auto in_cpu = input.tensor().cpu();
    auto out_cpu = output.tensor().cpu();
    auto in_data = in_cpu.data<float>();
    auto out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_EQ(in_data[i], out_data[i]);
    }
}

TEST_P(DropPathTest, TrainingModeOutputShape) {
    DropPath dp(0.3);
    dp.to(device);
    dp.train();
    auto input = Variable(randn({8, 16}, DType::Float32, device), false);
    auto output = dp.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 8);
    ASSERT_EQ(output.tensor().shape()[1], 16);
}

TEST_P(DropPathTest, Backward) {
    // Use a larger batch so the probability of every sample being dropped is
    // negligible — keeps EXPECT_GRAD_FLOWS deterministic with drop_rate=0.3.
    DropPath dp(0.3);
    dp.to(device);
    dp.train();
    auto input = Variable(randn({16, 8}, DType::Float32, device), true);
    auto output = dp.forward(input);
    auto loss = sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

INSTANTIATE_BACKEND_TESTS(DropPathTest);
