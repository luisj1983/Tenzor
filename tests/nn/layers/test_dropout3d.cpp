/**
 * @file test_dropout3d.cpp
 * @brief Tests for Dropout3d (spatial 3D dropout) layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class Dropout3dTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(Dropout3dTest, EvalModePassthrough) {
    Dropout3d dp(0.5);
    dp.to(device);
    dp.eval();
    auto input = Variable(randn({2, 3, 4, 4, 4}, DType::Float32, device), false);
    auto output = dp.forward(input);
    auto in_cpu = input.tensor().cpu();
    auto out_cpu = output.tensor().cpu();
    auto in_data = in_cpu.data<float>();
    auto out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_EQ(in_data[i], out_data[i]);
    }
}

TEST_P(Dropout3dTest, OutputShape) {
    Dropout3d dp(0.5);
    dp.to(device);
    dp.train();
    // N=2, C=3, D=4, H=4, W=4
    auto input = Variable(randn({2, 3, 4, 4, 4}, DType::Float32, device), false);
    auto output = dp.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 5);
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);
}

TEST_P(Dropout3dTest, ChannelwiseDropout) {
    Dropout3d dp(0.99);  // Very high drop rate
    dp.to(device);
    dp.train();
    auto input = Variable(ones({1, 10, 2, 2, 2}, DType::Float32, device), false);
    auto output = dp.forward(input);
    // With p=0.99, most channels should be zeroed out
    // Check that dropped channels have ALL spatial values zeroed
    auto out_cpu = output.tensor().cpu();
    auto out_data = out_cpu.data<float>();
    for (int64_t c = 0; c < 10; ++c) {
        bool first_val_zero = (out_data[c * 8] == 0.0f);
        // If channel is dropped, ALL spatial values should be zero
        if (first_val_zero) {
            for (int64_t s = 0; s < 8; ++s) {
                EXPECT_EQ(out_data[c * 8 + s], 0.0f)
                    << "Channel " << c << " spatial " << s << " should be zero";
            }
        }
    }
}

TEST_P(Dropout3dTest, Backward) {
    // Dropout3d zeros entire channels — use enough channels so the probability
    // that every channel in every sample is dropped is negligible (keeps
    // EXPECT_GRAD_FLOWS deterministic).
    Dropout3d dp(0.3);
    dp.to(device);
    dp.train();
    auto input = Variable(randn({2, 16, 4, 4, 4}, DType::Float32, device), true);
    auto output = dp.forward(input);
    auto loss = sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

INSTANTIATE_BACKEND_TESTS(Dropout3dTest);
