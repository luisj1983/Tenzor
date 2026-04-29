/**
 * @file test_dropout3d.cpp
 * @brief Tests for Dropout3d (spatial 3D dropout) layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class Dropout3dTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(Dropout3dTest, EvalModePassthrough) {
    Dropout3d dp(0.5);
    dp.eval();
    auto input = Variable(randn({2, 3, 4, 4, 4}, DType::Float32, Device::cpu()), false);
    auto output = dp.forward(input);
    auto in_data = input.tensor().data<float>();
    auto out_data = output.tensor().data<float>();
    for (int64_t i = 0; i < input.tensor().numel(); ++i) {
        EXPECT_EQ(in_data[i], out_data[i]);
    }
}

TEST_F(Dropout3dTest, OutputShape) {
    Dropout3d dp(0.5);
    dp.train();
    // N=2, C=3, D=4, H=4, W=4
    auto input = Variable(randn({2, 3, 4, 4, 4}, DType::Float32, Device::cpu()), false);
    auto output = dp.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 5);
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);
}

TEST_F(Dropout3dTest, ChannelwiseDropout) {
    Dropout3d dp(0.99);  // Very high drop rate
    dp.train();
    auto input = Variable(ones({1, 10, 2, 2, 2}, DType::Float32, Device::cpu()), false);
    auto output = dp.forward(input);
    // With p=0.99, most channels should be zeroed out
    // Check that dropped channels have ALL spatial values zeroed
    auto out_data = output.tensor().data<float>();
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

TEST_F(Dropout3dTest, Backward) {
    // Dropout3d zeros entire channels — use enough channels so the probability
    // that every channel in every sample is dropped is negligible (keeps
    // EXPECT_GRAD_FLOWS deterministic).
    Dropout3d dp(0.3);
    dp.train();
    auto input = Variable(randn({2, 16, 4, 4, 4}, DType::Float32, Device::cpu()), true);
    auto output = dp.forward(input);
    auto loss = sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}
