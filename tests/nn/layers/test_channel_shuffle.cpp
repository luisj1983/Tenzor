#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

class ChannelShuffleTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Test basic forward pass shape preservation
TEST_F(ChannelShuffleTest, ForwardShapePreserved) {
    ChannelShuffle cs(2);
    auto input = Variable(randn({1, 4, 8, 8}, DType::Float32, Device::cpu()), false);
    auto output = cs.forward(input);

    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

// Test that channels are actually shuffled
TEST_F(ChannelShuffleTest, ChannelReordering) {
    ChannelShuffle cs(2);

    // Create input with distinct values per channel: channel i filled with float(i)
    auto t = zeros({1, 4, 1, 1}, DType::Float32, Device::cpu());
    float* data = t.data<float>();
    data[0] = 0.0f;  // Group 0, channel 0
    data[1] = 1.0f;  // Group 0, channel 1
    data[2] = 2.0f;  // Group 1, channel 0
    data[3] = 3.0f;  // Group 1, channel 1

    auto input = Variable(t, false);
    auto output = cs.forward(input);
    auto out_cpu = output.tensor().to(Device::cpu());
    const float* out_data = out_cpu.data<float>();

    // With groups=2: (G=2, C/G=2) -> reshape to (1,2,2,1,1) -> permute -> (1,2,2,1,1)
    // Original order: [0,1,2,3] -> After shuffle: [0,2,1,3]
    EXPECT_FLOAT_EQ(out_data[0], 0.0f);
    EXPECT_FLOAT_EQ(out_data[1], 2.0f);
    EXPECT_FLOAT_EQ(out_data[2], 1.0f);
    EXPECT_FLOAT_EQ(out_data[3], 3.0f);
}

// Test gradient flow
TEST_F(ChannelShuffleTest, BackwardGradient) {
    ChannelShuffle cs(2);
    auto input = Variable(randn({2, 4, 3, 3}, DType::Float32, Device::cpu()), true);
    auto output = cs.forward(input);

    // Backward with ones gradient
    std::vector<int64_t> out_shape(output.shape().begin(), output.shape().end());
    auto grad_tensor = ones(out_shape, DType::Float32, Device::cpu());
    output.backward(grad_tensor);

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    // All gradients should be 1.0 since channel shuffle is a permutation
    auto grad_cpu = grad.to(Device::cpu());
    const float* grad_data = grad_cpu.data<float>();
    for (size_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_NEAR(grad_data[i], 1.0f, 1e-6) << "Gradient mismatch at index " << i;
    }
}

// Test that shuffle then unshuffle is identity
TEST_F(ChannelShuffleTest, DoubleShuffleWithInverseGroups) {
    int64_t groups = 3;
    int64_t channels = 6;
    int64_t cpg = channels / groups;  // 2

    ChannelShuffle cs1(groups);
    ChannelShuffle cs2(cpg);  // Inverse shuffle

    auto input = Variable(randn({1, channels, 4, 4}, DType::Float32, Device::cpu()), false);
    auto shuffled = cs1.forward(input);
    auto restored = cs2.forward(shuffled);

    auto in_data = input.tensor().to(Device::cpu()).data<float>();
    auto out_data = restored.tensor().to(Device::cpu()).data<float>();
    for (size_t i = 0; i < input.tensor().numel(); ++i) {
        EXPECT_NEAR(in_data[i], out_data[i], 1e-6);
    }
}

// Test invalid groups
TEST_F(ChannelShuffleTest, InvalidGroupsThrows) {
    EXPECT_THROW(ChannelShuffle(0), std::invalid_argument);
}

// Test channels not divisible by groups
TEST_F(ChannelShuffleTest, IndivisibleChannelsThrows) {
    ChannelShuffle cs(3);
    auto input = Variable(randn({1, 4, 8, 8}, DType::Float32, Device::cpu()), false);
    EXPECT_THROW(cs.forward(input), std::invalid_argument);
}

// Test with groups=1 (identity)
TEST_F(ChannelShuffleTest, GroupsOneIsIdentity) {
    ChannelShuffle cs(1);
    auto input = Variable(randn({2, 6, 4, 4}, DType::Float32, Device::cpu()), false);
    auto output = cs.forward(input);

    auto in_data = input.tensor().to(Device::cpu()).data<float>();
    auto out_data = output.tensor().to(Device::cpu()).data<float>();
    for (size_t i = 0; i < input.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(in_data[i], out_data[i]);
    }
}

// Test with 3D input (no batch dim)
TEST_F(ChannelShuffleTest, ThreeDimensionalInput) {
    ChannelShuffle cs(2);
    auto input = Variable(randn({4, 8, 8}, DType::Float32, Device::cpu()), false);
    auto output = cs.forward(input);

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 8);
}

// Test input too few dimensions
TEST_F(ChannelShuffleTest, TwoDimensionalThrows) {
    ChannelShuffle cs(2);
    auto input = Variable(randn({4, 8}, DType::Float32, Device::cpu()), false);
    EXPECT_THROW(cs.forward(input), std::invalid_argument);
}
