/**
 * @file test_channel_shuffle_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for ChannelShuffle layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class ChannelShuffleMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ChannelShuffleMultiDTypeTest, ForwardShapePreserved) {
    ChannelShuffle cs(2);
    convert_model(cs);
    auto input = createInput({1, 4, 8, 8}, false);
    auto output = cs.forward(input);

    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST_P(ChannelShuffleMultiDTypeTest, GroupsOneIsIdentity) {
    ChannelShuffle cs(1);
    convert_model(cs);
    auto input = createInput({2, 6, 4, 4}, false);
    auto output = cs.forward(input);

    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* in_data = in_cpu.data<float>();
    auto* out_data = out_cpu.data<float>();
    for (size_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_NEAR(in_data[i], out_data[i], atol());
    }
}

TEST_P(ChannelShuffleMultiDTypeTest, DoubleShuffleWithInverseGroups) {
    int64_t groups = 3;
    int64_t channels = 6;
    int64_t cpg = channels / groups;

    ChannelShuffle cs1(groups);
    ChannelShuffle cs2(cpg);
    convert_model(cs1);
    convert_model(cs2);

    auto input = createInput({1, channels, 4, 4}, false);
    auto shuffled = cs1.forward(input);
    auto restored = cs2.forward(shuffled);

    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto out_cpu = restored.tensor().to(Device::cpu()).to(DType::Float32);
    auto* in_data = in_cpu.data<float>();
    auto* out_data = out_cpu.data<float>();
    for (size_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_NEAR(in_data[i], out_data[i], atol());
    }
}

TEST_P(ChannelShuffleMultiDTypeTest, InvalidGroupsThrows) {
    EXPECT_THROW(ChannelShuffle(0), std::invalid_argument);
}

TEST_P(ChannelShuffleMultiDTypeTest, IndivisibleChannelsThrows) {
    ChannelShuffle cs(3);
    convert_model(cs);
    auto input = createInput({1, 4, 8, 8}, false);
    EXPECT_THROW(cs.forward(input), std::invalid_argument);
}

TEST_P(ChannelShuffleMultiDTypeTest, ThreeDimensionalInput) {
    ChannelShuffle cs(2);
    convert_model(cs);
    auto input = createInput({4, 8, 8}, false);
    auto output = cs.forward(input);

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 8);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ChannelShuffleMultiDTypeTest);
