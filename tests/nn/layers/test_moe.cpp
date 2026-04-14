/**
 * @file test_moe.cpp
 * @brief Tests for Mixture of Experts layer
 */

#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/nn/layers/moe.hpp>
#include <tenzor/autograd/variable.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class MoETest : public tenzor::testing::BackendTest {};

TEST_P(MoETest, Construction) {
    EXPECT_NO_THROW({
        MixtureOfExperts moe(64, 128, 4, 2);
        EXPECT_EQ(moe.num_experts(), 4);
        EXPECT_EQ(moe.top_k(), 2);
    });
}

TEST_P(MoETest, Construction_WithParams) {
    MixtureOfExperts moe(64, 128, 8, 2, 1.25, 0.01, 0.1);
    EXPECT_EQ(moe.num_experts(), 8);
    EXPECT_EQ(moe.top_k(), 2);
    EXPECT_DOUBLE_EQ(moe.capacity_factor(), 1.25);
}

TEST_P(MoETest, ForwardShape_2D) {
    MixtureOfExperts moe(32, 64, 4, 2);
    auto input = Variable(randn({8, 32}, DType::Float32, Device::cpu()), false);
    auto output = moe.forward(input);
    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 32);
}

TEST_P(MoETest, ForwardShape_3D) {
    MixtureOfExperts moe(32, 64, 4, 2);
    auto input = Variable(randn({2, 4, 32}, DType::Float32, Device::cpu()), false);
    auto output = moe.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 32);
}

TEST_P(MoETest, ForwardWithLoss) {
    MixtureOfExperts moe(32, 64, 4, 2);
    auto input = Variable(randn({8, 32}, DType::Float32, Device::cpu()), false);
    auto [output, aux_loss] = moe.forward_with_loss(input);

    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 32);

    // aux_loss should be a scalar (0-d or 1-element tensor)
    EXPECT_LE(aux_loss.tensor().numel(), 1);
}

TEST_P(MoETest, AuxLossNonNegative) {
    MixtureOfExperts moe(32, 64, 4, 2, 1.25, 0.01);
    auto input = Variable(randn({16, 32}, DType::Float32, Device::cpu()), false);
    auto [_, aux_loss] = moe.forward_with_loss(input);

    auto loss_cpu = aux_loss.tensor().to(Device::cpu());
    float loss_val = loss_cpu.template item<float>();
    EXPECT_GE(loss_val, 0.0f) << "Auxiliary loss should be non-negative";
}

INSTANTIATE_BACKEND_TESTS(MoETest);
