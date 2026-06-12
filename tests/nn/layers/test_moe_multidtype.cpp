/**
 * @file test_moe_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Mixture of Experts layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/layers/moe.hpp>
#include <tenzor/autograd/variable.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class MoEMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(MoEMultiDTypeTest, Construction) {
    EXPECT_NO_THROW({
        MixtureOfExperts moe(64, 128, 4, 2);
        EXPECT_EQ(moe.num_experts(), 4);
        EXPECT_EQ(moe.top_k(), 2);
    });
}

TEST_P(MoEMultiDTypeTest, ConstructionWithParams) {
    MixtureOfExperts moe(64, 128, 8, 2, 1.25, 0.01, 0.1);
    EXPECT_EQ(moe.num_experts(), 8);
    EXPECT_EQ(moe.top_k(), 2);
    EXPECT_DOUBLE_EQ(moe.capacity_factor(), 1.25);
}

TEST_P(MoEMultiDTypeTest, ForwardShape2D) {
    MixtureOfExperts moe(32, 64, 4, 2);
    convert_model(moe);

    auto input = createInput({8, 32}, false);
    auto output = moe.forward(input);
    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 32);
}

TEST_P(MoEMultiDTypeTest, ForwardShape3D) {
    MixtureOfExperts moe(32, 64, 4, 2);
    convert_model(moe);

    auto input = createInput({2, 4, 32}, false);
    auto output = moe.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 32);
}

TEST_P(MoEMultiDTypeTest, ForwardWithLoss) {
    MixtureOfExperts moe(32, 64, 4, 2);
    convert_model(moe);

    auto input = createInput({8, 32}, false);
    auto [output, aux_loss] = moe.forward_with_loss(input);

    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_LE(aux_loss.tensor().numel(), 1);
}

TEST_P(MoEMultiDTypeTest, AuxLossNonNegative) {
    MixtureOfExperts moe(32, 64, 4, 2, 1.25, 0.01);
    convert_model(moe);

    auto input = createInput({16, 32}, false);
    auto [_, aux_loss] = moe.forward_with_loss(input);

    auto loss_cpu = aux_loss.tensor().to(Device::cpu()).to(DType::Float32);
    float loss_val = loss_cpu.template item<float>();
    EXPECT_GE(loss_val, 0.0f) << "Auxiliary loss should be non-negative";
}

TEST_P(MoEMultiDTypeTest, OutputFinite) {
    MixtureOfExperts moe(32, 64, 4, 2);
    convert_model(moe);

    auto input = createInput({4, 32}, false);
    auto output = moe.forward(input);

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

// Phase 3 addition (Phase 3-followup #21 fix): MoE forward used raw tensor
// ops + Tensor accumulator throughout, dropping the autograd graph from
// input → expert outputs. Fixed in src/nn/layers/moe.cpp by switching to
// Variable-level reshape + accumulator + masking.
TEST_P(MoEMultiDTypeTest, BackwardGradPopulated) {
    MixtureOfExperts moe(16, 32, 4, 2);
    convert_model(moe);
    auto input = createInput({4, 16}, true);
    auto output = moe.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MoEMultiDTypeTest);
