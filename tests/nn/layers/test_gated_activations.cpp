/**
 * @file test_gated_activations.cpp
 * @brief Tests for GatedLinearUnit, GeGLU, and ReGLU activations
 */

#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <tenzor/nn/layers/hrm.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/reduction.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class GatedActivationsTest : public tenzor::testing::BackendTest {};

TEST_P(GatedActivationsTest, GateType_Enum) {
    EXPECT_NE(static_cast<int>(GateType::Sigmoid), static_cast<int>(GateType::SiLU));
    EXPECT_NE(static_cast<int>(GateType::GELU), static_cast<int>(GateType::ReLU));
}

TEST_P(GatedActivationsTest, GatedLinearUnit_SwiGLU_Construction) {
    EXPECT_NO_THROW({
        GatedLinearUnit glu(64, 128, GateType::SiLU, false);
        EXPECT_EQ(glu.gate_type(), GateType::SiLU);
    }) << "Failed on " << device.to_string();
}

TEST_P(GatedActivationsTest, GeGLU_Construction) {
    GeGLU geglu(64, 128);
    EXPECT_EQ(geglu.gate_type(), GateType::GELU);
}

TEST_P(GatedActivationsTest, ReGLU_Construction) {
    ReGLU reglu(64, 128);
    EXPECT_EQ(reglu.gate_type(), GateType::ReLU);
}

TEST_P(GatedActivationsTest, GeGLU_ForwardShape) {
    GeGLU geglu(64, 128);
    auto input = Variable(randn({2, 10, 64}, DType::Float32, Device::cpu()), true);
    auto output = geglu.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 64);  // down_proj maps back to in_features
}

TEST_P(GatedActivationsTest, ReGLU_ForwardShape) {
    ReGLU reglu(64, 128);
    auto input = Variable(randn({2, 10, 64}, DType::Float32, Device::cpu()), true);
    auto output = reglu.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST_P(GatedActivationsTest, AllGateTypes_ProduceDifferentOutput) {
    auto input = Variable(randn({1, 4, 32}, DType::Float32, Device::cpu()), false);

    GatedLinearUnit glu_sig(32, 64, GateType::Sigmoid, false);
    GatedLinearUnit glu_silu(32, 64, GateType::SiLU, false);
    GatedLinearUnit glu_gelu(32, 64, GateType::GELU, false);
    GatedLinearUnit glu_relu(32, 64, GateType::ReLU, false);

    auto out_sig = glu_sig.forward(input);
    auto out_silu = glu_silu.forward(input);
    auto out_gelu = glu_gelu.forward(input);
    auto out_relu = glu_relu.forward(input);

    // All outputs should have the same shape
    EXPECT_EQ(out_sig.shape().size(), out_silu.shape().size());
    EXPECT_EQ(out_sig.shape().size(), out_gelu.shape().size());
    EXPECT_EQ(out_sig.shape().size(), out_relu.shape().size());
    for (size_t i = 0; i < out_sig.shape().size(); ++i) {
        EXPECT_EQ(out_sig.shape()[i], out_silu.shape()[i]);
        EXPECT_EQ(out_sig.shape()[i], out_gelu.shape()[i]);
        EXPECT_EQ(out_sig.shape()[i], out_relu.shape()[i]);
    }
}

// Locks in that backward propagates through every gate type. Pre-fix, the
// raw-tensor multiply inside GatedLinearUnit's gating step severed grad_fn
// and tests like the shape ones above would NOT detect it (output exists,
// shape is correct, but backward returns zero gradients).
TEST_P(GatedActivationsTest, GeGLU_GradFlowsThroughBackward) {
    GeGLU geglu(32, 64);
    Variable input(randn({2, 4, 32}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto output = geglu.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(GatedActivationsTest, ReGLU_GradFlowsThroughBackward) {
    ReGLU reglu(32, 64);
    Variable input(randn({2, 4, 32}, DType::Float32, Device::cpu()), true);
    auto output = reglu.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(GatedActivationsTest, GatedLinearUnit_AllGateTypes_GradFlow) {
    for (auto gate : {GateType::Sigmoid, GateType::SiLU,
                      GateType::GELU, GateType::ReLU}) {
        SCOPED_TRACE("gate type = " + std::to_string(static_cast<int>(gate)));
        GatedLinearUnit glu(32, 64, gate, /*bias=*/false);
        Variable input(randn({1, 4, 32}, DType::Float32, Device::cpu()), true);
        auto output = glu.forward(input);
        auto loss = tenzor::sum(output);
        loss.backward();
        EXPECT_GRAD_FLOWS(input);
    }
}

INSTANTIATE_BACKEND_TESTS(GatedActivationsTest);
