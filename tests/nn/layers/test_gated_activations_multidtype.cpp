/**
 * @file test_gated_activations_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for GatedLinearUnit, GeGLU, ReGLU
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/layers/hrm.hpp>
#include <tenzor/autograd/variable.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class GatedActivationsMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(GatedActivationsMultiDTypeTest, GeGLU_Construction) {
    GeGLU geglu(64, 128);
    convert_model(geglu);
    EXPECT_EQ(geglu.gate_type(), GateType::GELU);
}

TEST_P(GatedActivationsMultiDTypeTest, ReGLU_Construction) {
    ReGLU reglu(64, 128);
    convert_model(reglu);
    EXPECT_EQ(reglu.gate_type(), GateType::ReLU);
}

TEST_P(GatedActivationsMultiDTypeTest, GeGLU_ForwardShape) {
    GeGLU geglu(64, 128);
    convert_model(geglu);
    auto input = createInput({2, 10, 64}, false);
    auto output = geglu.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST_P(GatedActivationsMultiDTypeTest, ReGLU_ForwardShape) {
    ReGLU reglu(64, 128);
    convert_model(reglu);
    auto input = createInput({2, 10, 64}, false);
    auto output = reglu.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST_P(GatedActivationsMultiDTypeTest, AllGateTypes_SameOutputShape) {
    auto input = createInput({1, 4, 32}, false);

    GatedLinearUnit glu_sig(32, 64, GateType::Sigmoid, false);
    GatedLinearUnit glu_silu(32, 64, GateType::SiLU, false);
    GatedLinearUnit glu_gelu(32, 64, GateType::GELU, false);
    GatedLinearUnit glu_relu(32, 64, GateType::ReLU, false);
    convert_model(glu_sig);
    convert_model(glu_silu);
    convert_model(glu_gelu);
    convert_model(glu_relu);

    auto out_sig = glu_sig.forward(input);
    auto out_silu = glu_silu.forward(input);
    auto out_gelu = glu_gelu.forward(input);
    auto out_relu = glu_relu.forward(input);

    for (size_t i = 0; i < out_sig.shape().size(); ++i) {
        EXPECT_EQ(out_sig.shape()[i], out_silu.shape()[i]);
        EXPECT_EQ(out_sig.shape()[i], out_gelu.shape()[i]);
        EXPECT_EQ(out_sig.shape()[i], out_relu.shape()[i]);
    }
}

TEST_P(GatedActivationsMultiDTypeTest, GatedLinearUnit_BackwardsCompat) {
    GatedLinearUnit glu_silu(64, 128, true, false);
    convert_model(glu_silu);
    EXPECT_EQ(glu_silu.gate_type(), GateType::SiLU);

    GatedLinearUnit glu_sigmoid(64, 128, false, false);
    convert_model(glu_sigmoid);
    EXPECT_EQ(glu_sigmoid.gate_type(), GateType::Sigmoid);
}

TEST_P(GatedActivationsMultiDTypeTest, OutputValuesFinite) {
    GeGLU geglu(32, 64);
    convert_model(geglu);
    auto input = createInput({2, 4, 32}, false);
    auto output = geglu.forward(input);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* d = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i])) << "Non-finite at " << i;
    }
}

// Phase 3 addition (Phase 3-followup #20 fix): GeGLU/ReGLU forward used raw
// tensor multiplication, dropping the autograd graph and producing zero
// input gradients. Fixed in src/nn/layers/hrm.cpp by switching to
// Variable-level operator*.
TEST_P(GatedActivationsMultiDTypeTest, GeGLU_BackwardGradPopulated) {
    GeGLU geglu(32, 64);
    convert_model(geglu);
    auto input = createInput({2, 4, 32}, true);
    auto output = geglu.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

TEST_P(GatedActivationsMultiDTypeTest, ReGLU_BackwardGradPopulated) {
    ReGLU reglu(32, 64);
    convert_model(reglu);
    // ReGLU sets negatives to zero — make sure inputs are positive enough
    // that some gradient flows through (not all halves zeroed).
    auto input = Variable((randn({2, 4, 32}, dtype(), device()) + 1.0f), true);
    auto output = reglu.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GatedActivationsMultiDTypeTest);
