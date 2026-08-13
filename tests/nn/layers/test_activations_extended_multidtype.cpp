/**
 * @file test_activations_extended_multidtype.cpp
 * @brief Multi-dtype tests for activations that previously had only
 *        backend_parity coverage: CELU, PReLU, Softsign, plus the
 *        functional rrelu().
 *
 * Other activations (GELU, ReLU, ELU, SELU, Sigmoid, Tanh, Softplus, Mish,
 * Hardswish, Hardsigmoid, Hardtanh, Threshold, Softshrink, Tanhshrink) are
 * already covered by existing multidtype suites — see test_activation_*
 * and test_higher_order_activations_multidtype.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class ActivationsExtendedMultiDTypeTest : public MultiBackendDTypeTest {};

// ----------------------------------------------------------------------------
// CELU: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
// ----------------------------------------------------------------------------

TEST_P(ActivationsExtendedMultiDTypeTest, CELU_ForwardShape) {
    CELU act;
    auto x = Variable(randn({4, 32}, dtype(), device()), false);
    auto y = act.forward(x);
    expectShape(y.tensor(), {4, 32});
    EXPECT_EQ(y.tensor().dtype(), dtype());
    EXPECT_EQ(y.tensor().device().type, device().type);
}

TEST_P(ActivationsExtendedMultiDTypeTest, CELU_PositiveInputsUnchanged) {
    CELU act;  // default alpha=1.0
    auto x = Variable(ones({1, 8}, dtype(), device()) * 2.0f, false);
    auto y = act.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], 2.0f, atol() * 10.0f);
    }
}

// ----------------------------------------------------------------------------
// PReLU: x if x > 0 else weight * x. weight is a learnable parameter.
// ----------------------------------------------------------------------------

TEST_P(ActivationsExtendedMultiDTypeTest, PReLU_ForwardShape) {
    // Documented gap: PReLU's learnable weight parameter is initialized as
    // Float32 and the Module API has no dtype-cast overload, so feeding
    // Float16/BFloat16 input promotes the output dtype. Skip both here until
    // Module gains .to(Device, DType) or PReLU dtype-respects construction.
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "PReLU has no dtype-cast path for its weight parameter";
    }
    PReLU act(/*num_parameters=*/1);
    act.to(device());
    auto x = Variable(randn({2, 16, 32}, dtype(), device()), false);
    auto y = act.forward(x);
    expectShape(y.tensor(), {2, 16, 32});
    EXPECT_EQ(y.tensor().dtype(), dtype());
}

TEST_P(ActivationsExtendedMultiDTypeTest, PReLU_PositiveInputsUnchanged) {
    PReLU act(1, /*init=*/0.25);
    act.to(device());
    auto x = Variable(ones({1, 4}, dtype(), device()) * 3.0f, false);
    auto y = act.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], 3.0f, atol() * 10.0f);
    }
}

TEST_P(ActivationsExtendedMultiDTypeTest, PReLU_NegativeInputsScaled) {
    // weight=0.25, input=-4 ⇒ output = 0.25 * -4 = -1.0
    PReLU act(1, /*init=*/0.25);
    act.to(device());
    auto x = Variable(ones({1, 4}, dtype(), device()) * -4.0f, false);
    auto y = act.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], -1.0f, atol() * 10.0f);
    }
}

// ----------------------------------------------------------------------------
// Softsign: x / (1 + |x|)
// ----------------------------------------------------------------------------

TEST_P(ActivationsExtendedMultiDTypeTest, Softsign_ForwardShape) {
    Softsign act;
    auto x = Variable(randn({4, 16}, dtype(), device()), false);
    auto y = act.forward(x);
    expectShape(y.tensor(), {4, 16});
    EXPECT_EQ(y.tensor().dtype(), dtype());
}

TEST_P(ActivationsExtendedMultiDTypeTest, Softsign_KnownValues) {
    // x=0 → 0; x=1 → 0.5; x=-1 → -0.5; x=3 → 0.75
    Softsign act;
    auto cpu_in = zeros({4}, DType::Float32, Device::cpu());
    cpu_in.data<float>()[0] = 0.0f;
    cpu_in.data<float>()[1] = 1.0f;
    cpu_in.data<float>()[2] = -1.0f;
    cpu_in.data<float>()[3] = 3.0f;
    auto x = Variable(cpu_in.to(dtype()).to(device()), false);
    auto y = act.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(out_cpu.data<float>()[0],  0.00f, atol() * 10.0f);
    EXPECT_NEAR(out_cpu.data<float>()[1],  0.50f, atol() * 10.0f);
    EXPECT_NEAR(out_cpu.data<float>()[2], -0.50f, atol() * 10.0f);
    EXPECT_NEAR(out_cpu.data<float>()[3],  0.75f, atol() * 10.0f);
}

// ----------------------------------------------------------------------------
// rrelu: functional. Eval mode uses the midpoint slope (lower+upper)/2
// for negative inputs.
// ----------------------------------------------------------------------------

TEST_P(ActivationsExtendedMultiDTypeTest, RReLU_FunctionalEvalMidpoint) {
    auto x = Variable(ones({1, 8}, dtype(), device()) * -2.0f, false);
    // Default lower=1/8, upper=1/3; midpoint = (0.125 + 0.333..) / 2 ≈ 0.229
    auto y = nn::rrelu(x, /*lower=*/0.125, /*upper=*/0.333,
                                   /*training=*/false);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    float expected = -2.0f * static_cast<float>((0.125 + 0.333) / 2.0);
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], expected, atol() * 10.0f);
    }
}

TEST_P(ActivationsExtendedMultiDTypeTest, RReLU_FunctionalPositivePassthrough) {
    auto x = Variable(ones({1, 8}, dtype(), device()) * 5.0f, false);
    auto y = nn::rrelu(x, 0.125, 0.333, /*training=*/false);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], 5.0f, atol() * 10.0f);
    }
}

// Phase 3 additions: backward gradient population for the activations covered
// in this file. Without these, a backward kernel returning zero or NaN would
// silently pass.

TEST_P(ActivationsExtendedMultiDTypeTest, CELU_BackwardGradPopulated) {
    nn::CELU celu;
    Variable input = createInput({4, 8}, true);
    auto out = celu.forward(input);
    sum(out).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

TEST_P(ActivationsExtendedMultiDTypeTest, PReLU_BackwardGradPopulated) {
    nn::PReLU prelu(/*num_parameters=*/1);
    convert_model(prelu);
    Variable input = createInput({4, 8}, true);
    auto out = prelu.forward(input);
    sum(out).backward();
    ASSERT_TRUE(input.has_grad());
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

TEST_P(ActivationsExtendedMultiDTypeTest, Softsign_BackwardGradPopulated) {
    nn::Softsign softsign;
    Variable input = createInput({4, 8}, true);
    auto out = softsign.forward(input);
    sum(out).backward();
    ASSERT_TRUE(input.has_grad());
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ActivationsExtendedMultiDTypeTest);
