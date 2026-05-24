/**
 * @file test_activation_backward_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for activation backward passes
 *
 * Covers backward gradient flow for: ELU, GELU, LeakyReLU, SELU, Mish, Swish, Dropout
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/layers/dropout.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"  // W.26: EXPECT_GRAD_FLOWS

using namespace tenzor;
using namespace tenzor::testing;

class ActivationBackwardMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Test forward + backward for a given activation module
    template <typename ActivationModule>
    void testActivationGradient(ActivationModule& act, const std::vector<int64_t>& shape,
                                const char* name) {
        auto input = createInput(shape, true);
        auto output = act.forward(input);

        expectDevice(output.tensor());
        expectDType(output.tensor());
        expectShape(output.tensor(), shape);

        auto out_shape = output.shape();
        std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
        auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

        EXPECT_NO_THROW({ output.backward(grad_output); })
            << name << " backward threw on " << device().to_string();

        ASSERT_TRUE(input.grad().has_value())
            << name << " backward did not produce gradient on " << device().to_string();
        expectShape(*input.grad(), shape);
    }
};

// ============================================================================
// ELU
// ============================================================================

TEST_P(ActivationBackwardMultiDTypeTest, ELUForwardBackward) {
    nn::ELU elu(1.0);
    testActivationGradient(elu, {4, 8}, "ELU");
}

TEST_P(ActivationBackwardMultiDTypeTest, ELUCustomAlpha) {
    nn::ELU elu(0.5);
    testActivationGradient(elu, {2, 16}, "ELU(alpha=0.5)");
}

// ============================================================================
// GELU
// ============================================================================

TEST_P(ActivationBackwardMultiDTypeTest, GELUForwardBackward) {
    nn::GELU gelu;
    testActivationGradient(gelu, {4, 8}, "GELU");
}

TEST_P(ActivationBackwardMultiDTypeTest, GELUTanhApprox) {
    nn::GELU gelu("tanh");
    testActivationGradient(gelu, {2, 16}, "GELU(tanh)");
}

// ============================================================================
// LeakyReLU
// ============================================================================

TEST_P(ActivationBackwardMultiDTypeTest, LeakyReLUForwardBackward) {
    nn::LeakyReLU lrelu(0.01);
    testActivationGradient(lrelu, {4, 8}, "LeakyReLU");
}

TEST_P(ActivationBackwardMultiDTypeTest, LeakyReLUCustomSlope) {
    nn::LeakyReLU lrelu(0.2);
    testActivationGradient(lrelu, {2, 16}, "LeakyReLU(0.2)");
}

// ============================================================================
// SELU
// ============================================================================

TEST_P(ActivationBackwardMultiDTypeTest, SELUForwardBackward) {
    nn::SELU selu;
    testActivationGradient(selu, {4, 8}, "SELU");
}

// ============================================================================
// Mish
// ============================================================================

TEST_P(ActivationBackwardMultiDTypeTest, MishForwardBackward) {
    nn::Mish mish;
    testActivationGradient(mish, {4, 8}, "Mish");
}

// ============================================================================
// Swish
// ============================================================================

TEST_P(ActivationBackwardMultiDTypeTest, SwishForwardBackward) {
    nn::Swish swish;
    testActivationGradient(swish, {4, 8}, "Swish");
}

// ============================================================================
// Dropout (training mode)
// ============================================================================

TEST_P(ActivationBackwardMultiDTypeTest, DropoutForwardBackward) {
    nn::Dropout dropout(0.5);
    dropout.train();

    auto input = createInput({4, 16}, true);
    auto output = dropout.forward(input);

    expectDevice(output.tensor());
    expectShape(output.tensor(), {4, 16});

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); })
        << "Dropout backward threw on " << device().to_string();

    EXPECT_GRAD_FLOWS(input);  // W.26
    ASSERT_TRUE(input.grad().has_value())
        << "Dropout backward did not produce gradient on " << device().to_string();
    expectShape(*input.grad(), {4, 16});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ActivationBackwardMultiDTypeTest);
