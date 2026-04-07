/**
 * @file test_conv3d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Conv3d operations
 *
 * Covers: Conv3d forward shape, stride/padding, backward gradient flow,
 * and weight gradient verification.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class Conv3dMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Forward Tests
// ============================================================================

TEST_P(Conv3dMultiDTypeTest, ForwardShapeBasic) {
    // Input: (N=1, C_in=1, D=8, H=8, W=8), kernel=3, stride=1, padding=1
    // Output D/H/W = floor((8 + 2*1 - 3) / 1) + 1 = 8
    nn::Conv3d conv(1, 16, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 1, 8, 8, 8}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 16, 8, 8, 8});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(Conv3dMultiDTypeTest, ForwardShapeWithStride) {
    // stride=2, padding=0: D_out = floor((8 - 3) / 2) + 1 = 3
    nn::Conv3d conv(2, 8, 3, 2, 0);
    convert_model(conv);

    auto input = createInput({1, 2, 8, 8, 8}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 8, 3, 3, 3});
    expectDevice(output.tensor());
}

TEST_P(Conv3dMultiDTypeTest, ForwardShapeNoBias) {
    nn::Conv3d conv(4, 8, 3, 1, 1, 1, 1, false);
    convert_model(conv);

    auto input = createInput({2, 4, 4, 4, 4}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {2, 8, 4, 4, 4});
}

// ============================================================================
// Backward Tests
// ============================================================================

TEST_P(Conv3dMultiDTypeTest, BackwardGradientFlow) {
    nn::Conv3d conv(2, 4, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 2, 4, 4, 4}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value())
        << "Conv3d backward did not produce input gradient on " << device().to_string();
    expectShape(*input.grad(), {1, 2, 4, 4, 4});
}

TEST_P(Conv3dMultiDTypeTest, WeightGradientExists) {
    nn::Conv3d conv(1, 4, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 1, 4, 4, 4}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);

    auto params = conv.parameters();
    ASSERT_FALSE(params.empty()) << "Conv3d has no parameters";
    // Weight parameter should have gradient
    ASSERT_TRUE(params[0]->grad().has_value())
        << "Conv3d weight gradient not produced on " << device().to_string();
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Conv3dMultiDTypeTest);
