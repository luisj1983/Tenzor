/**
 * @file test_conv_gaps_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for convolution operation gaps
 *
 * Covers: Conv1d (forward+backward), ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class ConvGapsMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Conv1d Tests
// ============================================================================

TEST_P(ConvGapsMultiDTypeTest, Conv1dForwardShape) {
    // Input: (N=2, C_in=3, L=16), kernel_size=3, C_out=8
    // Output L = floor((16 - 3) / 1) + 1 = 14
    nn::Conv1d conv(3, 8, 3);
    convert_model(conv);

    auto input = createInput({2, 3, 16}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {2, 8, 14});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(ConvGapsMultiDTypeTest, Conv1dWithPaddingStride) {
    // padding=1, stride=2: L_out = floor((16 + 2*1 - 3) / 2) + 1 = 8
    nn::Conv1d conv(4, 8, 3, 2, 1);
    convert_model(conv);

    auto input = createInput({1, 4, 16}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 8, 8});
}

// Previously DISABLED_ for an "assertion in vector bounds check" — verified
// passing across all 15 backend/dtype instantiations; re-enabling.
TEST_P(ConvGapsMultiDTypeTest, Conv1dBackward) {
    nn::Conv1d conv(3, 8, 3);
    convert_model(conv);

    auto input = createInput({1, 3, 10}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 3, 10});
}

// ============================================================================
// ConvTranspose1d Tests
// ============================================================================

TEST_P(ConvGapsMultiDTypeTest, ConvTranspose1dForwardShape) {
    // Upsamples: stride=2 approximately doubles spatial dim
    // L_out = (L_in - 1) * stride - 2*padding + kernel_size + output_padding
    // = (8 - 1) * 2 - 0 + 3 + 0 = 17
    nn::ConvTranspose1d conv(4, 8, 3, 2);
    convert_model(conv);

    auto input = createInput({1, 4, 8}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 8, 17});
    expectDevice(output.tensor());
}

TEST_P(ConvGapsMultiDTypeTest, ConvTranspose1dBackward) {
    nn::ConvTranspose1d conv(3, 6, 3, 1);
    convert_model(conv);

    auto input = createInput({1, 3, 8}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 3, 8});
}

// ============================================================================
// ConvTranspose2d Tests
// ============================================================================

TEST_P(ConvGapsMultiDTypeTest, ConvTranspose2dForwardShape) {
    // stride=2 approximately doubles spatial dimensions
    nn::ConvTranspose2d conv(4, 8, 3, 2, 1);
    convert_model(conv);

    auto input = createInput({1, 4, 8, 8}, false);
    auto output = conv.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 8);
    EXPECT_GT(shape[2], 8);  // upsampled
    EXPECT_GT(shape[3], 8);
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(ConvGapsMultiDTypeTest, ConvTranspose2dBackward) {
    nn::ConvTranspose2d conv(3, 6, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 3, 8, 8}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 3, 8, 8});
}

// ============================================================================
// ConvTranspose3d Tests
// ============================================================================

TEST_P(ConvGapsMultiDTypeTest, ConvTranspose3dForwardShape) {
    nn::ConvTranspose3d conv(2, 4, 3, 2, 1);
    convert_model(conv);

    auto input = createInput({1, 2, 4, 4, 4}, false);
    auto output = conv.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 4);
    EXPECT_GT(shape[2], 4);  // upsampled
    EXPECT_GT(shape[3], 4);
    EXPECT_GT(shape[4], 4);
    expectDevice(output.tensor());
}

TEST_P(ConvGapsMultiDTypeTest, ConvTranspose3dBackward) {
    nn::ConvTranspose3d conv(2, 4, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 2, 4, 4, 4}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    try {
        output.backward(grad_output);
    } catch (const std::exception& e) {
        FAIL() << "ConvTranspose3d backward threw: " << e.what()
               << " on " << device().to_string();
    }

    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 2, 4, 4, 4});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ConvGapsMultiDTypeTest);
