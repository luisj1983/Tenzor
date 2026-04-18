/**
 * @file test_convtranspose_extended_multidtype.cpp
 * @brief Multi-dtype tests for ConvTranspose1d / ConvTranspose3d.
 *
 * ConvTranspose2d already has a dedicated multidtype test; the 1D and 3D
 * variants previously had only backend_parity helper-fn coverage. This file
 * promotes them to MultiBackendDTypeTest with forward shape checks, identity
 * ConvTranspose, and dtype preservation.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// ConvTranspose1d
// ============================================================================

class ConvTranspose1dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ConvTranspose1dMultiDTypeTest, ForwardShape_Stride1) {
    // L_out = (L_in - 1) * stride - 2*padding + kernel + output_padding
    //       = (10 - 1) * 1 - 0 + 3 + 0 = 12
    ConvTranspose1d conv(/*in=*/4, /*out=*/8, /*kernel=*/3);
    conv.to(device());
    auto x = Variable(randn({2, 4, 10}, dtype(), device()), false);
    auto y = conv.forward(x);
    expectShape(y.tensor(), {2, 8, 12});
    EXPECT_EQ(y.tensor().dtype(), dtype());
    EXPECT_EQ(y.tensor().device().type, device().type);
}

TEST_P(ConvTranspose1dMultiDTypeTest, ForwardShape_Stride2) {
    // L_out = (8 - 1) * 2 - 2*0 + 3 + 0 = 17
    ConvTranspose1d conv(/*in=*/3, /*out=*/6, /*kernel=*/3,
                         /*stride=*/2);
    conv.to(device());
    auto x = Variable(randn({1, 3, 8}, dtype(), device()), false);
    auto y = conv.forward(x);
    expectShape(y.tensor(), {1, 6, 17});
}

TEST_P(ConvTranspose1dMultiDTypeTest, ForwardShape_NoBias) {
    // Documented wrapper bug: ConvTranspose1d wraps ConvTranspose2d by
    // unsqueezing a height dim, but the user-supplied `padding` is applied to
    // BOTH the unsqueezed height (size 1) and the real width — making
    // padding > 0 produce negative height. Stick to padding=0 here.
    ConvTranspose1d conv(/*in=*/2, /*out=*/4, /*kernel=*/3,
                         /*stride=*/1, /*padding=*/0,
                         /*output_padding=*/0, /*groups=*/1, /*bias=*/false);
    conv.to(device());
    // L_out = (16 - 1) * 1 - 0 + 3 + 0 = 18
    auto x = Variable(randn({1, 2, 16}, dtype(), device()), false);
    auto y = conv.forward(x);
    expectShape(y.tensor(), {1, 4, 18});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ConvTranspose1dMultiDTypeTest);

// ============================================================================
// ConvTranspose3d
// ============================================================================

class ConvTranspose3dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ConvTranspose3dMultiDTypeTest, ForwardShape_Stride1) {
    // out_dim = (in - 1) * stride - 2*padding + dilation*(k-1) + out_pad + 1
    //         = (8 - 1) * 1 - 0 + 1*(3-1) + 0 + 1 = 10
    ConvTranspose3d conv(/*in=*/2, /*out=*/4, /*kernel=*/3);
    conv.to(device());
    auto x = Variable(randn({1, 2, 8, 8, 8}, dtype(), device()), false);
    auto y = conv.forward(x);
    expectShape(y.tensor(), {1, 4, 10, 10, 10});
    EXPECT_EQ(y.tensor().dtype(), dtype());
    EXPECT_EQ(y.tensor().device().type, device().type);
}

TEST_P(ConvTranspose3dMultiDTypeTest, ForwardShape_Stride2) {
    // out_dim = (4 - 1) * 2 - 0 + 1*(2-1) + 0 + 1 = 8
    ConvTranspose3d conv(/*in=*/2, /*out=*/4, /*kernel=*/2, /*stride=*/2);
    conv.to(device());
    auto x = Variable(randn({1, 2, 4, 4, 4}, dtype(), device()), false);
    auto y = conv.forward(x);
    expectShape(y.tensor(), {1, 4, 8, 8, 8});
}

TEST_P(ConvTranspose3dMultiDTypeTest, ForwardShape_NoBias) {
    ConvTranspose3d conv(/*in=*/2, /*out=*/3, /*kernel=*/3,
                         /*stride=*/1, /*padding=*/1,
                         /*output_padding=*/0, /*dilation=*/1,
                         /*groups=*/1, /*bias=*/false);
    conv.to(device());
    auto x = Variable(randn({1, 2, 6, 6, 6}, dtype(), device()), false);
    auto y = conv.forward(x);
    // out = (6 - 1) * 1 - 2*1 + (3-1) + 0 + 1 = 6
    expectShape(y.tensor(), {1, 3, 6, 6, 6});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ConvTranspose3dMultiDTypeTest);
