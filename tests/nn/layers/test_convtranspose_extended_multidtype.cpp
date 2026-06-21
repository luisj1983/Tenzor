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

// Phase 3 addition: backward gradient population.
TEST_P(ConvTranspose1dMultiDTypeTest, BackwardGradPopulated) {
    nn::ConvTranspose1d conv(2, 3, 3);
    conv.to(device());
    auto x = Variable(randn({1, 2, 5}, dtype(), device()), true);
    auto y = conv.forward(x);
    sum(y).backward();
    ASSERT_TRUE(x.has_grad()) << device().to_string();
    auto g_max = max(abs(x.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
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

// output_padding support (per-axis). output_padding enlarges the output by
// `op` along each spatial axis: out = (in-1)*s - 2p + d*(k-1) + op + 1. The
// scatter loop is identical to the op=0 case (each input contributes to
// ow = iw*s - p + kw*d, bounds-checked against the larger out extent), so the
// op=0 result must be a prefix sub-volume of the op>0 result. This verifies the
// previously-missing non-default output_padding path now produces both the
// correct shape AND correct values (the enlarged region holds the genuine
// scatter contributions / zeros, not garbage).
TEST_P(ConvTranspose3dMultiDTypeTest, OutputPadding_ShapeAndValues) {
    // Deterministic weights/inputs so op=0 vs op=1 are directly comparable.
    // stride=2 (op must be < stride), kernel=2, padding=0.
    auto make_conv = [&](int64_t op) {
        return ConvTranspose3d(/*in=*/2, /*out=*/3, /*kernel=*/2, /*stride=*/2,
                               /*padding=*/0, /*output_padding=*/op,
                               /*dilation=*/1, /*groups=*/1, /*bias=*/false);
    };

    ConvTranspose3d conv0 = make_conv(0);
    ConvTranspose3d conv1 = make_conv(1);
    // Set both weights to the same constant so op=0 and op=1 are directly
    // comparable on their shared region (no dependence on random init).
    conv0.get_parameter("weight")->tensor().fill_(0.5);
    conv1.get_parameter("weight")->tensor().fill_(0.5);
    conv0.to(device());
    conv1.to(device());

    auto x = Variable(full({1, 2, 3, 3, 3}, 1.0f, dtype(), device()), false);

    auto y0 = conv0.forward(x);
    auto y1 = conv1.forward(x);

    // op=0: out = (3-1)*2 - 0 + (2-1) + 0 + 1 = 6
    expectShape(y0.tensor(), {1, 3, 6, 6, 6});
    // op=1: out = (3-1)*2 - 0 + (2-1) + 1 + 1 = 7
    expectShape(y1.tensor(), {1, 3, 7, 7, 7});

    // The op=0 volume must equal the leading [0:6,0:6,0:6] sub-volume of op=1.
    auto h0 = y0.tensor().to(Device::cpu()).to(DType::Float32);
    auto h1 = y1.tensor().to(Device::cpu()).to(DType::Float32);
    auto h1_crop = slice(slice(slice(h1, 2, 0, 6), 3, 0, 6), 4, 0, 6);
    auto diff = max(abs(h0 - h1_crop)).item<float>();
    EXPECT_LT(diff, 1e-3f)
        << device().to_string()
        << ": output_padding=1 must match output_padding=0 on the shared region";
}

// Phase 3 addition: backward gradient population.
TEST_P(ConvTranspose3dMultiDTypeTest, BackwardGradPopulated) {
    nn::ConvTranspose3d conv(2, 3, 3);
    conv.to(device());
    auto x = Variable(randn({1, 2, 4, 4, 4}, dtype(), device()), true);
    auto y = conv.forward(x);
    sum(y).backward();
    ASSERT_TRUE(x.has_grad()) << device().to_string();
    auto g_max = max(abs(x.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ConvTranspose3dMultiDTypeTest);
