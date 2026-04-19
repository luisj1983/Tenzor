/**
 * @file test_pool_conv_missing_multidtype.cpp
 * @brief Coverage for 1D / 3D pool + Conv3d / ConvTranspose OpIds.
 *
 * OpIds covered (named for the audit grep): AvgPool1dForward, AvgPool1dBackward,
 * AvgPool3dForward, AvgPool3dBackward, MaxPool1dForward, MaxPool1dBackward,
 * MaxPool3dForward, MaxPool3dBackward, AdaptiveAvgPool1dBackward,
 * AdaptiveAvgPool3dBackward, AdaptiveMaxPool1dBackward, AdaptiveMaxPool3dBackward,
 * Conv3dForward, Conv3dBackwardInput, Conv3dBackwardWeight, Conv3dBackwardBias,
 * ConvTranspose2dForward, ConvTranspose3dForward, MaxUnpool3dForward,
 * MaxUnpool3dBackward, MaxUnpool2dBackward, DeformableConv2dForward,
 * DeformableConv2dBackwardInput, DeformableConv2dBackwardWeight,
 * DeformableConv2dBackwardBias, AdaptiveAvgPool2dBackward, AdaptiveMaxPool2dBackward.
 *
 * Each test runs the forward through an nn layer on the fixture device; the
 * autograd backward path is triggered by a sum+backward so the Backward OpIds
 * get touched too. Shape assertions validate output geometry.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class PoolConvMissingMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    Tensor on_device(std::vector<int64_t> shape) {
        return randn(shape, DType::Float32, Device::cpu()).to(dtype()).to(device());
    }
};

#define PC_SKIP_INT() \
    do { if (dtype() == DType::Int32 || dtype() == DType::Int64 || \
             dtype() == DType::UInt8 || dtype() == DType::Int8  || \
             dtype() == DType::Bool) { \
            SKIP_WITH_REASON(SkipReason::DtypeUnsupportedOnBackend, \
                             "pool/conv ops have no integer dispatch"); \
        } } while (0)

// ---------------------------------------------------------------------------
// AvgPool1d / MaxPool1d (forward + backward via autograd)
// ---------------------------------------------------------------------------

TEST_P(PoolConvMissingMultiDTypeTest, AvgPool1dForwardBackward) {
    PC_SKIP_INT();
    nn::AvgPool1d pool(/*kernel_size=*/2, /*stride=*/2);
    auto input = Variable(on_device({1, 4, 8}), /*requires_grad=*/true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 4);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(PoolConvMissingMultiDTypeTest, MaxPool1dForwardBackward) {
    PC_SKIP_INT();
    nn::MaxPool1d pool(/*kernel_size=*/2, /*stride=*/2);
    auto input = Variable(on_device({1, 4, 8}), /*requires_grad=*/true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 4);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

// ---------------------------------------------------------------------------
// AvgPool3d / MaxPool3d (forward + backward via autograd)
// ---------------------------------------------------------------------------

TEST_P(PoolConvMissingMultiDTypeTest, AvgPool3dForwardBackward) {
    PC_SKIP_INT();
    nn::AvgPool3d pool(/*kernel=*/2, /*stride=*/2, /*padding=*/0);
    auto input = Variable(on_device({1, 2, 4, 4, 4}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(PoolConvMissingMultiDTypeTest, MaxPool3dForwardBackward) {
    PC_SKIP_INT();
    nn::MaxPool3d pool(/*kernel=*/2, /*stride=*/2, /*padding=*/0);
    auto input = Variable(on_device({1, 2, 4, 4, 4}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool1d / 3d and AdaptiveMaxPool1d / 3d (backward variants)
// ---------------------------------------------------------------------------

TEST_P(PoolConvMissingMultiDTypeTest, AdaptiveAvgPool1dBackward) {
    PC_SKIP_INT();
    nn::AdaptiveAvgPool1d pool(/*output_size=*/2);
    auto input = Variable(on_device({1, 3, 8}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(PoolConvMissingMultiDTypeTest, AdaptiveMaxPool1dBackward) {
    PC_SKIP_INT();
    nn::AdaptiveMaxPool1d pool(/*output_size=*/2);
    auto input = Variable(on_device({1, 3, 8}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(PoolConvMissingMultiDTypeTest, AdaptiveAvgPool3dBackward) {
    PC_SKIP_INT();
    nn::AdaptiveAvgPool3d pool(/*output_size=*/2);
    auto input = Variable(on_device({1, 2, 4, 4, 4}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[4], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(PoolConvMissingMultiDTypeTest, AdaptiveMaxPool3dBackward) {
    PC_SKIP_INT();
    nn::AdaptiveMaxPool3d pool(/*output_size=*/2);
    auto input = Variable(on_device({1, 2, 4, 4, 4}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[4], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(PoolConvMissingMultiDTypeTest, AdaptiveAvgPool2dBackward) {
    PC_SKIP_INT();
    nn::AdaptiveAvgPool2d pool(/*output_size=*/2);
    auto input = Variable(on_device({1, 3, 8, 8}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(PoolConvMissingMultiDTypeTest, AdaptiveMaxPool2dBackward) {
    PC_SKIP_INT();
    nn::AdaptiveMaxPool2d pool(/*output_size=*/2);
    auto input = Variable(on_device({1, 3, 8, 8}), true);
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape()[2], 2);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

// ---------------------------------------------------------------------------
// Conv3dForward / Conv3dBackward*
// ---------------------------------------------------------------------------

TEST_P(PoolConvMissingMultiDTypeTest, Conv3dForwardBackward) {
    PC_SKIP_INT();
    nn::Conv3d conv(/*in_channels=*/2, /*out_channels=*/4,
                    /*kernel=*/3, /*stride=*/1, /*padding=*/1);
    conv.to(device());
    auto input = Variable(on_device({1, 2, 4, 4, 4}), true);
    auto output = conv.forward(input);
    EXPECT_EQ(output.shape()[1], 4);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

// ---------------------------------------------------------------------------
// ConvTranspose2dForward / ConvTranspose3dForward
// ---------------------------------------------------------------------------

TEST_P(PoolConvMissingMultiDTypeTest, ConvTranspose2dForward) {
    PC_SKIP_INT();
    nn::ConvTranspose2d ct(/*in=*/2, /*out=*/4, /*kernel=*/3, /*stride=*/2,
                           /*padding=*/1);
    ct.to(device());
    auto input = Variable(on_device({1, 2, 4, 4}), true);
    auto output = ct.forward(input);
    EXPECT_EQ(output.shape()[1], 4);
}

TEST_P(PoolConvMissingMultiDTypeTest, ConvTranspose3dForward) {
    PC_SKIP_INT();
    nn::ConvTranspose3d ct(/*in=*/2, /*out=*/4, /*kernel=*/3, /*stride=*/2,
                           /*padding=*/1);
    ct.to(device());
    auto input = Variable(on_device({1, 2, 4, 4, 4}), true);
    auto output = ct.forward(input);
    EXPECT_EQ(output.shape()[1], 4);
}

// ---------------------------------------------------------------------------
// MaxUnpool2dBackward / MaxUnpool3dForward / MaxUnpool3dBackward
// ---------------------------------------------------------------------------

// MaxUnpool2d: pool with indices via direct dispatch of MaxPool2dForward,
// then unpool via F::max_unpool2d. Exercises MaxUnpool2dForward +
// MaxUnpool2dBackward via autograd.
TEST_P(PoolConvMissingMultiDTypeTest, MaxUnpool2dForwardBackward) {
    PC_SKIP_INT();
    auto input_tensor = on_device({1, 2, 4, 4});
    std::vector<Tensor> pool_inputs = {input_tensor};
    OpAttributes pool_attrs;
    pool_attrs.set(AttrKey::KernelSize, int64_t{2});
    pool_attrs.set(AttrKey::Stride, int64_t{2});
    pool_attrs.set(AttrKey::Padding, int64_t{0});
    auto pool_result = dispatch_to_device(OpId::MaxPool2dForward, device().type,
                                          pool_inputs, pool_attrs);
    ASSERT_EQ(pool_result.size(), 2u) << "MaxPool2dForward must return (output, indices)";
    Tensor pooled = pool_result[0];
    Tensor indices = pool_result[1];

    auto input = Variable(pooled, /*requires_grad=*/true);
    auto output = nn::functional::max_unpool2d(
        input, indices,
        /*kernel_size=*/{2, 2},
        /*stride=*/{2, 2},
        /*padding=*/{0, 0});
    expectShape(output.tensor(), {1, 2, 4, 4});
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
    // Backward through MaxUnpool2dBackward: grad_input shape must match input shape.
    EXPECT_EQ(input.grad()->shape()[2], 2);
    EXPECT_EQ(input.grad()->shape()[3], 2);
}

TEST_P(PoolConvMissingMultiDTypeTest, MaxUnpool3dForwardBackward) {
    PC_SKIP_INT();
    auto input_tensor = on_device({1, 2, 4, 4, 4});
    std::vector<Tensor> pool_inputs = {input_tensor};
    OpAttributes pool_attrs;
    pool_attrs.set(AttrKey::KernelSize, int64_t{2});
    pool_attrs.set(AttrKey::Stride, int64_t{2});
    pool_attrs.set(AttrKey::Padding, int64_t{0});
    auto pool_result = dispatch_to_device(OpId::MaxPool3dForward, device().type,
                                          pool_inputs, pool_attrs);
    ASSERT_EQ(pool_result.size(), 2u) << "MaxPool3dForward must return (output, indices)";
    Tensor pooled = pool_result[0];
    Tensor indices = pool_result[1];

    auto input = Variable(pooled, /*requires_grad=*/true);
    auto output = nn::functional::max_unpool3d(
        input, indices,
        /*kernel_size=*/{2, 2, 2},
        /*stride=*/{2, 2, 2},
        /*padding=*/{0, 0, 0});
    expectShape(output.tensor(), {1, 2, 4, 4, 4});
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->shape()[2], 2);
}

// ---------------------------------------------------------------------------
// DeformableConv2d forward + backward through the nn layer
// ---------------------------------------------------------------------------

TEST_P(PoolConvMissingMultiDTypeTest, DeformableConv2dFullCycle) {
    PC_SKIP_INT();
    nn::DeformableConv2d dcn(/*in=*/2, /*out=*/4, /*k=*/3,
                              /*stride=*/1, /*padding=*/1);
    // Module defaults to Float32 weights — move to both the target device
    // AND the target dtype so forward/backward stays dtype-consistent with
    // the inputs.
    dcn.to(dtype());
    dcn.to(device());
    auto input = Variable(on_device({1, 2, 6, 6}), true);
    auto offset = Variable(zeros({1, 18, 6, 6}, DType::Float32, Device::cpu())
                               .to(dtype()).to(device()), true);
    auto output = dcn.forward(input, offset);
    EXPECT_EQ(output.shape()[1], 4);
    sum(output).backward();
    ASSERT_TRUE(input.grad().has_value());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(PoolConvMissingMultiDTypeTest);
