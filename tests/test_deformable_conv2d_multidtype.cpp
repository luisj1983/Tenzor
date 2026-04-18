/**
 * @file test_deformable_conv2d_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for DeformableConv2d (DCNv2)
 */

#include "backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include <gtest/gtest.h>

using namespace tenzor;
using namespace tenzor::testing;

class DeformableConv2dTest : public BackendTest {};

// ============================================================================
// Forward shape tests
// ============================================================================

TEST_P(DeformableConv2dTest, ForwardShapeBasic) {
    int64_t N = 2, C_in = 4, H = 8, W = 8;
    int64_t C_out = 8, kH = 3, kW = 3;
    int64_t stride = 1, padding = 1;
    int64_t H_out = H, W_out = W; // padding=1 with 3x3 kernel preserves size

    auto input = randn({N, C_in, H, W}, DType::Float32, device);
    auto weight = randn({C_out, C_in, kH, kW}, DType::Float32, device);
    auto bias = randn({C_out}, DType::Float32, device);
    auto offset = randn({N, 2 * kH * kW, H_out, W_out}, DType::Float32, device);
    auto mask = randn({N, kH * kW, H_out, W_out}, DType::Float32, device);

    auto output = tenzor::ops::deformable_conv2d(input, offset, weight, bias, mask,
                                          stride, stride, padding, padding,
                                          1, 1, 1, 1);

    EXPECT_EQ(output.shape()[0], N);
    EXPECT_EQ(output.shape()[1], C_out);
    EXPECT_EQ(output.shape()[2], H_out);
    EXPECT_EQ(output.shape()[3], W_out);
}

TEST_P(DeformableConv2dTest, ForwardShapeWithStride) {
    int64_t N = 1, C_in = 4, H = 16, W = 16;
    int64_t C_out = 8, kH = 3, kW = 3;
    int64_t stride = 2, padding = 1;
    int64_t H_out = (H + 2 * padding - kH) / stride + 1; // 8
    int64_t W_out = (W + 2 * padding - kW) / stride + 1; // 8

    auto input = randn({N, C_in, H, W}, DType::Float32, device);
    auto weight = randn({C_out, C_in, kH, kW}, DType::Float32, device);
    auto bias = Tensor(); // no bias
    auto offset = randn({N, 2 * kH * kW, H_out, W_out}, DType::Float32, device);
    auto mask = Tensor(); // no mask (DCNv1)

    auto output = tenzor::ops::deformable_conv2d(input, offset, weight, bias, mask,
                                          stride, stride, padding, padding,
                                          1, 1, 1, 1);

    EXPECT_EQ(output.shape()[0], N);
    EXPECT_EQ(output.shape()[1], C_out);
    EXPECT_EQ(output.shape()[2], H_out);
    EXPECT_EQ(output.shape()[3], W_out);
}

// ============================================================================
// Correctness tests
// ============================================================================

TEST_P(DeformableConv2dTest, ZeroOffsetMatchesConv2d) {
    // With zero offsets and no mask, deformable conv2d should produce the
    // same result as regular conv2d
    int64_t N = 1, C_in = 2, H = 6, W = 6;
    int64_t C_out = 2, kH = 3, kW = 3;
    int64_t stride = 1, padding = 1;
    int64_t H_out = H, W_out = W;

    auto input = randn({N, C_in, H, W}, DType::Float32, device);
    auto weight = randn({C_out, C_in, kH, kW}, DType::Float32, device);
    auto bias = randn({C_out}, DType::Float32, device);
    auto offset = zeros({N, 2 * kH * kW, H_out, W_out}, DType::Float32, device);
    auto mask = Tensor(); // no mask

    auto dcn_output = tenzor::ops::deformable_conv2d(input, offset, weight, bias, mask,
                                              stride, stride, padding, padding,
                                              1, 1, 1, 1);

    // Compare against regular Conv2d dispatch
    OpAttributes conv_attrs;
    conv_attrs.set(AttrKey::Stride, stride);
    conv_attrs.set(AttrKey::Padding, padding);
    conv_attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));
    conv_attrs.set(AttrKey::StrideH, stride);
    conv_attrs.set(AttrKey::StrideW, stride);
    conv_attrs.set(AttrKey::PaddingH, padding);
    conv_attrs.set(AttrKey::PaddingW, padding);
    conv_attrs.set(AttrKey::DilationH, static_cast<int64_t>(1));
    conv_attrs.set(AttrKey::DilationW, static_cast<int64_t>(1));
    conv_attrs.set(AttrKey::Groups, static_cast<int64_t>(1));

    std::vector<Tensor> conv_inputs = {input, weight, bias};
    auto conv_output = dispatch(OpId::Conv2dForward, conv_inputs, conv_attrs)[0];

    expectTensorNear(dcn_output, conv_output, 1e-4f);
}

TEST_P(DeformableConv2dTest, NonZeroOffsetChangesOutput) {
    int64_t N = 1, C_in = 2, H = 6, W = 6;
    int64_t C_out = 2, kH = 3, kW = 3;
    int64_t stride = 1, padding = 1;
    int64_t H_out = H, W_out = W;

    auto input = randn({N, C_in, H, W}, DType::Float32, device);
    auto weight = randn({C_out, C_in, kH, kW}, DType::Float32, device);
    auto bias = randn({C_out}, DType::Float32, device);
    auto mask = Tensor();

    auto zero_offset = zeros({N, 2 * kH * kW, H_out, W_out}, DType::Float32, device);
    auto rand_offset = randn({N, 2 * kH * kW, H_out, W_out}, DType::Float32, device);

    auto out_zero = tenzor::ops::deformable_conv2d(input, zero_offset, weight, bias, mask,
                                            stride, stride, padding, padding,
                                            1, 1, 1, 1);
    auto out_rand = tenzor::ops::deformable_conv2d(input, rand_offset, weight, bias, mask,
                                            stride, stride, padding, padding,
                                            1, 1, 1, 1);

    // Outputs should differ
    auto diff = tenzor::sub(out_zero.to(Device::cpu()), out_rand.to(Device::cpu()));
    auto abs_diff = tenzor::abs(diff);
    auto max_diff = tenzor::max(abs_diff);
    EXPECT_GT(max_diff.to(Device::cpu()).data<float>()[0], 1e-6f)
        << "Non-zero offsets should produce different output";
}

TEST_P(DeformableConv2dTest, MaskModulatesOutput) {
    int64_t N = 1, C_in = 2, H = 6, W = 6;
    int64_t C_out = 2, kH = 3, kW = 3;
    int64_t stride = 1, padding = 1;
    int64_t H_out = H, W_out = W;

    auto input = randn({N, C_in, H, W}, DType::Float32, device);
    auto weight = randn({C_out, C_in, kH, kW}, DType::Float32, device);
    auto bias = Tensor();
    auto offset = zeros({N, 2 * kH * kW, H_out, W_out}, DType::Float32, device);

    // Mask of all zeros should produce zero output
    auto zero_mask = zeros({N, kH * kW, H_out, W_out}, DType::Float32, device);
    auto out_zero_mask = tenzor::ops::deformable_conv2d(input, offset, weight, bias, zero_mask,
                                                 stride, stride, padding, padding,
                                                 1, 1, 1, 1);

    auto cpu_out = out_zero_mask.to(Device::cpu());
    auto* data = cpu_out.data<float>();
    float max_val = 0.0f;
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        max_val = std::max(max_val, std::abs(data[i]));
    }
    EXPECT_LT(max_val, 1e-6f) << "Zero mask should produce zero output";
}

// ============================================================================
// NN Layer test
// ============================================================================

TEST_P(DeformableConv2dTest, NNLayerForward) {
    // DeformableConv2d has kernels on every backend (CPU, CUDA, ROCm, Vulkan,
    // OneAPI). The CPU-only gate was legacy from a time when the GPU paths
    // were under construction; the parity helper below verifies equivalence.
    nn::DeformableConv2d dcn(4, 8, 3, 1, 1);
    dcn.to(device);
    auto input = Variable(randn({1, 4, 8, 8}, DType::Float32, device), true);
    auto offset = Variable(zeros({1, 18, 8, 8}, DType::Float32, device), false);

    auto output = dcn.forward(input, offset);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

// ============================================================================
// Gradient flow test
// ============================================================================

TEST_P(DeformableConv2dTest, BackwardGradientFlow) {
    nn::DeformableConv2d dcn(2, 4, 3, 1, 1);
    dcn.to(device);
    auto input = Variable(randn({1, 2, 6, 6}, DType::Float32, device), true);
    auto offset = Variable(zeros({1, 18, 6, 6}, DType::Float32, device), true);
    auto mask = Variable(ones({1, 9, 6, 6}, DType::Float32, device), true);

    auto output = dcn.forward(input, offset, mask);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    auto grad_cpu = input.grad().value().to(Device::cpu());
    auto* grad_data = grad_cpu.data<float>();
    float grad_sum = 0.0f;
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        grad_sum += std::abs(grad_data[i]);
    }
    EXPECT_GT(grad_sum, 0.0f) << "Input gradient should be non-zero";
}

// Cross-backend gradient parity: build identical inputs/weights on CPU and
// the test device, run forward+backward, and compare gradients. This
// exercises the deformable_conv2d backward kernels (bilinear interp
// gradient + weight + offset) against the CPU reference.
TEST_P(DeformableConv2dTest, BackwardParityVsCPU) {
    if (device.type == Device::Type::CPU) GTEST_SKIP();

    const int64_t in_ch = 2, out_ch = 4, k = 3;
    nn::DeformableConv2d dcn_cpu(in_ch, out_ch, k, 1, 1);

    auto input = randn({1, in_ch, 6, 6}, DType::Float32, Device::cpu());
    auto offset = randn({1, 2*k*k, 6, 6}, DType::Float32, Device::cpu()) * 0.2f;
    auto mask = ones({1, k*k, 6, 6}, DType::Float32, Device::cpu());

    // CPU run
    auto in_cpu = Variable(input.clone(), true);
    auto off_cpu = Variable(offset.clone(), true);
    auto msk_cpu = Variable(mask.clone(), true);
    auto out_cpu = dcn_cpu.forward(in_cpu, off_cpu, msk_cpu);
    tenzor::sum(out_cpu).backward();

    // Device run — clone the module's state_dict so parameters match exactly.
    nn::DeformableConv2d dcn_dev(in_ch, out_ch, k, 1, 1);
    dcn_dev.load_state_dict(dcn_cpu.state_dict());
    dcn_dev.to(device);

    auto in_dev = Variable(input.to(device), true);
    auto off_dev = Variable(offset.to(device), true);
    auto msk_dev = Variable(mask.to(device), true);
    auto out_dev = dcn_dev.forward(in_dev, off_dev, msk_dev);
    tenzor::sum(out_dev).backward();
    device.synchronize();

    auto diff = [](const Tensor& a, const Tensor& b) {
        auto ac = a.to(Device::cpu());
        auto bc = b.to(Device::cpu());
        float m = 0;
        auto* ap = ac.data<float>();
        auto* bp = bc.data<float>();
        for (int64_t i = 0; i < ac.numel(); ++i) {
            float e = std::fabs(ap[i] - bp[i]);
            if (e > m) m = e;
        }
        return m;
    };

    // Forward must match (tight).
    EXPECT_LT(diff(out_cpu.tensor(), out_dev.tensor()), 1e-4f)
        << "DeformableConv2d forward diverges";

    ASSERT_TRUE(in_cpu.grad().has_value());
    ASSERT_TRUE(in_dev.grad().has_value());
    EXPECT_LT(diff(in_cpu.grad().value(), in_dev.grad().value()), 1e-3f)
        << "input gradient diverges";

    ASSERT_TRUE(off_cpu.grad().has_value());
    ASSERT_TRUE(off_dev.grad().has_value());
    EXPECT_LT(diff(off_cpu.grad().value(), off_dev.grad().value()), 1e-3f)
        << "offset gradient diverges";

    ASSERT_TRUE(msk_cpu.grad().has_value());
    ASSERT_TRUE(msk_dev.grad().has_value());
    EXPECT_LT(diff(msk_cpu.grad().value(), msk_dev.grad().value()), 1e-3f)
        << "mask gradient diverges";
}

INSTANTIATE_BACKEND_TESTS(DeformableConv2dTest);
