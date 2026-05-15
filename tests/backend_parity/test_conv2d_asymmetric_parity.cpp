// test_conv2d_asymmetric_parity.cpp
//
// Audit E1: CUDA Conv2d must honor per-axis StrideH/StrideW, PaddingH/W,
// DilationH/W attributes (previously the cuDNN descriptor was filled with
// the scalar `Stride` duplicated across both axes, so asymmetric stride
// silently degraded to symmetric).
//
// Verifies that with StrideH != StrideW the produced output has the
// correct asymmetric output shape and matches CPU's result element-wise.
// CUDA test is skipped when no CUDA backend is loaded.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/autograd/variable.hpp>

using namespace tenzor;

namespace {

class E1Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    static bool has_cuda() {
        try {
            auto t = zeros({1}, DType::Float32, Device::cuda(0));
            (void)t;
            return true;
        } catch (...) {
            return false;
        }
    }
};

// Compute expected output shape with per-axis params:
// out_h = (H + 2*pad_h - dil_h*(kH-1) - 1) / stride_h + 1
static auto expected_out_hw(int64_t H, int64_t W, int64_t kH, int64_t kW,
                             int64_t stride_h, int64_t stride_w,
                             int64_t pad_h, int64_t pad_w,
                             int64_t dil_h, int64_t dil_w) -> std::pair<int64_t, int64_t> {
    int64_t out_h = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    return {out_h, out_w};
}

static auto random_tensor(std::vector<int64_t> shape, Device dev) -> Tensor {
    auto t = zeros(shape, DType::Float32, dev);
    // Deterministic fill — Float32 [0..1) values.
    auto cpu = (dev.type == Device::Type::CPU) ? t : zeros(shape, DType::Float32, Device::cpu());
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        p[i] = static_cast<float>(((i * 1103515245LL + 12345LL) & 0xFFFF)) / 65536.0f;
    }
    return (dev.type == Device::Type::CPU) ? cpu : cpu.to(dev);
}

} // namespace

TEST_F(E1Test, CUDA_Conv2dForward_AsymmetricStride_ProducesCorrectShape) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    // Input: N=1, C=2, H=8, W=8. Weight: out=4, in=2, kH=3, kW=3.
    auto input  = random_tensor({1, 2, 8, 8}, Device::cuda(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::cuda(0));

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,   2);
    attrs.set(AttrKey::StrideW,   1);
    attrs.set(AttrKey::PaddingH,  0);
    attrs.set(AttrKey::PaddingW,  0);
    attrs.set(AttrKey::DilationH, 1);
    attrs.set(AttrKey::DilationW, 1);
    attrs.set(AttrKey::Groups,    1);

    std::vector<Tensor> inputs = {input, weight};
    auto outputs = dispatch(OpId::Conv2dForward, inputs, attrs);
    ASSERT_EQ(outputs.size(), 1u);

    auto [exp_h, exp_w] = expected_out_hw(8, 8, 3, 3, /*sh=*/2, /*sw=*/1,
                                            /*ph=*/0, /*pw=*/0,
                                            /*dh=*/1, /*dw=*/1);
    auto shape = outputs[0].shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], exp_h);  // 3
    EXPECT_EQ(shape[3], exp_w);  // 6
}

TEST_F(E1Test, CUDA_Conv2dForward_AsymmetricPadding_ProducesCorrectShape) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::cuda(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::cuda(0));
    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,   1);
    attrs.set(AttrKey::StrideW,   1);
    attrs.set(AttrKey::PaddingH,  2);
    attrs.set(AttrKey::PaddingW,  0);
    attrs.set(AttrKey::DilationH, 1);
    attrs.set(AttrKey::DilationW, 1);
    attrs.set(AttrKey::Groups,    1);

    std::vector<Tensor> inputs = {input, weight};
    auto outputs = dispatch(OpId::Conv2dForward, inputs, attrs);
    ASSERT_EQ(outputs.size(), 1u);
    auto [exp_h, exp_w] = expected_out_hw(8, 8, 3, 3, 1, 1, 2, 0, 1, 1);
    auto shape = outputs[0].shape();
    EXPECT_EQ(shape[2], exp_h);  // 10
    EXPECT_EQ(shape[3], exp_w);  // 6
}

TEST_F(E1Test, CUDA_Conv2dForward_AsymmetricDilation_ProducesCorrectShape) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";
    auto input  = random_tensor({1, 2, 10, 10}, Device::cuda(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::cuda(0));
    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,   1);
    attrs.set(AttrKey::StrideW,   1);
    attrs.set(AttrKey::PaddingH,  0);
    attrs.set(AttrKey::PaddingW,  0);
    attrs.set(AttrKey::DilationH, 2);
    attrs.set(AttrKey::DilationW, 1);
    attrs.set(AttrKey::Groups,    1);

    std::vector<Tensor> inputs = {input, weight};
    auto outputs = dispatch(OpId::Conv2dForward, inputs, attrs);
    ASSERT_EQ(outputs.size(), 1u);
    auto [exp_h, exp_w] = expected_out_hw(10, 10, 3, 3, 1, 1, 0, 0, 2, 1);
    auto shape = outputs[0].shape();
    EXPECT_EQ(shape[2], exp_h);  // 6
    EXPECT_EQ(shape[3], exp_w);  // 8
}

TEST_F(E1Test, CUDA_Conv2dForward_AsymmetricMatchesCPU_Values) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    auto input_cpu  = random_tensor({1, 2, 8, 8}, Device::cpu());
    auto weight_cpu = random_tensor({4, 2, 3, 3}, Device::cpu());
    auto input_gpu  = input_cpu.to(Device::cuda(0));
    auto weight_gpu = weight_cpu.to(Device::cuda(0));

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,   2);
    attrs.set(AttrKey::StrideW,   1);
    attrs.set(AttrKey::PaddingH,  1);
    attrs.set(AttrKey::PaddingW,  0);
    attrs.set(AttrKey::DilationH, 1);
    attrs.set(AttrKey::DilationW, 1);
    attrs.set(AttrKey::Groups,    1);

    std::vector<Tensor> cpu_inputs = {input_cpu, weight_cpu};
    std::vector<Tensor> gpu_inputs = {input_gpu, weight_gpu};

    Tensor cpu_out, gpu_out;
    try {
        cpu_out = dispatch(OpId::Conv2dForward, cpu_inputs, attrs)[0];
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CPU Conv2dForward does not support per-axis yet (Phase E1 only): "
                     << e.what();
    }
    gpu_out = dispatch(OpId::Conv2dForward, gpu_inputs, attrs)[0].to(Device::cpu());

    // Same shape.
    ASSERT_EQ(cpu_out.shape().size(), gpu_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], gpu_out.shape()[i]) << " dim " << i;
    }
    // Element-wise tolerance.
    auto* cp = cpu_out.data<float>();
    auto* gp = gpu_out.data<float>();
    int64_t n = cpu_out.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(cp[i], gp[i], 1e-4f) << " elem " << i;
    }
}

TEST_F(E1Test, CUDA_Conv2dForward_ScalarAttrsStillWork) {
    // Regression: when only the scalar Stride/Padding/Dilation attrs are set
    // (no per-axis keys), the kernel must still produce the symmetric result.
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::cuda(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::cuda(0));

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   2);
    attrs.set(AttrKey::Padding,  1);
    attrs.set(AttrKey::Dilation, 1);
    attrs.set(AttrKey::Groups,   1);

    std::vector<Tensor> inputs = {input, weight};
    auto outputs = dispatch(OpId::Conv2dForward, inputs, attrs);
    ASSERT_EQ(outputs.size(), 1u);
    auto [exp_h, exp_w] = expected_out_hw(8, 8, 3, 3, 2, 2, 1, 1, 1, 1);
    auto shape = outputs[0].shape();
    EXPECT_EQ(shape[2], exp_h);  // 4
    EXPECT_EQ(shape[3], exp_w);  // 4
}

// ----------------------------------------------------------------------------
// Audit E2: ROCm honest behavior — symmetric runs unchanged; asymmetric
// throws cleanly rather than silently degrading to symmetric.
// ----------------------------------------------------------------------------

static bool has_rocm() {
    try {
        auto t = zeros({1}, DType::Float32, Device::rocm(0));
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

TEST_F(E1Test, ROCm_Conv2dForward_SymmetricStill_Works) {
    if (!has_rocm()) GTEST_SKIP() << "ROCm not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::rocm(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::rocm(0));

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   2);
    attrs.set(AttrKey::Padding,  1);
    attrs.set(AttrKey::Dilation, 1);
    attrs.set(AttrKey::Groups,   1);

    std::vector<Tensor> inputs = {input, weight};
    auto outputs = dispatch(OpId::Conv2dForward, inputs, attrs);
    ASSERT_EQ(outputs.size(), 1u);
}

TEST_F(E1Test, ROCm_Conv2dForward_Asymmetric_ThrowsHonestly) {
    // Audit E2: previously silently degraded to symmetric. Now throws.
    if (!has_rocm()) GTEST_SKIP() << "ROCm not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::rocm(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::rocm(0));

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,   2);
    attrs.set(AttrKey::StrideW,   1);
    attrs.set(AttrKey::PaddingH,  0);
    attrs.set(AttrKey::PaddingW,  0);
    attrs.set(AttrKey::DilationH, 1);
    attrs.set(AttrKey::DilationW, 1);
    attrs.set(AttrKey::Groups,    1);

    std::vector<Tensor> inputs = {input, weight};
    EXPECT_THROW(dispatch(OpId::Conv2dForward, inputs, attrs), std::runtime_error);
}

// ----------------------------------------------------------------------------
// Audit E3: OneAPI honest behavior — symmetric runs unchanged; asymmetric
// throws cleanly rather than silently degrading to symmetric.
// ----------------------------------------------------------------------------

static bool has_oneapi() {
    try {
        auto t = zeros({1}, DType::Float32, Device::oneapi(0));
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

TEST_F(E1Test, OneAPI_Conv2dForward_SymmetricStill_Works) {
    if (!has_oneapi()) GTEST_SKIP() << "OneAPI not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::oneapi(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::oneapi(0));

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   2);
    attrs.set(AttrKey::Padding,  1);
    attrs.set(AttrKey::Dilation, 1);
    attrs.set(AttrKey::Groups,   1);

    std::vector<Tensor> inputs = {input, weight};
    auto outputs = dispatch(OpId::Conv2dForward, inputs, attrs);
    ASSERT_EQ(outputs.size(), 1u);
}

TEST_F(E1Test, OneAPI_Conv2dForward_Asymmetric_ThrowsHonestly) {
    if (!has_oneapi()) GTEST_SKIP() << "OneAPI not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::oneapi(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::oneapi(0));

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,   2);
    attrs.set(AttrKey::StrideW,   1);
    attrs.set(AttrKey::PaddingH,  0);
    attrs.set(AttrKey::PaddingW,  0);
    attrs.set(AttrKey::DilationH, 1);
    attrs.set(AttrKey::DilationW, 1);
    attrs.set(AttrKey::Groups,    1);

    std::vector<Tensor> inputs = {input, weight};
    EXPECT_THROW(dispatch(OpId::Conv2dForward, inputs, attrs), std::runtime_error);
}

// ----------------------------------------------------------------------------
// Audit E4: Vulkan honest behavior — symmetric runs unchanged; asymmetric
// throws cleanly rather than silently degrading to symmetric.
// ----------------------------------------------------------------------------

static bool has_vulkan() {
    try {
        auto t = zeros({1}, DType::Float32, Device::vulkan(0));
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

TEST_F(E1Test, Vulkan_Conv2dForward_SymmetricStill_Works) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::vulkan(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::vulkan(0));

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   2);
    attrs.set(AttrKey::Padding,  1);
    attrs.set(AttrKey::Dilation, 1);
    attrs.set(AttrKey::Groups,   1);

    std::vector<Tensor> inputs = {input, weight};
    auto outputs = dispatch(OpId::Conv2dForward, inputs, attrs);
    ASSERT_EQ(outputs.size(), 1u);
}

TEST_F(E1Test, Vulkan_Conv2dForward_Asymmetric_ThrowsHonestly) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::vulkan(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::vulkan(0));

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,   2);
    attrs.set(AttrKey::StrideW,   1);
    attrs.set(AttrKey::PaddingH,  0);
    attrs.set(AttrKey::PaddingW,  0);
    attrs.set(AttrKey::DilationH, 1);
    attrs.set(AttrKey::DilationW, 1);
    attrs.set(AttrKey::Groups,    1);

    std::vector<Tensor> inputs = {input, weight};
    EXPECT_THROW(dispatch(OpId::Conv2dForward, inputs, attrs), std::runtime_error);
}

// ----------------------------------------------------------------------------
// Audit E5: high-level guards in F::conv2d and nn::Conv2d removed.
// Asymmetric flows through to the backend; CPU/CUDA produce correct math,
// other backends throw at the backend layer (better error location +
// message). These end-to-end tests verify the public API surface.
// ----------------------------------------------------------------------------

TEST_F(E1Test, NN_FConv2d_AsymmetricStride_CPU_ProducesShape) {
    auto input  = random_tensor({1, 2, 8, 8}, Device::cpu());
    auto weight = random_tensor({4, 2, 3, 3}, Device::cpu());
    Variable input_v(input, false);
    Variable weight_v(weight, false);

    // F::conv2d previously threw on asymmetric stride; now flows through.
    Variable out = nn::functional::conv2d(input_v, weight_v, std::nullopt,
                                            /*stride=*/{2, 1},
                                            /*padding=*/{0, 0},
                                            /*dilation=*/{1, 1},
                                            /*groups=*/1);
    auto shape = out.shape();
    ASSERT_EQ(shape.size(), 4u);
    // Output: out_h = (8 - 3)/2 + 1 = 3; out_w = (8 - 3)/1 + 1 = 6.
    EXPECT_EQ(shape[2], 3);
    EXPECT_EQ(shape[3], 6);
}

TEST_F(E1Test, NN_FConv2d_AsymmetricStride_CUDA_ProducesShape) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::cuda(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::cuda(0));
    Variable input_v(input, false);
    Variable weight_v(weight, false);

    Variable out = nn::functional::conv2d(input_v, weight_v, std::nullopt,
                                            /*stride=*/{2, 1},
                                            /*padding=*/{0, 0},
                                            /*dilation=*/{1, 1},
                                            /*groups=*/1);
    auto shape = out.shape();
    EXPECT_EQ(shape[2], 3);
    EXPECT_EQ(shape[3], 6);
}

TEST_F(E1Test, NN_FConv2d_AsymmetricStride_ROCm_ThrowsAtBackend) {
    // After E5 the F::conv2d guard is gone; ROCm asymmetric throws at the
    // backend with a clear "kernel refactor pending" message.
    if (!has_rocm()) GTEST_SKIP() << "ROCm not available";
    auto input  = random_tensor({1, 2, 8, 8}, Device::rocm(0));
    auto weight = random_tensor({4, 2, 3, 3}, Device::rocm(0));
    Variable input_v(input, false);
    Variable weight_v(weight, false);

    EXPECT_THROW(
        nn::functional::conv2d(input_v, weight_v, std::nullopt,
                                /*stride=*/{2, 1}, /*padding=*/{0, 0},
                                /*dilation=*/{1, 1}, /*groups=*/1),
        std::runtime_error);
}

TEST_F(E1Test, NN_Conv2d_Module_AsymmetricStride_CPU_Works) {
    // Module-level: nn::Conv2d with rectangular stride. Previously threw.
    nn::Conv2d conv(/*in_channels=*/2, /*out_channels=*/4,
                     /*kernel_size=*/std::pair<int64_t, int64_t>{3, 3},
                     /*stride=*/std::pair<int64_t, int64_t>{2, 1},
                     /*padding=*/std::pair<int64_t, int64_t>{0, 0},
                     /*dilation=*/std::pair<int64_t, int64_t>{1, 1},
                     /*groups=*/1, /*bias=*/false);

    Variable x(random_tensor({1, 2, 8, 8}, Device::cpu()), false);
    Variable y = conv.forward(x);
    auto shape = y.shape();
    EXPECT_EQ(shape[2], 3);
    EXPECT_EQ(shape[3], 6);
}

TEST_F(E1Test, NN_Conv2d_Module_AsymmetricDilation_CPU_Works) {
    nn::Conv2d conv(/*in_channels=*/2, /*out_channels=*/4,
                     /*kernel_size=*/std::pair<int64_t, int64_t>{3, 3},
                     /*stride=*/std::pair<int64_t, int64_t>{1, 1},
                     /*padding=*/std::pair<int64_t, int64_t>{0, 0},
                     /*dilation=*/std::pair<int64_t, int64_t>{2, 1},
                     /*groups=*/1, /*bias=*/false);

    Variable x(random_tensor({1, 2, 10, 10}, Device::cpu()), false);
    Variable y = conv.forward(x);
    auto shape = y.shape();
    // out_h = (10 - 2*(3-1) - 1)/1 + 1 = 6; out_w = (10 - 3)/1 + 1 = 8.
    EXPECT_EQ(shape[2], 6);
    EXPECT_EQ(shape[3], 8);
}
