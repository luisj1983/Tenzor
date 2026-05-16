// test_oneapi_im2col_dtype_parity.cpp
//
// Wave F2: OneAPI im2col now natively supports F16/BF16 (no tensor-wide
// widen-narrow). The kernel is a pure data shuffle so SYCL templates
// directly on `sycl::half` (F16) and `uint16_t` (BF16 storage). col2im
// (which needs atomic_add) remains a documented follow-up.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

using namespace tenzor;

namespace {

class OneAPIIm2ColDtypeParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    static bool has_oneapi() {
        try {
            auto t = zeros({1}, DType::Float32, Device::oneapi(0));
            (void)t; return true;
        } catch (...) { return false; }
    }
};

static auto seq_f32(std::vector<int64_t> shape) -> Tensor {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>((i & 0x3F) - 32) / 32.0f;  // in [-1, 1]
    }
    return t;
}

static void check_close(const Tensor& a, const Tensor& b, float atol) {
    ASSERT_EQ(a.shape().size(), b.shape().size());
    for (size_t i = 0; i < a.shape().size(); ++i) {
        ASSERT_EQ(a.shape()[i], b.shape()[i]) << " dim " << i;
    }
    auto* ap = a.data<float>();
    auto* bp = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(ap[i], bp[i], atol) << " elem " << i;
    }
}

}  // namespace

TEST_F(OneAPIIm2ColDtypeParity, Im2Col_F16_NativeMatchesCPU) {
    if (!has_oneapi()) GTEST_SKIP() << "OneAPI not available";

    auto in_f32 = seq_f32({1, 2, 6, 6});  // (N, C, H, W)
    OpAttributes attrs;
    attrs.set(AttrKey::KernelSize, static_cast<int64_t>(3));
    attrs.set(AttrKey::Stride, static_cast<int64_t>(1));
    attrs.set(AttrKey::Padding, static_cast<int64_t>(1));
    attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));

    // F32 reference on CPU.
    std::vector<Tensor> cpu_inputs = {in_f32};
    Tensor cpu_out = dispatch(OpId::Unfold, cpu_inputs, attrs)[0];

    // Native F16 on OneAPI.
    auto in_f16_oneapi = in_f32.to(DType::Float16).to(Device::oneapi(0));
    std::vector<Tensor> oneapi_inputs = {in_f16_oneapi};
    Tensor oneapi_out_f16 = dispatch(OpId::Unfold, oneapi_inputs, attrs)[0];

    // Verify the result was returned in Float16 (no tensor-wide widen).
    ASSERT_EQ(oneapi_out_f16.dtype(), DType::Float16);

    // Compare values (F32 reference cast down to F16 vs native F16 output).
    Tensor oneapi_out_f32 = oneapi_out_f16.to(Device::cpu()).to(DType::Float32);
    Tensor cpu_out_f16_back = cpu_out.to(DType::Float16).to(DType::Float32);
    check_close(cpu_out_f16_back, oneapi_out_f32, 5e-3f);
}

TEST_F(OneAPIIm2ColDtypeParity, Im2Col_BF16_NativeMatchesCPU) {
    if (!has_oneapi()) GTEST_SKIP() << "OneAPI not available";

    auto in_f32 = seq_f32({1, 2, 6, 6});
    OpAttributes attrs;
    attrs.set(AttrKey::KernelSize, static_cast<int64_t>(3));
    attrs.set(AttrKey::Stride, static_cast<int64_t>(1));
    attrs.set(AttrKey::Padding, static_cast<int64_t>(1));
    attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));

    std::vector<Tensor> cpu_inputs = {in_f32};
    Tensor cpu_out = dispatch(OpId::Unfold, cpu_inputs, attrs)[0];

    auto in_bf16_oneapi = in_f32.to(DType::BFloat16).to(Device::oneapi(0));
    std::vector<Tensor> oneapi_inputs = {in_bf16_oneapi};
    Tensor oneapi_out_bf16 = dispatch(OpId::Unfold, oneapi_inputs, attrs)[0];

    ASSERT_EQ(oneapi_out_bf16.dtype(), DType::BFloat16);

    Tensor oneapi_out_f32 = oneapi_out_bf16.to(Device::cpu()).to(DType::Float32);
    Tensor cpu_out_bf16_back = cpu_out.to(DType::BFloat16).to(DType::Float32);
    check_close(cpu_out_bf16_back, oneapi_out_f32, 1e-2f);
}
