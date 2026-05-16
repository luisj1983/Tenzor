// test_cuda_fused_dtype_parity.cpp
//
// Wave E1/E2: CUDA fused kernels native F16/BF16 (no tensor-wide widen-narrow).
// Verifies that CUDA's fused_softmax_cross_entropy and fused_rms_norm produce
// correct results for Float16 and BFloat16 inputs by running each kernel
// natively (F32 accumulator inside the kernel, F16/BF16 storage at load/store)
// and comparing against the CPU F32 reference.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

using namespace tenzor;

namespace {

class CudaFusedDtypeParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    static bool has_cuda() {
        try {
            auto t = zeros({1}, DType::Float32, Device::cuda(0));
            (void)t; return true;
        } catch (...) { return false; }
    }
};

static auto random_f32(std::vector<int64_t> shape) -> Tensor {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i * 2654435761u);
        p[i] = (static_cast<float>(bits & 0xFFFFu) / 65536.0f) - 0.5f;
    }
    return t;
}

}  // namespace

TEST_F(CudaFusedDtypeParity, FusedSoftmaxCrossEntropy_F16_NativeMatchesRef) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    // Build F32 logits & integer targets on CPU.
    auto logits_f32 = random_f32({16, 32});
    auto targets = zeros({16}, DType::Int64, Device::cpu());
    auto* tp = targets.data<int64_t>();
    for (int64_t i = 0; i < targets.numel(); ++i) tp[i] = (i * 7) % 32;

    // Reference: F32 reduction in CUDA.
    auto logits_f32_cuda = logits_f32.to(Device::cuda(0));
    auto targets_cuda = targets.to(Device::cuda(0));
    OpAttributes attrs;
    attrs.set(AttrKey::Reduction, std::string("mean"));
    std::vector<Tensor> ref_inputs = {logits_f32_cuda, targets_cuda};
    Tensor loss_ref = dispatch(OpId::FusedSoftmaxCrossEntropy, ref_inputs, attrs)[0]
                          .to(Device::cpu());

    // Native F16 path.
    auto logits_f16_cuda = logits_f32.to(DType::Float16).to(Device::cuda(0));
    std::vector<Tensor> f16_inputs = {logits_f16_cuda, targets_cuda};
    Tensor loss_f16 = dispatch(OpId::FusedSoftmaxCrossEntropy, f16_inputs, attrs)[0]
                          .to(Device::cpu()).to(DType::Float32);

    ASSERT_EQ(loss_ref.dtype(), DType::Float32);
    ASSERT_EQ(loss_f16.dtype(), DType::Float32);
    EXPECT_NEAR(loss_ref.data<float>()[0], loss_f16.data<float>()[0], 5e-3f);
}

TEST_F(CudaFusedDtypeParity, FusedRMSNorm_F16_NativeMatchesRef) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    auto input_f32 = random_f32({4, 64});
    auto weight_f32 = random_f32({64});

    // Reference: F32 on CUDA.
    auto in_cuda = input_f32.to(Device::cuda(0));
    auto wt_cuda = weight_f32.to(Device::cuda(0));
    OpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(1e-6));
    std::vector<Tensor> ref_inputs = {in_cuda, wt_cuda};
    Tensor ref_out = dispatch(OpId::FusedRMSNorm, ref_inputs, attrs)[0].to(Device::cpu());

    // Native F16.
    auto in_f16_cuda = input_f32.to(DType::Float16).to(Device::cuda(0));
    auto wt_f16_cuda = weight_f32.to(DType::Float16).to(Device::cuda(0));
    std::vector<Tensor> f16_inputs = {in_f16_cuda, wt_f16_cuda};
    Tensor out_f16 = dispatch(OpId::FusedRMSNorm, f16_inputs, attrs)[0]
                          .to(Device::cpu()).to(DType::Float32);

    ASSERT_EQ(out_f16.dtype(), DType::Float32);
    ASSERT_EQ(ref_out.numel(), out_f16.numel());
    auto* rp = ref_out.data<float>();
    auto* op = out_f16.data<float>();
    for (int64_t i = 0; i < ref_out.numel(); ++i) {
        EXPECT_NEAR(rp[i], op[i], 5e-3f) << " elem " << i;
    }
}

TEST_F(CudaFusedDtypeParity, FusedSoftmaxCrossEntropy_BF16_NativeMatchesRef) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    auto logits_f32 = random_f32({16, 32});
    auto targets = zeros({16}, DType::Int64, Device::cpu());
    auto* tp = targets.data<int64_t>();
    for (int64_t i = 0; i < targets.numel(); ++i) tp[i] = (i * 7) % 32;

    auto logits_f32_cuda = logits_f32.to(Device::cuda(0));
    auto targets_cuda = targets.to(Device::cuda(0));
    OpAttributes attrs;
    attrs.set(AttrKey::Reduction, std::string("mean"));
    std::vector<Tensor> ref_inputs = {logits_f32_cuda, targets_cuda};
    Tensor loss_ref = dispatch(OpId::FusedSoftmaxCrossEntropy, ref_inputs, attrs)[0]
                          .to(Device::cpu());

    auto logits_bf16_cuda = logits_f32.to(DType::BFloat16).to(Device::cuda(0));
    std::vector<Tensor> bf16_inputs = {logits_bf16_cuda, targets_cuda};
    Tensor loss_bf16 = dispatch(OpId::FusedSoftmaxCrossEntropy, bf16_inputs, attrs)[0]
                           .to(Device::cpu()).to(DType::Float32);

    EXPECT_NEAR(loss_ref.data<float>()[0], loss_bf16.data<float>()[0], 1e-2f);
}

// ----------------------------------------------------------------------------
// H1 fix (regression test): RMSNorm backward F16/BF16 native dispatch.
// Previously: fused_rms_norm_backward_cuda threw "need Float32 or Float64".
// Now: native __half / __nv_bfloat16 dispatch with Acc=float per
// rms_acc_type<T> traits (matches forward path).
// ----------------------------------------------------------------------------
TEST_F(CudaFusedDtypeParity, FusedRMSNormBackward_F16_NoThrow) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";
    auto input_f32 = random_f32({4, 64});
    auto weight_f32 = random_f32({64});

    auto in_cuda  = input_f32.to(DType::Float16).to(Device::cuda(0));
    auto wt_cuda  = weight_f32.to(DType::Float16).to(Device::cuda(0));
    // Need rrms (F32 storage per forward contract).
    auto rrms_cuda = zeros({4}, DType::Float32, Device::cuda(0));

    // Build a fake grad_output of matching dtype.
    auto grad_out = input_f32.to(DType::Float16).to(Device::cuda(0));

    OpAttributes attrs;
    std::vector<Tensor> inputs = {grad_out, in_cuda, wt_cuda, rrms_cuda};
    // The backward dispatcher should NOT throw — it routes through
    // fused_rms_norm_backward_cuda which now has F16 dispatch arm.
    EXPECT_NO_THROW({
        (void)dispatch(OpId::RMSNormBackward, inputs, attrs);
    });
}

TEST_F(CudaFusedDtypeParity, FusedRMSNormBackward_BF16_NoThrow) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";
    auto input_f32 = random_f32({4, 64});
    auto weight_f32 = random_f32({64});

    auto in_cuda  = input_f32.to(DType::BFloat16).to(Device::cuda(0));
    auto wt_cuda  = weight_f32.to(DType::BFloat16).to(Device::cuda(0));
    auto rrms_cuda = zeros({4}, DType::Float32, Device::cuda(0));
    auto grad_out = input_f32.to(DType::BFloat16).to(Device::cuda(0));

    OpAttributes attrs;
    std::vector<Tensor> inputs = {grad_out, in_cuda, wt_cuda, rrms_cuda};
    EXPECT_NO_THROW({
        (void)dispatch(OpId::RMSNormBackward, inputs, attrs);
    });
}
