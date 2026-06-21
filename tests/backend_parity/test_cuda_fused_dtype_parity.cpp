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
#include "../multi_backend_dtype_fixture.hpp"  // FF.28: SKIP_WITH_REASON
#include <algorithm>
#include <cmath>

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
    if (!has_cuda()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

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
    if (!has_cuda()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

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
    if (!has_cuda()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

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
// Shared helper: run the RMSNorm backward in a given dtype on CUDA and return
// {grad_input, grad_weight} on CPU as Float32. rrms always travels as Float32
// (the forward contract stores reciprocal-RMS in F32), so the only dtype that
// varies between the F32 reference path and the native F16/BF16 path is the
// storage of grad_output / input / weight.
static void check_rmsnorm_backward_native_matches_ref(DType native_dtype, float tol) {
    auto input_f32  = random_f32({4, 64});
    auto weight_f32 = random_f32({64});
    auto grad_f32   = random_f32({4, 64});

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::Eps, static_cast<double>(1e-6));

    // Forward (F32 on CUDA) to obtain the correct rrms (F32) that backward needs.
    auto in_f32_cuda = input_f32.to(Device::cuda(0));
    auto wt_f32_cuda = weight_f32.to(Device::cuda(0));
    std::vector<Tensor> fwd_inputs = {in_f32_cuda, wt_f32_cuda};
    auto fwd_out = dispatch(OpId::FusedRMSNorm, fwd_inputs, fwd_attrs);
    ASSERT_GE(fwd_out.size(), 2u) << "FusedRMSNorm must return {output, rrms}";
    Tensor rrms_f32_cuda = fwd_out[1];  // F32 reciprocal RMS, on CUDA

    OpAttributes attrs;

    // Reference: F32 backward on CUDA.
    auto grad_f32_cuda = grad_f32.to(Device::cuda(0));
    std::vector<Tensor> ref_inputs = {grad_f32_cuda, in_f32_cuda, wt_f32_cuda, rrms_f32_cuda};
    auto ref = dispatch(OpId::RMSNormBackward, ref_inputs, attrs);
    ASSERT_GE(ref.size(), 2u);
    Tensor ref_gi = ref[0].to(Device::cpu()).to(DType::Float32);
    Tensor ref_gw = ref[1].to(Device::cpu()).to(DType::Float32);

    // Native F16/BF16 backward on CUDA (rrms stays F32 per contract).
    auto grad_nat_cuda = grad_f32.to(native_dtype).to(Device::cuda(0));
    auto in_nat_cuda   = input_f32.to(native_dtype).to(Device::cuda(0));
    auto wt_nat_cuda   = weight_f32.to(native_dtype).to(Device::cuda(0));
    std::vector<Tensor> nat_inputs = {grad_nat_cuda, in_nat_cuda, wt_nat_cuda, rrms_f32_cuda};
    auto nat = dispatch(OpId::RMSNormBackward, nat_inputs, attrs);
    ASSERT_GE(nat.size(), 2u);
    Tensor nat_gi = nat[0].to(Device::cpu()).to(DType::Float32);
    Tensor nat_gw = nat[1].to(Device::cpu()).to(DType::Float32);

    ASSERT_EQ(ref_gi.numel(), nat_gi.numel());
    ASSERT_EQ(ref_gw.numel(), nat_gw.numel());

    // Both grads must match the F32 reference to the dtype tolerance — a
    // backward returning zeros / NaNs / wrong-shape fails here (the old
    // EXPECT_NO_THROW would have passed it).
    {
        auto* rp = ref_gi.data<float>();
        auto* np = nat_gi.data<float>();
        double max_abs_gi = 0.0;
        for (int64_t i = 0; i < ref_gi.numel(); ++i) {
            EXPECT_NEAR(rp[i], np[i], tol) << "grad_input elem " << i;
            max_abs_gi = std::max(max_abs_gi, std::abs(static_cast<double>(rp[i])));
        }
        // The reference grad must be non-trivial, otherwise the comparison is vacuous.
        EXPECT_GT(max_abs_gi, 0.0) << "reference grad_input is identically zero";
    }
    {
        auto* rp = ref_gw.data<float>();
        auto* np = nat_gw.data<float>();
        double max_abs_gw = 0.0;
        for (int64_t i = 0; i < ref_gw.numel(); ++i) {
            EXPECT_NEAR(rp[i], np[i], tol) << "grad_weight elem " << i;
            max_abs_gw = std::max(max_abs_gw, std::abs(static_cast<double>(rp[i])));
        }
        EXPECT_GT(max_abs_gw, 0.0) << "reference grad_weight is identically zero";
    }
}

// H1 regression: RMSNorm backward F16 native dispatch must produce grads that
// match the F32 reference, not merely avoid throwing.
TEST_F(CudaFusedDtypeParity, FusedRMSNormBackward_F16_MatchesRef) {
    if (!has_cuda()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");
    check_rmsnorm_backward_native_matches_ref(DType::Float16, 5e-3f);
}

TEST_F(CudaFusedDtypeParity, FusedRMSNormBackward_BF16_MatchesRef) {
    if (!has_cuda()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");
    check_rmsnorm_backward_native_matches_ref(DType::BFloat16, 1e-2f);
}
