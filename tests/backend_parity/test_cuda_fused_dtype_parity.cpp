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
#include "parity_test_utils.hpp"
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
    static bool has_oneapi() {
        try {
            auto t = zeros({1}, DType::Float32, Device::oneapi(0));
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

// FINDING 17 follow-up: unlike the CPU-vs-CUDA tests in this file, the tests
// below compare CUDA's native-low-precision result against CUDA's OWN F32
// result — both sides require CUDA to compute at all (there is no CPU
// reference anywhere in the comparison), so a CPU-only host has no live
// value to check against a recorded golden even in principle. These
// necessarily stay CUDA-only; SKIP_WITH_REASON is the correct, honest
// outcome here, not a coverage gap to paper over.
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

// Regression: fused RMSNorm must produce the same result on a non-contiguous
// input as on its contiguous equivalent. The GPU kernels index storage flat, so
// a missing contiguity guard reads the wrong layout. Compares each available GPU
// backend's non-contiguous result against the CPU contiguous reference.
TEST_F(CudaFusedDtypeParity, FusedRMSNorm_NonContiguousMatchesContiguous) {
    auto input = random_f32({4, 8});   // logical [4,8], contiguous on CPU
    auto weight = random_f32({8});
    OpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(1e-6));

    std::vector<Tensor> ref_inputs = {input, weight};
    Tensor ref = dispatch(OpId::FusedRMSNorm, ref_inputs, attrs)[0]
                     .to(Device::cpu()).to(DType::Float64);

    // base[j,i] = input[i,j]; base.transpose(0,1) is a non-contiguous view whose
    // logical values equal `input`.
    Tensor base = input.transpose(0, 1).contiguous();  // [8,4] contiguous

    struct Cand { const char* name; Device dev; };
    std::vector<Cand> cands = {
        {"cuda", Device::cuda(0)}, {"oneapi", Device::oneapi(0)},
        {"rocm", Device::rocm(0)}, {"vulkan", Device::vulkan(0)},
    };
    int tested = 0;
    for (auto& c : cands) {
        try { auto probe = zeros({1}, DType::Float32, c.dev); (void)probe; }
        catch (...) { continue; }
        Tensor in_nc = base.to(c.dev).transpose(0, 1);  // [4,8] non-contiguous
        ASSERT_FALSE(in_nc.is_contiguous()) << c.name;
        std::vector<Tensor> inp = {in_nc, weight.to(c.dev)};
        Tensor out = dispatch(OpId::FusedRMSNorm, inp, attrs)[0]
                         .to(Device::cpu()).to(DType::Float64);
        ASSERT_EQ(out.numel(), ref.numel()) << c.name;
        const double* r = ref.data<double>();
        const double* o = out.data<double>();
        for (int64_t i = 0; i < ref.numel(); ++i)
            EXPECT_NEAR(o[i], r[i], 1e-3) << c.name << " elem " << i;
        ++tested;
    }
    if (tested == 0)
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "no GPU backend");
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
// OneAPI native F16/BF16 FusedSoftmaxCrossEntropy. Unlike FusedRMSNorm (where
// ROCm/OneAPI both widen-narrow via .to(Float32) — see the file-top comment),
// OneAPI's fused_softmax_cross_entropy_kernel has a genuine native sycl::half/
// bfloat16 code path (src/backends/oneapi/kernels/fused_ops.cpp) that reads
// half/bf16 directly and accumulates in float, exactly like CUDA's. That's a
// real on-device numeric path a CPU-only host can't stand in for, so it needs
// its own live check the same way the CUDA tests above do (findings.txt —
// this file was previously CUDA-only without confirming whether ROCm/OneAPI
// had a native path to test; verified here that OneAPI does for this op).
// ROCm's fused_softmax_cross_entropy_hip widen-narrows (calls the F32 kernel
// then casts back), so there is no native ROCm path to test here.
// ----------------------------------------------------------------------------
TEST_F(CudaFusedDtypeParity, FusedSoftmaxCrossEntropy_F16_NativeMatchesRef_OneAPI) {
    if (!has_oneapi()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "OneAPI not available");

    auto logits_f32 = random_f32({16, 32});
    auto targets = zeros({16}, DType::Int64, Device::cpu());
    auto* tp = targets.data<int64_t>();
    for (int64_t i = 0; i < targets.numel(); ++i) tp[i] = (i * 7) % 32;

    auto logits_f32_dev = logits_f32.to(Device::oneapi(0));
    auto targets_dev = targets.to(Device::oneapi(0));
    OpAttributes attrs;
    attrs.set(AttrKey::Reduction, std::string("mean"));
    std::vector<Tensor> ref_inputs = {logits_f32_dev, targets_dev};
    Tensor loss_ref = dispatch(OpId::FusedSoftmaxCrossEntropy, ref_inputs, attrs)[0]
                          .to(Device::cpu());

    auto logits_f16_dev = logits_f32.to(DType::Float16).to(Device::oneapi(0));
    std::vector<Tensor> f16_inputs = {logits_f16_dev, targets_dev};
    Tensor loss_f16 = dispatch(OpId::FusedSoftmaxCrossEntropy, f16_inputs, attrs)[0]
                          .to(Device::cpu()).to(DType::Float32);

    ASSERT_EQ(loss_ref.dtype(), DType::Float32);
    ASSERT_EQ(loss_f16.dtype(), DType::Float32);
    EXPECT_NEAR(loss_ref.data<float>()[0], loss_f16.data<float>()[0], 5e-3f);
}

TEST_F(CudaFusedDtypeParity, FusedSoftmaxCrossEntropy_BF16_NativeMatchesRef_OneAPI) {
    if (!has_oneapi()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "OneAPI not available");

    auto logits_f32 = random_f32({16, 32});
    auto targets = zeros({16}, DType::Int64, Device::cpu());
    auto* tp = targets.data<int64_t>();
    for (int64_t i = 0; i < targets.numel(); ++i) tp[i] = (i * 7) % 32;

    auto logits_f32_dev = logits_f32.to(Device::oneapi(0));
    auto targets_dev = targets.to(Device::oneapi(0));
    OpAttributes attrs;
    attrs.set(AttrKey::Reduction, std::string("mean"));
    std::vector<Tensor> ref_inputs = {logits_f32_dev, targets_dev};
    Tensor loss_ref = dispatch(OpId::FusedSoftmaxCrossEntropy, ref_inputs, attrs)[0]
                          .to(Device::cpu());

    auto logits_bf16_dev = logits_f32.to(DType::BFloat16).to(Device::oneapi(0));
    std::vector<Tensor> bf16_inputs = {logits_bf16_dev, targets_dev};
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

    // Reference: F32 backward on CUDA. Dispatch contract is
    // [grad_output, input, rrms, weight] (autograd saves [input, rrms, weight]).
    auto grad_f32_cuda = grad_f32.to(Device::cuda(0));
    std::vector<Tensor> ref_inputs = {grad_f32_cuda, in_f32_cuda, rrms_f32_cuda, wt_f32_cuda};
    auto ref = dispatch(OpId::RMSNormBackward, ref_inputs, attrs);
    ASSERT_GE(ref.size(), 2u);
    Tensor ref_gi = ref[0].to(Device::cpu()).to(DType::Float32);
    Tensor ref_gw = ref[1].to(Device::cpu()).to(DType::Float32);

    // Native F16/BF16 backward on CUDA (rrms stays F32 per contract).
    auto grad_nat_cuda = grad_f32.to(native_dtype).to(Device::cuda(0));
    auto in_nat_cuda   = input_f32.to(native_dtype).to(Device::cuda(0));
    auto wt_nat_cuda   = weight_f32.to(native_dtype).to(Device::cuda(0));
    std::vector<Tensor> nat_inputs = {grad_nat_cuda, in_nat_cuda, rrms_f32_cuda, wt_nat_cuda};
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

// F050: CUDA FusedConv2dBnReLU must honor dilation (was silently ignored) and the
// training path (batch stats + running-stat update), matching the CPU kernel.
namespace {
static Tensor positive_var(int64_t c) {
    auto t = zeros({c}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t i = 0; i < c; ++i) p[i] = 1.0f + 0.1f * static_cast<float>(i);
    return t;
}
}  // namespace

// FINDING 17: previously an unconditional SKIP_WITH_REASON() on a host
// without CUDA, with zero golden::* integration. On a CPU-only host, fall
// back to comparing against a recorded golden instead of skipping outright.
TEST_F(CudaFusedDtypeParity, FusedConv2dBnReLU_DilationMatchesCPU) {
    auto input  = random_f32({2, 3, 8, 8});
    auto weight = random_f32({4, 3, 3, 3});
    auto bias   = random_f32({4});
    auto gamma  = random_f32({4});
    auto beta   = random_f32({4});
    auto rm     = random_f32({4});
    auto rv     = positive_var(4);

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   static_cast<int64_t>(1));
    attrs.set(AttrKey::Padding,  static_cast<int64_t>(1));
    attrs.set(AttrKey::Dilation, static_cast<int64_t>(2));
    attrs.set(AttrKey::Eps,      static_cast<double>(1e-5));

    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::FusedConv2dBnReLU, ins, attrs)[0];
    };

    Device target = has_cuda() ? Device::cuda(0) : Device::cpu();
    ::tenzor::testing::test_operation_parity_single(
        op, {input, weight, bias, gamma, beta, rm, rv}, target, 1e-5f, 1e-4f,
        "FusedConv2dBnReLU_Dilation");
}

// F055: fused conv-activation with asymmetric stride/padding. The non-cuDNN CUDA
// fallback (native per-axis conv2d + activation) previously rejected asymmetric
// config; this checks the op-level result matches CPU (cuDNN serves it here, the
// native kernel is the non-cuDNN mirror of the same math).
TEST_F(CudaFusedDtypeParity, FusedConv2dReLU_AsymmetricMatchesCPU) {
    auto input  = random_f32({2, 3, 9, 11});
    auto weight = random_f32({4, 3, 3, 3});

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH,  static_cast<int64_t>(2));
    attrs.set(AttrKey::StrideW,  static_cast<int64_t>(1));
    attrs.set(AttrKey::PaddingH, static_cast<int64_t>(2));
    attrs.set(AttrKey::PaddingW, static_cast<int64_t>(1));
    attrs.set(AttrKey::Groups,   static_cast<int64_t>(1));

    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::FusedConv2dReLU, ins, attrs)[0];
    };

    Device target = has_cuda() ? Device::cuda(0) : Device::cpu();
    ::tenzor::testing::test_operation_parity_single(
        op, {input, weight}, target, 1e-5f, 1e-3f, "FusedConv2dReLU_Asymmetric");
}

// FINDING 17 follow-up: this test also asserts on the in-place-mutated
// running_mean/running_var side effect (lines below), which doesn't fit
// test_operation_parity_single's single-return-value golden contract.
// Left CUDA-only rather than force-fitting a workaround for the mutation
// check specifically.
TEST_F(CudaFusedDtypeParity, FusedConv2dBnReLU_TrainingMatchesCPU) {
    if (!has_cuda()) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");
    auto input  = random_f32({2, 3, 6, 6});
    auto weight = random_f32({5, 3, 3, 3});
    auto bias   = random_f32({5});
    auto gamma  = random_f32({5});
    auto beta   = random_f32({5});
    auto rm0    = random_f32({5});
    auto rv0    = positive_var(5);

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   static_cast<int64_t>(1));
    attrs.set(AttrKey::Padding,  static_cast<int64_t>(1));
    attrs.set(AttrKey::Momentum, static_cast<double>(0.1));
    attrs.set(AttrKey::Eps,      static_cast<double>(1e-5));
    attrs.set(AttrKey::Training, true);

    // Independent running-stat copies per backend (updated in place by the op).
    Tensor rm_cpu = rm0.clone(), rv_cpu = rv0.clone();
    std::vector<Tensor> cpu_in = {input, weight, bias, gamma, beta, rm_cpu, rv_cpu};
    Tensor ref = dispatch(OpId::FusedConv2dBnReLU, cpu_in, attrs)[0];

    std::vector<Tensor> cu_in = {
        input.to(Device::cuda(0)), weight.to(Device::cuda(0)), bias.to(Device::cuda(0)),
        gamma.to(Device::cuda(0)), beta.to(Device::cuda(0)),
        rm0.to(Device::cuda(0)), rv0.to(Device::cuda(0))};
    Tensor out = dispatch(OpId::FusedConv2dBnReLU, cu_in, attrs)[0].to(Device::cpu());

    ASSERT_EQ(ref.numel(), out.numel());
    const float* rp = ref.data<float>();
    const float* op = out.data<float>();
    for (int64_t i = 0; i < ref.numel(); ++i)
        EXPECT_NEAR(rp[i], op[i], 1e-4f) << "training conv-bn-relu elem " << i;

    // Running statistics must be updated identically on both backends.
    Tensor rm_cu = cu_in[5].to(Device::cpu());
    Tensor rv_cu = cu_in[6].to(Device::cpu());
    const float* rmc = rm_cpu.data<float>();  const float* rmg = rm_cu.data<float>();
    const float* rvc = rv_cpu.data<float>();  const float* rvg = rv_cu.data<float>();
    for (int64_t i = 0; i < rm_cpu.numel(); ++i) {
        EXPECT_NEAR(rmc[i], rmg[i], 1e-5f) << "running_mean elem " << i;
        EXPECT_NEAR(rvc[i], rvg[i], 1e-5f) << "running_var elem " << i;
    }
}
