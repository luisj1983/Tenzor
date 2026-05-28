/**
 * @file test_dtype_preservation.cpp
 * @brief S13 — sampler/loss/FFT dtype-preservation regression tests.
 *
 * Several CPU kernels historically silently downcast (or upcast) their
 * output dtype away from the input dtype:
 *
 *   1. bernoulli / normal_sample — hard-wired Float32 output regardless
 *      of probability/mean dtype, so Float64 probs lost precision.
 *   2. fused_softmax_cross_entropy — restricted to rank-2 logits, and
 *      its half-precision path returned Float32 grad_logits, breaking
 *      seq2seq use cases and silently changing the gradient dtype.
 *   3. fused_attention — Float64 silently downcast to Float32 with a
 *      one-shot WARN. Now runs natively in double.
 *   4. ctc_loss_forward — always cast log_probs to Float32. Now keeps
 *      Float64 natively.
 *   5. mfcc — epsilon=1e-10 was unrepresentable at Float16/BFloat16,
 *      so log(0) produced -inf. Fixed via widen_narrow_compute.
 *   6. fft2 / fftn — half-precision inputs only worked through the 1-D
 *      fft_kernel path. ND paths now mirror that handling.
 *
 * Tests assert *dtype contracts*, not numeric values — RNG and
 * floating-point differences make value comparison brittle and
 * orthogonal to the actual S13 fix.
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/fused_ops.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

#include <cmath>
#include <vector>

using namespace tenzor;

namespace {

// Tensor::shape() returns std::span<const int64_t>, which doesn't have a
// usable operator== for GTest. Convert to std::vector for the comparison.
inline std::vector<int64_t> shape_v(const Tensor& t) {
    auto s = t.shape();
    return std::vector<int64_t>(s.begin(), s.end());
}


class DtypePreservationEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new DtypePreservationEnv);

// ---------------------------------------------------------------------------
// Fix 1: bernoulli / normal_sample dtype preservation
// ---------------------------------------------------------------------------

TEST(DtypePreservation, BernoulliFloat64) {
    auto probs = full({100}, 0.5, DType::Float64);
    auto out = bernoulli(probs);
    EXPECT_EQ(out.dtype(), DType::Float64)
        << "bernoulli(Float64) should preserve Float64; got "
        << dtype_name(out.dtype());
    EXPECT_EQ(shape_v(out), shape_v(probs));
}

TEST(DtypePreservation, BernoulliFloat16) {
    auto probs = full({100}, 0.5, DType::Float16);
    auto out = bernoulli(probs);
    EXPECT_EQ(out.dtype(), DType::Float16)
        << "bernoulli(Float16) should preserve Float16; got "
        << dtype_name(out.dtype());
}

TEST(DtypePreservation, BernoulliFloat32StaysFloat32) {
    // Float32 must remain bit-identical to the legacy path.
    auto probs = full({64}, 0.5, DType::Float32);
    auto out = bernoulli(probs);
    EXPECT_EQ(out.dtype(), DType::Float32);
}

TEST(DtypePreservation, NormalSampleFloat64) {
    auto mean = full({100}, 0.0, DType::Float64);
    auto std_t = full({100}, 1.0, DType::Float64);
    auto out = normal(mean, std_t);
    EXPECT_EQ(out.dtype(), DType::Float64)
        << "normal(Float64, Float64) should preserve Float64; got "
        << dtype_name(out.dtype());
    EXPECT_EQ(shape_v(out), shape_v(mean));
}

TEST(DtypePreservation, NormalSampleBFloat16) {
    auto mean = full({64}, 0.0, DType::BFloat16);
    auto std_t = full({64}, 1.0, DType::BFloat16);
    auto out = normal(mean, std_t);
    EXPECT_EQ(out.dtype(), DType::BFloat16)
        << "normal(BFloat16, BFloat16) should preserve BFloat16";
}

// ---------------------------------------------------------------------------
// Fix 2: fused_softmax_cross_entropy dtype + rank generalisation
// ---------------------------------------------------------------------------

TEST(DtypePreservation, FusedSoftmaxCEDtypeRoundTripF16) {
    // Build logits at Float16 and target as Int64. The kernel must accept
    // half-precision logits and (when grad is requested) return grad_logits
    // at Float16 — the loss itself is allowed to stay Float32 by intent.
    const int64_t N = 4, C = 5;
    auto logits = randn({N, C}).to(DType::Float16);
    // Int64 targets — kernel reads `targets.data<int64_t>()` directly.
    std::vector<int64_t> tdata = {0, 1, 2, 3, 4};
    Tensor targets({N}, DType::Int64, Device::cpu());
    auto* tptr = targets.data<int64_t>();
    for (int64_t i = 0; i < N; ++i) tptr[i] = tdata[i];

    // Direct dispatch so we can inspect grad_logits (the public wrapper
    // returns only the loss).
    OpAttributes attrs;
    attrs.set(AttrKey::Reduction, std::string("mean"));
    attrs.set(AttrKey::ComputeGrad, true);
    std::vector<Tensor> inputs = {logits, targets};
    auto outputs = dispatch<OpId::FusedSoftmaxCrossEntropy>(inputs, attrs);
    ASSERT_GE(outputs.size(), 2u)
        << "fused_softmax_cross_entropy with ComputeGrad=true must return "
           "{loss, grad_logits}";
    EXPECT_EQ(outputs[1].dtype(), DType::Float16)
        << "grad_logits must narrow back to the input Float16 dtype; got "
        << dtype_name(outputs[1].dtype());
    EXPECT_EQ(shape_v(outputs[1]), shape_v(logits));
}

TEST(DtypePreservation, FusedSoftmaxCERank3) {
    // logits (N=2, T=3, C=5), targets (N=2, T=3): forward + grad must both
    // round-trip the original shape.
    const int64_t N = 2, T = 3, C = 5;
    auto logits = randn({N, T, C}, DType::Float32);
    // Build int64 targets in [0, C). Index column j = (i+1) % C for variety.
    Tensor targets({N, T}, DType::Int64, Device::cpu());
    auto* tptr = targets.data<int64_t>();
    for (int64_t i = 0; i < N * T; ++i) tptr[i] = i % C;

    // Public API: should accept rank-3 logits now.
    auto loss = tenzor::ops::fused_softmax_cross_entropy(logits, targets, "mean");
    EXPECT_EQ(loss.numel(), 1) << "mean reduction → scalar loss";
    EXPECT_EQ(loss.dtype(), DType::Float32);

    // grad_logits shape via direct dispatch
    OpAttributes attrs;
    attrs.set(AttrKey::Reduction, std::string("none"));
    attrs.set(AttrKey::ComputeGrad, true);
    std::vector<Tensor> inputs = {logits, targets};
    auto outputs = dispatch<OpId::FusedSoftmaxCrossEntropy>(inputs, attrs);
    ASSERT_GE(outputs.size(), 2u);
    // Per-sample loss for "none" should be (N, T).
    EXPECT_EQ(outputs[0].ndim(), 2);
    EXPECT_EQ(outputs[0].shape()[0], N);
    EXPECT_EQ(outputs[0].shape()[1], T);
    // grad_logits should match logits shape.
    EXPECT_EQ(shape_v(outputs[1]), shape_v(logits));
}

// ---------------------------------------------------------------------------
// Fix 3: fused_attention Float64 native
// ---------------------------------------------------------------------------

TEST(DtypePreservation, FusedAttentionFloat64) {
    // Tiny 4D attention with Float64 Q/K/V — output must stay Float64.
    const int64_t B = 1, H = 1, S = 2, D = 4;
    auto Q = randn({B, H, S, D}).to(DType::Float64);
    auto K = randn({B, H, S, D}).to(DType::Float64);
    auto V = randn({B, H, S, D}).to(DType::Float64);

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, 1.0 / std::sqrt(static_cast<double>(D)));
    attrs.set(AttrKey::Causal, false);
    std::vector<Tensor> inputs = {Q, K, V};
    auto outputs = dispatch<OpId::FusedAttention>(inputs, attrs);
    ASSERT_GE(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].dtype(), DType::Float64)
        << "fused_attention(Float64) must preserve Float64; got "
        << dtype_name(outputs[0].dtype());
    EXPECT_EQ(shape_v(outputs[0]), shape_v(Q));

    // Output must be finite (no NaN/Inf from a bad widen/narrow path).
    const double* out = outputs[0].data<double>();
    for (int64_t i = 0; i < outputs[0].numel(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "non-finite at " << i;
    }
}

// ---------------------------------------------------------------------------
// Fix 4: CTC loss Float64 native
// ---------------------------------------------------------------------------

TEST(DtypePreservation, CTCLossFloat64) {
    // T_max=4, N=1, C=3 (incl. blank=0). Uniform log-probs log(1/C).
    // Targets: [1, 2] (length 2).
    const int64_t T = 4, N = 1, C = 3;
    auto log_probs = full({T, N, C}, std::log(1.0 / static_cast<double>(C)),
                          DType::Float64);
    EXPECT_EQ(log_probs.dtype(), DType::Float64);

    // Targets, input/target lengths in Int32 (kernel-contracted).
    Tensor targets({N, 2}, DType::Int32, Device::cpu());
    targets.data<int32_t>()[0] = 1;
    targets.data<int32_t>()[1] = 2;
    Tensor input_lengths({N}, DType::Int32, Device::cpu());
    input_lengths.data<int32_t>()[0] = T;
    Tensor target_lengths({N}, DType::Int32, Device::cpu());
    target_lengths.data<int32_t>()[0] = 2;

    OpAttributes attrs;
    attrs.set(AttrKey::Blank, static_cast<int64_t>(0));
    attrs.set(AttrKey::ZeroInfinity, false);
    std::vector<Tensor> inputs = {log_probs, targets, input_lengths, target_lengths};
    auto outputs = dispatch<OpId::CTCLossForward>(inputs, attrs);
    ASSERT_EQ(outputs.size(), 2u)
        << "CTCLossForward returns {loss_per_sample, raw_grad}";
    EXPECT_EQ(outputs[0].dtype(), DType::Float64)
        << "CTC loss with Float64 log_probs must stay Float64; got "
        << dtype_name(outputs[0].dtype());
    EXPECT_EQ(outputs[1].dtype(), DType::Float64)
        << "CTC raw_grad with Float64 log_probs must stay Float64";

    EXPECT_EQ(shape_v(outputs[0]), (std::vector<int64_t>{N}));
    EXPECT_EQ(shape_v(outputs[1]), (std::vector<int64_t>{T, N, C}));

    // Loss should be finite and positive (uniform distribution, target
    // sequence achievable in T steps).
    const double loss = outputs[0].data<double>()[0];
    EXPECT_TRUE(std::isfinite(loss)) << "CTC loss must be finite";
    EXPECT_GT(loss, 0.0);
}

// ---------------------------------------------------------------------------
// Fix 5: MFCC half-precision survives epsilon shift
// ---------------------------------------------------------------------------

TEST(DtypePreservation, MFCCFloat16Recovery) {
    // A short Float16 waveform. The internal mel_spec contains very small
    // values; before S13, log(mel_spec + 1e-10) collapsed to log(0) = -inf
    // because the epsilon rounded to 0 at FP16. After the widen-narrow
    // fix the output should be finite.
    const int64_t n_samples = 1024;
    auto waveform = randn({n_samples}).to(DType::Float16);

    // n_fft=256, hop=128, n_mels=16, n_mfcc=8 — small enough for unit-test
    // budget.
    auto result = tenzor::fft::mfcc(
        waveform,
        /*sample_rate=*/16000,
        /*n_mfcc=*/8,
        /*n_mels=*/16,
        /*n_fft=*/256,
        /*hop_length=*/128);

    EXPECT_GT(result.numel(), 0);
    // We don't pin the output dtype (the downstream DCT may keep widened
    // precision), but every value must be finite — that's what proves the
    // epsilon-shift no longer underflows under half precision.
    auto result_f32 = (result.dtype() == DType::Float32)
                      ? result
                      : result.to(DType::Float32);
    const float* data = result_f32.data<float>();
    for (int64_t i = 0; i < result_f32.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(data[i]))
            << "MFCC[" << i << "] = " << data[i]
            << " — expected finite after Float16 widen-narrow on log step";
    }
}

// ---------------------------------------------------------------------------
// Fix 6: fft2 / fftn accept half-precision input
// ---------------------------------------------------------------------------

TEST(DtypePreservation, FFT2Float16Input) {
    // 2D Float16 input must not throw and should produce a Complex64 output.
    auto x = randn({4, 4}).to(DType::Float16);
    Tensor out;
    ASSERT_NO_THROW(out = tenzor::fft::fft2(x));
    EXPECT_EQ(out.dtype(), DType::Complex64)
        << "fft2(Float16) must widen internally to Complex64; got "
        << dtype_name(out.dtype());
    EXPECT_EQ(shape_v(out), shape_v(x));
}

TEST(DtypePreservation, FFTNFloat16Input) {
    // 3D Float16 input through fftn over the last two dims.
    auto x = randn({2, 4, 4}).to(DType::Float16);
    Tensor out;
    ASSERT_NO_THROW(out = tenzor::fft::fftn(x));
    EXPECT_EQ(out.dtype(), DType::Complex64)
        << "fftn(Float16) must widen internally to Complex64";
    EXPECT_EQ(shape_v(out), shape_v(x));
}

TEST(DtypePreservation, FFT2BFloat16Input) {
    // BFloat16 path mirrors Float16.
    auto x = randn({4, 4}).to(DType::BFloat16);
    Tensor out;
    ASSERT_NO_THROW(out = tenzor::fft::fft2(x));
    EXPECT_EQ(out.dtype(), DType::Complex64);
}

} // namespace
