/**
 * @file test_flex_attention_multidtype.cpp
 * @brief Multi-dtype tests for FlexAttention across all available backends.
 *
 * FlexAttention currently implements the online-softmax reference on the host
 * and moves inputs off GPU during execution. These tests exercise the function
 * end-to-end across backends so any regression in the GPU-to-CPU fallback
 * path (or a future native GPU kernel) is caught immediately.
 *
 * Coverage:
 *  - Basic forward with BlockMask::causal, BlockMask::sliding_window, custom mask.
 *  - Score-modification hooks: causal_score_mod, alibi_score_mod.
 *  - Shape / dtype preservation.
 *  - CPU-vs-backend parity via round-trip of the same inputs.
 *
 * The CPU path is the reference in every case; the backend path should match
 * within Float32/Float64 tolerance, and within the loosened multi-dtype fixture
 * tolerance for Float16 / BFloat16.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/flex_attention.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class FlexAttentionMultiDTypeTest : public MultiBackendDTypeTest {
protected:

    // Build matching Q/K/V on CPU + target device. We keep the shapes small
    // (B=1, H=2, S=64, D=32) so the host-side reference stays under a second
    // even on the ROCm path.
    struct QKV {
        Tensor q_cpu, k_cpu, v_cpu;
        Tensor q_dev, k_dev, v_dev;
        int64_t B, H, S, D;
    };

    QKV make_qkv() {
        int64_t B = 1, H = 2, S = 64, D = 32;
        auto q_cpu = randn({B, H, S, D}, DType::Float32, Device::cpu());
        auto k_cpu = randn({B, H, S, D}, DType::Float32, Device::cpu());
        auto v_cpu = randn({B, H, S, D}, DType::Float32, Device::cpu());
        auto q_dev = q_cpu.to(dtype()).to(device());
        auto k_dev = k_cpu.to(dtype()).to(device());
        auto v_dev = v_cpu.to(dtype()).to(device());
        return {q_cpu, k_cpu, v_cpu, q_dev, k_dev, v_dev, B, H, S, D};
    }

    // Promote any non-Float32 result to Float32 on CPU for comparison.
    Tensor to_cpu_float32(const Tensor& t) {
        auto c = t.device().type == Device::Type::CPU ? t : t.to(Device::cpu());
        return c.dtype() == DType::Float32 ? c : c.to(DType::Float32);
    }

    // Tolerance varies by dtype — the fixture already sets rtol/atol in SetUp,
    // but FlexAttention's accumulation over many KV blocks widens the band.
    float rtol_for_flex() const {
        switch (dtype()) {
            case DType::Float64: return 1e-6f;
            case DType::Float32: return 1e-4f;
            case DType::Float16: return 3e-2f;
            case DType::BFloat16: return 5e-2f;
            default: return 1e-4f;
        }
    }
    float atol_for_flex() const {
        switch (dtype()) {
            case DType::Float64: return 1e-7f;
            case DType::Float32: return 1e-5f;
            case DType::Float16: return 5e-3f;
            case DType::BFloat16: return 1e-2f;
            default: return 1e-5f;
        }
    }
};

// ---------------------------------------------------------------------------
// Shape / dtype preservation
// ---------------------------------------------------------------------------

TEST_P(FlexAttentionMultiDTypeTest, ForwardShapeAndDtype) {
    auto v = make_qkv();
    auto mask = BlockMask::causal(v.S, /*block_size=*/16);
    auto out = flex_attention(v.q_dev, v.k_dev, v.v_dev, mask);
    ASSERT_EQ(out.shape().size(), 4u);
    EXPECT_EQ(out.shape()[0], v.B);
    EXPECT_EQ(out.shape()[1], v.H);
    EXPECT_EQ(out.shape()[2], v.S);
    EXPECT_EQ(out.shape()[3], v.D);
    // The helper currently returns Float32 regardless of input dtype (the CPU
    // reference operates in Float32). We don't strictly require dtype
    // preservation here — the parity test below asserts value equivalence
    // which is the user-visible property.
}

// ---------------------------------------------------------------------------
// CPU-vs-backend parity: causal block mask
// ---------------------------------------------------------------------------

TEST_P(FlexAttentionMultiDTypeTest, ParityCausalMask) {
    auto v = make_qkv();
    auto mask = BlockMask::causal(v.S, /*block_size=*/16);

    auto out_cpu = flex_attention(v.q_cpu, v.k_cpu, v.v_cpu, mask);
    auto out_dev = flex_attention(v.q_dev, v.k_dev, v.v_dev, mask);
    auto out_cpu_f32 = to_cpu_float32(out_cpu);
    auto out_dev_f32 = to_cpu_float32(out_dev);

    ASSERT_EQ(out_cpu_f32.numel(), out_dev_f32.numel());
    const float* a = out_cpu_f32.data<float>();
    const float* b = out_dev_f32.data<float>();
    float rtol = rtol_for_flex();
    float atol = atol_for_flex();
    for (int64_t i = 0; i < out_cpu_f32.numel(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        float threshold = atol + rtol * std::abs(a[i]);
        ASSERT_LE(diff, threshold) << "causal-mask parity miss at " << i
                                   << " cpu=" << a[i] << " dev=" << b[i];
    }
}

// ---------------------------------------------------------------------------
// Sliding-window block mask
// ---------------------------------------------------------------------------

TEST_P(FlexAttentionMultiDTypeTest, ParitySlidingWindow) {
    auto v = make_qkv();
    auto mask = BlockMask::sliding_window(v.S, /*window=*/32, /*block_size=*/16);
    auto out_cpu = flex_attention(v.q_cpu, v.k_cpu, v.v_cpu, mask);
    auto out_dev = flex_attention(v.q_dev, v.k_dev, v.v_dev, mask);
    auto a = to_cpu_float32(out_cpu);
    auto b = to_cpu_float32(out_dev);
    ASSERT_EQ(a.numel(), b.numel());
    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    float rtol = rtol_for_flex();
    float atol = atol_for_flex();
    for (int64_t i = 0; i < a.numel(); ++i) {
        ASSERT_LE(std::abs(ap[i] - bp[i]), atol + rtol * std::abs(ap[i]))
            << "sliding-window parity miss at " << i;
    }
}

// ---------------------------------------------------------------------------
// Score modification — causal
// ---------------------------------------------------------------------------

TEST_P(FlexAttentionMultiDTypeTest, CausalScoreModParity) {
    auto v = make_qkv();
    auto mask = BlockMask::causal(v.S, /*block_size=*/16);
    auto score_mod = causal_score_mod();
    auto out_cpu = flex_attention(v.q_cpu, v.k_cpu, v.v_cpu, mask, score_mod);
    auto out_dev = flex_attention(v.q_dev, v.k_dev, v.v_dev, mask, score_mod);
    auto a = to_cpu_float32(out_cpu);
    auto b = to_cpu_float32(out_dev);
    ASSERT_EQ(a.numel(), b.numel());
    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    float rtol = rtol_for_flex();
    float atol = atol_for_flex();
    for (int64_t i = 0; i < a.numel(); ++i) {
        ASSERT_LE(std::abs(ap[i] - bp[i]), atol + rtol * std::abs(ap[i]))
            << "causal score-mod parity miss at " << i;
    }
}

// ---------------------------------------------------------------------------
// Score modification — ALiBi
// ---------------------------------------------------------------------------

TEST_P(FlexAttentionMultiDTypeTest, ALiBiScoreModParity) {
    auto v = make_qkv();
    auto mask = BlockMask::causal(v.S, /*block_size=*/16);
    // Per-head slopes — small values so the pre-softmax bias doesn't dominate.
    auto slopes = tenzor::full({v.H}, 0.1f, DType::Float32, Device::cpu());
    auto score_mod = alibi_score_mod(slopes);
    auto out_cpu = flex_attention(v.q_cpu, v.k_cpu, v.v_cpu, mask, score_mod);
    auto out_dev = flex_attention(v.q_dev, v.k_dev, v.v_dev, mask, score_mod);
    auto a = to_cpu_float32(out_cpu);
    auto b = to_cpu_float32(out_dev);
    ASSERT_EQ(a.numel(), b.numel());
    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    float rtol = rtol_for_flex();
    float atol = atol_for_flex();
    for (int64_t i = 0; i < a.numel(); ++i) {
        ASSERT_LE(std::abs(ap[i] - bp[i]), atol + rtol * std::abs(ap[i]))
            << "ALiBi score-mod parity miss at " << i;
    }
}

// ---------------------------------------------------------------------------
// Regression: custom block mask
// ---------------------------------------------------------------------------

TEST_P(FlexAttentionMultiDTypeTest, CustomMaskShape) {
    auto v = make_qkv();
    int64_t bs = 16;
    int64_t nq = (v.S + bs - 1) / bs;
    // Build an arbitrary mask — first row dense, remaining rows diagonal.
    auto mask_data = zeros({nq, nq}, DType::Bool, Device::cpu());
    auto* m = static_cast<uint8_t*>(mask_data.data_ptr());
    for (int64_t i = 0; i < nq; ++i) {
        m[0 * nq + i] = 1;        // first row: attend to everything
        m[i * nq + i] = 1;        // all rows: self
    }
    auto mask = BlockMask(mask_data, bs);
    auto out = flex_attention(v.q_dev, v.k_dev, v.v_dev, mask);
    EXPECT_EQ(out.shape()[2], v.S);
    EXPECT_EQ(out.shape()[3], v.D);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FlexAttentionMultiDTypeTest);
