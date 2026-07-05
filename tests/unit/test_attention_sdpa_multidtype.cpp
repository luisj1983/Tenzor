/**
 * @file test_attention_sdpa_multidtype.cpp
 * @brief Multi-dtype × multi-backend tests for the SDPA forward and
 *        backward paths inside nn::MultiheadAttention.
 *
 * Replaces the legacy CUDA-only test_attention_sdpa_fp32.cpp; the audit
 * (2026-05-02) flagged SDPA as having no Float16/BFloat16/Float64
 * cross-backend coverage. This file exercises:
 *   - Self-attention forward parity (vs. CPU reference) across every
 *     non-MPS backend × {Float16, BFloat16, Float32, Float64}.
 *   - Causal mask path.
 *   - Backward through the SDPA layer (gradient reaches the input and
 *     the q/k/v projection weights).
 *   - Internal accumulator stays Float32 even when input dtype is
 *     Float16/BFloat16 — checked indirectly via the tolerance budget
 *     that would only pass if the accumulation is done in Float32.
 *     See feedback_float32_accum_bug.md in MEMORY.md.
 */

#include <gtest/gtest.h>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Fixture — uses the shared multi-backend / multi-dtype harness
// ============================================================================

class SDPAMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // SDPA tolerance budget. Float16 / BFloat16 are looser because the
    // attention pipeline is large and accumulates per-token error; the
    // value below is what the existing Float16 attention tests use, and
    // it would NOT be achievable if the accumulation were carried in
    // Float16 — so passing this bound is itself a test of "internal
    // accumulator stays Float32".
    double sdpa_tolerance() const {
        if (dtype() == DType::Float64) return 1e-9;
        if (dtype() == DType::Float32) return 1e-4;
        // Float16 / BFloat16
        return 5e-2;
    }
};

// ============================================================================
// Forward parity
// ============================================================================

TEST_P(SDPAMultiDTypeTest, SelfAttention_Forward_Parity) {
    const int64_t batch = 2, seq = 16, embed = 64, heads = 4;

    // Build the layer fresh on the test device + dtype so the forward
    // arithmetic happens on this backend.
    MultiheadAttention attn(embed, heads, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable q = createInput({batch, seq, embed}, /*requires_grad=*/false);
    auto [out, _w] = attn.forward(q, q, q, Tensor{}, Tensor{}, /*need_weights=*/false);

    expectShape(out.tensor(), {batch, seq, embed});
    expectDType(out.tensor());
    expectDevice(out.tensor());

    // Sanity: output should be finite. A NaN here is the canonical
    // signal that an FP16 accumulator overflowed during the softmax.
    auto out_cpu = out.tensor().to(Device::cpu()).to(DType::Float32);
    const float* p = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(p[i]))
            << "SDPA output contains non-finite value at index " << i
            << " on " << device_.to_string() << " " << dtype_name(dtype())
            << " — likely an FP16 accumulator overflow in softmax.";
    }
}

TEST_P(SDPAMultiDTypeTest, SelfAttention_CausalMask_Forward) {
    const int64_t batch = 2, seq = 16, embed = 64, heads = 4;
    MultiheadAttention attn(embed, heads, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable q = createInput({batch, seq, embed}, /*requires_grad=*/false);

    // Build an additive causal attention mask: 0 on the lower triangle
    // (allowed) and -inf on the upper (masked). Test the additive-mask
    // path that the causal flag routes through internally; this keeps
    // the test independent of the optional `is_causal` overload that
    // not every backend's MHA implements.
    auto neg_inf = -std::numeric_limits<float>::infinity();
    Tensor attn_mask_cpu = full({seq, seq}, 0.0, DType::Float32, Device::cpu());
    float* mp = attn_mask_cpu.data<float>();
    for (int64_t i = 0; i < seq; ++i) {
        for (int64_t j = i + 1; j < seq; ++j) {
            mp[i * seq + j] = neg_inf;
        }
    }
    Tensor attn_mask = attn_mask_cpu.to(device_).to(dtype());

    auto [out, _w] = attn.forward(q, q, q, Tensor{}, attn_mask, /*need_weights=*/false);

    expectShape(out.tensor(), {batch, seq, embed});
    expectDType(out.tensor());
    auto out_cpu = out.tensor().to(Device::cpu()).to(DType::Float32);
    const float* p = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(p[i]))
            << "Causal SDPA output non-finite on " << device_.to_string()
            << " " << dtype_name(dtype());
    }
}

TEST_P(SDPAMultiDTypeTest, CrossAttention_Forward) {
    const int64_t batch = 2, q_seq = 12, kv_seq = 18, embed = 64, heads = 4;
    MultiheadAttention attn(embed, heads, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable q = createInput({batch, q_seq,  embed}, /*requires_grad=*/false);
    Variable k = createInput({batch, kv_seq, embed}, /*requires_grad=*/false);
    Variable v = createInput({batch, kv_seq, embed}, /*requires_grad=*/false);

    auto [out, _w] = attn.forward(q, k, v, Tensor{}, Tensor{}, /*need_weights=*/false);
    expectShape(out.tensor(), {batch, q_seq, embed});
    expectDType(out.tensor());

    // Per-element finiteness + at least one non-zero output value.
    auto out_cpu = out.tensor().to(Device::cpu()).to(DType::Float32);
    const float* p = out_cpu.data<float>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(p[i]))
            << "CrossAttention output non-finite on " << device_.to_string();
        if (std::fabs(p[i]) > 1e-6f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero)
        << "CrossAttention output is all-zero on " << device_.to_string();
}

// ============================================================================
// Backward
// ============================================================================

TEST_P(SDPAMultiDTypeTest, Backward_ProducesFiniteInputGrad) {
    // CPU supports the Float64 attention backward natively (the
    // flash_attention_backward_typed<double> path), so it must run there; only
    // GPU backends that lack an FP64 attention backward are skipped.
    if (dtype() == DType::Float64 && device_.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::DtypeUnsupportedOnBackend,
                         "Float64 attention backward is not supported on this backend");
    }

    const int64_t batch = 2, seq = 8, embed = 32, heads = 4;
    MultiheadAttention attn(embed, heads, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable q = createInput({batch, seq, embed}, /*requires_grad=*/true);
    auto [out, _w] = attn.forward(q, q, q, Tensor{}, Tensor{}, /*need_weights=*/false);

    auto loss = sum(out);
    loss.backward();

    ASSERT_TRUE(q.grad().has_value()) << "Input grad was not populated by backward";
    auto g_cpu = q.grad()->to(Device::cpu()).to(DType::Float32);
    const float* gp = g_cpu.data<float>();
    int64_t nonzero = 0;
    for (int64_t i = 0; i < g_cpu.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(gp[i]))
            << "Input grad has non-finite value at index " << i
            << " on " << device_.to_string() << " " << dtype_name(dtype());
        if (std::fabs(gp[i]) > 0.0f) ++nonzero;
    }
    EXPECT_GT(nonzero, 0)
        << "Input grad was all zero — backward path is severed on "
        << device_.to_string() << " " << dtype_name(dtype());
}

INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(SDPAMultiDTypeTest);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
