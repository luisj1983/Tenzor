/**
 * @file test_multihead_attention_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for nn::MultiheadAttention layer
 *
 * Standalone layer-level tests (constructor, state_dict, forward, backward,
 * masks). Complements the lower-level functional tests in
 * test_mha_functional_multidtype.cpp.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class MultiheadAttentionMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Constructor / parameter layout
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, ConstructorRegistersExpectedParameters) {
    nn::MultiheadAttention mha(16, 4, /*dropout=*/0.0, /*bias=*/true);
    convert_model(mha);

    auto params = mha.parameters();
    EXPECT_GT(params.size(), 0u) << "MultiheadAttention should register parameters";
    // embed_dim = 16 → at least weights on input/output projections.
    size_t total = countParameters(params);
    EXPECT_GE(total, static_cast<size_t>(16 * 16))
        << "Expected at least embed_dim^2 parameters in projections";
}

TEST_P(MultiheadAttentionMultiDTypeTest, StateDictRoundtrip) {
    nn::MultiheadAttention mha(16, 4);
    convert_model(mha);

    auto state = mha.state_dict();
    EXPECT_FALSE(state.empty());

    // Recreate and load — numeric contents should match.
    nn::MultiheadAttention mha2(16, 4);
    convert_model(mha2);
    mha2.load_state_dict(state);

    auto state2 = mha2.state_dict();
    EXPECT_EQ(state.size(), state2.size());
}

// ============================================================================
// Forward pass
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, ForwardSelfAttentionShape) {
    const int64_t batch = 2, seq = 4, embed = 16, num_heads = 4;
    nn::MultiheadAttention mha(embed, num_heads, 0.0, true, false, false,
                               0, 0, /*batch_first=*/true);
    convert_model(mha);

    Variable q = createInput({batch, seq, embed}, false);
    auto [out, weights] = mha.forward(q, q, q);

    expectShape(out.tensor(), {batch, seq, embed});
    expectDevice(out.tensor());
    expectDType(out.tensor());
}

TEST_P(MultiheadAttentionMultiDTypeTest, ForwardCrossAttentionShape) {
    const int64_t batch = 2, seq_q = 4, seq_k = 6, embed = 16, num_heads = 4;
    nn::MultiheadAttention mha(embed, num_heads, 0.0, true, false, false,
                               0, 0, /*batch_first=*/true);
    convert_model(mha);

    Variable q = createInput({batch, seq_q, embed}, false);
    Variable k = createInput({batch, seq_k, embed}, false);
    Variable v = createInput({batch, seq_k, embed}, false);

    auto [out, weights] = mha.forward(q, k, v);
    expectShape(out.tensor(), {batch, seq_q, embed});
    expectDevice(out.tensor());
}

// ============================================================================
// Backward / gradient flow
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, GradientFlowsToQuery) {
    const int64_t batch = 1, seq = 3, embed = 8, num_heads = 2;
    nn::MultiheadAttention mha(embed, num_heads, 0.0, true, false, false,
                               0, 0, /*batch_first=*/true);
    convert_model(mha);

    Variable q = createInput({batch, seq, embed}, true);
    auto [out, _] = mha.forward(q, q, q);
    auto grad = tenzor::ones({batch, seq, embed}, dtype_, device_);
    out.backward(grad);

    EXPECT_GRAD_FLOWS(q);
    expectShape(*q.grad(), {batch, seq, embed});
}

// ============================================================================
// Masking behavior
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, AttnMaskChangesOutput) {
    const int64_t batch = 1, seq = 4, embed = 8, num_heads = 2;
    nn::MultiheadAttention mha(embed, num_heads, 0.0, true, false, false,
                               0, 0, /*batch_first=*/true);
    convert_model(mha);

    Variable q = createInput({batch, seq, embed}, false);

    auto [out_unmasked, _] = mha.forward(q, q, q);

    // Lower-triangular causal mask via -inf above the diagonal.
    Tensor mask({seq, seq}, dtype_, Device::cpu());
    auto mask_f32 = tenzor::zeros({seq, seq}, DType::Float32, Device::cpu());
    auto* mf = mask_f32.data<float>();
    for (int64_t i = 0; i < seq; ++i) {
        for (int64_t j = 0; j < seq; ++j) {
            if (j > i) mf[i * seq + j] = -std::numeric_limits<float>::infinity();
        }
    }
    if (dtype_ != DType::Float32) mask_f32 = mask_f32.to(dtype_);
    mask = mask_f32.to(device_);

    auto [out_masked, __] = mha.forward(q, q, q, Tensor{}, mask);

    // Masked vs unmasked must differ on at least one output element.
    auto u = out_unmasked.tensor().to(Device::cpu()).to(DType::Float32);
    auto m = out_masked.tensor().to(Device::cpu()).to(DType::Float32);
    const auto* up = u.data<float>();
    const auto* mp = m.data<float>();
    bool any_diff = false;
    for (int64_t i = 0; i < u.numel(); ++i) {
        if (std::abs(up[i] - mp[i]) > atol_) { any_diff = true; break; }
    }
    EXPECT_TRUE(any_diff) << "Causal mask had no effect on output";
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MultiheadAttentionMultiDTypeTest);
