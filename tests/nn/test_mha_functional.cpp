/**
 * @file test_mha_functional.cpp
 * @brief Tests for nn::functional::multi_head_attention_forward
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/ops/creation.hpp>
#include "../backend_test_fixture.hpp"

namespace F = tenzor::nn::functional;
using namespace tenzor;

class MHAFunctionalTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Helpers for common dimensions
    static constexpr int64_t batch = 2;
    static constexpr int64_t seq_len = 4;
    static constexpr int64_t embed_dim = 8;
    static constexpr int64_t num_heads = 2;

    // Create projection weights and biases
    // in_proj_weight: (3 * embed_dim, embed_dim) for Q, K, V
    // in_proj_bias: (3 * embed_dim)
    // out_proj_weight: (embed_dim, embed_dim)
    // out_proj_bias: (embed_dim)
    struct Projections {
        Tensor in_proj_weight;
        Tensor in_proj_bias;
        Tensor out_proj_weight;
        Tensor out_proj_bias;
    };

    Projections make_projections() {
        return {
            rand({3 * embed_dim, embed_dim}, DType::Float32, device),
            zeros({3 * embed_dim}, DType::Float32, device),
            rand({embed_dim, embed_dim}, DType::Float32, device),
            zeros({embed_dim}, DType::Float32, device),
        };
    }
};

// ============================================================================
// Basic forward pass
// ============================================================================

TEST_P(MHAFunctionalTest, BasicForwardShape) {
    auto query = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto key   = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto value = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto proj  = make_projections();

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias);

    EXPECT_EQ(output.shape().size(), 3u);
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

// ============================================================================
// Different seq lengths for Q and K/V
// ============================================================================

TEST_P(MHAFunctionalTest, DifferentKVSeqLen) {
    int64_t kv_seq_len = 6;
    auto query = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto key   = rand({batch, kv_seq_len, embed_dim}, DType::Float32, device);
    auto value = rand({batch, kv_seq_len, embed_dim}, DType::Float32, device);
    auto proj  = make_projections();

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias);

    // Output seq_len follows query
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

// ============================================================================
// With attention mask
// ============================================================================

TEST_P(MHAFunctionalTest, WithAttentionMask) {
    auto query = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto key   = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto value = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto proj  = make_projections();

    // Additive mask: (seq_len, seq_len) — 0 means attend, -inf means mask
    auto mask = zeros({seq_len, seq_len}, DType::Float32, device);

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias,
        mask);

    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

// ============================================================================
// need_weights flag
// ============================================================================

TEST_P(MHAFunctionalTest, NeedWeightsTrue) {
    auto query = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto key   = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto value = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto proj  = make_projections();

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias,
        std::nullopt,  // no mask
        0.0,           // no dropout
        false,         // not training
        true);         // need_weights

    // Attention weights: (batch, num_heads, seq_q, seq_k)
    EXPECT_EQ(attn_weights.shape().size(), 4u);
    EXPECT_EQ(attn_weights.shape()[0], batch);
    EXPECT_EQ(attn_weights.shape()[1], num_heads);
    EXPECT_EQ(attn_weights.shape()[2], seq_len);
    EXPECT_EQ(attn_weights.shape()[3], seq_len);

    // Attention weights should sum to ~1 along last dim (softmax output)
    auto attn_weights_cpu = attn_weights.cpu();
    auto w_cpu = attn_weights_cpu.data<float>();
    int64_t head_dim_k = seq_len;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t q = 0; q < seq_len; ++q) {
                float row_sum = 0.0f;
                for (int64_t k = 0; k < head_dim_k; ++k) {
                    int64_t idx = ((b * num_heads + h) * seq_len + q) * head_dim_k + k;
                    row_sum += w_cpu[idx];
                }
                EXPECT_NEAR(row_sum, 1.0f, 1e-4f)
                    << "Attention weights should sum to 1 at b=" << b
                    << " h=" << h << " q=" << q;
            }
        }
    }
}

TEST_P(MHAFunctionalTest, NeedWeightsFalse) {
    auto query = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto key   = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto value = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto proj  = make_projections();

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias,
        std::nullopt,
        0.0,
        false,
        false);  // need_weights = false

    // Output should still be valid
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);

    // Attention weights should be empty (numel == 0)
    EXPECT_EQ(attn_weights.numel(), 0);
}

// ============================================================================
// Output values are finite
// ============================================================================

TEST_P(MHAFunctionalTest, OutputValuesFinite) {
    auto query = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto key   = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto value = rand({batch, seq_len, embed_dim}, DType::Float32, device);
    auto proj  = make_projections();

    auto [output, _] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias);

    auto output_cpu = output.cpu();
    auto* d = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i])) << "Non-finite value at index " << i;
    }
}

INSTANTIATE_BACKEND_TESTS(MHAFunctionalTest);
