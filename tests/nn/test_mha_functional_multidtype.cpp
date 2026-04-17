/**
 * @file test_mha_functional_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for nn::functional::multi_head_attention_forward
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/functional.hpp>

namespace F = tenzor::nn::functional;
using namespace tenzor;
using namespace tenzor::testing;

class MHAFunctionalMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    static constexpr int64_t batch = 2;
    static constexpr int64_t seq_len = 4;
    static constexpr int64_t embed_dim = 8;
    static constexpr int64_t num_heads = 2;

    struct Projections {
        Tensor in_proj_weight;
        Tensor in_proj_bias;
        Tensor out_proj_weight;
        Tensor out_proj_bias;
    };

    Projections make_projections() {
        return {
            createRandn({3 * embed_dim, embed_dim}),
            createZeros({3 * embed_dim}),
            createRandn({embed_dim, embed_dim}),
            createZeros({embed_dim}),
        };
    }
};

TEST_P(MHAFunctionalMultiDTypeTest, BasicForwardShape) {
    auto query = createRandn({batch, seq_len, embed_dim});
    auto key   = createRandn({batch, seq_len, embed_dim});
    auto value = createRandn({batch, seq_len, embed_dim});
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

TEST_P(MHAFunctionalMultiDTypeTest, DifferentKVSeqLen) {
    int64_t kv_seq_len = 6;
    auto query = createRandn({batch, seq_len, embed_dim});
    auto key   = createRandn({batch, kv_seq_len, embed_dim});
    auto value = createRandn({batch, kv_seq_len, embed_dim});
    auto proj  = make_projections();

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias);

    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_P(MHAFunctionalMultiDTypeTest, WithAttentionMask) {
    auto query = createRandn({batch, seq_len, embed_dim});
    auto key   = createRandn({batch, seq_len, embed_dim});
    auto value = createRandn({batch, seq_len, embed_dim});
    auto proj  = make_projections();
    auto mask  = createZeros({seq_len, seq_len});

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias,
        mask);

    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_P(MHAFunctionalMultiDTypeTest, NeedWeightsFalse) {
    auto query = createRandn({batch, seq_len, embed_dim});
    auto key   = createRandn({batch, seq_len, embed_dim});
    auto value = createRandn({batch, seq_len, embed_dim});
    auto proj  = make_projections();

    auto [output, attn_weights] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias,
        std::nullopt, 0.0, false, false);

    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
    EXPECT_EQ(attn_weights.numel(), 0);
}

TEST_P(MHAFunctionalMultiDTypeTest, OutputValuesFinite) {
    auto query = createRandn({batch, seq_len, embed_dim});
    auto key   = createRandn({batch, seq_len, embed_dim});
    auto value = createRandn({batch, seq_len, embed_dim});
    auto proj  = make_projections();

    auto [output, _] = F::multi_head_attention_forward(
        query, key, value, num_heads,
        proj.in_proj_weight, proj.in_proj_bias,
        proj.out_proj_weight, proj.out_proj_bias);

    auto out_cpu = output.to(Device::cpu()).to(DType::Float32);
    auto* d = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i])) << "Non-finite value at index " << i;
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MHAFunctionalMultiDTypeTest);
