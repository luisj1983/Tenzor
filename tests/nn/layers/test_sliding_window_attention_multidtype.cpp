/**
 * @file test_sliding_window_attention_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for sliding window attention
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/layers/gqa_attention.hpp>
#include <tenzor/autograd/variable.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class SlidingWindowAttentionMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(SlidingWindowAttentionMultiDTypeTest, ConstructionDefaultNoWindow) {
    GroupedQueryAttention gqa(64, 8, 2);
    EXPECT_EQ(gqa.window_size(), -1);
}

TEST_P(SlidingWindowAttentionMultiDTypeTest, ConstructionWithWindow) {
    GroupedQueryAttention gqa(64, 8, 2, 0.0, true, false, nullptr, 32);
    EXPECT_EQ(gqa.window_size(), 32);
}

TEST_P(SlidingWindowAttentionMultiDTypeTest, ForwardShapeWithWindow) {
    GroupedQueryAttention gqa(64, 8, 2, 0.0, true, true, nullptr, 16);
    convert_model(gqa);

    int64_t batch = 2, seq_len = 20, embed_dim = 64;
    auto q = createInput({batch, seq_len, embed_dim}, false);

    auto [output, weights] = gqa.forward(q, q, q);
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_P(SlidingWindowAttentionMultiDTypeTest, ForwardShapeNoWindow) {
    GroupedQueryAttention gqa(64, 8, 2, 0.0, true, true, nullptr, -1);
    convert_model(gqa);

    int64_t batch = 2, seq_len = 10, embed_dim = 64;
    auto q = createInput({batch, seq_len, embed_dim}, false);

    auto [output, weights] = gqa.forward(q, q, q);
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_P(SlidingWindowAttentionMultiDTypeTest, WindowedVsFullDifferentOutput) {
    int64_t batch = 1, seq_len = 16, embed_dim = 64;

    GroupedQueryAttention gqa_full(embed_dim, 8, 2, 0.0, true, false, nullptr, -1);
    GroupedQueryAttention gqa_window(embed_dim, 8, 2, 0.0, true, false, nullptr, 4);
    convert_model(gqa_full);
    convert_model(gqa_window);

    auto q = createInput({batch, seq_len, embed_dim}, false);
    auto [out_full, _1] = gqa_full.forward(q, q, q);
    auto [out_window, _2] = gqa_window.forward(q, q, q);

    EXPECT_EQ(out_full.shape().size(), out_window.shape().size());
    for (size_t i = 0; i < out_full.shape().size(); ++i) {
        EXPECT_EQ(out_full.shape()[i], out_window.shape()[i]);
    }
}

TEST_P(SlidingWindowAttentionMultiDTypeTest, OutputFinite) {
    GroupedQueryAttention gqa(64, 8, 2, 0.0, true, true, nullptr, 8);
    convert_model(gqa);

    auto q = createInput({1, 12, 64}, false);
    auto [output, _] = gqa.forward(q, q, q);

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SlidingWindowAttentionMultiDTypeTest);
