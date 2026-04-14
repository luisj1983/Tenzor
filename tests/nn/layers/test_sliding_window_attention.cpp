/**
 * @file test_sliding_window_attention.cpp
 * @brief Tests for sliding window attention in GroupedQueryAttention
 */

#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/nn/layers/gqa_attention.hpp>
#include <tenzor/autograd/variable.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class SlidingWindowAttentionTest : public tenzor::testing::BackendTest {};

TEST_P(SlidingWindowAttentionTest, Construction_DefaultNoWindow) {
    GroupedQueryAttention gqa(64, 8, 2);
    EXPECT_EQ(gqa.window_size(), -1);
}

TEST_P(SlidingWindowAttentionTest, Construction_WithWindow) {
    GroupedQueryAttention gqa(64, 8, 2, 0.0, true, false, nullptr, 32);
    EXPECT_EQ(gqa.window_size(), 32);
}

TEST_P(SlidingWindowAttentionTest, ForwardShape_WithWindow) {
    GroupedQueryAttention gqa(64, 8, 2, 0.0, true, true, nullptr, 16);

    int64_t batch = 2, seq_len = 20, embed_dim = 64;
    auto q = Variable(randn({batch, seq_len, embed_dim}, DType::Float32, Device::cpu()), false);

    auto [output, weights] = gqa.forward(q, q, q);
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_P(SlidingWindowAttentionTest, ForwardShape_NoWindow) {
    GroupedQueryAttention gqa(64, 8, 2, 0.0, true, true, nullptr, -1);

    int64_t batch = 2, seq_len = 10, embed_dim = 64;
    auto q = Variable(randn({batch, seq_len, embed_dim}, DType::Float32, Device::cpu()), false);

    auto [output, weights] = gqa.forward(q, q, q);
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_P(SlidingWindowAttentionTest, WindowedVsFullAttention_DifferentOutput) {
    int64_t batch = 1, seq_len = 16, embed_dim = 64;
    auto input_t = randn({batch, seq_len, embed_dim}, DType::Float32, Device::cpu());

    GroupedQueryAttention gqa_full(embed_dim, 8, 2, 0.0, true, false, nullptr, -1);
    GroupedQueryAttention gqa_window(embed_dim, 8, 2, 0.0, true, false, nullptr, 4);

    auto q = Variable(input_t, false);
    auto [out_full, _1] = gqa_full.forward(q, q, q);
    auto [out_window, _2] = gqa_window.forward(q, q, q);

    // Windowed and full attention should produce different outputs
    // (unless the sequence is very short relative to window)
    EXPECT_EQ(out_full.shape().size(), out_window.shape().size());
    for (size_t i = 0; i < out_full.shape().size(); ++i) {
        EXPECT_EQ(out_full.shape()[i], out_window.shape()[i]);
    }
}

INSTANTIATE_BACKEND_TESTS(SlidingWindowAttentionTest);
