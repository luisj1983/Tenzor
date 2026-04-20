/**
 * @file test_sparse_embedding.cpp
 * @brief Tests for SparseEmbedding layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sparse_embedding.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class SparseEmbeddingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(SparseEmbeddingTest, ForwardShape) {
    SparseEmbedding emb(100, 32);
    auto indices = randint(0, 100, {4, 5}, DType::Int64, Device::cpu());
    auto input = Variable(indices, false);
    auto output = emb.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 3);
    ASSERT_EQ(shape[0], 4);
    ASSERT_EQ(shape[1], 5);
    ASSERT_EQ(shape[2], 32);
}

TEST_F(SparseEmbeddingTest, DeterministicLookup) {
    SparseEmbedding emb(10, 4);
    // Same index should give same embedding
    auto idx1 = zeros({1, 1}, DType::Int64, Device::cpu());
    auto idx2 = zeros({1, 1}, DType::Int64, Device::cpu());
    auto out1 = emb.forward(Variable(idx1, false));
    auto out2 = emb.forward(Variable(idx2, false));
    auto d1 = out1.tensor().data<float>();
    auto d2 = out2.tensor().data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_EQ(d1[i], d2[i]);
    }
}

TEST_F(SparseEmbeddingTest, Backward) {
    SparseEmbedding emb(50, 16);
    auto indices = randint(0, 50, {2, 3}, DType::Int64, Device::cpu());
    auto input = Variable(indices, false);
    auto output = emb.forward(input);
    auto loss = sum(output);
    loss.backward();

    // Gradient flows to the weight table (indices are Int64 and cannot carry
    // grads). TESTING.md triad: has_grad, grad numel == param numel, and the
    // gradient actually has non-zero magnitude (catches silent-zero backward).
    ASSERT_TRUE(emb.weight().has_grad());
    ASSERT_EQ(emb.weight().grad()->numel(), emb.weight().tensor().numel());

    auto grad_data = emb.weight().grad()->data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < emb.weight().grad()->numel(); ++i) {
        ASSERT_FALSE(std::isnan(grad_data[i]));
        ASSERT_FALSE(std::isinf(grad_data[i]));
        max_abs = std::max(max_abs, std::abs(grad_data[i]));
    }
    EXPECT_GT(max_abs, 0.0f);
}
