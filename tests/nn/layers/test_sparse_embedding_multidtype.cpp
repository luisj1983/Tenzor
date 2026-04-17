/**
 * @file test_sparse_embedding_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for SparseEmbedding layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sparse_embedding.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class SparseEmbeddingMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);

        // Skip if sparse ops are not available on this backend
        try {
            auto test = tenzor::zeros({2, 2}, dtype(), device());
            (void)test;
        } catch (...) {
            GTEST_SKIP() << "Sparse ops not available on " << backend_name();
        }
    }
};

TEST_P(SparseEmbeddingMultiDTypeTest, ForwardShape) {
    SparseEmbedding emb(100, 32);
    convert_model(emb);

    auto indices = randint(0, 100, {4, 5}, DType::Int64, device());
    auto input = Variable(indices, false);
    auto output = emb.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 3);
    ASSERT_EQ(shape[0], 4);
    ASSERT_EQ(shape[1], 5);
    ASSERT_EQ(shape[2], 32);
}

TEST_P(SparseEmbeddingMultiDTypeTest, DeterministicLookup) {
    SparseEmbedding emb(10, 4);
    convert_model(emb);

    auto idx1 = zeros({1, 1}, DType::Int64, device());
    auto idx2 = zeros({1, 1}, DType::Int64, device());
    auto out1 = emb.forward(Variable(idx1, false));
    auto out2 = emb.forward(Variable(idx2, false));

    auto d1 = out1.tensor().to(Device::cpu()).to(DType::Float32);
    auto d2 = out2.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data1 = d1.data<float>();
    auto* data2 = d2.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(data1[i], data2[i], atol());
    }
}

TEST_P(SparseEmbeddingMultiDTypeTest, Backward) {
    SparseEmbedding emb(50, 16);
    convert_model(emb);

    auto indices = randint(0, 50, {2, 3}, DType::Int64, device());
    auto input = Variable(indices, false);
    auto output = emb.forward(input);
    auto loss = sum(output);
    EXPECT_NO_THROW(loss.backward());
}

TEST_P(SparseEmbeddingMultiDTypeTest, OutputFinite) {
    SparseEmbedding emb(20, 8);
    convert_model(emb);

    auto indices = randint(0, 20, {2, 4}, DType::Int64, device());
    auto output = emb.forward(Variable(indices, false));

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

TEST_P(SparseEmbeddingMultiDTypeTest, SingleIndex) {
    SparseEmbedding emb(10, 4);
    convert_model(emb);

    auto idx = zeros({1, 1}, DType::Int64, device());
    auto output = emb.forward(Variable(idx, false));
    EXPECT_EQ(output.tensor().shape()[0], 1);
    EXPECT_EQ(output.tensor().shape()[1], 1);
    EXPECT_EQ(output.tensor().shape()[2], 4);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SparseEmbeddingMultiDTypeTest);
