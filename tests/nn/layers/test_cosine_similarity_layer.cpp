/**
 * @file test_cosine_similarity_layer.cpp
 * @brief Tests for CosineSimilarity layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/distance.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class CosineSimilarityTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(CosineSimilarityTest, IdenticalInputs) {
    CosineSimilarity cs(1);
    auto x = Variable(randn({4, 8}, DType::Float32, Device::cpu()), false);
    auto result = cs(x, x);
    // Cosine similarity of identical vectors should be 1.0
    auto data = result.tensor().data<float>();
    for (int64_t i = 0; i < result.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, 1e-5f);
    }
}

TEST_F(CosineSimilarityTest, OrthogonalInputs) {
    CosineSimilarity cs(1);
    auto x = Variable(zeros({1, 4}, DType::Float32, Device::cpu()), false);
    auto y = Variable(zeros({1, 4}, DType::Float32, Device::cpu()), false);
    x.tensor().data<float>()[0] = 1.0f;
    y.tensor().data<float>()[1] = 1.0f;
    auto result = cs(x, y);
    // Orthogonal vectors -> similarity = 0
    EXPECT_NEAR(result.tensor().data<float>()[0], 0.0f, 1e-5f);
}

TEST_F(CosineSimilarityTest, OppositeInputs) {
    CosineSimilarity cs(1);
    auto t = randn({4, 8}, DType::Float32, Device::cpu());
    auto x = Variable(t, false);
    auto y = Variable(tenzor::neg(t), false);
    auto result = cs(x, y);
    // Opposite vectors -> similarity = -1
    auto data = result.tensor().data<float>();
    for (int64_t i = 0; i < result.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], -1.0f, 1e-5f);
    }
}

TEST_F(CosineSimilarityTest, OutputShape) {
    CosineSimilarity cs(1);
    auto x = Variable(randn({4, 8}, DType::Float32, Device::cpu()), false);
    auto y = Variable(randn({4, 8}, DType::Float32, Device::cpu()), false);
    auto result = cs(x, y);
    // Reduction along dim=1 -> shape [4]
    ASSERT_EQ(result.tensor().shape().size(), 1);
    ASSERT_EQ(result.tensor().shape()[0], 4);
}
