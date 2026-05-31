/**
 * @file test_cosine_similarity_layer.cpp
 * @brief Tests for CosineSimilarity layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/distance.hpp>
#include "../../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class CosineSimilarityTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(CosineSimilarityTest, IdenticalInputs) {
    CosineSimilarity cs(1);
    auto x = Variable(randn({4, 8}, DType::Float32, device), false);
    auto result = cs(x, x);
    // Cosine similarity of identical vectors should be 1.0
    auto host = result.tensor().cpu();
    auto data = host.data<float>();
    for (int64_t i = 0; i < host.numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, 1e-5f);
    }
}

TEST_P(CosineSimilarityTest, OrthogonalInputs) {
    CosineSimilarity cs(1);
    auto x_cpu = zeros({1, 4}, DType::Float32, Device::cpu());
    auto y_cpu = zeros({1, 4}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = 1.0f;
    y_cpu.data<float>()[1] = 1.0f;
    auto x = Variable(x_cpu.to(device), false);
    auto y = Variable(y_cpu.to(device), false);
    auto result = cs(x, y);
    // Orthogonal vectors -> similarity = 0
    auto host = result.tensor().cpu();
    EXPECT_NEAR(host.data<float>()[0], 0.0f, 1e-5f);
}

TEST_P(CosineSimilarityTest, OppositeInputs) {
    CosineSimilarity cs(1);
    auto t = randn({4, 8}, DType::Float32, device);
    auto x = Variable(t, false);
    auto y = Variable(tenzor::neg(t), false);
    auto result = cs(x, y);
    // Opposite vectors -> similarity = -1
    auto host = result.tensor().cpu();
    auto data = host.data<float>();
    for (int64_t i = 0; i < host.numel(); ++i) {
        EXPECT_NEAR(data[i], -1.0f, 1e-5f);
    }
}

TEST_P(CosineSimilarityTest, OutputShape) {
    CosineSimilarity cs(1);
    auto x = Variable(randn({4, 8}, DType::Float32, device), false);
    auto y = Variable(randn({4, 8}, DType::Float32, device), false);
    auto result = cs(x, y);
    // Reduction along dim=1 -> shape [4]
    ASSERT_EQ(result.tensor().shape().size(), 1);
    ASSERT_EQ(result.tensor().shape()[0], 4);
}

INSTANTIATE_BACKEND_TESTS(CosineSimilarityTest);
