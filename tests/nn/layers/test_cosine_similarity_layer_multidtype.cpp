/**
 * @file test_cosine_similarity_layer_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for CosineSimilarity layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/distance.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class CosineSimilarityMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(CosineSimilarityMultiDTypeTest, IdenticalInputs) {
    CosineSimilarity cs(1);
    auto x = createInput({4, 8}, false);
    auto result = cs(x, x);
    auto data = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* d = data.data<float>();
    for (int64_t i = 0; i < data.numel(); ++i) {
        EXPECT_NEAR(d[i], 1.0f, atol());
    }
}

TEST_P(CosineSimilarityMultiDTypeTest, OutputShape) {
    CosineSimilarity cs(1);
    auto x = createInput({4, 8}, false);
    auto y = createInput({4, 8}, false);
    auto result = cs(x, y);
    ASSERT_EQ(result.tensor().shape().size(), 1);
    ASSERT_EQ(result.tensor().shape()[0], 4);
}

TEST_P(CosineSimilarityMultiDTypeTest, OrthogonalInputs) {
    CosineSimilarity cs(1);
    auto xt = createZeros({1, 4});
    auto yt = createZeros({1, 4});
    // Set values on CPU then move
    auto xt_cpu = tenzor::zeros({1, 4}, DType::Float32, Device::cpu());
    auto yt_cpu = tenzor::zeros({1, 4}, DType::Float32, Device::cpu());
    xt_cpu.data<float>()[0] = 1.0f;
    yt_cpu.data<float>()[1] = 1.0f;
    auto x = Variable(xt_cpu.to(dtype()).to(device()), false);
    auto y = Variable(yt_cpu.to(dtype()).to(device()), false);
    auto result = cs(x, y);
    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(out.data<float>()[0], 0.0f, atol());
}

TEST_P(CosineSimilarityMultiDTypeTest, OppositeInputs) {
    CosineSimilarity cs(1);
    auto t = createRandn({4, 8});
    auto x = Variable(t, false);
    auto y = Variable(tenzor::neg(t), false);
    auto result = cs(x, y);
    auto data = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* d = data.data<float>();
    for (int64_t i = 0; i < data.numel(); ++i) {
        EXPECT_NEAR(d[i], -1.0f, atol());
    }
}

TEST_P(CosineSimilarityMultiDTypeTest, DTypePreserved) {
    CosineSimilarity cs(1);
    auto x = createInput({4, 8}, false);
    auto y = createInput({4, 8}, false);
    auto result = cs(x, y);
    expectDType(result.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(CosineSimilarityMultiDTypeTest);
