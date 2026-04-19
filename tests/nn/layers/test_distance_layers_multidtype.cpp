/**
 * @file test_distance_layers_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for distance layers.
 *
 * Covers PairwiseDistance and CosineSimilarity (Bilinear is exercised via
 * the existing Linear test infrastructure). Forward shape + backward
 * gradient population.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/distance.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DistanceLayersMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(DistanceLayersMultiDTypeTest, PairwiseDistance_L2_ForwardShape) {
    nn::PairwiseDistance pd(/*p=*/2.0);
    Variable x1 = createInput({4, 8}, false);
    Variable x2 = createInput({4, 8}, false);
    auto out = pd.forward(x1, x2);
    expectShape(out.tensor(), {4});
    expectDType(out.tensor());
}

TEST_P(DistanceLayersMultiDTypeTest, PairwiseDistance_L1_BackwardGradPopulated) {
    nn::PairwiseDistance pd(/*p=*/1.0);
    Variable x1 = createInput({3, 5}, true);
    Variable x2 = createInput({3, 5}, true);
    auto out = pd.forward(x1, x2);
    sum(out).backward();
    ASSERT_TRUE(x1.has_grad()) << device().to_string();
    ASSERT_TRUE(x2.has_grad()) << device().to_string();
    auto g1 = max(abs(x1.grad()->to(Device::cpu()).to(DType::Float32)));
    auto g2 = max(abs(x2.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g1.item<float>(), 0.0f);
    EXPECT_GT(g2.item<float>(), 0.0f);
}

TEST_P(DistanceLayersMultiDTypeTest, PairwiseDistance_L2_BackwardGradPopulated) {
    nn::PairwiseDistance pd(/*p=*/2.0);
    Variable x1 = createInput({3, 5}, true);
    Variable x2 = createInput({3, 5}, true);
    auto out = pd.forward(x1, x2);
    sum(out).backward();
    ASSERT_TRUE(x1.has_grad());
    auto g1 = max(abs(x1.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g1.item<float>(), 0.0f) << "L2 input grad all-zero on " << device().to_string();
}

TEST_P(DistanceLayersMultiDTypeTest, CosineSimilarity_ForwardShape) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "cosine_similarity has no half-precision dispatch";
    }
    nn::CosineSimilarity cs(/*dim=*/-1);
    Variable x1 = createInput({4, 8}, false);
    Variable x2 = createInput({4, 8}, false);
    auto out = cs.forward(x1, x2);
    expectShape(out.tensor(), {4});
}

TEST_P(DistanceLayersMultiDTypeTest, CosineSimilarity_BackwardGradPopulated) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "cosine_similarity has no half-precision dispatch";
    }
    nn::CosineSimilarity cs;
    Variable x1 = createInput({3, 5}, true);
    Variable x2 = createInput({3, 5}, true);
    auto out = cs.forward(x1, x2);
    sum(out).backward();
    ASSERT_TRUE(x1.has_grad());
    ASSERT_TRUE(x2.has_grad());
    auto g1 = max(abs(x1.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g1.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DistanceLayersMultiDTypeTest);
