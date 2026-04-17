/**
 * @file test_histogramdd_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for histogramdd reduction operation
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/creation.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class HistogramddMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(HistogramddMultiDTypeTest, Basic2DKnownData) {
    // histogramdd operates on CPU
    auto input = tenzor::zeros({4, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.1f; d[1] = 0.1f;
    d[2] = 0.9f; d[3] = 0.9f;
    d[4] = 0.1f; d[5] = 0.9f;
    d[6] = 0.9f; d[7] = 0.1f;

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    EXPECT_EQ(counts.shape().size(), 2u);
    EXPECT_EQ(counts.shape()[0], 2);
    EXPECT_EQ(counts.shape()[1], 2);
    EXPECT_EQ(edges.size(), 2u);
}

TEST_P(HistogramddMultiDTypeTest, AutoRange) {
    auto input = tenzor::zeros({6, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 1.0f; d[1] = 1.0f; d[2] = 3.0f; d[3] = 3.0f;
    d[4] = 5.0f; d[5] = 5.0f; d[6] = 7.0f; d[7] = 7.0f;
    d[8] = 9.0f; d[9] = 9.0f; d[10] = 2.0f; d[11] = 8.0f;

    auto [counts, edges] = histogramdd(input, {5, 5});
    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(counts.shape()[1], 5);

    auto* c = counts.data<int64_t>();
    int64_t total = 0;
    for (int64_t i = 0; i < counts.numel(); ++i) total += c[i];
    EXPECT_EQ(total, 6);
}

TEST_P(HistogramddMultiDTypeTest, ExplicitRanges) {
    auto input = tenzor::zeros({3, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.5f; d[1] = 0.5f;
    d[2] = 1.5f; d[3] = 1.5f;
    d[4] = 2.5f; d[5] = 2.5f;

    auto [counts, edges] = histogramdd(input, {3, 3},
        std::vector<std::pair<double,double>>{{0.0, 3.0}, {0.0, 3.0}});

    auto* c = counts.data<int64_t>();
    EXPECT_EQ(c[0 * 3 + 0], 1);
    EXPECT_EQ(c[1 * 3 + 1], 1);
    EXPECT_EQ(c[2 * 3 + 2], 1);
}

TEST_P(HistogramddMultiDTypeTest, SingleSample) {
    auto input = tenzor::zeros({1, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.5f; d[1] = 0.5f;

    auto [counts, edges] = histogramdd(input, {3, 3},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    auto* c = counts.data<int64_t>();
    int64_t total = 0;
    for (int64_t i = 0; i < counts.numel(); ++i) total += c[i];
    EXPECT_EQ(total, 1);
}

TEST_P(HistogramddMultiDTypeTest, DensityNormalization) {
    auto input = tenzor::zeros({4, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.25f; d[1] = 0.25f;
    d[2] = 0.75f; d[3] = 0.75f;
    d[4] = 0.25f; d[5] = 0.75f;
    d[6] = 0.75f; d[7] = 0.25f;

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}},
        true);

    auto* c = counts.data<float>();
    float integral = 0.0f;
    double bin_vol = 0.5 * 0.5;
    for (int64_t i = 0; i < counts.numel(); ++i) {
        integral += static_cast<float>(c[i] * bin_vol);
    }
    EXPECT_NEAR(integral, 1.0f, 1e-4f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HistogramddMultiDTypeTest);
