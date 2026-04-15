/**
 * @file test_histogramdd.cpp
 * @brief Tests for the histogramdd reduction operation
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>

using namespace tenzor;

class HistogramddTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// ============================================================================
// Basic 2D histogram with known data
// ============================================================================

TEST_F(HistogramddTest, Basic2DKnownData) {
    // 4 samples, 2 dimensions: (0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1)
    auto input = zeros({4, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.1f; d[1] = 0.1f;  // sample 0
    d[2] = 0.9f; d[3] = 0.9f;  // sample 1
    d[4] = 0.1f; d[5] = 0.9f;  // sample 2
    d[6] = 0.9f; d[7] = 0.1f;  // sample 3

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    EXPECT_EQ(counts.shape().size(), 2u);
    EXPECT_EQ(counts.shape()[0], 2);
    EXPECT_EQ(counts.shape()[1], 2);
    EXPECT_EQ(edges.size(), 2u);

    // Each quadrant should have 1 sample
    auto* c = counts.data<int64_t>();
    int64_t total = c[0] + c[1] + c[2] + c[3];
    EXPECT_EQ(total, 4);
}

// ============================================================================
// Auto-range detection
// ============================================================================

TEST_F(HistogramddTest, AutoRange) {
    // Data in [0,10] x [0,10], let auto-range figure it out
    auto input = zeros({6, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0]  = 1.0f; d[1]  = 1.0f;
    d[2]  = 3.0f; d[3]  = 3.0f;
    d[4]  = 5.0f; d[5]  = 5.0f;
    d[6]  = 7.0f; d[7]  = 7.0f;
    d[8]  = 9.0f; d[9]  = 9.0f;
    d[10] = 2.0f; d[11] = 8.0f;

    auto [counts, edges] = histogramdd(input, {5, 5});  // auto-range

    EXPECT_EQ(counts.shape().size(), 2u);
    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(counts.shape()[1], 5);
    EXPECT_EQ(edges.size(), 2u);

    // All 6 samples should be accounted for
    auto* c = counts.data<int64_t>();
    int64_t total = 0;
    for (int64_t i = 0; i < counts.numel(); ++i) total += c[i];
    EXPECT_EQ(total, 6);
}

// ============================================================================
// Explicit ranges
// ============================================================================

TEST_F(HistogramddTest, ExplicitRanges) {
    auto input = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.5f; d[1] = 0.5f;
    d[2] = 1.5f; d[3] = 1.5f;
    d[4] = 2.5f; d[5] = 2.5f;

    auto [counts, edges] = histogramdd(input, {3, 3},
        std::vector<std::pair<double,double>>{{0.0, 3.0}, {0.0, 3.0}});

    EXPECT_EQ(counts.shape()[0], 3);
    EXPECT_EQ(counts.shape()[1], 3);

    // Each sample falls in a different bin along the diagonal
    auto* c = counts.data<int64_t>();
    EXPECT_EQ(c[0 * 3 + 0], 1);  // bin (0,0)
    EXPECT_EQ(c[1 * 3 + 1], 1);  // bin (1,1)
    EXPECT_EQ(c[2 * 3 + 2], 1);  // bin (2,2)
}

// ============================================================================
// Density normalization
// ============================================================================

TEST_F(HistogramddTest, DensityNormalization) {
    auto input = zeros({4, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.25f; d[1] = 0.25f;
    d[2] = 0.75f; d[3] = 0.75f;
    d[4] = 0.25f; d[5] = 0.75f;
    d[6] = 0.75f; d[7] = 0.25f;

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}},
        /*density=*/true);

    // With density=true, integral over bins should equal 1
    // bin_volume = (0.5) * (0.5) = 0.25 for each bin
    auto* c = counts.data<float>();
    float integral = 0.0f;
    double bin_vol = 0.5 * 0.5;
    for (int64_t i = 0; i < counts.numel(); ++i) {
        integral += static_cast<float>(c[i] * bin_vol);
    }
    EXPECT_NEAR(integral, 1.0f, 1e-4f);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(HistogramddTest, SingleSample) {
    auto input = zeros({1, 2}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    d[0] = 0.5f; d[1] = 0.5f;

    auto [counts, edges] = histogramdd(input, {3, 3},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    auto* c = counts.data<int64_t>();
    int64_t total = 0;
    for (int64_t i = 0; i < counts.numel(); ++i) total += c[i];
    EXPECT_EQ(total, 1);
}

TEST_F(HistogramddTest, AllSameValue) {
    auto input = full({10, 2}, 0.5f, DType::Float32, Device::cpu());

    auto [counts, edges] = histogramdd(input, {4, 4},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    // All 10 samples should land in the same bin
    auto* c = counts.data<int64_t>();
    int64_t max_val = 0;
    for (int64_t i = 0; i < counts.numel(); ++i) {
        max_val = std::max(max_val, c[i]);
    }
    EXPECT_EQ(max_val, 10);
}

// ============================================================================
// Multi-dtype: Float64
// ============================================================================

TEST_F(HistogramddTest, Float64Basic) {
    auto input = zeros({3, 2}, DType::Float64, Device::cpu());
    auto* d = const_cast<double*>(input.data<double>());
    d[0] = 0.1; d[1] = 0.1;
    d[2] = 0.5; d[3] = 0.5;
    d[4] = 0.9; d[5] = 0.9;

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    EXPECT_EQ(counts.shape()[0], 2);
    EXPECT_EQ(counts.shape()[1], 2);

    // Verify total count
    float total = 0.0f;
    auto counts_f32 = counts.to(DType::Float32);
    auto* c = counts_f32.data<float>();
    for (int64_t i = 0; i < counts_f32.numel(); ++i) total += c[i];
    EXPECT_NEAR(total, 3.0f, 1e-5f);
}
