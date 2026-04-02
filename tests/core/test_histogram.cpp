/**
 * @file test_histogram.cpp
 * @brief Tests for histogram computation
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

class HistogramTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(HistogramTest, UniformData) {
    // 10 values uniformly in [0, 10)
    auto input = tenzor::arange(0.0f, 10.0f, 1.0f);
    auto [counts, edges] = tenzor::histogram(input, /*bins=*/5, /*min=*/0.0, /*max=*/10.0);

    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(edges.shape()[0], 6);  // bins + 1 edges

    // Each bin should have 2 elements
    auto* c = counts.data<int64_t>();
    for (int64_t i = 0; i < 5; i++) {
        EXPECT_EQ(c[i], 2);
    }
}

TEST_F(HistogramTest, AutoRange) {
    // When min==max==0, should auto-detect range from data
    auto input = tenzor::arange(0.0f, 5.0f, 1.0f);
    auto [counts, edges] = tenzor::histogram(input, /*bins=*/5);

    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(edges.shape()[0], 6);

    // Total count should equal input size
    int64_t total = 0;
    auto* c = counts.data<int64_t>();
    for (int64_t i = 0; i < 5; i++) {
        total += c[i];
    }
    EXPECT_EQ(total, 5);
}

TEST_F(HistogramTest, SingleBin) {
    auto input = tenzor::randn({100});
    auto [counts, edges] = tenzor::histogram(input, /*bins=*/1);

    EXPECT_EQ(counts.shape()[0], 1);
    EXPECT_EQ(counts.data<int64_t>()[0], 100);
}
