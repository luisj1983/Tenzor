/**
 * @file test_histogram.cpp
 * @brief Tests for histogram computation
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class HistogramTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(HistogramTest, UniformData) {
    // 10 values uniformly in [0, 10)
    auto input = tenzor::arange(0.0f, 10.0f, 1.0f, DType::Float32, device);
    auto [counts, edges] = tenzor::histogram(input, /*bins=*/5, /*min=*/0.0, /*max=*/10.0);

    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(edges.shape()[0], 6);  // bins + 1 edges

    // Each bin should have 2 elements
    auto counts_cpu = counts.cpu();
    auto* c = counts_cpu.data<int64_t>();
    for (int64_t i = 0; i < 5; i++) {
        EXPECT_EQ(c[i], 2);
    }
}

TEST_P(HistogramTest, AutoRange) {
    // When min==max==0, should auto-detect range from data
    auto input = tenzor::arange(0.0f, 5.0f, 1.0f, DType::Float32, device);
    auto [counts, edges] = tenzor::histogram(input, /*bins=*/5);

    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(edges.shape()[0], 6);

    // Total count should equal input size
    int64_t total = 0;
    auto counts_cpu = counts.cpu();
    auto* c = counts_cpu.data<int64_t>();
    for (int64_t i = 0; i < 5; i++) {
        total += c[i];
    }
    EXPECT_EQ(total, 5);
}

TEST_P(HistogramTest, SingleBin) {
    auto input = tenzor::randn({100}, DType::Float32, device);
    auto [counts, edges] = tenzor::histogram(input, /*bins=*/1);

    EXPECT_EQ(counts.shape()[0], 1);
    auto counts_cpu = counts.cpu();
    EXPECT_EQ(counts_cpu.data<int64_t>()[0], 100);
}

INSTANTIATE_BACKEND_TESTS(HistogramTest);
