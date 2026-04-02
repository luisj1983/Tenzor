/**
 * @file test_cdist.cpp
 * @brief Tests for pairwise distance computation
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;

class CDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(CDistTest, EuclideanDistance2D) {
    // 2 points in 3D space
    auto x1 = tenzor::zeros({2, 3});
    auto x2 = tenzor::ones({3, 3});

    auto result = tenzor::cdist(x1, x2, 2.0);

    // Shape: (2, 3) - pairwise distances
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 3);

    // Distance from origin to (1,1,1) = sqrt(3)
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], std::sqrt(3.0f), 1e-4);
}

TEST_F(CDistTest, ManhattanDistance) {
    auto x1 = tenzor::zeros({2, 3});
    auto x2 = tenzor::ones({2, 3});

    auto result = tenzor::cdist(x1, x2, 1.0);

    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);

    // L1 distance from origin to (1,1,1) = 3
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 3.0f, 1e-4);
}

TEST_F(CDistTest, BatchedCDist) {
    auto x1 = tenzor::randn({4, 5, 3});
    auto x2 = tenzor::randn({4, 6, 3});

    auto result = tenzor::cdist(x1, x2, 2.0);

    EXPECT_EQ(result.ndim(), 3);
    EXPECT_EQ(result.shape()[0], 4);
    EXPECT_EQ(result.shape()[1], 5);
    EXPECT_EQ(result.shape()[2], 6);

    // All distances should be non-negative
    auto* data = result.data<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
        EXPECT_GE(data[i], 0.0f);
    }
}

TEST_F(CDistTest, SelfDistance) {
    auto x = tenzor::randn({3, 2});
    auto result = tenzor::cdist(x, x, 2.0);

    // Diagonal should be zero (distance from point to itself)
    auto* data = result.data<float>();
    for (int64_t i = 0; i < 3; i++) {
        EXPECT_NEAR(data[i * 3 + i], 0.0f, 1e-4);
    }
}
