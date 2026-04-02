/**
 * @file test_bucketize.cpp
 * @brief Tests for bucketize operation
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/advanced.hpp"

using namespace tenzor;

class BucketizeTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(BucketizeTest, BasicBucketize) {
    // Boundaries: [1, 3, 5, 7]
    auto boundaries = tenzor::zeros({4});
    boundaries.data<float>()[0] = 1.0f;
    boundaries.data<float>()[1] = 3.0f;
    boundaries.data<float>()[2] = 5.0f;
    boundaries.data<float>()[3] = 7.0f;

    auto input = tenzor::zeros({5});
    input.data<float>()[0] = 0.0f;
    input.data<float>()[1] = 2.0f;
    input.data<float>()[2] = 4.0f;
    input.data<float>()[3] = 6.0f;
    input.data<float>()[4] = 8.0f;

    auto result = tenzor::bucketize(input, boundaries);

    EXPECT_EQ(result.dtype(), DType::Int64);
    EXPECT_EQ(result.shape()[0], 5);

    auto* data = result.data<int64_t>();
    EXPECT_EQ(data[0], 0);  // 0 < 1
    EXPECT_EQ(data[1], 1);  // 1 <= 2 < 3
    EXPECT_EQ(data[2], 2);  // 3 <= 4 < 5
    EXPECT_EQ(data[3], 3);  // 5 <= 6 < 7
    EXPECT_EQ(data[4], 4);  // 7 <= 8
}

TEST_F(BucketizeTest, RightClosed) {
    auto boundaries = tenzor::zeros({3});
    boundaries.data<float>()[0] = 1.0f;
    boundaries.data<float>()[1] = 3.0f;
    boundaries.data<float>()[2] = 5.0f;

    auto input = tenzor::zeros({3});
    input.data<float>()[0] = 1.0f;
    input.data<float>()[1] = 3.0f;
    input.data<float>()[2] = 5.0f;

    // right=false (default): values at boundary go to next bucket
    auto left = tenzor::bucketize(input, boundaries, /*right=*/false);
    // right=true: values at boundary stay in current bucket
    auto right_res = tenzor::bucketize(input, boundaries, /*right=*/true);

    // Just verify they produce valid indices
    auto* l = left.data<int64_t>();
    auto* r = right_res.data<int64_t>();
    EXPECT_GE(l[0], 0);
    EXPECT_GE(r[0], 0);
    EXPECT_LE(l[0], 3);
    EXPECT_LE(r[0], 3);
}

TEST_F(BucketizeTest, MultidimensionalInput) {
    auto boundaries = tenzor::zeros({3});
    boundaries.data<float>()[0] = 2.0f;
    boundaries.data<float>()[1] = 4.0f;
    boundaries.data<float>()[2] = 6.0f;

    auto input = tenzor::randn({3, 4}) * 4.0f + 3.0f;

    auto result = tenzor::bucketize(input, boundaries);
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 4);

    // All indices should be in [0, 3]
    auto* data = result.data<int64_t>();
    for (int64_t i = 0; i < result.numel(); i++) {
        EXPECT_GE(data[i], 0);
        EXPECT_LE(data[i], 3);
    }
}
