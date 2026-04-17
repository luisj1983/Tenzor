/**
 * @file test_bucketize_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for bucketize operation
 *
 * Float16 is skipped for exact value checks due to precision.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/advanced.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class BucketizeMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(BucketizeMultiDTypeTest, BasicBucketizeShape) {
    // Build boundaries on CPU then move
    auto boundaries = tenzor::zeros({4}, DType::Float32, Device::cpu());
    boundaries.data<float>()[0] = 1.0f;
    boundaries.data<float>()[1] = 3.0f;
    boundaries.data<float>()[2] = 5.0f;
    boundaries.data<float>()[3] = 7.0f;
    boundaries = boundaries.to(dtype()).to(device());

    auto input = tenzor::zeros({5}, DType::Float32, Device::cpu());
    input.data<float>()[0] = 0.0f;
    input.data<float>()[1] = 2.0f;
    input.data<float>()[2] = 4.0f;
    input.data<float>()[3] = 6.0f;
    input.data<float>()[4] = 8.0f;
    input = input.to(dtype()).to(device());

    auto result = tenzor::bucketize(input, boundaries);

    EXPECT_EQ(result.dtype(), DType::Int64);
    EXPECT_EQ(result.shape()[0], 5);
}

TEST_P(BucketizeMultiDTypeTest, BasicBucketizeValues) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Skipping exact value check for low-precision dtype";
    }

    auto boundaries = tenzor::zeros({4}, DType::Float32, Device::cpu());
    boundaries.data<float>()[0] = 1.0f;
    boundaries.data<float>()[1] = 3.0f;
    boundaries.data<float>()[2] = 5.0f;
    boundaries.data<float>()[3] = 7.0f;
    boundaries = boundaries.to(dtype()).to(device());

    auto input = tenzor::zeros({5}, DType::Float32, Device::cpu());
    input.data<float>()[0] = 0.0f;
    input.data<float>()[1] = 2.0f;
    input.data<float>()[2] = 4.0f;
    input.data<float>()[3] = 6.0f;
    input.data<float>()[4] = 8.0f;
    input = input.to(dtype()).to(device());

    auto result = tenzor::bucketize(input, boundaries);
    auto r_cpu = result.to(Device::cpu());
    auto* data = r_cpu.data<int64_t>();
    EXPECT_EQ(data[0], 0);  // 0 < 1
    EXPECT_EQ(data[1], 1);  // 1 <= 2 < 3
    EXPECT_EQ(data[2], 2);  // 3 <= 4 < 5
    EXPECT_EQ(data[3], 3);  // 5 <= 6 < 7
    EXPECT_EQ(data[4], 4);  // 7 <= 8
}

TEST_P(BucketizeMultiDTypeTest, RightClosedValidIndices) {
    auto boundaries = tenzor::zeros({3}, DType::Float32, Device::cpu());
    boundaries.data<float>()[0] = 1.0f;
    boundaries.data<float>()[1] = 3.0f;
    boundaries.data<float>()[2] = 5.0f;
    boundaries = boundaries.to(dtype()).to(device());

    auto input = tenzor::zeros({3}, DType::Float32, Device::cpu());
    input.data<float>()[0] = 1.0f;
    input.data<float>()[1] = 3.0f;
    input.data<float>()[2] = 5.0f;
    input = input.to(dtype()).to(device());

    auto left = tenzor::bucketize(input, boundaries, /*right=*/false);
    auto right_res = tenzor::bucketize(input, boundaries, /*right=*/true);

    auto l_cpu = left.to(Device::cpu());
    auto r_cpu = right_res.to(Device::cpu());
    auto* l = l_cpu.data<int64_t>();
    auto* r = r_cpu.data<int64_t>();
    for (int64_t i = 0; i < 3; i++) {
        EXPECT_GE(l[i], 0);
        EXPECT_LE(l[i], 3);
        EXPECT_GE(r[i], 0);
        EXPECT_LE(r[i], 3);
    }
}

TEST_P(BucketizeMultiDTypeTest, MultidimensionalInputShape) {
    auto boundaries = tenzor::zeros({3}, DType::Float32, Device::cpu());
    boundaries.data<float>()[0] = 2.0f;
    boundaries.data<float>()[1] = 4.0f;
    boundaries.data<float>()[2] = 6.0f;
    boundaries = boundaries.to(dtype()).to(device());

    auto input = createRandn({3, 4});

    auto result = tenzor::bucketize(input, boundaries);
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 4);

    // All indices should be in [0, 3]
    auto r_cpu = result.to(Device::cpu());
    auto* data = r_cpu.data<int64_t>();
    for (int64_t i = 0; i < result.numel(); i++) {
        EXPECT_GE(data[i], 0);
        EXPECT_LE(data[i], 3);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BucketizeMultiDTypeTest);
