/**
 * @file test_bucketize.cpp
 * @brief Tests for bucketize operation — parameterized across every
 *        available backend. Previously CPU-only.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/advanced.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class BucketizeTest : public tenzor::testing::BackendTest {};

TEST_P(BucketizeTest, BasicBucketize) {
    // Boundaries: [1, 3, 5, 7]
    auto boundaries_cpu = tenzor::zeros({4}, DType::Float32, Device::cpu());
    float* bp = boundaries_cpu.data<float>();
    bp[0] = 1.0f; bp[1] = 3.0f; bp[2] = 5.0f; bp[3] = 7.0f;
    auto boundaries = boundaries_cpu.to(device);

    auto input_cpu = tenzor::zeros({5}, DType::Float32, Device::cpu());
    float* ip = input_cpu.data<float>();
    ip[0] = 0.0f; ip[1] = 2.0f; ip[2] = 4.0f; ip[3] = 6.0f; ip[4] = 8.0f;
    auto input = input_cpu.to(device);

    auto result = tenzor::bucketize(input, boundaries);
    EXPECT_EQ(result.dtype(), DType::Int64);
    EXPECT_EQ(result.shape()[0], 5);

    auto r_cpu = result.to(Device::cpu()).contiguous();
    const int64_t* data = r_cpu.data<int64_t>();
    EXPECT_EQ(data[0], 0);  // 0 < 1
    EXPECT_EQ(data[1], 1);  // 1 <= 2 < 3
    EXPECT_EQ(data[2], 2);  // 3 <= 4 < 5
    EXPECT_EQ(data[3], 3);  // 5 <= 6 < 7
    EXPECT_EQ(data[4], 4);  // 7 <= 8
}

TEST_P(BucketizeTest, RightClosed) {
    auto boundaries_cpu = tenzor::zeros({3}, DType::Float32, Device::cpu());
    float* bp = boundaries_cpu.data<float>();
    bp[0] = 1.0f; bp[1] = 3.0f; bp[2] = 5.0f;
    auto boundaries = boundaries_cpu.to(device);

    auto input_cpu = tenzor::zeros({3}, DType::Float32, Device::cpu());
    float* ip = input_cpu.data<float>();
    ip[0] = 1.0f; ip[1] = 3.0f; ip[2] = 5.0f;
    auto input = input_cpu.to(device);

    auto left = tenzor::bucketize(input, boundaries, /*right=*/false);
    auto right_res = tenzor::bucketize(input, boundaries, /*right=*/true);

    auto l_cpu = left.to(Device::cpu()).contiguous();
    auto r_cpu = right_res.to(Device::cpu()).contiguous();
    const int64_t* l = l_cpu.data<int64_t>();
    const int64_t* r = r_cpu.data<int64_t>();
    EXPECT_GE(l[0], 0);
    EXPECT_GE(r[0], 0);
    EXPECT_LE(l[0], 3);
    EXPECT_LE(r[0], 3);
}

TEST_P(BucketizeTest, MultidimensionalInput) {
    auto boundaries_cpu = tenzor::zeros({3}, DType::Float32, Device::cpu());
    float* bp = boundaries_cpu.data<float>();
    bp[0] = 2.0f; bp[1] = 4.0f; bp[2] = 6.0f;
    auto boundaries = boundaries_cpu.to(device);

    auto input = tenzor::randn({3, 4}, DType::Float32, device) * 4.0f + 3.0f;

    auto result = tenzor::bucketize(input, boundaries);
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 4);

    auto r_cpu = result.to(Device::cpu()).contiguous();
    const int64_t* data = r_cpu.data<int64_t>();
    for (int64_t i = 0; i < r_cpu.numel(); i++) {
        EXPECT_GE(data[i], 0);
        EXPECT_LE(data[i], 3);
    }
}

INSTANTIATE_BACKEND_TESTS(BucketizeTest);
