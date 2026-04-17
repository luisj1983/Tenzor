/**
 * @file test_cdist_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for pairwise distance computation
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class CDistMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(CDistMultiDTypeTest, EuclideanDistance2DShape) {
    auto x1 = createZeros({2, 3});
    auto x2 = createOnes({3, 3});

    auto result = tenzor::cdist(x1, x2, 2.0);

    expectShape(result, {2, 3});
    expectDevice(result);
    expectDType(result);
}

TEST_P(CDistMultiDTypeTest, EuclideanDistance2DValues) {
    auto x1 = createZeros({2, 3});
    auto x2 = createOnes({3, 3});

    auto result = tenzor::cdist(x1, x2, 2.0);

    // Distance from origin to (1,1,1) = sqrt(3)
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* data = r_cpu.data<float>();
    EXPECT_NEAR(data[0], std::sqrt(3.0f), atol() + 1e-2f);
}

TEST_P(CDistMultiDTypeTest, ManhattanDistance) {
    auto x1 = createZeros({2, 3});
    auto x2 = createOnes({2, 3});

    auto result = tenzor::cdist(x1, x2, 1.0);

    expectShape(result, {2, 2});
    expectDevice(result);
    expectDType(result);

    // L1 distance from origin to (1,1,1) = 3
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* data = r_cpu.data<float>();
    EXPECT_NEAR(data[0], 3.0f, atol() + 1e-2f);
}

TEST_P(CDistMultiDTypeTest, BatchedCDistShape) {
    auto x1 = createRandn({4, 5, 3});
    auto x2 = createRandn({4, 6, 3});

    auto result = tenzor::cdist(x1, x2, 2.0);

    expectShape(result, {4, 5, 6});
    expectDevice(result);
    expectDType(result);
}

TEST_P(CDistMultiDTypeTest, BatchedCDistNonNegative) {
    auto x1 = createRandn({4, 5, 3});
    auto x2 = createRandn({4, 6, 3});

    auto result = tenzor::cdist(x1, x2, 2.0);

    // All distances should be non-negative
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* data = r_cpu.data<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
        EXPECT_GE(data[i], -atol());
    }
}

TEST_P(CDistMultiDTypeTest, SelfDistanceDiagonalZero) {
    auto x = createRandn({3, 2});
    auto result = tenzor::cdist(x, x, 2.0);

    expectShape(result, {3, 3});

    // Diagonal should be zero (distance from point to itself)
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* data = r_cpu.data<float>();
    for (int64_t i = 0; i < 3; i++) {
        EXPECT_NEAR(data[i * 3 + i], 0.0f, atol() + 1e-2f);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(CDistMultiDTypeTest);
