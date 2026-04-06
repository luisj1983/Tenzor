/**
 * @file test_dtype_conversion_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for dtype conversion (cast) operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DTypeConversionMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(DTypeConversionMultiDTypeTest, ToFloat32) {
    auto t = createOnes({4, 4});
    auto result = t.to(DType::Float32);
    EXPECT_EQ(result.dtype(), DType::Float32);
    expectDevice(result);
    expectShape(result, {4, 4});
}

TEST_P(DTypeConversionMultiDTypeTest, ToFloat64) {
    auto t = createOnes({4, 4});
    auto result = t.to(DType::Float64);
    EXPECT_EQ(result.dtype(), DType::Float64);
    expectDevice(result);
}

TEST_P(DTypeConversionMultiDTypeTest, ToSameDTypeNoop) {
    auto t = createRandn({4, 4});
    auto result = t.to(dtype());
    EXPECT_EQ(result.dtype(), dtype());
}

TEST_P(DTypeConversionMultiDTypeTest, RoundTripFloat32) {
    auto t = createRandn({8});
    auto f32 = t.to(DType::Float32);
    auto back = f32.to(dtype());
    EXPECT_EQ(back.dtype(), dtype());
    expectShape(back, {8});
}

TEST_P(DTypeConversionMultiDTypeTest, ValuePreservation) {
    auto t = createOnes({4});
    auto f32 = t.to(DType::Float32).to(Device::cpu());
    auto* d = f32.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(d[i], 1.0f, 0.01f);
    }
}

TEST_P(DTypeConversionMultiDTypeTest, DeviceTransfer) {
    auto t = createRandn({4, 4});
    auto cpu_t = t.to(Device::cpu());
    EXPECT_EQ(cpu_t.device().type, Device::Type::CPU);
    expectShape(cpu_t, {4, 4});

    auto back = cpu_t.to(device());
    expectDevice(back);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DTypeConversionMultiDTypeTest);
