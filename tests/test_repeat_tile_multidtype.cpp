/**
 * @file test_repeat_tile_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for repeat and tile operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/ops/creation.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class RepeatTileMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(RepeatTileMultiDTypeTest, RepeatBasic) {
    auto t = createOnes({2, 3});
    auto result = tenzor::repeat(t, {2, 3});
    expectShape(result, {4, 9});
    expectDevice(result);
    expectDType(result);
}

TEST_P(RepeatTileMultiDTypeTest, RepeatSingleDim) {
    auto t = createOnes({4});
    auto result = tenzor::repeat(t, {3});
    expectShape(result, {12});
}

TEST_P(RepeatTileMultiDTypeTest, TileBasic) {
    auto t = createOnes({2, 3});
    auto result = tenzor::tile(t, {2, 3});
    expectShape(result, {4, 9});
    expectDevice(result);
    expectDType(result);
}

TEST_P(RepeatTileMultiDTypeTest, TileHigherDims) {
    auto t = createOnes({2, 3, 4});
    auto result = tenzor::tile(t, {1, 2, 3});
    expectShape(result, {2, 6, 12});
}

TEST_P(RepeatTileMultiDTypeTest, RepeatPreservesValues) {
    // Repeating ones should still give ones
    auto t = createOnes({3});
    auto result = tenzor::repeat(t, {4});
    expectShape(result, {12});

    auto r_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* d = r_f32.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(d[i], 1.0f, atol());
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RepeatTileMultiDTypeTest);
