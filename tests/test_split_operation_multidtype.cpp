/**
 * @file test_split_operation_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for split and chunk operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/ops/creation.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SplitOpMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(SplitOpMultiDTypeTest, SplitEven) {
    auto t = createRandn({6, 4});
    auto parts = tenzor::split(t, 2, 0);  // split into size-2 chunks
    EXPECT_EQ(parts.size(), 3);
    for (auto& p : parts) {
        expectShape(p, {2, 4});
        expectDevice(p);
        expectDType(p);
    }
}

TEST_P(SplitOpMultiDTypeTest, SplitDim1) {
    auto t = createRandn({3, 8});
    auto parts = tenzor::split(t, 4, 1);
    EXPECT_EQ(parts.size(), 2);
    expectShape(parts[0], {3, 4});
    expectShape(parts[1], {3, 4});
}

TEST_P(SplitOpMultiDTypeTest, ChunkEven) {
    auto t = createRandn({12, 4});
    auto parts = tenzor::chunk(t, 3, 0);
    EXPECT_EQ(parts.size(), 3);
    for (auto& p : parts) {
        expectShape(p, {4, 4});
    }
}

TEST_P(SplitOpMultiDTypeTest, ChunkUneven) {
    // 10 elements split into 3 chunks: 4, 4, 2
    auto t = createRandn({10, 3});
    auto parts = tenzor::chunk(t, 3, 0);
    EXPECT_EQ(parts.size(), 3);
    expectShape(parts[0], {4, 3});
    expectShape(parts[1], {4, 3});
    expectShape(parts[2], {2, 3});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SplitOpMultiDTypeTest);
