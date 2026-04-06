/**
 * @file test_indexing_operator_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for indexing operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/ops/creation.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class IndexingOpMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(IndexingOpMultiDTypeTest, SelectDim0) {
    auto t = createRandn({4, 3});
    auto result = tenzor::select(t, 0, 2);
    expectShape(result, {3});
    expectDevice(result);
    expectDType(result);
}

TEST_P(IndexingOpMultiDTypeTest, SelectDim1) {
    auto t = createRandn({4, 3});
    auto result = tenzor::select(t, 1, 1);
    expectShape(result, {4});
}

TEST_P(IndexingOpMultiDTypeTest, NarrowDim0) {
    auto t = createRandn({10, 5});
    auto result = tenzor::narrow(t, 0, 2, 4);
    expectShape(result, {4, 5});
    expectDevice(result);
}

TEST_P(IndexingOpMultiDTypeTest, SliceBasic) {
    auto t = createRandn({8, 4});
    auto result = tenzor::slice(t, 0, 1, 5);
    expectShape(result, {4, 4});
}

TEST_P(IndexingOpMultiDTypeTest, WhereCondition) {
    auto cond = tenzor::gt(createRandn({3, 4}),
                           tenzor::zeros({3, 4}, dtype(), device()));
    auto x = createOnes({3, 4});
    auto y = tenzor::zeros({3, 4}, dtype(), device());
    auto result = tenzor::where(cond, x, y);
    expectShape(result, {3, 4});
    expectDevice(result);
}

TEST_P(IndexingOpMultiDTypeTest, MaskedFill) {
    auto t = createRandn({4, 4});
    auto mask = tenzor::gt(t, tenzor::zeros({4, 4}, dtype(), device()));
    auto result = tenzor::masked_fill(t, mask, 0.0f);
    expectShape(result, {4, 4});
    expectDevice(result);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(IndexingOpMultiDTypeTest);
