/**
 * @file test_transform_new_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for new transform ops
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class TransformNewMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(TransformNewMultiDTypeTest, MoveaxisEquivalentToMovedim) {
    auto input = createRandn({2, 3, 4});
    auto result_moveaxis = moveaxis(input, {0, 2}, {2, 0});
    auto result_movedim  = movedim(input, {0, 2}, {2, 0});

    EXPECT_EQ(result_moveaxis.ndim(), result_movedim.ndim());
    for (int64_t i = 0; i < result_moveaxis.ndim(); ++i) {
        EXPECT_EQ(result_moveaxis.shape()[i], result_movedim.shape()[i]);
    }
}

TEST_P(TransformNewMultiDTypeTest, NarrowCopyCorrectShape) {
    auto input = createRandn({10});
    auto result = narrow_copy(input, 0, 3, 4);
    EXPECT_EQ(result.shape()[0], 4);
}

TEST_P(TransformNewMultiDTypeTest, NarrowCopyIsNotView) {
    auto input = createOnes({8});
    auto result = narrow_copy(input, 0, 2, 3);
    // result should be independent of input
    EXPECT_EQ(result.shape()[0], 3);
}

TEST_P(TransformNewMultiDTypeTest, ColumnStack1DInputs) {
    auto a = createOnes({3});
    auto b = createOnes({3});

    auto result = column_stack({a, b});
    EXPECT_EQ(result.shape().size(), 2u);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_P(TransformNewMultiDTypeTest, RowStackMatchesVstack) {
    auto a = createRandn({2, 3});
    auto b = createRandn({4, 3});

    auto result_row   = row_stack({a, b});
    auto result_vstack = vstack({a, b});

    EXPECT_EQ(result_row.shape()[0], 6);
    EXPECT_EQ(result_row.shape()[1], 3);
    EXPECT_EQ(result_vstack.shape()[0], 6);
}

TEST_P(TransformNewMultiDTypeTest, BroadcastTensorsBasic) {
    auto a = createOnes({3, 1});
    auto b = createOnes({1, 4});

    auto results = broadcast_tensors({a, b});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].shape()[0], 3);
    EXPECT_EQ(results[0].shape()[1], 4);
    EXPECT_EQ(results[1].shape()[0], 3);
    EXPECT_EQ(results[1].shape()[1], 4);
}

TEST_P(TransformNewMultiDTypeTest, BroadcastTensorsIncompatibleShapes) {
    auto a = createOnes({3, 4});
    auto b = createOnes({5, 4});
    EXPECT_THROW(broadcast_tensors({a, b}), std::exception);
}

TEST_P(TransformNewMultiDTypeTest, DTypePreserved) {
    auto input = createRandn({2, 3, 4});
    auto result = moveaxis(input, {0}, {2});
    expectDType(result);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(TransformNewMultiDTypeTest);
