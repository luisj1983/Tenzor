/**
 * @file test_transform_new.cpp
 * @brief Tests for new transform operations: moveaxis, narrow_copy, column_stack,
 *        row_stack, broadcast_tensors
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>

using namespace tenzor;

class TransformNewTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// ============================================================================
// moveaxis
// ============================================================================

TEST_F(TransformNewTest, MoveaxisEquivalentToMovedim) {
    auto input = rand({2, 3, 4}, DType::Float32, Device::cpu());

    auto result_moveaxis = moveaxis(input, {0, 2}, {2, 0});
    auto result_movedim  = movedim(input, {0, 2}, {2, 0});

    EXPECT_EQ(result_moveaxis.ndim(), result_movedim.ndim());
    for (int64_t i = 0; i < result_moveaxis.ndim(); ++i) {
        EXPECT_EQ(result_moveaxis.shape()[i], result_movedim.shape()[i]);
    }

    auto* a = result_moveaxis.data<float>();
    auto* b = result_movedim.data<float>();
    for (int64_t i = 0; i < result_moveaxis.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "Mismatch at index " << i;
    }
}

TEST_F(TransformNewTest, MoveaxisTransposeLike) {
    // Moving axis 0 -> 1 and 1 -> 0 is a transpose for 2D
    auto input = rand({3, 5}, DType::Float32, Device::cpu());
    auto result = moveaxis(input, {0, 1}, {1, 0});

    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.shape()[1], 3);
}

// ============================================================================
// narrow_copy
// ============================================================================

TEST_F(TransformNewTest, NarrowCopyCorrectData) {
    auto input = zeros({10}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input.data<float>());
    for (int i = 0; i < 10; ++i) d[i] = static_cast<float>(i);

    auto result = narrow_copy(input, 0, 3, 4);  // elements 3,4,5,6

    EXPECT_EQ(result.shape()[0], 4);
    auto* r = result.data<float>();
    EXPECT_FLOAT_EQ(r[0], 3.0f);
    EXPECT_FLOAT_EQ(r[1], 4.0f);
    EXPECT_FLOAT_EQ(r[2], 5.0f);
    EXPECT_FLOAT_EQ(r[3], 6.0f);
}

TEST_F(TransformNewTest, NarrowCopyIsNotView) {
    auto input = ones({8}, DType::Float32, Device::cpu());
    auto result = narrow_copy(input, 0, 2, 3);

    // Modify original; result should be unaffected
    auto* d = const_cast<float*>(input.data<float>());
    d[2] = 999.0f;
    d[3] = 999.0f;
    d[4] = 999.0f;

    auto* r = result.data<float>();
    EXPECT_FLOAT_EQ(r[0], 1.0f);
    EXPECT_FLOAT_EQ(r[1], 1.0f);
    EXPECT_FLOAT_EQ(r[2], 1.0f);
}

TEST_F(TransformNewTest, NarrowCopy2D) {
    auto input = rand({4, 6}, DType::Float32, Device::cpu());
    auto result = narrow_copy(input, 1, 1, 3);  // columns 1,2,3

    EXPECT_EQ(result.shape()[0], 4);
    EXPECT_EQ(result.shape()[1], 3);
}

// ============================================================================
// column_stack
// ============================================================================

TEST_F(TransformNewTest, ColumnStack1DInputs) {
    auto a = ones({3}, DType::Float32, Device::cpu());
    auto b = full({3}, 2.0f, DType::Float32, Device::cpu());
    auto c = full({3}, 3.0f, DType::Float32, Device::cpu());

    auto result = column_stack({a, b, c});

    // 1D inputs become columns -> (3, 3)
    EXPECT_EQ(result.shape().size(), 2u);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 3);

    auto* r = result.data<float>();
    // First column: all 1s
    EXPECT_FLOAT_EQ(r[0 * 3 + 0], 1.0f);
    EXPECT_FLOAT_EQ(r[1 * 3 + 0], 1.0f);
    EXPECT_FLOAT_EQ(r[2 * 3 + 0], 1.0f);
    // Second column: all 2s
    EXPECT_FLOAT_EQ(r[0 * 3 + 1], 2.0f);
}

TEST_F(TransformNewTest, ColumnStack2DInputs) {
    auto a = ones({3, 2}, DType::Float32, Device::cpu());
    auto b = full({3, 1}, 5.0f, DType::Float32, Device::cpu());

    auto result = column_stack({a, b});

    // 2D inputs cat along dim 1 -> (3, 3)
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 3);
}

// ============================================================================
// row_stack (alias for vstack)
// ============================================================================

TEST_F(TransformNewTest, RowStackMatchesVstack) {
    auto a = rand({2, 3}, DType::Float32, Device::cpu());
    auto b = rand({4, 3}, DType::Float32, Device::cpu());

    auto result_row   = row_stack({a, b});
    auto result_vstack = vstack({a, b});

    EXPECT_EQ(result_row.ndim(), result_vstack.ndim());
    EXPECT_EQ(result_row.shape()[0], 6);
    EXPECT_EQ(result_row.shape()[1], 3);

    auto* r1 = result_row.data<float>();
    auto* r2 = result_vstack.data<float>();
    for (int64_t i = 0; i < result_row.numel(); ++i) {
        EXPECT_FLOAT_EQ(r1[i], r2[i]) << "Mismatch at index " << i;
    }
}

// ============================================================================
// broadcast_tensors
// ============================================================================

TEST_F(TransformNewTest, BroadcastTensorsBasic) {
    auto a = ones({3, 1}, DType::Float32, Device::cpu());
    auto b = ones({1, 4}, DType::Float32, Device::cpu());

    auto results = broadcast_tensors({a, b});

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].shape()[0], 3);
    EXPECT_EQ(results[0].shape()[1], 4);
    EXPECT_EQ(results[1].shape()[0], 3);
    EXPECT_EQ(results[1].shape()[1], 4);
}

TEST_F(TransformNewTest, BroadcastTensorsMultiple) {
    auto a = ones({2, 1, 5}, DType::Float32, Device::cpu());
    auto b = ones({3, 1}, DType::Float32, Device::cpu());
    auto c = ones({2, 3, 5}, DType::Float32, Device::cpu());

    auto results = broadcast_tensors({a, b, c});

    ASSERT_EQ(results.size(), 3u);
    for (auto& r : results) {
        EXPECT_EQ(r.shape()[0], 2);
        EXPECT_EQ(r.shape()[1], 3);
        EXPECT_EQ(r.shape()[2], 5);
    }
}

TEST_F(TransformNewTest, BroadcastTensorsIncompatibleShapes) {
    auto a = ones({3, 4}, DType::Float32, Device::cpu());
    auto b = ones({5, 4}, DType::Float32, Device::cpu());

    EXPECT_THROW(broadcast_tensors({a, b}), std::exception);
}
