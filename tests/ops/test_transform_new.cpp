/**
 * @file test_transform_new.cpp
 * @brief Multi-backend tests for new transform operations.
 *
 * Migrated from CPU-only `::testing::Test` to BackendTest. Covers
 * moveaxis, narrow_copy, column_stack, row_stack, broadcast_tensors on
 * every backend that registers them.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class TransformNewTest : public BackendTest {};

// ============================================================================
// moveaxis
// ============================================================================

TEST_P(TransformNewTest, MoveaxisEquivalentToMovedim) {
    auto input = rand({2, 3, 4}, DType::Float32, device);

    auto result_moveaxis = moveaxis(input, {0, 2}, {2, 0});
    auto result_movedim  = movedim(input, {0, 2}, {2, 0});

    EXPECT_EQ(result_moveaxis.ndim(), result_movedim.ndim());
    for (int64_t i = 0; i < result_moveaxis.ndim(); ++i) {
        EXPECT_EQ(result_moveaxis.shape()[i], result_movedim.shape()[i]);
    }

    auto a_cpu = result_moveaxis.to(Device::cpu()).contiguous();
    auto b_cpu = result_movedim.to(Device::cpu()).contiguous();
    auto* a = a_cpu.data<float>();
    auto* b = b_cpu.data<float>();
    for (int64_t i = 0; i < result_moveaxis.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i])
            << "Mismatch at index " << i << " on " << device.to_string();
    }
}

TEST_P(TransformNewTest, MoveaxisTransposeLike) {
    // Moving axis 0 -> 1 and 1 -> 0 is a transpose for 2D
    auto input = rand({3, 5}, DType::Float32, device);
    auto result = moveaxis(input, {0, 1}, {1, 0});

    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.shape()[1], 3);
}

// ============================================================================
// narrow_copy
// ============================================================================

TEST_P(TransformNewTest, NarrowCopyCorrectData) {
    // Build the populated input on CPU then move to device — data<float>()
    // for direct write requires CPU storage. The narrow_copy op runs on the
    // target device after the move.
    auto input_cpu = zeros({10}, DType::Float32, Device::cpu());
    auto* d = const_cast<float*>(input_cpu.data<float>());
    for (int i = 0; i < 10; ++i) d[i] = static_cast<float>(i);
    auto input = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);

    auto result = narrow_copy(input, 0, 3, 4);  // elements 3,4,5,6

    EXPECT_EQ(result.shape()[0], 4);
    auto result_cpu = result.to(Device::cpu()).contiguous();
    auto* r = result_cpu.data<float>();
    EXPECT_FLOAT_EQ(r[0], 3.0f);
    EXPECT_FLOAT_EQ(r[1], 4.0f);
    EXPECT_FLOAT_EQ(r[2], 5.0f);
    EXPECT_FLOAT_EQ(r[3], 6.0f);
}

TEST_P(TransformNewTest, NarrowCopyIsNotView) {
    // Allocate input on CPU so we can mutate it in-place; the copy semantics
    // we're verifying don't depend on which backend ran narrow_copy.
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

TEST_P(TransformNewTest, NarrowCopy2D) {
    auto input = rand({4, 6}, DType::Float32, device);
    auto result = narrow_copy(input, 1, 1, 3);  // columns 1,2,3

    EXPECT_EQ(result.shape()[0], 4);
    EXPECT_EQ(result.shape()[1], 3);
}

// ============================================================================
// column_stack
// ============================================================================

TEST_P(TransformNewTest, ColumnStack1DInputs) {
    auto a = ones({3}, DType::Float32, device);
    auto b = full({3}, 2.0f, DType::Float32, device);
    auto c = full({3}, 3.0f, DType::Float32, device);

    auto result = column_stack({a, b, c});

    // 1D inputs become columns -> (3, 3)
    EXPECT_EQ(result.shape().size(), 2u);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 3);

    auto result_cpu = result.to(Device::cpu()).contiguous();
    auto* r = result_cpu.data<float>();
    // First column: all 1s
    EXPECT_FLOAT_EQ(r[0 * 3 + 0], 1.0f);
    EXPECT_FLOAT_EQ(r[1 * 3 + 0], 1.0f);
    EXPECT_FLOAT_EQ(r[2 * 3 + 0], 1.0f);
    // Second column: all 2s
    EXPECT_FLOAT_EQ(r[0 * 3 + 1], 2.0f);
}

TEST_P(TransformNewTest, ColumnStack2DInputs) {
    auto a = ones({3, 2}, DType::Float32, device);
    auto b = full({3, 1}, 5.0f, DType::Float32, device);

    auto result = column_stack({a, b});

    // 2D inputs cat along dim 1 -> (3, 3)
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 3);
}

// ============================================================================
// row_stack (alias for vstack)
// ============================================================================

TEST_P(TransformNewTest, RowStackMatchesVstack) {
    auto a = rand({2, 3}, DType::Float32, device);
    auto b = rand({4, 3}, DType::Float32, device);

    auto result_row   = row_stack({a, b});
    auto result_vstack = vstack({a, b});

    EXPECT_EQ(result_row.ndim(), result_vstack.ndim());
    EXPECT_EQ(result_row.shape()[0], 6);
    EXPECT_EQ(result_row.shape()[1], 3);

    auto r1_cpu = result_row.to(Device::cpu()).contiguous();
    auto r2_cpu = result_vstack.to(Device::cpu()).contiguous();
    auto* r1 = r1_cpu.data<float>();
    auto* r2 = r2_cpu.data<float>();
    for (int64_t i = 0; i < result_row.numel(); ++i) {
        EXPECT_FLOAT_EQ(r1[i], r2[i])
            << "Mismatch at index " << i << " on " << device.to_string();
    }
}

// ============================================================================
// broadcast_tensors
// ============================================================================

TEST_P(TransformNewTest, BroadcastTensorsBasic) {
    auto a = ones({3, 1}, DType::Float32, device);
    auto b = ones({1, 4}, DType::Float32, device);

    auto results = broadcast_tensors({a, b});

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].shape()[0], 3);
    EXPECT_EQ(results[0].shape()[1], 4);
    EXPECT_EQ(results[1].shape()[0], 3);
    EXPECT_EQ(results[1].shape()[1], 4);
}

TEST_P(TransformNewTest, BroadcastTensorsMultiple) {
    auto a = ones({2, 1, 5}, DType::Float32, device);
    auto b = ones({3, 1}, DType::Float32, device);
    auto c = ones({2, 3, 5}, DType::Float32, device);

    auto results = broadcast_tensors({a, b, c});

    ASSERT_EQ(results.size(), 3u);
    for (auto& r : results) {
        EXPECT_EQ(r.shape()[0], 2);
        EXPECT_EQ(r.shape()[1], 3);
        EXPECT_EQ(r.shape()[2], 5);
    }
}

TEST_P(TransformNewTest, BroadcastTensorsIncompatibleShapes) {
    auto a = ones({3, 4}, DType::Float32, device);
    auto b = ones({5, 4}, DType::Float32, device);

    EXPECT_THROW(broadcast_tensors({a, b}), std::exception);
}

INSTANTIATE_BACKEND_TESTS(TransformNewTest);
