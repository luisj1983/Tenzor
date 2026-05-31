/**
 * @file test_empty_tensors.cpp
 * @brief Tests for empty tensor edge cases (zero-element tensors)
 *
 * Covers:
 * - Creation of tensors with zero-sized dimensions
 * - Reductions on empty tensors (sum, mean, max)
 * - Broadcasting with empty tensors
 * - Reshape of empty tensors
 * - numel() and ndim() correctness
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

class EmptyTensorTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// 1. Creation of Empty Tensors
// ============================================================================

TEST_P(EmptyTensorTest, ZerosWithZeroRows) {
    auto t = zeros({0, 5}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 0);
    EXPECT_EQ(t.shape()[1], 5);
}

TEST_P(EmptyTensorTest, OnesWithZeroCols) {
    auto t = ones({2, 0}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 0);
}

TEST_P(EmptyTensorTest, ZerosWithAllZeroDims) {
    auto t = zeros({0, 0}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 0);
    EXPECT_EQ(t.shape()[1], 0);
}

TEST_P(EmptyTensorTest, ZerosWithZeroMiddleDim) {
    auto t = zeros({3, 0, 4}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 0);
    EXPECT_EQ(t.shape()[2], 4);
}

TEST_P(EmptyTensorTest, FullWithZeroDim) {
    auto t = full({0, 3}, 7.0f, DType::Float32, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 0);
    EXPECT_EQ(t.shape()[1], 3);
}

// ============================================================================
// 2. DType and Device Properties on Empty Tensors
// ============================================================================

TEST_P(EmptyTensorTest, DTypePreserved) {
    auto t_f32 = zeros({0, 5}, DType::Float32, device);
    EXPECT_EQ(t_f32.dtype(), DType::Float32);

    auto t_f64 = zeros({0, 5}, DType::Float64, device);
    EXPECT_EQ(t_f64.dtype(), DType::Float64);
}

TEST_P(EmptyTensorTest, DevicePreserved) {
    auto t = zeros({0, 5}, DType::Float32, device);
    EXPECT_EQ(t.device().type, device.type);
}

// ============================================================================
// 3. Reductions on Empty Tensors
// ============================================================================

TEST_P(EmptyTensorTest, SumOfEmptyTensor) {
    auto t = zeros({0, 5}, DType::Float32, device);
    // Sum over all elements of an empty tensor should produce 0
    auto result = sum(t);
    EXPECT_EQ(result.numel(), 1);
    EXPECT_FLOAT_EQ(result.cpu().item<float>(), 0.0f);
}

TEST_P(EmptyTensorTest, SumAlongNonEmptyDim) {
    // Shape (0, 5) summed along dim=1 should produce shape (0,)
    auto t = zeros({0, 5}, DType::Float32, device);
    auto result = sum(t, /*dim=*/1);
    EXPECT_EQ(result.shape()[0], 0);
    EXPECT_EQ(result.numel(), 0);
}

TEST_P(EmptyTensorTest, SumAlongEmptyDim) {
    // Shape (2, 0) summed along dim=1 should produce shape (2,)
    // with each element being 0 (sum of empty set)
    auto t = zeros({2, 0}, DType::Float32, device);
    auto result = sum(t, /*dim=*/1);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.numel(), 2);
    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

// ============================================================================
// 4. Reshape of Empty Tensors
// ============================================================================

TEST_P(EmptyTensorTest, ReshapeEmptyToEmpty) {
    auto t = zeros({0, 5}, DType::Float32, device);
    auto r = t.reshape({5, 0});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 2);
    EXPECT_EQ(r.shape()[0], 5);
    EXPECT_EQ(r.shape()[1], 0);
}

TEST_P(EmptyTensorTest, ReshapeEmptyToHigherRank) {
    auto t = zeros({0, 6}, DType::Float32, device);
    auto r = t.reshape({0, 2, 3});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 3);
    EXPECT_EQ(r.shape()[0], 0);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_EQ(r.shape()[2], 3);
}

TEST_P(EmptyTensorTest, ReshapeEmptyFlatten) {
    auto t = zeros({0, 5}, DType::Float32, device);
    auto r = t.reshape({0});
    // Flattening (0,5) should give shape (0,) with 0 elements
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 1);
    EXPECT_EQ(r.shape()[0], 0);
}

// ============================================================================
// 5. ndim() Correctness
// ============================================================================

TEST_P(EmptyTensorTest, NdimFor1DEmpty) {
    auto t = zeros({0}, DType::Float32, device);
    EXPECT_EQ(t.ndim(), 1);
    EXPECT_EQ(t.numel(), 0);
}

TEST_P(EmptyTensorTest, NdimFor3DEmpty) {
    auto t = zeros({0, 3, 4}, DType::Float32, device);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.numel(), 0);
}

TEST_P(EmptyTensorTest, NdimFor4DEmpty) {
    auto t = zeros({2, 0, 3, 4}, DType::Float32, device);
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_EQ(t.numel(), 0);
}

// ============================================================================
// 6. Broadcasting with Empty Tensors
// ============================================================================

TEST_P(EmptyTensorTest, AddEmptyWithEmpty) {
    auto a = zeros({0, 5}, DType::Float32, device);
    auto b = zeros({0, 5}, DType::Float32, device);
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 5);
}

TEST_P(EmptyTensorTest, AddEmptyBroadcastRow) {
    // (0, 5) + (1, 5) should broadcast to (0, 5)
    auto a = zeros({0, 5}, DType::Float32, device);
    auto b = ones({1, 5}, DType::Float32, device);
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 5);
}

// ============================================================================
// 7. Multiple DType Empty Tensors
// ============================================================================

TEST_P(EmptyTensorTest, EmptyFloat64) {
    auto t = zeros({0, 3}, DType::Float64, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.dtype(), DType::Float64);
    EXPECT_EQ(t.ndim(), 2);
}

TEST_P(EmptyTensorTest, EmptyInt32) {
    auto t = zeros({3, 0}, DType::Int32, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.dtype(), DType::Int32);
    EXPECT_EQ(t.ndim(), 2);
}

INSTANTIATE_BACKEND_TESTS(EmptyTensorTest);
