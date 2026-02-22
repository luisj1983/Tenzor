/**
 * @file test_view_slice_semantics.cpp
 * @brief Tests for view and slice correctness
 *
 * Covers:
 * - Slice produces shared-storage views with correct offset
 * - .contiguous() on slices creates independent copy
 * - In-place ops on views affect original
 * - Chained slices work correctly
 * - Slice backward gradient
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;

class ViewSliceTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};

bool ViewSliceTest::initialized = false;

// ============================================================================
// 1. Slice Produces Views with Correct Offset
// ============================================================================

TEST_F(ViewSliceTest, SliceBasicShape) {
    // Create a 4x5 tensor and slice rows 1..3
    auto t = zeros({4, 5}, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 20; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto s = t.slice(0, 1, 3);  // rows 1 and 2
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 5);
    EXPECT_EQ(s.numel(), 10);
}

TEST_F(ViewSliceTest, SliceDataCorrectness) {
    auto t = zeros({4, 5}, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 20; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto s = t.slice(0, 1, 3);  // rows 1 and 2
    auto s_contig = s.contiguous();
    auto* s_data = s_contig.data<float>();

    // Row 1 of original: [5, 6, 7, 8, 9]
    EXPECT_FLOAT_EQ(s_data[0], 5.0f);
    EXPECT_FLOAT_EQ(s_data[1], 6.0f);
    EXPECT_FLOAT_EQ(s_data[2], 7.0f);
    EXPECT_FLOAT_EQ(s_data[3], 8.0f);
    EXPECT_FLOAT_EQ(s_data[4], 9.0f);

    // Row 2 of original: [10, 11, 12, 13, 14]
    EXPECT_FLOAT_EQ(s_data[5], 10.0f);
    EXPECT_FLOAT_EQ(s_data[6], 11.0f);
    EXPECT_FLOAT_EQ(s_data[7], 12.0f);
    EXPECT_FLOAT_EQ(s_data[8], 13.0f);
    EXPECT_FLOAT_EQ(s_data[9], 14.0f);
}

TEST_F(ViewSliceTest, SliceAlongDim1) {
    auto t = zeros({3, 6}, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 18; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto s = t.slice(1, 2, 5);  // cols 2, 3, 4
    EXPECT_EQ(s.shape()[0], 3);
    EXPECT_EQ(s.shape()[1], 3);
    EXPECT_EQ(s.numel(), 9);
}

// ============================================================================
// 2. .contiguous() Creates Independent Copy
// ============================================================================

TEST_F(ViewSliceTest, ContiguousCreatesIndependentCopy) {
    auto t = zeros({4, 4}, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto s = t.slice(0, 1, 3);        // view of rows 1-2
    auto c = s.contiguous();           // independent copy
    EXPECT_EQ(c.numel(), 8);

    // Modify the original tensor
    data[4] = 999.0f;  // first element of row 1

    // The contiguous copy should NOT be affected
    auto* c_data = c.data<float>();
    EXPECT_FLOAT_EQ(c_data[0], 4.0f);  // original value before modification
}

// ============================================================================
// 3. In-place Operations on Views Affect Original
// ============================================================================

TEST_F(ViewSliceTest, InplaceAddOnViewAffectsOriginal) {
    auto t = zeros({4, 3}, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Get a slice (view) of rows 1-2
    auto s = t.slice(0, 1, 3);

    // Perform in-place add on the slice
    auto increment = ones({2, 3}, DType::Float32, Device::cpu());
    add_(s, increment);

    // Original tensor row 1 should be modified: [3+1, 4+1, 5+1] = [4, 5, 6]
    EXPECT_FLOAT_EQ(data[3], 4.0f);
    EXPECT_FLOAT_EQ(data[4], 5.0f);
    EXPECT_FLOAT_EQ(data[5], 6.0f);

    // Original tensor row 2 should be modified: [6+1, 7+1, 8+1] = [7, 8, 9]
    EXPECT_FLOAT_EQ(data[6], 7.0f);
    EXPECT_FLOAT_EQ(data[7], 8.0f);
    EXPECT_FLOAT_EQ(data[8], 9.0f);

    // Row 0 should NOT be modified
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
}

// ============================================================================
// 4. Chained Slices
// ============================================================================

TEST_F(ViewSliceTest, ChainedSlicesShape) {
    auto t = zeros({10, 8}, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 80; ++i) {
        data[i] = static_cast<float>(i);
    }

    // First slice: rows 2..7 -> shape (5, 8)
    auto s1 = t.slice(0, 2, 7);
    EXPECT_EQ(s1.shape()[0], 5);
    EXPECT_EQ(s1.shape()[1], 8);

    // Second slice on s1: rows 1..3 -> shape (2, 8)
    auto s2 = s1.slice(0, 1, 3);
    EXPECT_EQ(s2.shape()[0], 2);
    EXPECT_EQ(s2.shape()[1], 8);

    // s2 should correspond to original rows 3 and 4
    auto s2_contig = s2.contiguous();
    auto* s2_data = s2_contig.data<float>();
    // Row 3 of original starts at index 24
    EXPECT_FLOAT_EQ(s2_data[0], 24.0f);
    // Row 4 of original starts at index 32
    EXPECT_FLOAT_EQ(s2_data[8], 32.0f);
}

TEST_F(ViewSliceTest, ChainedSlicesOnDifferentDims) {
    auto t = zeros({6, 8}, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 48; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Slice rows 1..4 -> shape (3, 8)
    auto s1 = t.slice(0, 1, 4);
    EXPECT_EQ(s1.shape()[0], 3);
    EXPECT_EQ(s1.shape()[1], 8);

    // Slice cols 2..6 from s1 -> shape (3, 4)
    auto s2 = s1.slice(1, 2, 6);
    EXPECT_EQ(s2.shape()[0], 3);
    EXPECT_EQ(s2.shape()[1], 4);
}

// ============================================================================
// 5. Slice Ndim and Numel
// ============================================================================

TEST_F(ViewSliceTest, SlicePreservesNdim) {
    auto t = zeros({5, 4, 3}, DType::Float32, Device::cpu());
    auto s = t.slice(0, 1, 3);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 4);
    EXPECT_EQ(s.shape()[2], 3);
    EXPECT_EQ(s.numel(), 24);
}

TEST_F(ViewSliceTest, SliceSingleRow) {
    auto t = zeros({5, 4}, DType::Float32, Device::cpu());
    auto s = t.slice(0, 2, 3);  // single row
    EXPECT_EQ(s.shape()[0], 1);
    EXPECT_EQ(s.shape()[1], 4);
    EXPECT_EQ(s.numel(), 4);
}

// ============================================================================
// 6. Slice Backward Gradient
// ============================================================================

TEST_F(ViewSliceTest, SliceBackwardGradient) {
    // x is a 4x3 Variable
    auto x_data = ones({4, 3}, DType::Float32, Device::cpu());
    auto* x_ptr = x_data.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        x_ptr[i] = static_cast<float>(i + 1);
    }

    Variable x(x_data, true);

    // Slice rows 1..3 (2 rows), then sum
    auto x_tensor = x.tensor();
    auto sliced = x_tensor.slice(0, 1, 3);
    Variable sliced_var(sliced, true);

    // Use autograd sum for gradient computation
    auto loss = tenzor::sum(sliced_var);
    loss.backward();

    // The gradient of sliced_var should be all ones (gradient of sum)
    ASSERT_TRUE(sliced_var.has_grad());
    auto grad = sliced_var.grad()->to(Device::cpu());
    auto* grad_data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 1.0f);
    }
}

// ============================================================================
// 7. DType Preservation Through Slice
// ============================================================================

TEST_F(ViewSliceTest, SlicePreservesDType) {
    auto t_f32 = zeros({4, 3}, DType::Float32, Device::cpu());
    auto s_f32 = t_f32.slice(0, 0, 2);
    EXPECT_EQ(s_f32.dtype(), DType::Float32);

    auto t_f64 = zeros({4, 3}, DType::Float64, Device::cpu());
    auto s_f64 = t_f64.slice(0, 0, 2);
    EXPECT_EQ(s_f64.dtype(), DType::Float64);
}

// ============================================================================
// 8. Contiguous on Already-Contiguous Tensor
// ============================================================================

TEST_F(ViewSliceTest, ContiguousOnContiguousTensorIsNoOp) {
    auto t = ones({3, 4}, DType::Float32, Device::cpu());
    auto c = t.contiguous();
    // Calling contiguous on an already-contiguous tensor should be efficient
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);
    EXPECT_EQ(c.numel(), 12);
}
