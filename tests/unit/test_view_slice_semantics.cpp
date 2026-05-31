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

#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

namespace {

class ViewSliceTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Build a Float32 tensor on `device` filled with 0..n-1 (row-major).
    Tensor iota(std::initializer_list<int64_t> shape) {
        auto host = zeros(shape, DType::Float32, Device::cpu());
        auto* data = host.data<float>();
        for (int64_t i = 0; i < host.numel(); ++i) {
            data[i] = static_cast<float>(i);
        }
        return host.to(device);
    }
};

// ============================================================================
// 1. Slice Produces Views with Correct Offset
// ============================================================================

TEST_P(ViewSliceTest, SliceBasicShape) {
    // Create a 4x5 tensor and slice rows 1..3
    auto t = iota({4, 5});

    auto s = t.slice(0, 1, 3);  // rows 1 and 2
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 5);
    EXPECT_EQ(s.numel(), 10);
}

TEST_P(ViewSliceTest, SliceDataCorrectness) {
    auto t = iota({4, 5});

    auto s = t.slice(0, 1, 3);  // rows 1 and 2
    auto s_contig = s.contiguous().cpu();
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

TEST_P(ViewSliceTest, SliceAlongDim1) {
    auto t = iota({3, 6});

    auto s = t.slice(1, 2, 5);  // cols 2, 3, 4
    EXPECT_EQ(s.shape()[0], 3);
    EXPECT_EQ(s.shape()[1], 3);
    EXPECT_EQ(s.numel(), 9);
}

// ============================================================================
// 2. .contiguous() Creates Independent Copy
// ============================================================================

TEST_P(ViewSliceTest, ContiguousCreatesIndependentCopy) {
    auto t = iota({4, 4});

    // Slicing rows of a row-major tensor leaves it contiguous, and
    // `.contiguous()` on an already-contiguous tensor is a no-op view
    // (matches PyTorch). To exercise the copy path, transpose first so
    // the resulting view is genuinely non-contiguous.
    auto s = t.transpose(0, 1);        // {4, 4} column-major view
    EXPECT_FALSE(s.is_contiguous());
    auto c = s.contiguous();           // independent copy
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.numel(), 16);

    // c[0] is t.T[0][0] = t[0][0] = 0.
    auto c_host = c.cpu();
    auto* c_data = c_host.data<float>();
    ASSERT_FLOAT_EQ(c_data[0], 0.0f);

    // Mutate the original in place via a contiguous view onto row 0 (which
    // includes element [0][0]). c — being an independent copy — must not
    // change.
    auto bump = ones({1, 4}, DType::Float32, device);
    auto row0 = t.slice(0, 0, 1);  // contiguous {1,4} view onto t row 0
    add_(row0, bump);              // t[0][*] += 1, so t[0][0] -> 1.0

    auto c_after = c.cpu();
    EXPECT_FLOAT_EQ(c_after.data<float>()[0], 0.0f);
}

// ============================================================================
// 3. In-place Operations on Views Affect Original
// ============================================================================

TEST_P(ViewSliceTest, InplaceAddOnViewAffectsOriginal) {
    auto t = zeros({4, 3}, DType::Float32, device);
    {
        auto host = zeros({4, 3}, DType::Float32, Device::cpu());
        auto* h = host.data<float>();
        for (int64_t i = 0; i < 12; ++i) {
            h[i] = static_cast<float>(i);
        }
        add_(t, host.to(device));
    }

    // Get a slice (view) of rows 1-2
    auto s = t.slice(0, 1, 3);

    // Perform in-place add on the slice
    auto increment = ones({2, 3}, DType::Float32, device);
    add_(s, increment);

    auto t_host = t.cpu();
    auto* data = t_host.data<float>();

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

TEST_P(ViewSliceTest, ChainedSlicesShape) {
    auto t = iota({10, 8});

    // First slice: rows 2..7 -> shape (5, 8)
    auto s1 = t.slice(0, 2, 7);
    EXPECT_EQ(s1.shape()[0], 5);
    EXPECT_EQ(s1.shape()[1], 8);

    // Second slice on s1: rows 1..3 -> shape (2, 8)
    auto s2 = s1.slice(0, 1, 3);
    EXPECT_EQ(s2.shape()[0], 2);
    EXPECT_EQ(s2.shape()[1], 8);

    // s2 should correspond to original rows 3 and 4
    auto s2_contig = s2.contiguous().cpu();
    auto* s2_data = s2_contig.data<float>();
    // Row 3 of original starts at index 24
    EXPECT_FLOAT_EQ(s2_data[0], 24.0f);
    // Row 4 of original starts at index 32
    EXPECT_FLOAT_EQ(s2_data[8], 32.0f);
}

TEST_P(ViewSliceTest, ChainedSlicesOnDifferentDims) {
    auto t = iota({6, 8});

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

TEST_P(ViewSliceTest, SlicePreservesNdim) {
    auto t = zeros({5, 4, 3}, DType::Float32, device);
    auto s = t.slice(0, 1, 3);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 4);
    EXPECT_EQ(s.shape()[2], 3);
    EXPECT_EQ(s.numel(), 24);
}

TEST_P(ViewSliceTest, SliceSingleRow) {
    auto t = zeros({5, 4}, DType::Float32, device);
    auto s = t.slice(0, 2, 3);  // single row
    EXPECT_EQ(s.shape()[0], 1);
    EXPECT_EQ(s.shape()[1], 4);
    EXPECT_EQ(s.numel(), 4);
}

// ============================================================================
// 6. Slice Backward Gradient
// ============================================================================

TEST_P(ViewSliceTest, SliceBackwardGradient) {
    // x is a 4x3 Variable
    auto x_host = ones({4, 3}, DType::Float32, Device::cpu());
    auto* x_ptr = x_host.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        x_ptr[i] = static_cast<float>(i + 1);
    }
    auto x_data = x_host.to(device);

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

TEST_P(ViewSliceTest, SlicePreservesDType) {
    auto t_f32 = zeros({4, 3}, DType::Float32, device);
    auto s_f32 = t_f32.slice(0, 0, 2);
    EXPECT_EQ(s_f32.dtype(), DType::Float32);

    auto t_f64 = zeros({4, 3}, DType::Float64, device);
    auto s_f64 = t_f64.slice(0, 0, 2);
    EXPECT_EQ(s_f64.dtype(), DType::Float64);
}

// ============================================================================
// 8. Contiguous on Already-Contiguous Tensor
// ============================================================================

TEST_P(ViewSliceTest, ContiguousOnContiguousTensorIsNoOp) {
    auto t = ones({3, 4}, DType::Float32, device);
    auto c = t.contiguous();
    // Calling contiguous on an already-contiguous tensor should be efficient
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);
    EXPECT_EQ(c.numel(), 12);
}

INSTANTIATE_BACKEND_TESTS(ViewSliceTest);

}  // namespace
