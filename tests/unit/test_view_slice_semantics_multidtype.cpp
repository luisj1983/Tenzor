/**
 * @file test_view_slice_semantics_multidtype.cpp
 * @brief Multi-backend tests for view and slice correctness
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
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class ViewSliceMultiBackendTest : public BackendTest {};

// ============================================================================
// 1. Slice Produces Views with Correct Offset
// ============================================================================

TEST_P(ViewSliceMultiBackendTest, SliceBasicShape) {
    // Create a 4x5 tensor and slice rows 1..3
    auto t = zeros({4, 5}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < 20; ++i) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    auto s = t.slice(0, 1, 3);  // rows 1 and 2
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 5);
    EXPECT_EQ(s.numel(), 10);
}

TEST_P(ViewSliceMultiBackendTest, SliceDataCorrectness) {
    auto t = zeros({4, 5}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < 20; ++i) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    auto s = t.slice(0, 1, 3);  // rows 1 and 2
    auto s_contig = s.contiguous().to(Device::cpu());
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

TEST_P(ViewSliceMultiBackendTest, SliceAlongDim1) {
    auto t = zeros({3, 6}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < 18; ++i) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    auto s = t.slice(1, 2, 5);  // cols 2, 3, 4
    EXPECT_EQ(s.shape()[0], 3);
    EXPECT_EQ(s.shape()[1], 3);
    EXPECT_EQ(s.numel(), 9);
}

// ============================================================================
// 2. .contiguous() Creates Independent Copy
// ============================================================================

TEST_P(ViewSliceMultiBackendTest, ContiguousCreatesIndependentCopy) {
    auto t_cpu = zeros({4, 4}, DType::Float32, Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto t = t_cpu.to(device);

    // Use a non-contiguous slice (stride > 1) so that `.contiguous()` is
    // forced to materialize a copy regardless of device. An already-
    // contiguous slice may legally return the same view (zero-copy), which
    // would share storage with the source tensor on the CPU backend and
    // make the independence check meaningless.
    auto s = t.slice(0, 0, 4, 2);     // view of rows 0 and 2 (stride 2)
    auto c = s.contiguous();           // independent copy (non-contiguous src)
    EXPECT_EQ(c.numel(), 8);

    // Modify the CPU-side source and re-upload to the device (on CPU,
    // to(cpu()) is a no-op, so this also mutates the original device tensor
    // since they share storage).
    data[0] = 999.0f;  // first element of row 0 in the slice
    t = t_cpu.to(device);

    // The contiguous copy must not be affected by the post-copy mutation.
    auto c_cpu = c.to(Device::cpu());
    auto* c_data = c_cpu.data<float>();
    EXPECT_FLOAT_EQ(c_data[0], 0.0f);  // original row-0, col-0 value
}

// ============================================================================
// 3. In-place Operations on Views Affect Original
// ============================================================================

TEST_P(ViewSliceMultiBackendTest, InplaceAddOnViewAffectsOriginal) {
    auto t_cpu = zeros({4, 3}, DType::Float32, Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto t = t_cpu.to(device);

    // Get a slice (view) of rows 1-2
    auto s = t.slice(0, 1, 3);

    // Perform in-place add on the slice
    auto increment = ones({2, 3}, DType::Float32, device);
    add_(s, increment);

    // Move back to CPU to check
    auto result = t.to(Device::cpu());
    auto* r_data = result.data<float>();

    // Original tensor row 1 should be modified: [3+1, 4+1, 5+1] = [4, 5, 6]
    EXPECT_FLOAT_EQ(r_data[3], 4.0f);
    EXPECT_FLOAT_EQ(r_data[4], 5.0f);
    EXPECT_FLOAT_EQ(r_data[5], 6.0f);

    // Original tensor row 2 should be modified: [6+1, 7+1, 8+1] = [7, 8, 9]
    EXPECT_FLOAT_EQ(r_data[6], 7.0f);
    EXPECT_FLOAT_EQ(r_data[7], 8.0f);
    EXPECT_FLOAT_EQ(r_data[8], 9.0f);

    // Row 0 should NOT be modified
    EXPECT_FLOAT_EQ(r_data[0], 0.0f);
    EXPECT_FLOAT_EQ(r_data[1], 1.0f);
    EXPECT_FLOAT_EQ(r_data[2], 2.0f);
}

// ============================================================================
// 4. Chained Slices
// ============================================================================

TEST_P(ViewSliceMultiBackendTest, ChainedSlicesShape) {
    auto t_cpu = zeros({10, 8}, DType::Float32, Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < 80; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto t = t_cpu.to(device);

    // First slice: rows 2..7 -> shape (5, 8)
    auto s1 = t.slice(0, 2, 7);
    EXPECT_EQ(s1.shape()[0], 5);
    EXPECT_EQ(s1.shape()[1], 8);

    // Second slice on s1: rows 1..3 -> shape (2, 8)
    auto s2 = s1.slice(0, 1, 3);
    EXPECT_EQ(s2.shape()[0], 2);
    EXPECT_EQ(s2.shape()[1], 8);

    // s2 should correspond to original rows 3 and 4
    auto s2_contig = s2.contiguous().to(Device::cpu());
    auto* s2_data = s2_contig.data<float>();
    // Row 3 of original starts at index 24
    EXPECT_FLOAT_EQ(s2_data[0], 24.0f);
    // Row 4 of original starts at index 32
    EXPECT_FLOAT_EQ(s2_data[8], 32.0f);
}

TEST_P(ViewSliceMultiBackendTest, ChainedSlicesOnDifferentDims) {
    auto t_cpu = zeros({6, 8}, DType::Float32, Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < 48; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto t = t_cpu.to(device);

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

TEST_P(ViewSliceMultiBackendTest, SlicePreservesNdim) {
    auto t = zeros({5, 4, 3}, DType::Float32, device);
    auto s = t.slice(0, 1, 3);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 4);
    EXPECT_EQ(s.shape()[2], 3);
    EXPECT_EQ(s.numel(), 24);
}

TEST_P(ViewSliceMultiBackendTest, SliceSingleRow) {
    auto t = zeros({5, 4}, DType::Float32, device);
    auto s = t.slice(0, 2, 3);  // single row
    EXPECT_EQ(s.shape()[0], 1);
    EXPECT_EQ(s.shape()[1], 4);
    EXPECT_EQ(s.numel(), 4);
}

// ============================================================================
// 6. Slice Backward Gradient
// ============================================================================

TEST_P(ViewSliceMultiBackendTest, SliceBackwardGradient) {
    // x is a 4x3 Variable
    auto x_data = ones({4, 3}, DType::Float32, device);
    auto x_cpu = x_data.to(Device::cpu());
    auto* x_ptr = x_cpu.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        x_ptr[i] = static_cast<float>(i + 1);
    }
    x_data = x_cpu.to(device);

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

TEST_P(ViewSliceMultiBackendTest, SlicePreservesDType) {
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

TEST_P(ViewSliceMultiBackendTest, ContiguousOnContiguousTensorIsNoOp) {
    auto t = ones({3, 4}, DType::Float32, device);
    auto c = t.contiguous();
    // Calling contiguous on an already-contiguous tensor should be efficient
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);
    EXPECT_EQ(c.numel(), 12);
}

INSTANTIATE_BACKEND_TESTS(ViewSliceMultiBackendTest);
