// Phase 6.2: SDDMM (sampled dense-dense matmul) sanity tests.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/ops/creation.hpp>

namespace tenzor {
namespace {

class SDDMMTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }

    // Build a CSR mask with two non-zero entries: (0, 1) and (1, 2)
    // in a 3x3 matrix. That lets us predict the output exactly.
    SparseTensor make_2x3_csr_mask() {
        // row_ptr: [0, 1, 2, 2]  -> row 0 has 1 nnz, row 1 has 1 nnz, row 2 has 0
        auto row_ptr = zeros({4}, DType::Int64, Device::cpu());
        auto* rp = row_ptr.data<int64_t>();
        rp[0] = 0; rp[1] = 1; rp[2] = 2; rp[3] = 2;
        // col_ind: [1, 2]
        auto col_ind = zeros({2}, DType::Int64, Device::cpu());
        auto* ci = col_ind.data<int64_t>();
        ci[0] = 1; ci[1] = 2;
        // values: placeholder — SDDMM overwrites them
        auto values = ones({2}, DType::Float32, Device::cpu());
        return SparseTensor::sparse_csr(row_ptr, col_ind, values, {3, 3});
    }
};

TEST_F(SDDMMTest, ComputesDotProductsAtMaskPositions) {
    // A: 3 rows x 2 cols, B: 3 rows x 2 cols (K=2).
    // A[0] = [1, 2], A[1] = [3, 4], A[2] = [5, 6]
    // B[0] = [1, 0], B[1] = [0, 1], B[2] = [1, 1]
    //
    // Expected dot products at mask positions:
    //   (0, 1): A[0] . B[1] = 1*0 + 2*1 = 2
    //   (1, 2): A[1] . B[2] = 3*1 + 4*1 = 7
    auto A = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* a = A.data<float>();
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4; a[4] = 5; a[5] = 6;

    auto B = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* b = B.data<float>();
    b[0] = 1; b[1] = 0; b[2] = 0; b[3] = 1; b[4] = 1; b[5] = 1;

    auto mask = make_2x3_csr_mask();
    auto result = sparse::sddmm(mask, A, B);

    // Result should have 2 non-zeros at the same (row, col) positions as
    // the mask, with values [2, 7].
    const auto& vals = result.values();
    ASSERT_EQ(vals.numel(), 2);
    EXPECT_FLOAT_EQ(vals.data<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(vals.data<float>()[1], 7.0f);

    // The row_ptr and col_ind of the result should equal the mask's.
    EXPECT_EQ(result.crow_indices().numel(), mask.crow_indices().numel());
    EXPECT_EQ(result.col_indices().numel(), mask.col_indices().numel());
}

TEST_F(SDDMMTest, RejectsShapeMismatch) {
    auto mask = make_2x3_csr_mask();
    auto A = zeros({3, 2}, DType::Float32, Device::cpu());
    auto B_bad = zeros({3, 3}, DType::Float32, Device::cpu());  // K mismatch
    EXPECT_THROW(sparse::sddmm(mask, A, B_bad), std::runtime_error);

    auto A_bad = zeros({4, 2}, DType::Float32, Device::cpu());   // M mismatch
    auto B = zeros({3, 2}, DType::Float32, Device::cpu());
    EXPECT_THROW(sparse::sddmm(mask, A_bad, B), std::runtime_error);
}

TEST_F(SDDMMTest, RejectsDtypeMismatch) {
    auto mask = make_2x3_csr_mask();
    auto A = zeros({3, 2}, DType::Float32, Device::cpu());
    auto B = zeros({3, 2}, DType::Float64, Device::cpu());
    EXPECT_THROW(sparse::sddmm(mask, A, B), std::runtime_error);
}

TEST_F(SDDMMTest, Float64Roundtrip) {
    // Smoke test on Float64 path.
    auto A = zeros({3, 2}, DType::Float64, Device::cpu());
    A.data<double>()[0] = 1.0;
    A.data<double>()[1] = 2.0;
    A.data<double>()[2] = 3.0;
    A.data<double>()[3] = 4.0;
    A.data<double>()[4] = 5.0;
    A.data<double>()[5] = 6.0;

    auto B = zeros({3, 2}, DType::Float64, Device::cpu());
    B.data<double>()[0] = 1.0;
    B.data<double>()[1] = 0.0;
    B.data<double>()[2] = 0.0;
    B.data<double>()[3] = 1.0;
    B.data<double>()[4] = 1.0;
    B.data<double>()[5] = 1.0;

    // Build a mask with Float64 values so dtype matches on that side.
    auto row_ptr = zeros({4}, DType::Int64, Device::cpu());
    auto* rp = row_ptr.data<int64_t>();
    rp[0] = 0; rp[1] = 1; rp[2] = 2; rp[3] = 2;
    auto col_ind = zeros({2}, DType::Int64, Device::cpu());
    auto* ci = col_ind.data<int64_t>();
    ci[0] = 1; ci[1] = 2;
    auto values = ones({2}, DType::Float64, Device::cpu());
    auto mask = SparseTensor::sparse_csr(row_ptr, col_ind, values, {3, 3});

    auto result = sparse::sddmm(mask, A, B);
    ASSERT_EQ(result.values().numel(), 2);
    EXPECT_DOUBLE_EQ(result.values().data<double>()[0], 2.0);
    EXPECT_DOUBLE_EQ(result.values().data<double>()[1], 7.0);
}

} // namespace
} // namespace tenzor
