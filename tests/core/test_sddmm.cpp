// SDDMM (sampled dense-dense matmul) sanity tests — parameterized across
// every available backend. Previously CPU-only.

#include <gtest/gtest.h>

#include <tenzor/ops/creation.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/tenzor.hpp>

#include "../backend_test_fixture.hpp"

namespace tenzor {
namespace {

class SDDMMTest : public tenzor::testing::BackendTest {
protected:

    // Build a CSR mask with two non-zero entries: (0, 1) and (1, 2) in a
    // 3x3 matrix, allocated on `device` so sddmm stays on-backend.
    SparseTensor make_2x3_csr_mask(DType values_dtype = DType::Float32) {
        auto row_ptr_cpu = zeros({4}, DType::Int64, Device::cpu());
        auto* rp = row_ptr_cpu.data<int64_t>();
        rp[0] = 0; rp[1] = 1; rp[2] = 2; rp[3] = 2;
        auto col_ind_cpu = zeros({2}, DType::Int64, Device::cpu());
        auto* ci = col_ind_cpu.data<int64_t>();
        ci[0] = 1; ci[1] = 2;
        auto values = ones({2}, values_dtype, device);
        return SparseTensor::sparse_csr(row_ptr_cpu.to(device),
                                        col_ind_cpu.to(device),
                                        values, {3, 3});
    }
};

TEST_P(SDDMMTest, ComputesDotProductsAtMaskPositions) {
    // A: 3 rows x 2 cols, B: 3 rows x 2 cols (K=2).
    // (0, 1): A[0] . B[1] = 2,   (1, 2): A[1] . B[2] = 7
    auto A_cpu = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* a = A_cpu.data<float>();
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4; a[4] = 5; a[5] = 6;
    auto A = A_cpu.to(device);

    auto B_cpu = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* b = B_cpu.data<float>();
    b[0] = 1; b[1] = 0; b[2] = 0; b[3] = 1; b[4] = 1; b[5] = 1;
    auto B = B_cpu.to(device);

    auto mask = make_2x3_csr_mask();
    auto result = sparse::sddmm(mask, A, B);

    const auto vals_cpu = result.values().to(Device::cpu()).contiguous();
    ASSERT_EQ(vals_cpu.numel(), 2);
    EXPECT_FLOAT_EQ(vals_cpu.data<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(vals_cpu.data<float>()[1], 7.0f);

    EXPECT_EQ(result.crow_indices().numel(), mask.crow_indices().numel());
    EXPECT_EQ(result.col_indices().numel(), mask.col_indices().numel());
}

TEST_P(SDDMMTest, RejectsShapeMismatch) {
    auto mask = make_2x3_csr_mask();
    auto A = zeros({3, 2}, DType::Float32, device);
    auto B_bad = zeros({3, 3}, DType::Float32, device);  // K mismatch
    EXPECT_THROW(sparse::sddmm(mask, A, B_bad), std::runtime_error);

    auto A_bad = zeros({4, 2}, DType::Float32, device);   // M mismatch
    auto B = zeros({3, 2}, DType::Float32, device);
    EXPECT_THROW(sparse::sddmm(mask, A_bad, B), std::runtime_error);
}

TEST_P(SDDMMTest, RejectsDtypeMismatch) {
    auto mask = make_2x3_csr_mask();
    auto A = zeros({3, 2}, DType::Float32, device);
    auto B = zeros({3, 2}, DType::Float64, device);
    EXPECT_THROW(sparse::sddmm(mask, A, B), std::runtime_error);
}

TEST_P(SDDMMTest, Float64Roundtrip) {
    auto A_cpu = zeros({3, 2}, DType::Float64, Device::cpu());
    auto* a = A_cpu.data<double>();
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6;
    auto A = A_cpu.to(device);

    auto B_cpu = zeros({3, 2}, DType::Float64, Device::cpu());
    auto* b = B_cpu.data<double>();
    b[0]=1; b[1]=0; b[2]=0; b[3]=1; b[4]=1; b[5]=1;
    auto B = B_cpu.to(device);

    auto mask = make_2x3_csr_mask(DType::Float64);
    auto result = sparse::sddmm(mask, A, B);
    auto vals_cpu = result.values().to(Device::cpu()).contiguous();
    ASSERT_EQ(vals_cpu.numel(), 2);
    EXPECT_DOUBLE_EQ(vals_cpu.data<double>()[0], 2.0);
    EXPECT_DOUBLE_EQ(vals_cpu.data<double>()[1], 7.0);
}

INSTANTIATE_BACKEND_TESTS(SDDMMTest);

}  // namespace
}  // namespace tenzor
