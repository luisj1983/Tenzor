/**
 * @file test_sddmm_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for SDDMM (sampled dense-dense matmul)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SDDMMMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Build a CSR mask with two non-zero entries: (0, 1) and (1, 2)
    // in a 3x3 matrix.
    SparseTensor make_2x3_csr_mask() {
        auto row_ptr = zeros({4}, DType::Int64, device());
        auto rp_cpu = row_ptr.to(Device::cpu());
        auto* rp = rp_cpu.data<int64_t>();
        rp[0] = 0; rp[1] = 1; rp[2] = 2; rp[3] = 2;
        row_ptr = rp_cpu.to(device());

        auto col_ind = zeros({2}, DType::Int64, device());
        auto ci_cpu = col_ind.to(Device::cpu());
        auto* ci = ci_cpu.data<int64_t>();
        ci[0] = 1; ci[1] = 2;
        col_ind = ci_cpu.to(device());

        auto values = ones({2}, dtype(), device());
        return SparseTensor::sparse_csr(row_ptr, col_ind, values, {3, 3});
    }
};

TEST_P(SDDMMMultiDTypeTest, ComputesDotProductsShape) {
    // A: 3x2, B: 3x2
    auto A = createZeros({3, 2});
    auto B = createZeros({3, 2});

    // Fill A on CPU
    auto a_cpu = A.to(Device::cpu()).to(DType::Float32);
    auto* a = a_cpu.data<float>();
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4; a[4] = 5; a[5] = 6;
    A = a_cpu.to(dtype()).to(device());

    // Fill B on CPU
    auto b_cpu = B.to(Device::cpu()).to(DType::Float32);
    auto* b = b_cpu.data<float>();
    b[0] = 1; b[1] = 0; b[2] = 0; b[3] = 1; b[4] = 1; b[5] = 1;
    B = b_cpu.to(dtype()).to(device());

    auto mask = make_2x3_csr_mask();
    auto result = sparse::sddmm(mask, A, B);

    const auto& vals = result.values();
    ASSERT_EQ(vals.numel(), 2);
}

TEST_P(SDDMMMultiDTypeTest, ComputesDotProductsValues) {
    auto A = createZeros({3, 2});
    auto B = createZeros({3, 2});

    auto a_cpu = A.to(Device::cpu()).to(DType::Float32);
    auto* a = a_cpu.data<float>();
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4; a[4] = 5; a[5] = 6;
    A = a_cpu.to(dtype()).to(device());

    auto b_cpu = B.to(Device::cpu()).to(DType::Float32);
    auto* b = b_cpu.data<float>();
    b[0] = 1; b[1] = 0; b[2] = 0; b[3] = 1; b[4] = 1; b[5] = 1;
    B = b_cpu.to(dtype()).to(device());

    auto mask = make_2x3_csr_mask();
    auto result = sparse::sddmm(mask, A, B);

    // Expected: (0,1): A[0].B[1] = 2, (1,2): A[1].B[2] = 7
    auto vals_cpu = result.values().to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(vals_cpu.data<float>()[0], 2.0f, atol() + 1e-3f);
    EXPECT_NEAR(vals_cpu.data<float>()[1], 7.0f, atol() + 1e-3f);
}

TEST_P(SDDMMMultiDTypeTest, StructureMatchesMask) {
    auto A = createOnes({3, 2});
    auto B = createOnes({3, 2});

    auto mask = make_2x3_csr_mask();
    auto result = sparse::sddmm(mask, A, B);

    EXPECT_EQ(result.crow_indices().numel(), mask.crow_indices().numel());
    EXPECT_EQ(result.col_indices().numel(), mask.col_indices().numel());
}

TEST_P(SDDMMMultiDTypeTest, RejectsShapeMismatch) {
    auto mask = make_2x3_csr_mask();
    auto A = createZeros({3, 2});
    auto B_bad = createZeros({3, 3});  // K mismatch
    EXPECT_THROW(sparse::sddmm(mask, A, B_bad), std::runtime_error);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SDDMMMultiDTypeTest);
