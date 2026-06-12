/**
 * @file test_sparse_bsr_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for BSR (Block Sparse Row) tensor format
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SparseBSRMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfSparseUnavailable() {
        // Smoke-construct a minimal sparse tensor on the current device.
        // Backend availability is already gated upstream by the
        // MultiBackendDTypeTest fixture, so any failure here is a real
        // sparse_bsr construction bug and must propagate as a test failure
        // rather than be swallowed as "not available".
        auto row_ptr = tenzor::zeros({2}, DType::Int64, device());
        auto col_ind = tenzor::zeros({0}, DType::Int64, device());
        auto values = tenzor::zeros({0, 2, 2}, DType::Float32, device());
        (void)SparseTensor::sparse_bsr(row_ptr, col_ind, values,
                                        {2, 2}, {2, 2});
    }
};

TEST_P(SparseBSRMultiDTypeTest, Construction) {
    skipIfSparseUnavailable();

    // Create a simple BSR tensor: 4x4 matrix with 2x2 blocks
    auto bsr_row_ptr = tenzor::zeros({3}, DType::Int64, device());
    auto bsr_col_ind = tenzor::zeros({2}, DType::Int64, device());

    // Fill index arrays on CPU then move
    auto row_ptr_cpu = tenzor::zeros({3}, DType::Int64);
    row_ptr_cpu.data<int64_t>()[0] = 0;
    row_ptr_cpu.data<int64_t>()[1] = 1;
    row_ptr_cpu.data<int64_t>()[2] = 2;
    bsr_row_ptr = row_ptr_cpu.to(device());

    auto col_ind_cpu = tenzor::zeros({2}, DType::Int64);
    col_ind_cpu.data<int64_t>()[0] = 0;
    col_ind_cpu.data<int64_t>()[1] = 1;
    bsr_col_ind = col_ind_cpu.to(device());

    // Values: 2 blocks of 2x2, created in Float32 then cast
    auto values_cpu = tenzor::zeros({2, 2, 2}, DType::Float32);
    auto* v = values_cpu.data<float>();
    v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
    v[4] = 5; v[5] = 6; v[6] = 7; v[7] = 8;
    auto values = values_cpu.to(dtype()).to(device());

    auto bsr = SparseTensor::sparse_bsr(bsr_row_ptr, bsr_col_ind, values,
                                         {4, 4}, {2, 2});

    EXPECT_EQ(bsr.layout(), SparseLayout::BSR);
    EXPECT_EQ(bsr.shape()[0], 4);
    EXPECT_EQ(bsr.shape()[1], 4);
    EXPECT_EQ(bsr.block_size().first, 2);
    EXPECT_EQ(bsr.block_size().second, 2);
}

TEST_P(SparseBSRMultiDTypeTest, ToDenseConversion) {
    skipIfSparseUnavailable();

    auto row_ptr_cpu = tenzor::zeros({3}, DType::Int64);
    row_ptr_cpu.data<int64_t>()[0] = 0;
    row_ptr_cpu.data<int64_t>()[1] = 1;
    row_ptr_cpu.data<int64_t>()[2] = 2;

    auto col_ind_cpu = tenzor::zeros({2}, DType::Int64);
    col_ind_cpu.data<int64_t>()[0] = 0;
    col_ind_cpu.data<int64_t>()[1] = 1;

    auto values_cpu = tenzor::zeros({2, 2, 2}, DType::Float32);
    auto* v = values_cpu.data<float>();
    v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
    v[4] = 5; v[5] = 6; v[6] = 7; v[7] = 8;

    auto bsr_row_ptr = row_ptr_cpu.to(device());
    auto bsr_col_ind = col_ind_cpu.to(device());
    auto values = values_cpu.to(dtype()).to(device());

    auto bsr = SparseTensor::sparse_bsr(bsr_row_ptr, bsr_col_ind, values,
                                         {4, 4}, {2, 2});

    auto dense = bsr.to_dense();
    EXPECT_EQ(dense.shape()[0], 4);
    EXPECT_EQ(dense.shape()[1], 4);

    // Compare on CPU in Float32
    auto d_cpu = dense.to(Device::cpu()).to(DType::Float32);
    auto* d = d_cpu.data<float>();
    // Block (0,0)
    EXPECT_NEAR(d[0*4+0], 1.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[0*4+1], 2.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[1*4+0], 3.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[1*4+1], 4.0f, atol() + 1e-2f);
    // Block (1,1)
    EXPECT_NEAR(d[2*4+2], 5.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[2*4+3], 6.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[3*4+2], 7.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[3*4+3], 8.0f, atol() + 1e-2f);
    // Zeros elsewhere
    EXPECT_NEAR(d[0*4+2], 0.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[0*4+3], 0.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[2*4+0], 0.0f, atol() + 1e-2f);
    EXPECT_NEAR(d[2*4+1], 0.0f, atol() + 1e-2f);
}

TEST_P(SparseBSRMultiDTypeTest, ShapePreservedAcrossDevices) {
    skipIfSparseUnavailable();

    auto row_ptr_cpu = tenzor::zeros({3}, DType::Int64);
    row_ptr_cpu.data<int64_t>()[0] = 0;
    row_ptr_cpu.data<int64_t>()[1] = 1;
    row_ptr_cpu.data<int64_t>()[2] = 2;

    auto col_ind_cpu = tenzor::zeros({2}, DType::Int64);
    col_ind_cpu.data<int64_t>()[0] = 0;
    col_ind_cpu.data<int64_t>()[1] = 1;

    auto values_cpu = tenzor::zeros({2, 2, 2}, DType::Float32);
    auto* v = values_cpu.data<float>();
    v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
    v[4] = 5; v[5] = 6; v[6] = 7; v[7] = 8;

    auto bsr = SparseTensor::sparse_bsr(
        row_ptr_cpu.to(device()),
        col_ind_cpu.to(device()),
        values_cpu.to(dtype()).to(device()),
        {4, 4}, {2, 2});

    EXPECT_EQ(bsr.shape()[0], 4);
    EXPECT_EQ(bsr.shape()[1], 4);
    EXPECT_EQ(bsr.block_size().first, 2);
    EXPECT_EQ(bsr.block_size().second, 2);
}

TEST_P(SparseBSRMultiDTypeTest, EmptyBSR) {
    skipIfSparseUnavailable();

    // 4x4 matrix with no nonzero blocks
    auto row_ptr_cpu = tenzor::zeros({3}, DType::Int64);
    // All zeros: no blocks in any row
    auto col_ind_cpu = tenzor::zeros({0}, DType::Int64);
    auto values_cpu = tenzor::zeros({0, 2, 2}, DType::Float32);

    auto bsr = SparseTensor::sparse_bsr(
        row_ptr_cpu.to(device()),
        col_ind_cpu.to(device()),
        values_cpu.to(dtype()).to(device()),
        {4, 4}, {2, 2});

    EXPECT_EQ(bsr.layout(), SparseLayout::BSR);
    EXPECT_EQ(bsr.shape()[0], 4);
    EXPECT_EQ(bsr.shape()[1], 4);

    auto dense = bsr.to_dense();
    auto d_cpu = dense.to(Device::cpu()).to(DType::Float32);
    auto* d = d_cpu.data<float>();
    for (int64_t i = 0; i < 16; ++i) {
        EXPECT_NEAR(d[i], 0.0f, atol() + 1e-2f);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SparseBSRMultiDTypeTest);
