/**
 * @file test_sparse_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for sparse tensor operations
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SparseMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        // Check if sparse ops are available on this backend
        try {
            auto indices = Tensor({2, int64_t(1)}, DType::Int64, Device::cpu());
            auto values = Tensor({int64_t(1)}, DType::Float32, Device::cpu());
            indices.data<int64_t>()[0] = 0;
            indices.data<int64_t>()[1] = 0;
            values.data<float>()[0] = 1.0f;
            auto sp = SparseTensor::sparse_coo(indices, values, {2, 2});
            sp.to_dense();
        } catch (...) {
            GTEST_SKIP() << "Sparse ops not available on " << backend_name();
        }
    }
};

TEST_P(SparseMultiDTypeTest, COOConstruction) {
    auto indices = Tensor({2, int64_t(3)}, DType::Int64, Device::cpu());
    auto values = Tensor({int64_t(3)}, dtype(), Device::cpu());

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 2;
    idx[3] = 1; idx[4] = 2; idx[5] = 3;

    // Fill values via Float32 then convert
    auto vals_f32 = Tensor({int64_t(3)}, DType::Float32, Device::cpu());
    auto* v = vals_f32.data<float>();
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;
    values = vals_f32.to(dtype());

    auto sparse = SparseTensor::sparse_coo(indices, values, {3, 4});

    EXPECT_EQ(sparse.layout(), SparseLayout::COO);
    EXPECT_EQ(sparse.nnz(), 3);
    EXPECT_EQ(sparse.shape()[0], 3);
    EXPECT_EQ(sparse.shape()[1], 4);
}

TEST_P(SparseMultiDTypeTest, COOToDense) {
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, Device::cpu());
    auto vals_f32 = Tensor({int64_t(2)}, DType::Float32, Device::cpu());

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;
    idx[2] = 1; idx[3] = 0;

    auto* v = vals_f32.data<float>();
    v[0] = 1.0f; v[1] = 2.0f;
    auto values = vals_f32.to(dtype());

    auto sparse = SparseTensor::sparse_coo(indices, values, {2, 2});
    auto dense = sparse.to_dense().to(Device::cpu()).to(DType::Float32);

    auto* d = dense.data<float>();
    EXPECT_NEAR(d[0], 0.0f, atol());
    EXPECT_NEAR(d[1], 1.0f, atol());
    EXPECT_NEAR(d[2], 2.0f, atol());
    EXPECT_NEAR(d[3], 0.0f, atol());
}

TEST_P(SparseMultiDTypeTest, SpMMCorrectness) {
    // A (2x3 sparse) * B (3x2 dense) = C (2x2)
    auto indices = Tensor({2, int64_t(3)}, DType::Int64, Device::cpu());
    auto vals_f32 = Tensor({int64_t(3)}, DType::Float32, Device::cpu());

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1;
    idx[3] = 0; idx[4] = 2; idx[5] = 1;

    auto* v = vals_f32.data<float>();
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;
    auto values = vals_f32.to(dtype());

    auto A = SparseTensor::sparse_coo(indices, values, {2, 3});

    auto B_cpu = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* bp = B_cpu.data<float>();
    bp[0] = 1.0f; bp[1] = 4.0f;
    bp[2] = 2.0f; bp[3] = 5.0f;
    bp[4] = 3.0f; bp[5] = 6.0f;
    auto B = B_cpu.to(dtype()).to(device());

    auto C = sparse::spmm(A, B).to(Device::cpu()).to(DType::Float32);
    auto* cp = C.data<float>();
    EXPECT_NEAR(cp[0], 7.0f, atol());
    EXPECT_NEAR(cp[1], 16.0f, atol());
    EXPECT_NEAR(cp[2], 6.0f, atol());
    EXPECT_NEAR(cp[3], 15.0f, atol());
}

TEST_P(SparseMultiDTypeTest, SpMVCorrectness) {
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, Device::cpu());
    auto vals_f32 = Tensor({int64_t(2)}, DType::Float32, Device::cpu());

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;
    idx[2] = 0; idx[3] = 1;

    auto* v = vals_f32.data<float>();
    v[0] = 1.0f; v[1] = 2.0f;
    auto values = vals_f32.to(dtype());

    auto A = SparseTensor::sparse_coo(indices, values, {2, 2});

    auto x_cpu = Tensor({int64_t(2)}, DType::Float32, Device::cpu());
    auto* xp = x_cpu.data<float>();
    xp[0] = 3.0f; xp[1] = 4.0f;
    auto x = x_cpu.to(dtype()).to(device());

    auto y = sparse::spmv(A, x).to(Device::cpu()).to(DType::Float32);
    auto* yp = y.data<float>();
    EXPECT_NEAR(yp[0], 3.0f, atol());
    EXPECT_NEAR(yp[1], 8.0f, atol());
}

TEST_P(SparseMultiDTypeTest, SparseDenseAdd) {
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, Device::cpu());
    auto vals_f32 = Tensor({int64_t(2)}, DType::Float32, Device::cpu());

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;
    idx[2] = 0; idx[3] = 1;

    auto* v = vals_f32.data<float>();
    v[0] = 1.0f; v[1] = 2.0f;
    auto values = vals_f32.to(dtype());

    auto sp = SparseTensor::sparse_coo(indices, values, {2, 2});

    auto dense_cpu = Tensor({2, 2}, DType::Float32, Device::cpu());
    auto* dp = dense_cpu.data<float>();
    dp[0] = 10.0f; dp[1] = 20.0f; dp[2] = 30.0f; dp[3] = 40.0f;
    auto dense = dense_cpu.to(dtype()).to(device());

    auto result = sparse::add(sp, dense).to(Device::cpu()).to(DType::Float32);
    auto* rp = result.data<float>();
    EXPECT_NEAR(rp[0], 11.0f, atol());
    EXPECT_NEAR(rp[1], 20.0f, atol());
    EXPECT_NEAR(rp[2], 30.0f, atol());
    EXPECT_NEAR(rp[3], 42.0f, atol());
}

TEST_P(SparseMultiDTypeTest, EmptySparse) {
    auto indices = Tensor({2, int64_t(0)}, DType::Int64, Device::cpu());
    auto values = Tensor({int64_t(0)}, dtype(), Device::cpu());
    auto sp = SparseTensor::sparse_coo(indices, values, {3, 3});

    EXPECT_EQ(sp.nnz(), 0);

    auto dense = sp.to_dense().to(Device::cpu()).to(DType::Float32);
    auto* d = dense.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(d[i], 0.0f, atol());
    }
}

TEST_P(SparseMultiDTypeTest, CoalesceDuplicates) {
    auto indices = Tensor({2, int64_t(3)}, DType::Int64, Device::cpu());
    auto vals_f32 = Tensor({int64_t(3)}, DType::Float32, Device::cpu());

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1;
    idx[3] = 0; idx[4] = 0; idx[5] = 1;

    auto* v = vals_f32.data<float>();
    v[0] = 3.0f; v[1] = 7.0f; v[2] = 5.0f;
    auto values = vals_f32.to(dtype());

    auto sparse = SparseTensor::sparse_coo(indices, values, {2, 2});
    auto coalesced = sparse.coalesce();
    EXPECT_TRUE(coalesced.is_coalesced());

    auto dense = coalesced.to_dense().to(Device::cpu()).to(DType::Float32);
    auto* d = dense.data<float>();
    EXPECT_NEAR(d[0], 10.0f, atol());
    EXPECT_NEAR(d[3], 5.0f, atol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SparseMultiDTypeTest);
