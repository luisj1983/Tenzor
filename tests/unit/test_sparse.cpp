#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SparseTest : public BackendTest {};

// ============================================================================
// COO Construction and Accessors
// ============================================================================

TEST_P(SparseTest, COOConstruction) {
    // 3x4 sparse matrix with 3 non-zeros
    auto indices = Tensor({2, int64_t(3)}, DType::Int64, device);
    auto values = Tensor({int64_t(3)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 2;  // rows
    idx[3] = 1; idx[4] = 2; idx[5] = 3;  // cols

    auto* v = values.data<float>();
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;

    auto sparse = SparseTensor::sparse_coo(indices, values, {3, 4});

    EXPECT_EQ(sparse.layout(), SparseLayout::COO);
    EXPECT_EQ(sparse.nnz(), 3);
    EXPECT_EQ(sparse.sparse_dim(), 2);
    EXPECT_EQ(sparse.shape()[0], 3);
    EXPECT_EQ(sparse.shape()[1], 4);
    EXPECT_EQ(sparse.dtype(), DType::Float32);
}

// ============================================================================
// CSR Construction and Accessors
// ============================================================================

TEST_P(SparseTest, CSRConstruction) {
    // 3x4 CSR with 4 non-zeros
    auto crow = Tensor({int64_t(4)}, DType::Int64, device);
    auto col = Tensor({int64_t(4)}, DType::Int64, device);
    auto values = Tensor({int64_t(4)}, DType::Float32, device);

    auto* cp = crow.data<int64_t>();
    cp[0] = 0; cp[1] = 1; cp[2] = 3; cp[3] = 4;

    auto* colp = col.data<int64_t>();
    colp[0] = 0; colp[1] = 1; colp[2] = 2; colp[3] = 3;

    auto* vp = values.data<float>();
    vp[0] = 1.0f; vp[1] = 2.0f; vp[2] = 3.0f; vp[3] = 4.0f;

    auto sparse = SparseTensor::sparse_csr(crow, col, values, {3, 4});

    EXPECT_EQ(sparse.layout(), SparseLayout::CSR);
    EXPECT_EQ(sparse.nnz(), 4);
    EXPECT_TRUE(sparse.is_coalesced());
}

// ============================================================================
// to_dense roundtrip
// ============================================================================

TEST_P(SparseTest, COOToDense) {
    // Build sparse [[0, 1], [2, 0]]
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, device);
    auto values = Tensor({int64_t(2)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;  // rows
    idx[2] = 1; idx[3] = 0;  // cols

    auto* v = values.data<float>();
    v[0] = 1.0f; v[1] = 2.0f;

    auto sparse = SparseTensor::sparse_coo(indices, values, {2, 2});
    auto dense = sparse.to_dense().to(Device::cpu());

    auto* d = dense.data<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f);  // [0,0]
    EXPECT_FLOAT_EQ(d[1], 1.0f);  // [0,1]
    EXPECT_FLOAT_EQ(d[2], 2.0f);  // [1,0]
    EXPECT_FLOAT_EQ(d[3], 0.0f);  // [1,1]
}

TEST_P(SparseTest, CSRToDense) {
    // Build sparse [[5, 0], [0, 7]]
    auto crow = Tensor({int64_t(3)}, DType::Int64, device);
    auto col = Tensor({int64_t(2)}, DType::Int64, device);
    auto values = Tensor({int64_t(2)}, DType::Float32, device);

    auto* cp = crow.data<int64_t>();
    cp[0] = 0; cp[1] = 1; cp[2] = 2;

    auto* colp = col.data<int64_t>();
    colp[0] = 0; colp[1] = 1;

    auto* vp = values.data<float>();
    vp[0] = 5.0f; vp[1] = 7.0f;

    auto sparse = SparseTensor::sparse_csr(crow, col, values, {2, 2});
    auto dense = sparse.to_dense().to(Device::cpu());

    auto* d = dense.data<float>();
    EXPECT_FLOAT_EQ(d[0], 5.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 0.0f);
    EXPECT_FLOAT_EQ(d[3], 7.0f);
}

// ============================================================================
// Dense -> Sparse -> Dense roundtrip
// ============================================================================

TEST_P(SparseTest, DenseSparseRoundtrip) {
    // Create a dense matrix with some zeros
    auto dense = zeros({3, 3}, DType::Float32, device);
    auto* dp = dense.data<float>();
    dp[0] = 1.0f;  // [0,0]
    dp[4] = 2.0f;  // [1,1]
    dp[8] = 3.0f;  // [2,2]

    auto sparse = to_sparse(dense);
    EXPECT_EQ(sparse.nnz(), 3);

    auto recovered = sparse.to_dense().to(Device::cpu());
    auto* rp = recovered.data<float>();

    EXPECT_FLOAT_EQ(rp[0], 1.0f);
    EXPECT_FLOAT_EQ(rp[1], 0.0f);
    EXPECT_FLOAT_EQ(rp[4], 2.0f);
    EXPECT_FLOAT_EQ(rp[8], 3.0f);
}

// ============================================================================
// COO <-> CSR conversion
// ============================================================================

TEST_P(SparseTest, COOToCSRConversion) {
    // Build COO and convert to CSR
    auto indices = Tensor({2, int64_t(3)}, DType::Int64, device);
    auto values = Tensor({int64_t(3)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 2;  // rows
    idx[3] = 0; idx[4] = 1; idx[5] = 2;  // cols (diagonal)

    auto* v = values.data<float>();
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;

    auto coo = SparseTensor::sparse_coo(indices, values, {3, 3});
    auto csr = coo.to_csr();

    EXPECT_EQ(csr.layout(), SparseLayout::CSR);
    EXPECT_EQ(csr.nnz(), 3);

    // Convert back and check dense
    auto dense = csr.to_dense().to(Device::cpu());
    auto* d = dense.data<float>();
    EXPECT_FLOAT_EQ(d[0], 1.0f);  // [0,0]
    EXPECT_FLOAT_EQ(d[4], 2.0f);  // [1,1]
    EXPECT_FLOAT_EQ(d[8], 3.0f);  // [2,2]
}

TEST_P(SparseTest, CSRToCOOConversion) {
    auto crow = Tensor({int64_t(3)}, DType::Int64, device);
    auto col = Tensor({int64_t(2)}, DType::Int64, device);
    auto values = Tensor({int64_t(2)}, DType::Float32, device);

    auto* cp = crow.data<int64_t>();
    cp[0] = 0; cp[1] = 1; cp[2] = 2;

    auto* colp = col.data<int64_t>();
    colp[0] = 1; colp[1] = 0;

    auto* vp = values.data<float>();
    vp[0] = 10.0f; vp[1] = 20.0f;

    auto csr = SparseTensor::sparse_csr(crow, col, values, {2, 2});
    auto coo = csr.to_coo();

    EXPECT_EQ(coo.layout(), SparseLayout::COO);
    EXPECT_EQ(coo.nnz(), 2);

    // Verify via to_dense
    auto dense = coo.to_dense().to(Device::cpu());
    auto* d = dense.data<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 10.0f);
    EXPECT_FLOAT_EQ(d[2], 20.0f);
    EXPECT_FLOAT_EQ(d[3], 0.0f);
}

// ============================================================================
// Coalesce
// ============================================================================

TEST_P(SparseTest, CoalesceDuplicates) {
    // Two entries at same position should be summed
    auto indices = Tensor({2, int64_t(3)}, DType::Int64, device);
    auto values = Tensor({int64_t(3)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1;  // rows: two at row 0
    idx[3] = 0; idx[4] = 0; idx[5] = 1;  // cols: two at (0,0), one at (1,1)

    auto* v = values.data<float>();
    v[0] = 3.0f; v[1] = 7.0f; v[2] = 5.0f;

    auto sparse = SparseTensor::sparse_coo(indices, values, {2, 2});
    EXPECT_FALSE(sparse.is_coalesced());

    auto coalesced = sparse.coalesce();
    EXPECT_TRUE(coalesced.is_coalesced());
    EXPECT_EQ(coalesced.nnz(), 2);  // (0,0) merged, (1,1) separate

    auto dense = coalesced.to_dense().to(Device::cpu());
    auto* d = dense.data<float>();
    EXPECT_FLOAT_EQ(d[0], 10.0f);  // 3+7
    EXPECT_FLOAT_EQ(d[3], 5.0f);
}

// ============================================================================
// SpMM: sparse-dense matrix multiply
// ============================================================================

TEST_P(SparseTest, SpMMCorrectness) {
    // A (2x3 sparse) * B (3x2 dense) = C (2x2 dense)
    // A = [[1, 0, 2], [0, 3, 0]]
    auto indices = Tensor({2, int64_t(3)}, DType::Int64, device);
    auto values = Tensor({int64_t(3)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1;  // rows
    idx[3] = 0; idx[4] = 2; idx[5] = 1;  // cols

    auto* v = values.data<float>();
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;

    auto A = SparseTensor::sparse_coo(indices, values, {2, 3});

    // B = [[1, 4], [2, 5], [3, 6]]
    auto B = zeros({3, 2}, DType::Float32, device);
    auto* bp = B.data<float>();
    bp[0] = 1.0f; bp[1] = 4.0f;
    bp[2] = 2.0f; bp[3] = 5.0f;
    bp[4] = 3.0f; bp[5] = 6.0f;

    auto C = sparse::spmm(A, B).to(Device::cpu());

    // C = A*B = [[1*1+2*3, 1*4+2*6], [3*2, 3*5]] = [[7, 16], [6, 15]]
    auto* cp = C.data<float>();
    EXPECT_FLOAT_EQ(cp[0], 7.0f);
    EXPECT_FLOAT_EQ(cp[1], 16.0f);
    EXPECT_FLOAT_EQ(cp[2], 6.0f);
    EXPECT_FLOAT_EQ(cp[3], 15.0f);
}

// ============================================================================
// SpMV: sparse-dense matrix-vector multiply
// ============================================================================

TEST_P(SparseTest, SpMVCorrectness) {
    // A = [[1, 0], [0, 2]], x = [3, 4]
    // y = A*x = [3, 8]
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, device);
    auto values = Tensor({int64_t(2)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;  // rows
    idx[2] = 0; idx[3] = 1;  // cols

    auto* v = values.data<float>();
    v[0] = 1.0f; v[1] = 2.0f;

    auto A = SparseTensor::sparse_coo(indices, values, {2, 2});

    auto x = Tensor({int64_t(2)}, DType::Float32, device);
    auto* xp = x.data<float>();
    xp[0] = 3.0f; xp[1] = 4.0f;

    auto y = sparse::spmv(A, x).to(Device::cpu());
    auto* yp = y.data<float>();
    EXPECT_FLOAT_EQ(yp[0], 3.0f);
    EXPECT_FLOAT_EQ(yp[1], 8.0f);
}

// ============================================================================
// Sparse-dense addition
// ============================================================================

TEST_P(SparseTest, SparseDenseAdd) {
    // sparse [[1, 0], [0, 2]] + dense [[10, 20], [30, 40]]
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, device);
    auto values = Tensor({int64_t(2)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;
    idx[2] = 0; idx[3] = 1;

    auto* v = values.data<float>();
    v[0] = 1.0f; v[1] = 2.0f;

    auto sp = SparseTensor::sparse_coo(indices, values, {2, 2});

    auto dense = Tensor({2, 2}, DType::Float32, device);
    auto* dp = dense.data<float>();
    dp[0] = 10.0f; dp[1] = 20.0f; dp[2] = 30.0f; dp[3] = 40.0f;

    auto result = sparse::add(sp, dense).to(Device::cpu());
    auto* rp = result.data<float>();
    EXPECT_FLOAT_EQ(rp[0], 11.0f);
    EXPECT_FLOAT_EQ(rp[1], 20.0f);
    EXPECT_FLOAT_EQ(rp[2], 30.0f);
    EXPECT_FLOAT_EQ(rp[3], 42.0f);
}

// ============================================================================
// Sparse-sparse addition
// ============================================================================

TEST_P(SparseTest, SparseSparseAdd) {
    // A = [[1, 0], [0, 0]], B = [[0, 0], [0, 2]]
    auto idx_a = Tensor({2, int64_t(1)}, DType::Int64, device);
    auto val_a = Tensor({int64_t(1)}, DType::Float32, device);
    idx_a.data<int64_t>()[0] = 0;
    idx_a.data<int64_t>()[1] = 0;
    val_a.data<float>()[0] = 1.0f;
    auto A = SparseTensor::sparse_coo(idx_a, val_a, {2, 2});

    auto idx_b = Tensor({2, int64_t(1)}, DType::Int64, device);
    auto val_b = Tensor({int64_t(1)}, DType::Float32, device);
    idx_b.data<int64_t>()[0] = 1;
    idx_b.data<int64_t>()[1] = 1;
    val_b.data<float>()[0] = 2.0f;
    auto B = SparseTensor::sparse_coo(idx_b, val_b, {2, 2});

    auto C = sparse::add(A, B);
    EXPECT_EQ(C.nnz(), 2);

    auto dense = C.to_dense().to(Device::cpu());
    auto* d = dense.data<float>();
    EXPECT_FLOAT_EQ(d[0], 1.0f);
    EXPECT_FLOAT_EQ(d[3], 2.0f);
}

// ============================================================================
// Scalar multiplication
// ============================================================================

TEST_P(SparseTest, ScalarMul) {
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, device);
    auto values = Tensor({int64_t(2)}, DType::Float32, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;
    idx[2] = 0; idx[3] = 1;

    auto* v = values.data<float>();
    v[0] = 3.0f; v[1] = 4.0f;

    auto sp = SparseTensor::sparse_coo(indices, values, {2, 2});
    auto scaled = sparse::mul(sp, 2.5);

    EXPECT_EQ(scaled.nnz(), 2);

    auto dense = scaled.to_dense().to(Device::cpu());
    auto* d = dense.data<float>();
    EXPECT_FLOAT_EQ(d[0], 7.5f);
    EXPECT_FLOAT_EQ(d[3], 10.0f);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_P(SparseTest, EmptySparse) {
    auto indices = Tensor({2, int64_t(0)}, DType::Int64, device);
    auto values = Tensor({int64_t(0)}, DType::Float32, device);
    auto sp = SparseTensor::sparse_coo(indices, values, {3, 3});

    EXPECT_EQ(sp.nnz(), 0);

    auto dense = sp.to_dense().to(Device::cpu());
    auto* d = dense.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(d[i], 0.0f);
    }
}

TEST_P(SparseTest, SingleElement) {
    auto indices = Tensor({2, int64_t(1)}, DType::Int64, device);
    auto values = Tensor({int64_t(1)}, DType::Float32, device);

    indices.data<int64_t>()[0] = 0;
    indices.data<int64_t>()[1] = 0;
    values.data<float>()[0] = 42.0f;

    auto sp = SparseTensor::sparse_coo(indices, values, {1, 1});
    auto dense = sp.to_dense().to(Device::cpu());
    EXPECT_FLOAT_EQ(dense.data<float>()[0], 42.0f);
}

TEST_P(SparseTest, Float64Support) {
    auto indices = Tensor({2, int64_t(2)}, DType::Int64, device);
    auto values = Tensor({int64_t(2)}, DType::Float64, device);

    auto* idx = indices.data<int64_t>();
    idx[0] = 0; idx[1] = 1;
    idx[2] = 0; idx[3] = 1;

    auto* v = values.data<double>();
    v[0] = 1.5; v[1] = 2.5;

    auto sp = SparseTensor::sparse_coo(indices, values, {2, 2});
    auto dense = sp.to_dense().to(Device::cpu());
    auto* d = dense.data<double>();
    EXPECT_DOUBLE_EQ(d[0], 1.5);
    EXPECT_DOUBLE_EQ(d[3], 2.5);
}

// ============================================================================
// Instantiate for CPU backend
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    CPU, SparseTest,
    ::testing::Values("cpu"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);
