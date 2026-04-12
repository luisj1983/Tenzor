/**
 * @file test_sparse_parity.cpp
 * @brief Sparse operation parity tests across backends
 *
 * Verifies that sparse tensor operations (SpMM, SpMV, sparse-dense conversion)
 * produce identical results across all backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Sparse-Dense Interaction Parity
// ============================================================================

TEST(SparseParity, IdentityMatmul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Identity matrix @ dense vector — exercises sparse-like paths
    auto identity = eye(8, 8, DType::Float32, Device::cpu());
    auto dense = randn({8, 4}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {identity, dense}, 1e-5f, 1e-6f, "Identity @ Dense (sparse-like)");
}

TEST(SparseParity, SparsePatternMatmul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Mostly-zero matrix @ dense — tests numerical accumulation with many zeros
    auto sparse_like = zeros({8, 8}, DType::Float32, Device::cpu());
    // Set diagonal and a few off-diagonal entries
    auto sparse_data = sparse_like.data<float>();
    for (int i = 0; i < 8; ++i) {
        sparse_data[i * 8 + i] = 1.0f;           // diagonal
        if (i + 1 < 8) sparse_data[i * 8 + i + 1] = 0.5f;  // super-diagonal
    }
    auto dense = randn({8, 4}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {sparse_like, dense}, 1e-5f, 1e-6f, "Sparse-pattern matmul");
}

// ============================================================================
// True SparseTensor Operation Tests
// ============================================================================

// Helper to create a small CSR sparse matrix for tests
static auto make_test_csr() -> SparseTensor {
    // 3x3 matrix:
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    auto crow = zeros({4}, DType::Int64, Device::cpu());
    crow.data<int64_t>()[0] = 0; crow.data<int64_t>()[1] = 2;
    crow.data<int64_t>()[2] = 3; crow.data<int64_t>()[3] = 5;

    auto cols = zeros({5}, DType::Int64, Device::cpu());
    cols.data<int64_t>()[0] = 0; cols.data<int64_t>()[1] = 2;
    cols.data<int64_t>()[2] = 1;
    cols.data<int64_t>()[3] = 0; cols.data<int64_t>()[4] = 2;

    auto vals = zeros({5}, DType::Float32, Device::cpu());
    vals.data<float>()[0] = 1.0f; vals.data<float>()[1] = 2.0f;
    vals.data<float>()[2] = 3.0f;
    vals.data<float>()[3] = 4.0f; vals.data<float>()[4] = 5.0f;

    return SparseTensor::sparse_csr(crow, cols, vals, {3, 3});
}

TEST(SparseParity, SpMM_CSR_Small) {
    auto sparse = make_test_csr();
    auto dense = randn({3, 4}, DType::Float32, Device::cpu());

    auto result = sparse::spmm(sparse, dense);

    auto sparse_dense = sparse.to_dense();
    auto ref = matmul(sparse_dense, dense);

    auto max_err = tenzor::max(tenzor::abs(sub(result, ref))).data<float>()[0];
    EXPECT_LT(max_err, 1e-5f) << "SpMM should match dense matmul";
}

TEST(SparseParity, SpMV) {
    auto sparse = make_test_csr();
    auto vec = randn({3}, DType::Float32, Device::cpu());

    auto result = sparse::spmv(sparse, vec);

    auto sparse_dense = sparse.to_dense();
    auto ref = matmul(sparse_dense, vec.reshape({3, 1})).reshape({3});

    auto max_err = tenzor::max(tenzor::abs(sub(result, ref))).data<float>()[0];
    EXPECT_LT(max_err, 1e-5f) << "SpMV should match dense matvec";
}

TEST(SparseParity, SparseToDenseRoundtrip) {
    // Create dense → sparse → dense, verify equality
    auto original = zeros({4, 4}, DType::Float32, Device::cpu());
    auto data = original.data<float>();
    data[0] = 1.0f; data[5] = 2.0f; data[10] = 3.0f; data[15] = 4.0f; // diagonal

    auto sparse = SparseTensor::from_dense(original);
    auto recovered = sparse.to_dense();

    auto max_err = tenzor::max(tenzor::abs(sub(recovered, original))).data<float>()[0];
    EXPECT_LT(max_err, 1e-7f) << "Dense→Sparse→Dense roundtrip should be exact";
}

TEST(SparseParity, SparseScalarMul) {
    auto crow = zeros({3}, DType::Int64, Device::cpu());
    crow.data<int64_t>()[0] = 0; crow.data<int64_t>()[1] = 1; crow.data<int64_t>()[2] = 2;

    auto cols = zeros({2}, DType::Int64, Device::cpu());
    cols.data<int64_t>()[0] = 0; cols.data<int64_t>()[1] = 1;

    auto vals = zeros({2}, DType::Float32, Device::cpu());
    vals.data<float>()[0] = 3.0f; vals.data<float>()[1] = 7.0f;

    auto sparse = SparseTensor::sparse_csr(crow, cols, vals, {2, 2});
    auto scaled = sparse::mul(sparse, 2.0);
    auto scaled_dense = scaled.to_dense();

    EXPECT_NEAR(scaled_dense.data<float>()[0], 6.0f, 1e-6f);
    EXPECT_NEAR(scaled_dense.data<float>()[3], 14.0f, 1e-6f);
}

TEST(SparseParity, SparseAddSparseDense) {
    auto identity = eye(4, 4, DType::Float32, Device::cpu());
    auto sparse = SparseTensor::from_dense(identity);
    auto dense = ones({4, 4}, DType::Float32, Device::cpu());

    auto result = sparse::add(sparse, dense);

    // Result should be identity + ones = ones with diagonal=2
    auto expected = add(identity, dense);
    auto max_err = tenzor::max(tenzor::abs(sub(result, expected))).data<float>()[0];
    EXPECT_LT(max_err, 1e-6f) << "Sparse + Dense should match dense addition";
}

// ============================================================================
// CUDA SparseAdd Tests — verifies on-device kernel matches CPU reference
// ============================================================================

TEST(SparseParity, SparseAddCUDA_Float32) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    auto sparse = make_test_csr();
    auto dense = randn({3, 3}, DType::Float32, Device::cpu());

    // CPU reference
    auto cpu_result = sparse::add(sparse, dense);

    // CUDA: move both to GPU, compute, bring back
    auto sparse_gpu = sparse.to(Device::cuda());
    auto dense_gpu = dense.to(Device::cuda());
    auto cuda_result = sparse::add(sparse_gpu, dense_gpu).to(Device::cpu());

    auto max_err = tenzor::max(tenzor::abs(sub(cuda_result, cpu_result))).data<float>()[0];
    EXPECT_LT(max_err, 1e-5f) << "CUDA SparseAdd Float32 should match CPU";
}

TEST(SparseParity, SparseAddCUDA_Float64) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    // Larger matrix to exercise more threads
    auto identity = eye(32, 32, DType::Float64, Device::cpu());
    auto sparse = SparseTensor::from_dense(identity);
    auto dense = randn({32, 32}, DType::Float64, Device::cpu());

    auto cpu_result = sparse::add(sparse, dense);

    auto sparse_gpu = sparse.to(Device::cuda());
    auto dense_gpu = dense.to(Device::cuda());
    auto cuda_result = sparse::add(sparse_gpu, dense_gpu).to(Device::cpu());

    auto max_err = tenzor::max(tenzor::abs(sub(cuda_result, cpu_result))).data<double>()[0];
    EXPECT_LT(max_err, 1e-12) << "CUDA SparseAdd Float64 should match CPU";
}

TEST(SparseParity, SparseAddCUDA_EmptySparse) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    // Empty sparse (all zeros) + dense should return dense unchanged
    auto zero_matrix = zeros({4, 4}, DType::Float32, Device::cpu());
    auto sparse = SparseTensor::from_dense(zero_matrix);
    auto dense = randn({4, 4}, DType::Float32, Device::cpu());

    auto sparse_gpu = sparse.to(Device::cuda());
    auto dense_gpu = dense.to(Device::cuda());
    auto cuda_result = sparse::add(sparse_gpu, dense_gpu).to(Device::cpu());

    auto max_err = tenzor::max(tenzor::abs(sub(cuda_result, dense))).data<float>()[0];
    EXPECT_LT(max_err, 1e-6f) << "SparseAdd with empty sparse should return dense unchanged";
}

TEST(SparseParity, SparseAddCUDA_FullRank) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA not available";

    // Fully dense sparse matrix (every element nonzero) + dense
    auto full = randn({8, 8}, DType::Float32, Device::cpu());
    // Make sure there are no exact zeros
    auto full_data = full.data<float>();
    for (int i = 0; i < 64; ++i) {
        if (full_data[i] == 0.0f) full_data[i] = 0.1f;
    }
    auto sparse = SparseTensor::from_dense(full);
    auto dense = randn({8, 8}, DType::Float32, Device::cpu());

    auto cpu_result = sparse::add(sparse, dense);

    auto sparse_gpu = sparse.to(Device::cuda());
    auto dense_gpu = dense.to(Device::cuda());
    auto cuda_result = sparse::add(sparse_gpu, dense_gpu).to(Device::cpu());

    auto max_err = tenzor::max(tenzor::abs(sub(cuda_result, cpu_result))).data<float>()[0];
    EXPECT_LT(max_err, 1e-5f) << "CUDA SparseAdd with full-rank sparse should match CPU";
}

TEST(SparseParity, FormatConversion_COO_CSR) {
    auto original = eye(4, 4, DType::Float32, Device::cpu());
    auto sparse_coo = SparseTensor::from_dense(original, SparseLayout::COO);
    auto sparse_csr = sparse_coo.to_csr();
    auto dense_from_csr = sparse_csr.to_dense();

    auto max_err = tenzor::max(tenzor::abs(sub(dense_from_csr, original))).data<float>()[0];
    EXPECT_LT(max_err, 1e-7f) << "COO→CSR→Dense should preserve values";
}

TEST(SparseParity, EmptySparseTensor) {
    // Sparse tensor with no nonzero elements — create via from_dense of zeros
    auto zero_matrix = zeros({4, 4}, DType::Float32, Device::cpu());
    auto sparse = SparseTensor::from_dense(zero_matrix);
    auto dense = sparse.to_dense();

    auto sum_val = tenzor::sum(dense).data<float>()[0];
    EXPECT_NEAR(sum_val, 0.0f, 1e-7f) << "Empty sparse tensor should be all zeros";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
