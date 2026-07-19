/**
 * @file test_sparse_parity.cpp
 * @brief Sparse operation parity tests across backends
 *
 * Verifies that sparse tensor operations (SpMM, SpMV, sparse-dense conversion)
 * produce identical results across all backends.
 */

#include <gtest/gtest.h>
#include <complex>
#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class SparseParity : public BackendTest {};
// ============================================================================
// Sparse-Dense Interaction Parity
// ============================================================================

TEST_P(SparseParity, IdentityMatmul) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity");

    // Identity matrix @ dense vector — exercises sparse-like paths
    auto identity = eye(8, 8, DType::Float32, Device::cpu());
    auto dense = randn({8, 4}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {identity, dense}, 1e-5f, 1e-6f, "Identity @ Dense (sparse-like)");
}

TEST_P(SparseParity, SparsePatternMatmul) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity");

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

TEST_P(SparseParity, SpMM_CSR_Small) {
    auto sparse = make_test_csr();
    auto dense = randn({3, 4}, DType::Float32, Device::cpu());

    auto result = sparse::spmm(sparse, dense);

    auto sparse_dense = sparse.to_dense();
    auto ref = matmul(sparse_dense, dense);

    auto max_err = tenzor::max(tenzor::abs(sub(result, ref))).data<float>()[0];
    EXPECT_LT(max_err, 1e-5f) << "SpMM should match dense matmul";
}

TEST_P(SparseParity, SpMV) {
    auto sparse = make_test_csr();
    auto vec = randn({3}, DType::Float32, Device::cpu());

    auto result = sparse::spmv(sparse, vec);

    auto sparse_dense = sparse.to_dense();
    auto ref = matmul(sparse_dense, vec.reshape({3, 1})).reshape({3});

    auto max_err = tenzor::max(tenzor::abs(sub(result, ref))).data<float>()[0];
    EXPECT_LT(max_err, 1e-5f) << "SpMV should match dense matvec";
}

TEST_P(SparseParity, SparseToDenseRoundtrip) {
    // Create dense → sparse → dense, verify equality
    auto original = zeros({4, 4}, DType::Float32, Device::cpu());
    auto data = original.data<float>();
    data[0] = 1.0f; data[5] = 2.0f; data[10] = 3.0f; data[15] = 4.0f; // diagonal

    auto sparse = SparseTensor::from_dense(original);
    auto recovered = sparse.to_dense();

    auto max_err = tenzor::max(tenzor::abs(sub(recovered, original))).data<float>()[0];
    EXPECT_LT(max_err, 1e-7f) << "Dense→Sparse→Dense roundtrip should be exact";
}

TEST_P(SparseParity, SparseScalarMul) {
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

TEST_P(SparseParity, SparseAddSparseDense) {
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
// These tests intrinsically target the CUDA SparseAdd kernel (they hardcode
// Device::cuda()), so they live in a non-parameterized TEST_F fixture rather
// than inside the backend-parameterized SparseParity suite. Putting CUDA-only
// tests in a parameterized suite with an inline has_cuda() guard causes the
// test to appear 5× (once per backend parameter) with 4 silent skips — the
// anti-pattern TESTING.md calls out. One TEST_F + SKIP_IF_NO_CUDA is honest.

class SparseAddcudaTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

TEST_F(SparseAddcudaTest, Float32) {
    SKIP_IF_NO_CUDA;

    auto sparse = make_test_csr();
    auto dense = randn({3, 3}, DType::Float32, Device::cpu());

    auto cpu_result = sparse::add(sparse, dense);

    auto sparse_gpu = sparse.to(Device::cuda());
    auto dense_gpu = dense.to(Device::cuda());
    auto cuda_result = sparse::add(sparse_gpu, dense_gpu).to(Device::cpu());

    auto max_err = tenzor::max(tenzor::abs(sub(cuda_result, cpu_result))).data<float>()[0];
    EXPECT_LT(max_err, 1e-5f) << "CUDA SparseAdd Float32 should match CPU";
}

TEST_F(SparseAddcudaTest, Float64) {
    SKIP_IF_NO_CUDA;

    // Larger matrix to exercise more threads.
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

TEST_F(SparseAddcudaTest, EmptySparse) {
    SKIP_IF_NO_CUDA;

    // Empty sparse (all zeros) + dense should return dense unchanged.
    auto zero_matrix = zeros({4, 4}, DType::Float32, Device::cpu());
    auto sparse = SparseTensor::from_dense(zero_matrix);
    auto dense = randn({4, 4}, DType::Float32, Device::cpu());

    auto sparse_gpu = sparse.to(Device::cuda());
    auto dense_gpu = dense.to(Device::cuda());
    auto cuda_result = sparse::add(sparse_gpu, dense_gpu).to(Device::cpu());

    auto max_err = tenzor::max(tenzor::abs(sub(cuda_result, dense))).data<float>()[0];
    EXPECT_LT(max_err, 1e-6f) << "SparseAdd with empty sparse should return dense unchanged";
}

TEST_F(SparseAddcudaTest, FullRank) {
    SKIP_IF_NO_CUDA;

    // Fully dense sparse matrix (every element nonzero) + dense.
    auto full = randn({8, 8}, DType::Float32, Device::cpu());
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

TEST_P(SparseParity, FormatConversion_COO_CSR) {
    auto original = eye(4, 4, DType::Float32, Device::cpu());
    auto sparse_coo = SparseTensor::from_dense(original, SparseLayout::COO);
    auto sparse_csr = sparse_coo.to_csr();
    auto dense_from_csr = sparse_csr.to_dense();

    auto max_err = tenzor::max(tenzor::abs(sub(dense_from_csr, original))).data<float>()[0];
    EXPECT_LT(max_err, 1e-7f) << "COO→CSR→Dense should preserve values";
}

TEST_P(SparseParity, EmptySparseTensor) {
    // Sparse tensor with no nonzero elements — create via from_dense of zeros
    auto zero_matrix = zeros({4, 4}, DType::Float32, Device::cpu());
    auto sparse = SparseTensor::from_dense(zero_matrix);
    auto dense = sparse.to_dense();

    auto sum_val = tenzor::sum(dense).data<float>()[0];
    EXPECT_NEAR(sum_val, 0.0f, 1e-7f) << "Empty sparse tensor should be all zeros";
}

// ============================================================================
// SpGEMM, DenseToSparse, sparse triangular solve (plan Phase 3.5)
// ============================================================================

TEST_P(SparseParity, DenseToSparse_Roundtrip_AllBackends) {
    // Sparse pattern we control: 4x4 with two nonzeros per row.
    auto dense = zeros({4, 4}, DType::Float32, Device::cpu());
    auto* d = dense.data<float>();
    d[0]=1.0f; d[5]=2.0f; d[10]=3.0f; d[15]=4.0f; d[1]=0.5f; d[6]=0.25f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity");

    for (const auto& dev : backends) {
        try {
            auto dev_dense = dense.to(dev);
            auto sp = tenzor::to_sparse(dev_dense);
            auto back = sp.to_dense();
            dev.synchronize();
            SCOPED_TRACE(std::string("to_sparse roundtrip on ") + backend_name(dev));
            EXPECT_TENSORS_CLOSE(dense, back.to(Device::cpu()), 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "to_sparse failed on " << backend_name(dev) << ": "
                      << e.what() << std::endl;
        }
    }
}

// Previously DISABLED_ — sparse::spgemm on OneAPI hung in infinite recursion
// (the oneMKL lambda called back into sparse::spgemm, which re-dispatched to
// the same lambda). Fixed by having the backend lambdas call `oneapi::
// spgemm_kernel` directly, and by routing spgemm through the SYCL-native
// Int64 path instead of the dense-intermediate oneMKL path whose helpers
// assume Int32 CSR indices.
TEST_P(SparseParity, SpGEMM) {
    // A: 4x5 sparse, B: 5x3 sparse; product C = A @ B is 4x3 sparse.
    auto A_dense = zeros({4, 5}, DType::Float32, Device::cpu());
    auto B_dense = zeros({5, 3}, DType::Float32, Device::cpu());
    auto* a = A_dense.data<float>();
    auto* b = B_dense.data<float>();
    a[0]=1; a[6]=2; a[12]=3; a[18]=4;  // diagonal-ish
    b[0]=1; b[4]=2; b[8]=3;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity");

    // CPU is the reference backend — a throw here is a real bug, so let it propagate.
    Tensor ref_dense;
    {
        auto A_sparse = tenzor::to_sparse(A_dense);
        auto B_sparse = tenzor::to_sparse(B_dense);
        auto C_sparse = tenzor::sparse::spgemm(A_sparse, B_sparse);
        ref_dense = C_sparse.to_dense();
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_sp = tenzor::to_sparse(A_dense.to(backends[i]));
            auto B_sp = tenzor::to_sparse(B_dense.to(backends[i]));
            auto C_sp = tenzor::sparse::spgemm(A_sp, B_sp);
            auto C_dense = C_sp.to_dense();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("spgemm on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_dense, C_dense.to(Device::cpu()),
                                 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "spgemm failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(SparseParity, SparseTriangularSolve) {
    // Build lower-triangular sparse L with nonzero diagonal; solve L @ x = b.
    auto L_dense = zeros({4, 4}, DType::Float32, Device::cpu());
    auto* l = L_dense.data<float>();
    l[0]=2.0f;  l[5]=3.0f;  l[10]=4.0f; l[15]=5.0f;   // diag
    l[4]=1.0f;  l[8]=0.5f;  l[9]=1.5f;                // below diag
    auto b_dense = ones({4, 1}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity");

    // CPU is the reference backend — a throw here is a real bug, so let it propagate.
    Tensor ref;
    {
        auto L_sparse = tenzor::to_sparse(L_dense);
        ref = tenzor::sparse::sparse_triangular_solve(L_sparse, b_dense, false);
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto L_sp = tenzor::to_sparse(L_dense.to(backends[i]));
            auto x = tenzor::sparse::sparse_triangular_solve(
                L_sp, b_dense.to(backends[i]), false);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("sparse_triangular_solve on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, x.to(Device::cpu()), 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "sparse_triangular_solve failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

// Note: plan called for SparseTrsv and SparseTrsm separately. The codebase
// exposes a single `sparse_triangular_solve(L, b)` which works for both the
// vector (b is [N, 1]) and matrix-RHS (b is [N, K]) cases. The test above
// covers the vector case; extend to matrix-RHS below.
TEST_P(SparseParity, SparseTriangularSolve_Matrix) {
    auto L_dense = zeros({4, 4}, DType::Float32, Device::cpu());
    auto* l = L_dense.data<float>();
    l[0]=2.0f;  l[5]=3.0f;  l[10]=4.0f; l[15]=5.0f;
    l[4]=1.0f;  l[8]=0.5f;
    auto B_dense = randn({4, 3}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity");

    // CPU is the reference backend — a throw here is a real bug, so let it propagate.
    Tensor ref;
    {
        auto L_sparse = tenzor::to_sparse(L_dense);
        ref = tenzor::sparse::sparse_triangular_solve(L_sparse, B_dense, false);
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto L_sp = tenzor::to_sparse(L_dense.to(backends[i]));
            auto X = tenzor::sparse::sparse_triangular_solve(
                L_sp, B_dense.to(backends[i]), false);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("sparse_triangular_solve_matrix on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, X.to(Device::cpu()), 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "sparse_triangular_solve_matrix failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

// ============================================================================
// Sparse triangular solve — Complex64/Complex128 parity
// ============================================================================
// Same lower-triangular nonzero pattern as the Float32 tests above:
// [ (0,0)         0            0         0    ]
// [ (1,0)      (1,1)           0         0    ]
// [ (2,0)      (2,1)        (2,2)        0    ]
// [   0           0            0       (3,3)  ]
// Built directly via SparseTensor::sparse_csr (rather than through
// to_sparse(dense)) so these tests exercise SparseTrsv/SparseTrsm's own
// complex-dtype dispatch without also depending on DenseToSparse's complex
// support, which is a separate code path and out of scope here.

static auto make_complex64_lower_triangular_csr() -> SparseTensor {
    auto crow = zeros({5}, DType::Int64, Device::cpu());
    auto* cp = crow.data<int64_t>();
    cp[0] = 0; cp[1] = 1; cp[2] = 3; cp[3] = 6; cp[4] = 7;

    auto cols = zeros({7}, DType::Int64, Device::cpu());
    auto* clp = cols.data<int64_t>();
    clp[0] = 0;                    // row 0
    clp[1] = 0; clp[2] = 1;        // row 1
    clp[3] = 0; clp[4] = 1; clp[5] = 2;  // row 2
    clp[6] = 3;                    // row 3

    auto vals = zeros({7}, DType::Complex64, Device::cpu());
    auto* vp = vals.data<std::complex<float>>();
    vp[0] = {2.0f, 0.0f};    // (0,0) diag
    vp[1] = {1.0f, 0.5f};    // (1,0)
    vp[2] = {3.0f, -1.0f};   // (1,1) diag
    vp[3] = {0.5f, 0.25f};   // (2,0)
    vp[4] = {1.5f, -0.5f};   // (2,1)
    vp[5] = {4.0f, 1.0f};    // (2,2) diag
    vp[6] = {5.0f, -2.0f};   // (3,3) diag

    return SparseTensor::sparse_csr(crow, cols, vals, {4, 4});
}

static auto make_complex128_lower_triangular_csr() -> SparseTensor {
    auto crow = zeros({5}, DType::Int64, Device::cpu());
    auto* cp = crow.data<int64_t>();
    cp[0] = 0; cp[1] = 1; cp[2] = 3; cp[3] = 6; cp[4] = 7;

    auto cols = zeros({7}, DType::Int64, Device::cpu());
    auto* clp = cols.data<int64_t>();
    clp[0] = 0;
    clp[1] = 0; clp[2] = 1;
    clp[3] = 0; clp[4] = 1; clp[5] = 2;
    clp[6] = 3;

    auto vals = zeros({7}, DType::Complex128, Device::cpu());
    auto* vp = vals.data<std::complex<double>>();
    vp[0] = {2.0, 0.0};
    vp[1] = {1.0, 0.5};
    vp[2] = {3.0, -1.0};
    vp[3] = {0.5, 0.25};
    vp[4] = {1.5, -0.5};
    vp[5] = {4.0, 1.0};
    vp[6] = {5.0, -2.0};

    return SparseTensor::sparse_csr(crow, cols, vals, {4, 4});
}

TEST_P(SparseParity, SparseTriangularSolve_Complex64) {
    auto L_sparse = make_complex64_lower_triangular_csr();
    auto b_dense = zeros({4}, DType::Complex64, Device::cpu());
    auto* bp = b_dense.data<std::complex<float>>();
    bp[0] = {1.0f, 1.0f};
    bp[1] = {2.0f, 0.0f};
    bp[2] = {0.0f, 3.0f};
    bp[3] = {-1.0f, 2.0f};

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity (complex64)");

    // CPU is the reference backend — a throw here is a real bug, so let it propagate.
    Tensor ref = tenzor::sparse::sparse_triangular_solve(L_sparse, b_dense, false);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto L_sp = L_sparse.to(backends[i]);
            auto x = tenzor::sparse::sparse_triangular_solve(
                L_sp, b_dense.to(backends[i]), false);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("sparse_triangular_solve complex64 on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, x.to(Device::cpu()), 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "sparse_triangular_solve (complex64) failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

TEST_P(SparseParity, SparseTriangularSolve_Complex64_Matrix) {
    auto L_sparse = make_complex64_lower_triangular_csr();
    auto B_dense = zeros({4, 2}, DType::Complex64, Device::cpu());
    auto* bp = B_dense.data<std::complex<float>>();
    bp[0] = {1.0f, 1.0f};   bp[1] = {0.0f, 1.0f};
    bp[2] = {2.0f, 0.0f};   bp[3] = {1.0f, -1.0f};
    bp[4] = {0.0f, 3.0f};   bp[5] = {2.0f, 2.0f};
    bp[6] = {-1.0f, 2.0f};  bp[7] = {-2.0f, 0.0f};

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity (complex64 matrix)");

    // CPU is the reference backend — a throw here is a real bug, so let it propagate.
    Tensor ref = tenzor::sparse::sparse_triangular_solve(L_sparse, B_dense, false);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto L_sp = L_sparse.to(backends[i]);
            auto X = tenzor::sparse::sparse_triangular_solve(
                L_sp, B_dense.to(backends[i]), false);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("sparse_triangular_solve_matrix complex64 on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, X.to(Device::cpu()), 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "sparse_triangular_solve_matrix (complex64) failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

TEST_P(SparseParity, SparseTriangularSolve_Complex128) {
    auto L_sparse = make_complex128_lower_triangular_csr();
    auto b_dense = zeros({4}, DType::Complex128, Device::cpu());
    auto* bp = b_dense.data<std::complex<double>>();
    bp[0] = {1.0, 1.0};
    bp[1] = {2.0, 0.0};
    bp[2] = {0.0, 3.0};
    bp[3] = {-1.0, 2.0};

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity (complex128)");

    // CPU is the reference backend — a throw here is a real bug, so let it propagate.
    Tensor ref = tenzor::sparse::sparse_triangular_solve(L_sparse, b_dense, false);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto L_sp = L_sparse.to(backends[i]);
            auto x = tenzor::sparse::sparse_triangular_solve(
                L_sp, b_dense.to(backends[i]), false);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("sparse_triangular_solve complex128 on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, x.to(Device::cpu()), 1e-9f, 1e-10f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "sparse_triangular_solve (complex128) failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

TEST_P(SparseParity, SparseTriangularSolve_Complex128_Matrix) {
    auto L_sparse = make_complex128_lower_triangular_csr();
    auto B_dense = zeros({4, 2}, DType::Complex128, Device::cpu());
    auto* bp = B_dense.data<std::complex<double>>();
    bp[0] = {1.0, 1.0};   bp[1] = {0.0, 1.0};
    bp[2] = {2.0, 0.0};   bp[3] = {1.0, -1.0};
    bp[4] = {0.0, 3.0};   bp[5] = {2.0, 2.0};
    bp[6] = {-1.0, 2.0};  bp[7] = {-2.0, 0.0};

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sparse parity (complex128 matrix)");

    // CPU is the reference backend — a throw here is a real bug, so let it propagate.
    Tensor ref = tenzor::sparse::sparse_triangular_solve(L_sparse, B_dense, false);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto L_sp = L_sparse.to(backends[i]);
            auto X = tenzor::sparse::sparse_triangular_solve(
                L_sp, B_dense.to(backends[i]), false);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("sparse_triangular_solve_matrix complex128 on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, X.to(Device::cpu()), 1e-9f, 1e-10f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "sparse_triangular_solve_matrix (complex128) failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

INSTANTIATE_BACKEND_TESTS(SparseParity);


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
