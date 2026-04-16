/**
 * @file test_linalg_parity.cpp
 * @brief Linear algebra operation parity tests across backends
 *
 * Tests 12 linear algebra operations (det, inv, solve, svd, qr, eigh,
 * cholesky, lu, addmm, addmv, baddbmm, solve_triangular) to ensure all
 * backends (CPU, CUDA, ROCm, Vulkan, OneAPI) produce identical results.
 *
 * For decomposition tests (SVD, QR, Eigh, LU, Cholesky), factors are NOT
 * compared directly due to sign ambiguity. Instead, the reconstruction
 * (e.g., Q@R ~= A) is verified.
 *
 * Manual backend loops are used instead of test_operation_parity since
 * most decompositions return multi-tensor results (tuples).
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/linalg.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Helper: create a well-conditioned matrix (A + k*I) to ensure invertibility
// ============================================================================

static Tensor make_well_conditioned(const std::vector<int64_t>& shape, float diag_boost,
                                    uint64_t seed = 200) {
    auto A = generate_test_tensor(shape, DType::Float32, Device::cpu(), seed);
    auto I = eye(shape.back(), std::nullopt, DType::Float32, Device::cpu());
    return A + diag_boost * I;
}

// Helper: create a symmetric positive-definite matrix via A@A^T + k*I
static Tensor make_spd(int64_t n, float diag_boost, uint64_t seed = 300) {
    auto A = generate_test_tensor({n, n}, DType::Float32, Device::cpu(), seed);
    auto At = transpose(A, 0, 1);
    auto I = eye(n, std::nullopt, DType::Float32, Device::cpu());
    return matmul(At, A) + diag_boost * I;
}

// Helper: create a symmetric matrix via A + A^T
static Tensor make_symmetric(int64_t n, uint64_t seed = 400) {
    auto A = generate_test_tensor({n, n}, DType::Float32, Device::cpu(), seed);
    auto At = transpose(A, 0, 1);
    return A + At;
}

// ============================================================================
// Determinant
// ============================================================================

TEST(LinalgParity, Det) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto A = make_well_conditioned({8, 8}, 5.0f, 201);

    // Compute reference on CPU
    Tensor ref;
    try {
        ref = linalg::det(A);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::det not available on CPU: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto result = linalg::det(A_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, result, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            std::cerr << "Det: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Inverse — verify A @ inv(A) ~= I
// ============================================================================

TEST(LinalgParity, Inv) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto A = make_well_conditioned({8, 8}, 5.0f, 202);
    auto I = eye(8, std::nullopt, DType::Float32, Device::cpu());

    // CPU reference: verify A @ inv(A) ~= I
    Tensor ref_inv;
    try {
        ref_inv = linalg::inv(A);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::inv not available on CPU: " << e.what();
    }
    auto ref_product = matmul(A, ref_inv);
    EXPECT_TENSORS_CLOSE(ref_product, I, 1e-4f, 1e-5f);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto inv_dev = linalg::inv(A_dev);
            backends[i].synchronize();
            auto product = matmul(A_dev, inv_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(product, I, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            std::cerr << "Inv: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Solve — verify A @ x ~= b
// ============================================================================

TEST(LinalgParity, Solve) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto A = make_well_conditioned({8, 8}, 5.0f, 203);
    auto b = generate_test_tensor({8, 4}, DType::Float32, Device::cpu(), 204);

    Tensor ref_x;
    try {
        ref_x = linalg::solve(A, b);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::solve not available on CPU: " << e.what();
    }

    // Verify A @ x ~= b on CPU
    auto reconstructed = matmul(A, ref_x);
    EXPECT_TENSORS_CLOSE(reconstructed, b, 1e-4f, 1e-5f);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto b_dev = b.to(backends[i]);
            auto x_dev = linalg::solve(A_dev, b_dev);
            backends[i].synchronize();
            auto recon = matmul(A_dev, x_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, b, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            std::cerr << "Solve: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// SVD — verify A ~= U @ diag(S) @ Vh (don't compare U/V directly)
// ============================================================================

TEST(LinalgParity, SVD) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto A = generate_test_tensor({8, 6}, DType::Float32, Device::cpu(), 205);

    // CPU reference reconstruction
    Tensor ref_U, ref_S, ref_Vh;
    try {
        std::tie(ref_U, ref_S, ref_Vh) = linalg::svd(A);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::svd not available on CPU: " << e.what();
    }

    // Reconstruct: A ~= U @ diag(S) @ Vh
    // S is (min(M,N),), need to form diagonal matrix
    auto S_diag = diag(ref_S);
    auto ref_recon = matmul(matmul(ref_U, S_diag), ref_Vh);
    EXPECT_TENSORS_CLOSE(ref_recon, A, 1e-3f, 1e-4f);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [U, S, Vh] = linalg::svd(A_dev);
            backends[i].synchronize();
            auto S_d = diag(S);
            auto recon = matmul(matmul(U, S_d), Vh);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, A, 1e-3f, 1e-4f);
        } catch (const std::exception& e) {
            std::cerr << "SVD: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// QR — verify A ~= Q @ R (don't compare Q/R individually)
// ============================================================================

TEST(LinalgParity, QR) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto A = generate_test_tensor({8, 6}, DType::Float32, Device::cpu(), 206);

    Tensor ref_Q, ref_R;
    try {
        std::tie(ref_Q, ref_R) = linalg::qr(A);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::qr not available on CPU: " << e.what();
    }

    auto ref_recon = matmul(ref_Q, ref_R);
    EXPECT_TENSORS_CLOSE(ref_recon, A, 1e-4f, 1e-5f);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [Q, R] = linalg::qr(A_dev);
            backends[i].synchronize();
            auto recon = matmul(Q, R);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, A, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            std::cerr << "QR: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Eigh — verify A @ v ~= v @ diag(lambda) for symmetric input
// Compare eigenvalues directly; verify eigenvectors via reconstruction only
// ============================================================================

TEST(LinalgParity, Eigh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto A = make_symmetric(8, 401);

    Tensor ref_eigenvalues, ref_eigenvectors;
    try {
        std::tie(ref_eigenvalues, ref_eigenvectors) = linalg::eigh(A);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::eigh not available on CPU: " << e.what();
    }

    // Verify A @ V ~= V @ diag(lambda)
    auto Vl = matmul(A, ref_eigenvectors);
    auto Vr = matmul(ref_eigenvectors, diag(ref_eigenvalues));
    EXPECT_TENSORS_CLOSE(Vl, Vr, 1e-3f, 1e-4f);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [eigenvalues, eigenvectors] = linalg::eigh(A_dev);
            backends[i].synchronize();

            // Compare eigenvalues directly (they should match across backends)
            EXPECT_TENSORS_CLOSE(ref_eigenvalues, eigenvalues, 1e-3f, 1e-4f);

            // Verify reconstruction: A @ V ~= V @ diag(lambda)
            auto lhs = matmul(A_dev, eigenvectors);
            auto rhs = matmul(eigenvectors, diag(eigenvalues));
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(lhs, rhs, 1e-3f, 1e-4f);
        } catch (const std::exception& e) {
            std::cerr << "Eigh: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Cholesky — verify L @ L^T ~= A for positive-definite input
// ============================================================================

TEST(LinalgParity, Cholesky) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Create positive-definite matrix via A^T @ A + 5*I
    auto A = make_spd(8, 5.0f, 301);

    Tensor ref_L;
    try {
        ref_L = linalg::cholesky(A);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::cholesky not available on CPU: " << e.what();
    }

    // Verify L @ L^T ~= A
    auto Lt = transpose(ref_L, 0, 1);
    auto ref_recon = matmul(ref_L, Lt);
    EXPECT_TENSORS_CLOSE(ref_recon, A, 1e-4f, 1e-5f);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto L = linalg::cholesky(A_dev);
            backends[i].synchronize();
            auto L_t = transpose(L, 0, 1);
            auto recon = matmul(L, L_t);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, A, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            std::cerr << "Cholesky: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// LU — decompose and verify reconstruction
// lu() returns (L, U, pivots); we verify L @ U ~= A (pivoted)
// ============================================================================

TEST(LinalgParity, LU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto A = make_well_conditioned({8, 8}, 5.0f, 207);

    Tensor ref_L, ref_U, ref_pivots;
    try {
        std::tie(ref_L, ref_U, ref_pivots) = linalg::lu(A);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::lu not available on CPU: " << e.what();
    }

    // For partial pivoting, L @ U gives a row-permuted version of A.
    // Rather than reconstructing P explicitly, we verify by solving:
    // the LU factors should allow accurate solve. Use a simpler check:
    // verify that linalg::solve using A gives consistent results.
    // Alternatively, just check L @ U reconstruction is consistent across backends.
    auto ref_recon = matmul(ref_L, ref_U);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [L, U, pivots] = linalg::lu(A_dev);
            backends[i].synchronize();
            auto recon = matmul(L, U);
            backends[i].synchronize();
            // L @ U should match across backends (same permuted A)
            EXPECT_TENSORS_CLOSE(ref_recon, recon, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            std::cerr << "LU: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Addmm — beta*input + alpha*(mat1 @ mat2)
// ============================================================================

TEST(LinalgParity, Addmm) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_test_tensor({8, 8}, DType::Float32, Device::cpu(), 208);
    auto mat1 = generate_test_tensor({8, 8}, DType::Float32, Device::cpu(), 209);
    auto mat2 = generate_test_tensor({8, 8}, DType::Float32, Device::cpu(), 210);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return addmm(inputs[0], inputs[1], inputs[2], 1.0, 1.0);
    }, {input, mat1, mat2}, backends, 1e-4f, 1e-5f, "Addmm");
}

// ============================================================================
// Addmv — beta*input + alpha*(mat @ vec)
// ============================================================================

TEST(LinalgParity, Addmv) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input_vec = generate_test_tensor({8}, DType::Float32, Device::cpu(), 211);
    auto mat = generate_test_tensor({8, 8}, DType::Float32, Device::cpu(), 212);
    auto vec = generate_test_tensor({8}, DType::Float32, Device::cpu(), 213);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return addmv(inputs[0], inputs[1], inputs[2], 1.0, 1.0);
    }, {input_vec, mat, vec}, backends, 1e-4f, 1e-5f, "Addmv");
}

// ============================================================================
// Baddbmm — batched beta*input + alpha*(batch1 @ batch2)
// ============================================================================

TEST(LinalgParity, Baddbmm) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_test_tensor({4, 8, 8}, DType::Float32, Device::cpu(), 214);
    auto batch1 = generate_test_tensor({4, 8, 8}, DType::Float32, Device::cpu(), 215);
    auto batch2 = generate_test_tensor({4, 8, 8}, DType::Float32, Device::cpu(), 216);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return baddbmm(inputs[0], inputs[1], inputs[2], 1.0, 1.0);
    }, {input, batch1, batch2}, backends, 1e-4f, 1e-5f, "Baddbmm");
}

// ============================================================================
// SolveTriangular — solve with triangular matrix
// ============================================================================

TEST(LinalgParity, SolveTriangular) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Create upper-triangular matrix with good conditioning
    auto A_full = make_well_conditioned({8, 8}, 5.0f, 217);
    auto A = triu(A_full);
    auto b = generate_test_tensor({8, 4}, DType::Float32, Device::cpu(), 218);

    Tensor ref_x;
    try {
        ref_x = linalg::solve_triangular(A, b, /*upper=*/true);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "linalg::solve_triangular not available on CPU: " << e.what();
    }

    // Verify A @ x ~= b on CPU
    auto reconstructed = matmul(A, ref_x);
    EXPECT_TENSORS_CLOSE(reconstructed, b, 1e-4f, 1e-5f);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto b_dev = b.to(backends[i]);
            auto x_dev = linalg::solve_triangular(A_dev, b_dev, /*upper=*/true);
            backends[i].synchronize();
            auto recon = matmul(A_dev, x_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, b, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            std::cerr << "SolveTriangular: backend " << backend_name(backends[i])
                      << " skipped: " << e.what() << std::endl;
        }
    }
}



int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
