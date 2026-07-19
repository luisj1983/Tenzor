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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class LinalgParity : public BackendTest {};
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

TEST_P(LinalgParity, Det) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    auto A = make_well_conditioned({8, 8}, 5.0f, 201);

    // Compute reference on CPU. The CPU path is ground truth: if linalg::det
    // throws here it is a real bug (missing kernel / wrong result), so let the
    // exception propagate as a test failure rather than masking it as a skip.
    Tensor ref = linalg::det(A);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto result = linalg::det(A_dev);
            backends[i].synchronize();
            auto result_cpu = result.to(Device::cpu());
            // Different backends disagree on whether det of a 2D input is
            // scalar (shape []) or shape-[1]. Flatten both to shape [1] for
            // comparison. det of an 8x8 matrix is the product of 8 pivots
            // from an LU factorization; pivot-order / multiply-add noise
            // compounds to about 1e-4 * |det|.
            auto ref_flat = ref.reshape({1});
            auto res_flat = result_cpu.reshape({1});
            EXPECT_TRUE(tensors_close(ref_flat, res_flat, 1e-3f, 1e-1f))
                << "Det on " << backend_name(backends[i]);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Det failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// Inverse — verify A @ inv(A) ~= I
// ============================================================================

TEST_P(LinalgParity, Inv) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    auto A = make_well_conditioned({8, 8}, 5.0f, 202);
    auto I = eye(8, std::nullopt, DType::Float32, Device::cpu());

    // CPU reference: verify A @ inv(A) ~= I. CPU is ground truth — a throw here
    // is a real bug, so let it propagate as a failure.
    Tensor ref_inv = linalg::inv(A);
    auto ref_product = matmul(A, ref_inv);
    EXPECT_TENSORS_CLOSE(ref_product, I, 1e-4f, 1e-5f);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto inv_dev = linalg::inv(A_dev);
            backends[i].synchronize();
            auto product = matmul(A_dev, inv_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(product, I, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Inv failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// Solve — verify A @ x ~= b
// ============================================================================

TEST_P(LinalgParity, Solve) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    auto A = make_well_conditioned({8, 8}, 5.0f, 203);
    auto b = generate_test_tensor({8, 4}, DType::Float32, Device::cpu(), 204);

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_x = linalg::solve(A, b);

    // Verify A @ x ~= b on CPU
    auto reconstructed = matmul(A, ref_x);
    EXPECT_TENSORS_CLOSE(reconstructed, b, 1e-4f, 1e-5f);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto b_dev = b.to(backends[i]);
            auto x_dev = linalg::solve(A_dev, b_dev);
            backends[i].synchronize();
            auto recon = matmul(A_dev, x_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, b, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Solve failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// SVD — verify A ~= U @ diag(S) @ Vh (don't compare U/V directly)
// ============================================================================

TEST_P(LinalgParity, SVD) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    auto A = generate_test_tensor({8, 6}, DType::Float32, Device::cpu(), 205);

    // CPU reference reconstruction. Use reduced SVD (full_matrices=false) so
    // U is [M, K] and Vh is [K, N] with K = min(M, N), letting us reconstruct
    // with a single [K, K] diag(S) — full SVD returns U=[M, M] which won't
    // conform to S_diag's [K, K].
    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_U, ref_S, ref_Vh;
    std::tie(ref_U, ref_S, ref_Vh) = linalg::svd(A, /*full_matrices=*/false);

    // Reconstruct: A ~= U @ diag(S) @ Vh
    auto S_diag = diag(ref_S);
    auto ref_recon = matmul(matmul(ref_U, S_diag), ref_Vh);
    EXPECT_TENSORS_CLOSE(ref_recon, A, 1e-3f, 1e-4f);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [U, S, Vh] = linalg::svd(A_dev, /*full_matrices=*/false);
            backends[i].synchronize();
            auto S_d = diag(S);
            auto recon = matmul(matmul(U, S_d), Vh);
            backends[i].synchronize();
            EXPECT_TRUE(tensors_close(recon.to(Device::cpu()), A, 1e-3f, 1e-4f))
                << "SVD reconstruction on " << backend_name(backends[i]);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "SVD failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// QR — verify A ~= Q @ R (don't compare Q/R individually)
// ============================================================================

TEST_P(LinalgParity, QR) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    auto A = generate_test_tensor({8, 6}, DType::Float32, Device::cpu(), 206);

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_Q, ref_R;
    std::tie(ref_Q, ref_R) = linalg::qr(A);

    auto ref_recon = matmul(ref_Q, ref_R);
    EXPECT_TENSORS_CLOSE(ref_recon, A, 1e-4f, 1e-5f);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [Q, R] = linalg::qr(A_dev);
            backends[i].synchronize();
            auto recon = matmul(Q, R);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, A, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "QR failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// Eigh — verify A @ v ~= v @ diag(lambda) for symmetric input
// Compare eigenvalues directly; verify eigenvectors via reconstruction only
// ============================================================================

TEST_P(LinalgParity, Eigh) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    auto A = make_symmetric(8, 401);

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_eigenvalues, ref_eigenvectors;
    std::tie(ref_eigenvalues, ref_eigenvectors) = linalg::eigh(A);

    // Verify A @ V ~= V @ diag(lambda)
    auto Vl = matmul(A, ref_eigenvectors);
    auto Vr = matmul(ref_eigenvectors, diag(ref_eigenvalues));
    EXPECT_TENSORS_CLOSE(Vl, Vr, 1e-3f, 1e-4f);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [eigenvalues, eigenvectors] = linalg::eigh(A_dev);
            backends[i].synchronize();

            // Eigenvalue ordering is implementation-defined (LAPACKE ascending,
            // some GPU solvers may differ). Sort both before comparing.
            auto [ref_sorted_v, _r] = sort(Variable(ref_eigenvalues, false), 0);
            auto [dev_sorted_v, _d] = sort(Variable(eigenvalues.to(Device::cpu()), false), 0);
            EXPECT_TRUE(tensors_close(ref_sorted_v.tensor(), dev_sorted_v.tensor(), 1e-3f, 1e-4f))
                << "Eigh eigenvalues on " << backend_name(backends[i]);

            // Verify reconstruction: A @ V ~= V @ diag(lambda)
            auto lhs = matmul(A_dev, eigenvectors);
            auto rhs = matmul(eigenvectors, diag(eigenvalues));
            backends[i].synchronize();
            EXPECT_TRUE(tensors_close(lhs.to(Device::cpu()), rhs.to(Device::cpu()), 1e-3f, 1e-4f))
                << "Eigh reconstruction on " << backend_name(backends[i]);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Eigh failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// Eigh, large batched (n > 128) — exercises the Vulkan tiled tridiagonal-QR
// path whose Givens-rotation buffer previously overflowed (W1 regression test).
// ============================================================================

TEST_P(LinalgParity, Eigh_LargeBatched) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    const int64_t n = 140;   // > 128 to hit the tiled/medium path; kept modest for speed
    const int64_t batch = 2; // > 1 to exercise per-batch buffer base offset

    // Batched symmetric matrix (batch, n, n).
    auto X = generate_test_tensor({batch, n, n}, DType::Float32, Device::cpu(), 911);
    auto A = X + transpose(X, 1, 2);

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_vals, ref_vecs;
    std::tie(ref_vals, ref_vecs) = linalg::eigh(A);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [vals, vecs] = linalg::eigh(A_dev);
            backends[i].synchronize();

            // Eigenvalues: sort along last dim before comparing (order is impl-defined).
            auto [ref_sorted, _r] = sort(Variable(ref_vals, false), -1);
            auto [dev_sorted, _d] = sort(Variable(vals.to(Device::cpu()), false), -1);
            EXPECT_TRUE(tensors_close(ref_sorted.tensor(), dev_sorted.tensor(), 1e-2f, 1e-3f))
                << "Eigh_LargeBatched eigenvalues on " << backend_name(backends[i]);

            // Reconstruction A @ V ~= V @ diag(lambda) per batch — the eigenvectors are
            // the part that the buffer overflow corrupted. Reconstruct via column
            // scaling (rhs[...,i,j] = V[...,i,j] * lambda[...,j]) rather than diag_embed.
            auto lhs = matmul(A_dev, vecs);
            auto rhs = mul(vecs, unsqueeze(vals, 1));  // vals (batch,n) -> (batch,1,n)
            backends[i].synchronize();
            EXPECT_TRUE(tensors_close(lhs.to(Device::cpu()), rhs.to(Device::cpu()), 1e-2f, 1e-3f))
                << "Eigh_LargeBatched reconstruction on " << backend_name(backends[i]);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Eigh_LargeBatched failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// Eig (non-symmetric) — verify the eigenpair relation A v = lambda v per
// eigenvector on each GPU backend (W2: CUDA eigenvectors were Schur vectors).
// Complex conjugate pairs: V[:,k] + i V[:,k+1] with eigenvalue WR[k] + i WI[k].
// ============================================================================

TEST_P(LinalgParity, Eig_NonSymmetric) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    const int64_t n = 6;
    // Diagonal + skew-symmetric => guaranteed complex conjugate eigenvalue pairs,
    // which exercises the complex-eigenvector unpack/back-substitution path.
    Tensor A({n, n}, DType::Float32, Device::cpu());
    {
        float* d = A.data<float>();
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                d[i * n + j] = (i == j) ? static_cast<float>(i + 1)
                             : (i < j ? 1.0f + 0.3f * static_cast<float>(j)
                                      : -(1.0f + 0.3f * static_cast<float>(i)));
    }

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor WRc, WIc, Vc;
    std::tie(WRc, WIc, Vc) = linalg::eig(A);

    auto* Ap = A.data<float>();
    auto eigvals_sorted = [n](const Tensor& wr, const Tensor& wi) {
        std::vector<std::pair<float, float>> ev;
        const float* r = wr.data<float>();
        const float* im = wi.data<float>();
        for (int64_t i = 0; i < n; ++i) ev.emplace_back(r[i], im[i]);
        std::sort(ev.begin(), ev.end());
        return ev;
    };
    auto ref_ev = eigvals_sorted(WRc, WIc);

    for (size_t b = 1; b < backends.size(); ++b) {
        // Non-symmetric eig: CPU (LAPACK geev) is ground truth. CUDA, ROCm,
        // OneAPI, and Vulkan all run the SAME native EISPACK-hqr2 eigensolver
        // (orthogonal Hessenberg + Francis double-shift QR with exceptional
        // shifts + strevc-style eigenvector back-substitution, including
        // analytic 2x2 real-Schur-block extraction for complex-conjugate
        // pairs) — none of cuSOLVER/rocSOLVER/oneMKL/Vulkan expose a GPU
        // geev, and CUDA deliberately abandoned cusolverDnXgeev (see
        // linalg.cu) because its real->complex contract does not honour
        // LAPACK's adjacent-conjugate-pair ordering. All four GPU backends
        // compute correct eigenvalues AND right eigenvectors, verified
        // against the CPU reference below.
        try {
            auto Ad = A.to(backends[b]);
            auto [WR, WI, V] = linalg::eig(Ad);
            backends[b].synchronize();
            Tensor wr = WR.to(Device::cpu()).contiguous();
            Tensor wi = WI.to(Device::cpu()).contiguous();
            Tensor v  = V.to(Device::cpu()).contiguous();
            const float* wrp = wr.data<float>();
            const float* wip = wi.data<float>();
            const float* vp  = v.data<float>();  // row-major (n, n), column k = eigenvector k

            // Eigenvalues (as a multiset, sorted) must match CPU.
            auto dev_ev = eigvals_sorted(wr, wi);
            // CUDA, ROCm, and OneAPI's native EISPACK-hqr2 eigensolvers (see
            // linalg.cu / linalg.hip.cpp / linalg.cpp) have been verified —
            // both by hand-tracing the analytic 2x2 real-Schur-block formula
            // (trace/det/discriminant -> complex-conjugate pair, matching
            // LAPACK dlanv2) and by numerically replaying this exact matrix
            // in Float32 and Float64 — to correctly extract complex-
            // conjugate eigenvalue pairs. Vulkan's port of the same
            // algorithm has likewise been verified. A gross eigenvalue
            // mismatch on any backend is therefore a real regression, not a
            // known limitation, and must fail loudly rather than silently
            // skip.
            bool eig_matches = true;
            for (int64_t i = 0; i < n; ++i) {
                if (std::abs(dev_ev[i].first - ref_ev[i].first) > 1e-2f ||
                    std::abs(std::abs(dev_ev[i].second) - std::abs(ref_ev[i].second)) > 1e-2f) {
                    eig_matches = false;
                    break;
                }
            }
            if (!eig_matches) {
                ADD_FAILURE() << "Eig_NonSymmetric eigenvalues mismatch on "
                              << backend_name(backends[b]);
                continue;
            }
            for (int64_t i = 0; i < n; ++i) {
                EXPECT_NEAR(dev_ev[i].first, ref_ev[i].first, 1e-3f)
                    << "Eig eigenvalue[" << i << "].re on " << backend_name(backends[b]);
                EXPECT_NEAR(std::abs(dev_ev[i].second), std::abs(ref_ev[i].second), 1e-3f)
                    << "Eig eigenvalue[" << i << "].im on " << backend_name(backends[b]);
            }

            // Eigenpair residual: A v = lambda v (handles complex conjugate pairs).
            for (int64_t k = 0; k < n; ++k) {
                if (wip[k] < 0.0f) continue;  // second column of a pair
                float maxres = 0.0f;
                if (wip[k] == 0.0f) {
                    for (int64_t row = 0; row < n; ++row) {
                        float av = 0.0f;
                        for (int64_t j = 0; j < n; ++j) av += Ap[row * n + j] * vp[j * n + k];
                        maxres = std::max(maxres, std::abs(av - wrp[k] * vp[row * n + k]));
                    }
                } else {
                    float lr = wrp[k], li = wip[k];
                    for (int64_t row = 0; row < n; ++row) {
                        float avr = 0.0f, avi = 0.0f;
                        for (int64_t j = 0; j < n; ++j) {
                            avr += Ap[row * n + j] * vp[j * n + k];
                            avi += Ap[row * n + j] * vp[j * n + (k + 1)];
                        }
                        float vr = vp[row * n + k], vi = vp[row * n + (k + 1)];
                        // A(vr+i vi) = (lr+i li)(vr+i vi)
                        maxres = std::max(maxres, std::abs(avr - (lr * vr - li * vi)));
                        maxres = std::max(maxres, std::abs(avi - (lr * vi + li * vr)));
                    }
                }
                EXPECT_LT(maxres, 1e-3f)
                    << "Eig residual A v != lambda v for eigenvalue " << k
                    << " on " << backend_name(backends[b]);
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Eig_NonSymmetric failed on " << backend_name(backends[b])
                      << ": " << e.what();
        }
    }
}

// Larger, batched non-symmetric eig (Float64): exercises the deflation search,
// exceptional shifts, and complex-pair back-substitution at scale. The check is
// self-validating — per-eigenpair residual ||A v - lambda v|| / ||v|| against
// the SAME input matrix — so it is robust to eigenvalue ordering and eigenvector
// sign/scale, and needs no CPU eigenvalue comparison.
TEST_P(LinalgParity, Eig_NonSymmetric_LargeBatched) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    const int64_t B = 3, n = 24;
    Tensor A({B, n, n}, DType::Float64, Device::cpu());
    {
        // Build each batch as an orthogonal similarity Q*A0*Q^T of a
        // block-diagonal A0 (2x2 rotation-scaling blocks => distinct complex
        // conjugate pairs, plus a few distinct real eigenvalues). Orthogonal Q
        // preserves eigenvector conditioning, so the true eigenpair residual is
        // ~machine precision and every correct solver must reproduce it.
        double* d = A.data<double>();
        for (int64_t b = 0; b < B; ++b) {
            std::vector<double> M(n * n, 0.0);
            int blk = 0, rl = 0;
            for (int64_t i = 0; i < n; ++i) {
                if (i + 1 < n && (i % 3 != 2)) {       // 2x2 complex block
                    // Well-separated, distinct eigenvalues: (2+blk+0.5b) ± (0.5+0.1blk)i.
                    double a = 2.0 + 1.0 * blk + 0.5 * b;
                    double bb = 0.5 + 0.1 * blk;
                    M[i * n + i] = a;          M[i * n + (i + 1)] = bb;
                    M[(i + 1) * n + i] = -bb;  M[(i + 1) * n + (i + 1)] = a;
                    ++blk;
                    ++i;
                } else {                                // distinct real eigenvalue, far away
                    M[i * n + i] = -5.0 - 1.5 * rl;
                    ++rl;
                }
            }
            // Orthogonal similarity via a sweep of Givens rotations.
            for (int64_t s = 0; s < n; ++s) {
                int64_t p = s % n, qd = (s * 7 + 3) % n;
                if (p == qd) continue;
                double th = 0.3 + 0.07 * s + 0.05 * b;
                double c = std::cos(th), sn = std::sin(th);
                for (int64_t j = 0; j < n; ++j) {       // rows p,qd
                    double mp = M[p * n + j], mq = M[qd * n + j];
                    M[p * n + j] = c * mp - sn * mq;
                    M[qd * n + j] = sn * mp + c * mq;
                }
                for (int64_t i = 0; i < n; ++i) {       // cols p,qd (transpose)
                    double mp = M[i * n + p], mq = M[i * n + qd];
                    M[i * n + p] = c * mp - sn * mq;
                    M[i * n + qd] = sn * mp + c * mq;
                }
            }
            for (int64_t i = 0; i < n * n; ++i) d[b * n * n + i] = M[i];
        }
    }

    auto residual_ok = [&](const Tensor& WR, const Tensor& WI, const Tensor& V,
                           const std::string& name) {
        Tensor wr = WR.to(Device::cpu()).contiguous();
        Tensor wi = WI.to(Device::cpu()).contiguous();
        Tensor v  = V.to(Device::cpu()).contiguous();
        const double* Ap = A.data<double>();
        const double* wrp = wr.data<double>();
        const double* wip = wi.data<double>();
        const double* vp  = v.data<double>();
        for (int64_t b = 0; b < B; ++b) {
            const double* Ab = Ap + b * n * n;
            const double* wrb = wrp + b * n;
            const double* wib = wip + b * n;
            const double* vb  = vp + b * n * n;
            for (int64_t k = 0; k < n; ++k) {
                if (wib[k] < 0.0) continue;  // imag column of a pair
                double vnorm = 0.0, res = 0.0;
                double lr = wrb[k], li = wib[k];
                for (int64_t row = 0; row < n; ++row) {
                    double vr = vb[row * n + k];
                    double vi = (li != 0.0) ? vb[row * n + (k + 1)] : 0.0;
                    vnorm += vr * vr + vi * vi;
                    double avr = 0.0, avi = 0.0;
                    for (int64_t j = 0; j < n; ++j) {
                        avr += Ab[row * n + j] * vb[j * n + k];
                        if (li != 0.0) avi += Ab[row * n + j] * vb[j * n + (k + 1)];
                    }
                    double dr = avr - (lr * vr - li * vi);
                    double di = avi - (lr * vi + li * vr);
                    res += dr * dr + di * di;
                }
                ASSERT_GT(vnorm, 0.0) << name << " degenerate eigenvector k=" << k << " b=" << b;
                EXPECT_LT(std::sqrt(res / vnorm), 1e-6)
                    << name << " eig residual k=" << k << " b=" << b;
            }
        }
    };

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor WRc, WIc, Vc;
    std::tie(WRc, WIc, Vc) = linalg::eig(A);
    residual_ok(WRc, WIc, Vc, "cpu");

    for (size_t b = 1; b < backends.size(); ++b) {
        try {
            auto Ad = A.to(backends[b]);
            auto [WR, WI, V] = linalg::eig(Ad);
            backends[b].synchronize();
            residual_ok(WR, WI, V, backend_name(backends[b]));
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Eig_NonSymmetric_LargeBatched: backend "
                          << backend_name(backends[b]) << " threw: " << e.what();
        }
    }
}

// ============================================================================
// Cholesky — verify L @ L^T ~= A for positive-definite input
// ============================================================================

TEST_P(LinalgParity, Cholesky) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    // Create positive-definite matrix via A^T @ A + 5*I
    auto A = make_spd(8, 5.0f, 301);

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_L = linalg::cholesky(A);

    // Verify L @ L^T ~= A
    auto Lt = transpose(ref_L, 0, 1);
    auto ref_recon = matmul(ref_L, Lt);
    EXPECT_TENSORS_CLOSE(ref_recon, A, 1e-4f, 1e-5f);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto L = linalg::cholesky(A_dev);
            backends[i].synchronize();
            auto L_t = transpose(L, 0, 1);
            auto recon = matmul(L, L_t);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(recon, A, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Cholesky failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// LU — decompose and verify reconstruction
// lu() returns (L, U, pivots); we verify L @ U ~= A (pivoted)
// ============================================================================

TEST_P(LinalgParity, LU) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    auto A = make_well_conditioned({8, 8}, 5.0f, 207);

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_L, ref_U, ref_pivots;
    std::tie(ref_L, ref_U, ref_pivots) = linalg::lu(A);

    // For partial pivoting, L @ U gives a row-permuted version of A.
    // Rather than reconstructing P explicitly, we verify by solving:
    // the LU factors should allow accurate solve. Use a simpler check:
    // verify that linalg::solve using A gives consistent results.
    // Alternatively, just check L @ U reconstruction is consistent across backends.
    auto ref_recon = matmul(ref_L, ref_U);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto [L, U, pivots] = linalg::lu(A_dev);
            backends[i].synchronize();
            auto recon = matmul(L, U);
            backends[i].synchronize();
            // L @ U should match across backends (same permuted A)
            EXPECT_TENSORS_CLOSE(ref_recon, recon, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LU failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// ============================================================================
// Addmm — beta*input + alpha*(mat1 @ mat2)
// ============================================================================

TEST_P(LinalgParity, Addmm) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

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

TEST_P(LinalgParity, Addmv) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

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

TEST_P(LinalgParity, Baddbmm) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

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

TEST_P(LinalgParity, SolveTriangular) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg parity");

    // Create upper-triangular matrix with good conditioning
    auto A_full = make_well_conditioned({8, 8}, 5.0f, 217);
    auto A = triu(A_full).contiguous();
    auto b = generate_test_tensor({8, 4}, DType::Float32, Device::cpu(), 218);

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_x = linalg::solve_triangular(A, b, /*upper=*/true);

    // Verify A @ x ~= b on CPU
    auto reconstructed = matmul(A, ref_x);
    EXPECT_TENSORS_CLOSE(reconstructed, b, 1e-4f, 1e-5f);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto A_dev = A.to(backends[i]);
            auto b_dev = b.to(backends[i]);
            auto x_dev = linalg::solve_triangular(A_dev, b_dev, /*upper=*/true);
            backends[i].synchronize();
            auto recon = matmul(A_dev, x_dev);
            backends[i].synchronize();
            EXPECT_TRUE(tensors_close(recon.to(Device::cpu()), b, 1e-4f, 1e-5f))
                << "SolveTriangular on " << backend_name(backends[i]);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "SolveTriangular failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}

// Pinv, LstSq, MatrixExp moved to test_linalg_extended_parity.cpp (plan 3.6).

// Release audit (#15): the OneAPI non-oneMKL SYCL fallback previously capped
// matrix size at N=90 (f32) / N=64 (f64). The arbitrary-N global-memory kernels
// must handle N above that cap. Validate inv / solve / qr / cholesky by
// reconstruction at N=128 on every backend. On a non-oneMKL OneAPI build this
// exercises the new global-memory fallback end-to-end; on oneMKL builds it
// confirms the oneMKL path at the same size.
TEST_P(LinalgParity, LargeN_ArbitrarySize) {
    if (device.type == Device::Type::CPU) {
        GTEST_SKIP() << "CPU is the reference backend";
    }
    const int64_t N = 128;  // > 90 (f32 cap) and > 64 (f64 cap)

    auto A    = make_well_conditioned({N, N}, static_cast<float>(N), 555);
    auto I    = eye(N, std::nullopt, DType::Float32, Device::cpu());
    auto spd  = make_spd(N, static_cast<float>(N), 556);
    auto bvec = generate_test_tensor({N, 1}, DType::Float32, Device::cpu(), 557);
    const std::string name = backend_name(device);

    // inv: A @ inv(A) ~= I
    {
        auto Adev = A.to(device);
        auto invd = linalg::inv(Adev);
        auto prod = matmul(Adev, invd).to(Device::cpu());
        EXPECT_TRUE(tensors_close(prod, I, 1e-2f, 1e-2f)) << "inv N=" << N << " on " << name;
    }
    // solve: A @ x ~= b
    {
        auto Adev = A.to(device);
        auto x = linalg::solve(Adev, bvec.to(device));
        auto recon = matmul(Adev, x).to(Device::cpu());
        EXPECT_TRUE(tensors_close(recon, bvec, 1e-2f, 1e-2f)) << "solve N=" << N << " on " << name;
    }
    // qr: Q @ R ~= A
    {
        auto Adev = A.to(device);
        auto [Q, R] = linalg::qr(Adev);
        auto qr_recon = matmul(Q, R).to(Device::cpu());
        EXPECT_TRUE(tensors_close(qr_recon, A, 1e-2f, 1e-2f)) << "qr N=" << N << " on " << name;
    }
    // cholesky (SPD): L @ L^T ~= A
    {
        auto Sdev = spd.to(device);
        auto L = linalg::cholesky(Sdev);
        auto LLt = matmul(L, transpose(L, 0, 1)).to(Device::cpu());
        EXPECT_TRUE(tensors_close(LLt, spd, 1e-1f, 1e-2f)) << "cholesky N=" << N << " on " << name;
    }
}

INSTANTIATE_BACKEND_TESTS(LinalgParity);


int main(int argc, char** argv) {
    // Force IEEE 754 FP32 on CUDA matmul — cuBLAS defaults to TF32 on
    // Ampere+ which silently drops ~13 mantissa bits and blows past the
    // 1e-4 rtol these tests enforce. Matches test_cross_backend_pairs and
    // test_operation_parity.
    setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/1);
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
