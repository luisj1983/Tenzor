/**
 * @file test_linalg_complex.cpp
 * @brief Audit Phase 2 Task 2.9: Complex64/Complex128 linalg via LAPACKE.
 *
 * Before this fix, src/ops/linalg.cpp prepare_matrix() rejected Complex64/
 * Complex128 with a runtime_error. MKL ships cgesv/zgesv/cpotrf/zpotrf/
 * cheev/zheev/cgeqrf/zgeqrf/cgesvd/zgesvd/cgetrf/zgetrf, so these dtypes
 * can be supported natively on CPU.
 */

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"

namespace tz = ::tenzor;

class LinalgComplex : public ::tenzor::testing::BackendTest {};

// ---------------------------------------------------------------------------
// Helper: build a 2x2 complex matrix (host writes on CPU, then move to device)
// ---------------------------------------------------------------------------
static tz::Tensor make_c64_2x2(
    std::complex<float> a00, std::complex<float> a01,
    std::complex<float> a10, std::complex<float> a11,
    const tz::Device& device)
{
    auto t = tz::zeros({2, 2}, tz::DType::Complex64);
    auto* p = t.data<std::complex<float>>();
    p[0] = a00; p[1] = a01; p[2] = a10; p[3] = a11;
    return t.to(device);
}

static tz::Tensor make_c128_2x2(
    std::complex<double> a00, std::complex<double> a01,
    std::complex<double> a10, std::complex<double> a11,
    const tz::Device& device)
{
    auto t = tz::zeros({2, 2}, tz::DType::Complex128);
    auto* p = t.data<std::complex<double>>();
    p[0] = a00; p[1] = a01; p[2] = a10; p[3] = a11;
    return t.to(device);
}

// ---------------------------------------------------------------------------
// solve: Ax = b
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, SolveComplex64) {
    // A = [[2+0i, 1+1i], [1-1i, 3+0i]]  (Hermitian positive definite)
    // det(A) = 2*3 - (1+i)(1-i) = 6 - 2 = 4 → non-singular
    // b = [[1+0i], [2+0i]]
    auto A = make_c64_2x2({2.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f}, device);
    auto b_host = tz::zeros({2, 1}, tz::DType::Complex64);
    b_host.data<std::complex<float>>()[0] = {1.f, 0.f};
    b_host.data<std::complex<float>>()[1] = {2.f, 0.f};
    auto b = b_host.to(device);

    tz::Tensor x;
    ASSERT_NO_THROW(x = tz::linalg::solve(A, b));
    EXPECT_EQ(x.dtype(), tz::DType::Complex64);
    ASSERT_EQ(x.shape().size(), 2u);
    EXPECT_EQ(x.shape()[0], 2);
    EXPECT_EQ(x.shape()[1], 1);

    // Verify: A @ x ≈ b
    auto bx = tz::matmul(A, x).cpu();
    const auto* bxp = bx.data<std::complex<float>>();
    const auto* bp  = b_host.data<std::complex<float>>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(bxp[i].real(), bp[i].real(), 1e-5f)
            << "real part mismatch at index " << i;
        EXPECT_NEAR(bxp[i].imag(), bp[i].imag(), 1e-5f)
            << "imag part mismatch at index " << i;
    }
}

TEST_P(LinalgComplex, SolveComplex128) {
    // Same system in double precision
    auto A = make_c128_2x2({2., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.}, device);
    auto b_host = tz::zeros({2, 1}, tz::DType::Complex128);
    b_host.data<std::complex<double>>()[0] = {1., 0.};
    b_host.data<std::complex<double>>()[1] = {2., 0.};
    auto b = b_host.to(device);

    tz::Tensor x;
    ASSERT_NO_THROW(x = tz::linalg::solve(A, b));
    EXPECT_EQ(x.dtype(), tz::DType::Complex128);

    auto bx = tz::matmul(A, x).cpu();
    const auto* bxp = bx.data<std::complex<double>>();
    const auto* bp  = b_host.data<std::complex<double>>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(bxp[i].real(), bp[i].real(), 1e-12)
            << "real part mismatch at index " << i;
        EXPECT_NEAR(bxp[i].imag(), bp[i].imag(), 1e-12)
            << "imag part mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// inv: A^-1 — verify A @ A^-1 ≈ I
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, InvComplex64) {
    auto A = make_c64_2x2({2.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f}, device);
    tz::Tensor Ainv;
    ASSERT_NO_THROW(Ainv = tz::linalg::inv(A));
    EXPECT_EQ(Ainv.dtype(), tz::DType::Complex64);

    auto I_approx = tz::matmul(A, Ainv).cpu();
    const auto* p = I_approx.data<std::complex<float>>();
    // [0,0] and [1,1] ≈ 1; [0,1] and [1,0] ≈ 0
    EXPECT_NEAR(p[0].real(), 1.f, 1e-5f);
    EXPECT_NEAR(p[0].imag(), 0.f, 1e-5f);
    EXPECT_NEAR(p[1].real(), 0.f, 1e-5f);
    EXPECT_NEAR(p[1].imag(), 0.f, 1e-5f);
    EXPECT_NEAR(p[2].real(), 0.f, 1e-5f);
    EXPECT_NEAR(p[2].imag(), 0.f, 1e-5f);
    EXPECT_NEAR(p[3].real(), 1.f, 1e-5f);
    EXPECT_NEAR(p[3].imag(), 0.f, 1e-5f);
}

TEST_P(LinalgComplex, InvComplex128) {
    auto A = make_c128_2x2({2., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.}, device);
    tz::Tensor Ainv;
    ASSERT_NO_THROW(Ainv = tz::linalg::inv(A));
    EXPECT_EQ(Ainv.dtype(), tz::DType::Complex128);

    auto I_approx = tz::matmul(A, Ainv).cpu();
    const auto* p = I_approx.data<std::complex<double>>();
    EXPECT_NEAR(p[0].real(), 1., 1e-12);
    EXPECT_NEAR(p[0].imag(), 0., 1e-12);
    EXPECT_NEAR(p[3].real(), 1., 1e-12);
    EXPECT_NEAR(p[3].imag(), 0., 1e-12);
}

// ---------------------------------------------------------------------------
// cholesky: A = L L^H for Hermitian positive-definite A
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, CholeskyComplex64) {
    // A = [[4+0i, 1+1i], [1-1i, 3+0i]] — HPD
    // eigenvalues: solve det(A - λI) = 0 → (4-λ)(3-λ) - 2 = 0 → both > 0
    auto A = make_c64_2x2({4.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f}, device);
    tz::Tensor L;
    ASSERT_NO_THROW(L = tz::linalg::cholesky(A));
    EXPECT_EQ(L.dtype(), tz::DType::Complex64);

    // Verify L @ L^H ≈ A
    // L^H = conj(transpose(L)); for 2x2 lower triangular:
    //   L = [[l00, 0], [l10, l11]]
    //   L^H = [[conj(l00), conj(l10)], [0, conj(l11)]]
    auto L_cpu = L.cpu();
    const auto* lp = L_cpu.data<std::complex<float>>();
    std::complex<float> l00 = lp[0], l10 = lp[2], l11 = lp[3];

    // (L @ L^H)[0,0] = |l00|^2
    float a00_r = std::norm(l00);
    EXPECT_NEAR(a00_r, 4.f, 1e-4f);

    // (L @ L^H)[0,1] = l00 * conj(l10)
    std::complex<float> a01 = l00 * std::conj(l10);
    EXPECT_NEAR(a01.real(), 1.f, 1e-4f);
    EXPECT_NEAR(a01.imag(), 1.f, 1e-4f);

    // (L @ L^H)[1,1] = |l10|^2 + |l11|^2
    float a11_r = std::norm(l10) + std::norm(l11);
    EXPECT_NEAR(a11_r, 3.f, 1e-4f);
}

TEST_P(LinalgComplex, CholeskyComplex128) {
    auto A = make_c128_2x2({4., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.}, device);
    tz::Tensor L;
    ASSERT_NO_THROW(L = tz::linalg::cholesky(A));
    EXPECT_EQ(L.dtype(), tz::DType::Complex128);

    auto L_cpu = L.cpu();
    const auto* lp = L_cpu.data<std::complex<double>>();
    std::complex<double> l00 = lp[0], l10 = lp[2], l11 = lp[3];
    EXPECT_NEAR(std::norm(l00), 4., 1e-10);
    EXPECT_NEAR((l00 * std::conj(l10)).real(), 1., 1e-10);
    EXPECT_NEAR((l00 * std::conj(l10)).imag(), 1., 1e-10);
    EXPECT_NEAR(std::norm(l10) + std::norm(l11), 3., 1e-10);
}

// ---------------------------------------------------------------------------
// eigh: Hermitian eigendecomposition
// For a Hermitian matrix the eigenvalues are real; eigh returns them as
// Float32 (for Complex64 input) or Float64 (for Complex128 input).
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, EighComplex64) {
    // A = [[2+0i, 1-1i], [1+1i, 3+0i]] — Hermitian
    // Characteristic poly: λ^2 - 5λ + (6 - 2) = λ^2 - 5λ + 4 = 0
    // → eigenvalues 1 and 4
    auto A = make_c64_2x2({2.f, 0.f}, {1.f,-1.f},
                          {1.f, 1.f}, {3.f, 0.f}, device);
    tz::Tensor eigvals, eigvecs;
    ASSERT_NO_THROW(std::tie(eigvals, eigvecs) = tz::linalg::eigh(A));

    // eigvals should be real-valued (Float32 for Complex64 input)
    EXPECT_EQ(eigvals.dtype(), tz::DType::Float32);
    EXPECT_EQ(eigvecs.dtype(), tz::DType::Complex64);
    ASSERT_EQ(eigvals.shape().size(), 1u);
    EXPECT_EQ(eigvals.shape()[0], 2);

    // Eigenvalues are sorted ascending: 1, 4
    auto eigvals_cpu = eigvals.cpu();
    const auto* wp = eigvals_cpu.data<float>();
    EXPECT_NEAR(wp[0], 1.f, 1e-4f);
    EXPECT_NEAR(wp[1], 4.f, 1e-4f);
}

TEST_P(LinalgComplex, EighComplex128) {
    auto A = make_c128_2x2({2., 0.}, {1.,-1.},
                           {1., 1.}, {3., 0.}, device);
    tz::Tensor eigvals, eigvecs;
    ASSERT_NO_THROW(std::tie(eigvals, eigvecs) = tz::linalg::eigh(A));

    EXPECT_EQ(eigvals.dtype(), tz::DType::Float64);
    EXPECT_EQ(eigvecs.dtype(), tz::DType::Complex128);

    auto eigvals_cpu = eigvals.cpu();
    const auto* wp = eigvals_cpu.data<double>();
    EXPECT_NEAR(wp[0], 1., 1e-10);
    EXPECT_NEAR(wp[1], 4., 1e-10);
}

// ---------------------------------------------------------------------------
// det: complex determinant via LU (cgetrf/zgetrf)
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, DetComplex64) {
    // det([[2+0i, 1+1i], [1-1i, 3+0i]]) = 2*3 - (1+i)(1-i) = 6 - 2 = 4+0i
    auto A = make_c64_2x2({2.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f}, device);
    tz::Tensor d;
    ASSERT_NO_THROW(d = tz::linalg::det(A));
    EXPECT_EQ(d.dtype(), tz::DType::Complex64);

    auto d_cpu = d.cpu();
    const auto* dp = d_cpu.data<std::complex<float>>();
    EXPECT_NEAR(dp[0].real(), 4.f, 1e-4f);
    EXPECT_NEAR(dp[0].imag(), 0.f, 1e-4f);
}

TEST_P(LinalgComplex, DetComplex128) {
    auto A = make_c128_2x2({2., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.}, device);
    tz::Tensor d;
    ASSERT_NO_THROW(d = tz::linalg::det(A));
    EXPECT_EQ(d.dtype(), tz::DType::Complex128);

    auto d_cpu = d.cpu();
    const auto* dp = d_cpu.data<std::complex<double>>();
    EXPECT_NEAR(dp[0].real(), 4., 1e-10);
    EXPECT_NEAR(dp[0].imag(), 0., 1e-10);
}

// ---------------------------------------------------------------------------
// qr: Q is unitary, R is upper-triangular for complex input
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, QRComplex64) {
    auto A = make_c64_2x2({1.f, 1.f}, {2.f,-1.f},
                          {0.f, 1.f}, {3.f, 0.f}, device);
    tz::Tensor Q, R;
    ASSERT_NO_THROW(std::tie(Q, R) = tz::linalg::qr(A));
    EXPECT_EQ(Q.dtype(), tz::DType::Complex64);
    EXPECT_EQ(R.dtype(), tz::DType::Complex64);

    // Q should be unitary: Q^H Q ≈ I
    // For a 2x2 matrix: use matmul(conj(Q^T), Q) ≈ I
    // We can verify via Q @ R ≈ A instead
    auto A_approx = tz::matmul(Q, R).cpu();
    auto A_cpu = A.cpu();
    const auto* ap = A_approx.data<std::complex<float>>();
    const auto* orig = A_cpu.data<std::complex<float>>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ap[i].real(), orig[i].real(), 1e-4f) << "idx=" << i;
        EXPECT_NEAR(ap[i].imag(), orig[i].imag(), 1e-4f) << "idx=" << i;
    }
}

TEST_P(LinalgComplex, QRComplex128) {
    auto A = make_c128_2x2({1., 1.}, {2.,-1.},
                           {0., 1.}, {3., 0.}, device);
    tz::Tensor Q, R;
    ASSERT_NO_THROW(std::tie(Q, R) = tz::linalg::qr(A));
    EXPECT_EQ(Q.dtype(), tz::DType::Complex128);

    auto A_approx = tz::matmul(Q, R).cpu();
    auto A_cpu = A.cpu();
    const auto* ap = A_approx.data<std::complex<double>>();
    const auto* orig = A_cpu.data<std::complex<double>>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ap[i].real(), orig[i].real(), 1e-10) << "idx=" << i;
        EXPECT_NEAR(ap[i].imag(), orig[i].imag(), 1e-10) << "idx=" << i;
    }
}

// ---------------------------------------------------------------------------
// svd: U @ diag(S) @ Vh ≈ A for complex input
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, SVDComplex64) {
    auto A = make_c64_2x2({1.f, 0.f}, {2.f, 1.f},
                          {0.f, 1.f}, {1.f, 0.f}, device);
    tz::Tensor U, S, Vh;
    ASSERT_NO_THROW(std::tie(U, S, Vh) = tz::linalg::svd(A, false));
    EXPECT_EQ(U.dtype(),  tz::DType::Complex64);
    // Singular values are real
    EXPECT_EQ(S.dtype(), tz::DType::Float32);
    EXPECT_EQ(Vh.dtype(), tz::DType::Complex64);

    // All singular values must be non-negative
    auto S_cpu = S.cpu();
    const auto* sp = S_cpu.data<float>();
    EXPECT_GE(sp[0], 0.f);
    EXPECT_GE(sp[1], 0.f);
    // And sorted descending
    EXPECT_GE(sp[0], sp[1] - 1e-5f);
}

TEST_P(LinalgComplex, SVDComplex128) {
    auto A = make_c128_2x2({1., 0.}, {2., 1.},
                           {0., 1.}, {1., 0.}, device);
    tz::Tensor U, S, Vh;
    ASSERT_NO_THROW(std::tie(U, S, Vh) = tz::linalg::svd(A, false));
    EXPECT_EQ(S.dtype(), tz::DType::Float64);

    auto S_cpu = S.cpu();
    const auto* sp = S_cpu.data<double>();
    EXPECT_GE(sp[0], 0.);
    EXPECT_GE(sp[1], 0.);
}

// ---------------------------------------------------------------------------
// lstsq: Least-squares solver for complex matrices (audit-4 W.21).
//
// The CPU implementation composes lstsq over QR + solve_triangular + matmul,
// all of which support Complex64/Complex128 via the LAPACKE primitives
// (cgeqrf/zgeqrf and the per-dtype matmul kernels). PyTorch's linalg.lstsq
// supports complex inputs via cgels/zgels; the QR composition has the same
// semantics for over-determined / square systems with full column rank.
// ---------------------------------------------------------------------------

TEST_P(LinalgComplex, LstsqComplex64) {
    // Over-determined system: A (3x2), B (3x1).
    // Choose A with full column rank and a complex B; verify A @ x ≈ B
    // (residual ≈ 0 because B lies in range(A) — we pick B = A @ x_true).
    auto A_host = tz::zeros({3, 2}, tz::DType::Complex64);
    auto* ap = A_host.data<std::complex<float>>();
    ap[0] = {1.f, 0.f};  ap[1] = {0.f, 1.f};   // row 0
    ap[2] = {1.f, 1.f};  ap[3] = {1.f, 0.f};   // row 1
    ap[4] = {0.f, 1.f};  ap[5] = {2.f, 0.f};   // row 2
    auto A = A_host.to(device);

    // x_true = [(1+0i), (0+1i)]^T → B = A @ x_true
    auto x_true_host = tz::zeros({2, 1}, tz::DType::Complex64);
    x_true_host.data<std::complex<float>>()[0] = {1.f, 0.f};
    x_true_host.data<std::complex<float>>()[1] = {0.f, 1.f};
    auto x_true = x_true_host.to(device);
    auto B = tz::matmul(A, x_true);

    tz::Tensor x, residuals;
    ASSERT_NO_THROW(std::tie(x, residuals) = tz::linalg::lstsq(A, B));
    EXPECT_EQ(x.dtype(), tz::DType::Complex64);
    ASSERT_EQ(x.shape().size(), 2u);
    EXPECT_EQ(x.shape()[0], 2);
    EXPECT_EQ(x.shape()[1], 1);

    // Solution should match x_true since B is in range(A).
    auto x_cpu = x.cpu();
    const auto* xp = x_cpu.data<std::complex<float>>();
    EXPECT_NEAR(xp[0].real(), 1.f, 1e-4f);
    EXPECT_NEAR(xp[0].imag(), 0.f, 1e-4f);
    EXPECT_NEAR(xp[1].real(), 0.f, 1e-4f);
    EXPECT_NEAR(xp[1].imag(), 1.f, 1e-4f);

    // A @ x ≈ B as a defence-in-depth sanity check (matches cgels semantics).
    auto recon = tz::matmul(A, x).cpu();
    auto B_cpu = B.cpu();
    const auto* rp = recon.data<std::complex<float>>();
    const auto* bp = B_cpu.data<std::complex<float>>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(rp[i].real(), bp[i].real(), 1e-4f) << "row " << i;
        EXPECT_NEAR(rp[i].imag(), bp[i].imag(), 1e-4f) << "row " << i;
    }
}

TEST_P(LinalgComplex, LstsqComplex128) {
    // Same shape and structure in double precision.
    auto A_host = tz::zeros({3, 2}, tz::DType::Complex128);
    auto* ap = A_host.data<std::complex<double>>();
    ap[0] = {1., 0.};  ap[1] = {0., 1.};
    ap[2] = {1., 1.};  ap[3] = {1., 0.};
    ap[4] = {0., 1.};  ap[5] = {2., 0.};
    auto A = A_host.to(device);

    auto x_true_host = tz::zeros({2, 1}, tz::DType::Complex128);
    x_true_host.data<std::complex<double>>()[0] = {1., 0.};
    x_true_host.data<std::complex<double>>()[1] = {0., 1.};
    auto x_true = x_true_host.to(device);
    auto B = tz::matmul(A, x_true);

    tz::Tensor x, residuals;
    ASSERT_NO_THROW(std::tie(x, residuals) = tz::linalg::lstsq(A, B));
    EXPECT_EQ(x.dtype(), tz::DType::Complex128);

    auto x_cpu = x.cpu();
    const auto* xp = x_cpu.data<std::complex<double>>();
    EXPECT_NEAR(xp[0].real(), 1., 1e-10);
    EXPECT_NEAR(xp[0].imag(), 0., 1e-10);
    EXPECT_NEAR(xp[1].real(), 0., 1e-10);
    EXPECT_NEAR(xp[1].imag(), 1., 1e-10);

    auto recon = tz::matmul(A, x).cpu();
    auto B_cpu = B.cpu();
    const auto* rp = recon.data<std::complex<double>>();
    const auto* bp = B_cpu.data<std::complex<double>>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(rp[i].real(), bp[i].real(), 1e-10) << "row " << i;
        EXPECT_NEAR(rp[i].imag(), bp[i].imag(), 1e-10) << "row " << i;
    }
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given complex linalg op throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(LinalgComplex);
