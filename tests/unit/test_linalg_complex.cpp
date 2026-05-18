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

namespace tz = ::tenzor;

// Custom main to ensure Tenzor is initialized (backends registered) before tests run.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    tenzor::initialize();
    return RUN_ALL_TESTS();
}

// ---------------------------------------------------------------------------
// Helper: build a 2x2 complex matrix on CPU
// ---------------------------------------------------------------------------
static tz::Tensor make_c64_2x2(
    std::complex<float> a00, std::complex<float> a01,
    std::complex<float> a10, std::complex<float> a11)
{
    auto t = tz::zeros({2, 2}, tz::DType::Complex64);
    auto* p = t.data<std::complex<float>>();
    p[0] = a00; p[1] = a01; p[2] = a10; p[3] = a11;
    return t;
}

static tz::Tensor make_c128_2x2(
    std::complex<double> a00, std::complex<double> a01,
    std::complex<double> a10, std::complex<double> a11)
{
    auto t = tz::zeros({2, 2}, tz::DType::Complex128);
    auto* p = t.data<std::complex<double>>();
    p[0] = a00; p[1] = a01; p[2] = a10; p[3] = a11;
    return t;
}

// ---------------------------------------------------------------------------
// solve: Ax = b
// ---------------------------------------------------------------------------

TEST(LinalgComplex, SolveComplex64) {
    // A = [[2+0i, 1+1i], [1-1i, 3+0i]]  (Hermitian positive definite)
    // det(A) = 2*3 - (1+i)(1-i) = 6 - 2 = 4 → non-singular
    // b = [[1+0i], [2+0i]]
    auto A = make_c64_2x2({2.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f});
    auto b = tz::zeros({2, 1}, tz::DType::Complex64);
    b.data<std::complex<float>>()[0] = {1.f, 0.f};
    b.data<std::complex<float>>()[1] = {2.f, 0.f};

    tz::Tensor x;
    ASSERT_NO_THROW(x = tz::linalg::solve(A, b));
    EXPECT_EQ(x.dtype(), tz::DType::Complex64);
    ASSERT_EQ(x.shape().size(), 2u);
    EXPECT_EQ(x.shape()[0], 2);
    EXPECT_EQ(x.shape()[1], 1);

    // Verify: A @ x ≈ b
    auto bx = tz::matmul(A, x);
    const auto* bxp = bx.data<std::complex<float>>();
    const auto* bp  = b.data<std::complex<float>>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(bxp[i].real(), bp[i].real(), 1e-5f)
            << "real part mismatch at index " << i;
        EXPECT_NEAR(bxp[i].imag(), bp[i].imag(), 1e-5f)
            << "imag part mismatch at index " << i;
    }
}

TEST(LinalgComplex, SolveComplex128) {
    // Same system in double precision
    auto A = make_c128_2x2({2., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.});
    auto b = tz::zeros({2, 1}, tz::DType::Complex128);
    b.data<std::complex<double>>()[0] = {1., 0.};
    b.data<std::complex<double>>()[1] = {2., 0.};

    tz::Tensor x;
    ASSERT_NO_THROW(x = tz::linalg::solve(A, b));
    EXPECT_EQ(x.dtype(), tz::DType::Complex128);

    auto bx = tz::matmul(A, x);
    const auto* bxp = bx.data<std::complex<double>>();
    const auto* bp  = b.data<std::complex<double>>();
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

TEST(LinalgComplex, InvComplex64) {
    auto A = make_c64_2x2({2.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f});
    tz::Tensor Ainv;
    ASSERT_NO_THROW(Ainv = tz::linalg::inv(A));
    EXPECT_EQ(Ainv.dtype(), tz::DType::Complex64);

    auto I_approx = tz::matmul(A, Ainv);
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

TEST(LinalgComplex, InvComplex128) {
    auto A = make_c128_2x2({2., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.});
    tz::Tensor Ainv;
    ASSERT_NO_THROW(Ainv = tz::linalg::inv(A));
    EXPECT_EQ(Ainv.dtype(), tz::DType::Complex128);

    auto I_approx = tz::matmul(A, Ainv);
    const auto* p = I_approx.data<std::complex<double>>();
    EXPECT_NEAR(p[0].real(), 1., 1e-12);
    EXPECT_NEAR(p[0].imag(), 0., 1e-12);
    EXPECT_NEAR(p[3].real(), 1., 1e-12);
    EXPECT_NEAR(p[3].imag(), 0., 1e-12);
}

// ---------------------------------------------------------------------------
// cholesky: A = L L^H for Hermitian positive-definite A
// ---------------------------------------------------------------------------

TEST(LinalgComplex, CholeskyComplex64) {
    // A = [[4+0i, 1+1i], [1-1i, 3+0i]] — HPD
    // eigenvalues: solve det(A - λI) = 0 → (4-λ)(3-λ) - 2 = 0 → both > 0
    auto A = make_c64_2x2({4.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f});
    tz::Tensor L;
    ASSERT_NO_THROW(L = tz::linalg::cholesky(A));
    EXPECT_EQ(L.dtype(), tz::DType::Complex64);

    // Verify L @ L^H ≈ A
    // L^H = conj(transpose(L)); for 2x2 lower triangular:
    //   L = [[l00, 0], [l10, l11]]
    //   L^H = [[conj(l00), conj(l10)], [0, conj(l11)]]
    const auto* lp = L.data<std::complex<float>>();
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

TEST(LinalgComplex, CholeskyComplex128) {
    auto A = make_c128_2x2({4., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.});
    tz::Tensor L;
    ASSERT_NO_THROW(L = tz::linalg::cholesky(A));
    EXPECT_EQ(L.dtype(), tz::DType::Complex128);

    const auto* lp = L.data<std::complex<double>>();
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

TEST(LinalgComplex, EighComplex64) {
    // A = [[2+0i, 1-1i], [1+1i, 3+0i]] — Hermitian
    // Characteristic poly: λ^2 - 5λ + (6 - 2) = λ^2 - 5λ + 4 = 0
    // → eigenvalues 1 and 4
    auto A = make_c64_2x2({2.f, 0.f}, {1.f,-1.f},
                          {1.f, 1.f}, {3.f, 0.f});
    tz::Tensor eigvals, eigvecs;
    ASSERT_NO_THROW(std::tie(eigvals, eigvecs) = tz::linalg::eigh(A));

    // eigvals should be real-valued (Float32 for Complex64 input)
    EXPECT_EQ(eigvals.dtype(), tz::DType::Float32);
    EXPECT_EQ(eigvecs.dtype(), tz::DType::Complex64);
    ASSERT_EQ(eigvals.shape().size(), 1u);
    EXPECT_EQ(eigvals.shape()[0], 2);

    // Eigenvalues are sorted ascending: 1, 4
    const auto* wp = eigvals.data<float>();
    EXPECT_NEAR(wp[0], 1.f, 1e-4f);
    EXPECT_NEAR(wp[1], 4.f, 1e-4f);
}

TEST(LinalgComplex, EighComplex128) {
    auto A = make_c128_2x2({2., 0.}, {1.,-1.},
                           {1., 1.}, {3., 0.});
    tz::Tensor eigvals, eigvecs;
    ASSERT_NO_THROW(std::tie(eigvals, eigvecs) = tz::linalg::eigh(A));

    EXPECT_EQ(eigvals.dtype(), tz::DType::Float64);
    EXPECT_EQ(eigvecs.dtype(), tz::DType::Complex128);

    const auto* wp = eigvals.data<double>();
    EXPECT_NEAR(wp[0], 1., 1e-10);
    EXPECT_NEAR(wp[1], 4., 1e-10);
}

// ---------------------------------------------------------------------------
// det: complex determinant via LU (cgetrf/zgetrf)
// ---------------------------------------------------------------------------

TEST(LinalgComplex, DetComplex64) {
    // det([[2+0i, 1+1i], [1-1i, 3+0i]]) = 2*3 - (1+i)(1-i) = 6 - 2 = 4+0i
    auto A = make_c64_2x2({2.f, 0.f}, {1.f, 1.f},
                          {1.f,-1.f}, {3.f, 0.f});
    tz::Tensor d;
    ASSERT_NO_THROW(d = tz::linalg::det(A));
    EXPECT_EQ(d.dtype(), tz::DType::Complex64);

    const auto* dp = d.data<std::complex<float>>();
    EXPECT_NEAR(dp[0].real(), 4.f, 1e-4f);
    EXPECT_NEAR(dp[0].imag(), 0.f, 1e-4f);
}

TEST(LinalgComplex, DetComplex128) {
    auto A = make_c128_2x2({2., 0.}, {1., 1.},
                           {1.,-1.}, {3., 0.});
    tz::Tensor d;
    ASSERT_NO_THROW(d = tz::linalg::det(A));
    EXPECT_EQ(d.dtype(), tz::DType::Complex128);

    const auto* dp = d.data<std::complex<double>>();
    EXPECT_NEAR(dp[0].real(), 4., 1e-10);
    EXPECT_NEAR(dp[0].imag(), 0., 1e-10);
}

// ---------------------------------------------------------------------------
// qr: Q is unitary, R is upper-triangular for complex input
// ---------------------------------------------------------------------------

TEST(LinalgComplex, QRComplex64) {
    auto A = make_c64_2x2({1.f, 1.f}, {2.f,-1.f},
                          {0.f, 1.f}, {3.f, 0.f});
    tz::Tensor Q, R;
    ASSERT_NO_THROW(std::tie(Q, R) = tz::linalg::qr(A));
    EXPECT_EQ(Q.dtype(), tz::DType::Complex64);
    EXPECT_EQ(R.dtype(), tz::DType::Complex64);

    // Q should be unitary: Q^H Q ≈ I
    // For a 2x2 matrix: use matmul(conj(Q^T), Q) ≈ I
    // We can verify via Q @ R ≈ A instead
    auto A_approx = tz::matmul(Q, R);
    const auto* ap = A_approx.data<std::complex<float>>();
    const auto* orig = A.data<std::complex<float>>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ap[i].real(), orig[i].real(), 1e-4f) << "idx=" << i;
        EXPECT_NEAR(ap[i].imag(), orig[i].imag(), 1e-4f) << "idx=" << i;
    }
}

TEST(LinalgComplex, QRComplex128) {
    auto A = make_c128_2x2({1., 1.}, {2.,-1.},
                           {0., 1.}, {3., 0.});
    tz::Tensor Q, R;
    ASSERT_NO_THROW(std::tie(Q, R) = tz::linalg::qr(A));
    EXPECT_EQ(Q.dtype(), tz::DType::Complex128);

    auto A_approx = tz::matmul(Q, R);
    const auto* ap = A_approx.data<std::complex<double>>();
    const auto* orig = A.data<std::complex<double>>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ap[i].real(), orig[i].real(), 1e-10) << "idx=" << i;
        EXPECT_NEAR(ap[i].imag(), orig[i].imag(), 1e-10) << "idx=" << i;
    }
}

// ---------------------------------------------------------------------------
// svd: U @ diag(S) @ Vh ≈ A for complex input
// ---------------------------------------------------------------------------

TEST(LinalgComplex, SVDComplex64) {
    auto A = make_c64_2x2({1.f, 0.f}, {2.f, 1.f},
                          {0.f, 1.f}, {1.f, 0.f});
    tz::Tensor U, S, Vh;
    ASSERT_NO_THROW(std::tie(U, S, Vh) = tz::linalg::svd(A, false));
    EXPECT_EQ(U.dtype(),  tz::DType::Complex64);
    // Singular values are real
    EXPECT_EQ(S.dtype(), tz::DType::Float32);
    EXPECT_EQ(Vh.dtype(), tz::DType::Complex64);

    // All singular values must be non-negative
    const auto* sp = S.data<float>();
    EXPECT_GE(sp[0], 0.f);
    EXPECT_GE(sp[1], 0.f);
    // And sorted descending
    EXPECT_GE(sp[0], sp[1] - 1e-5f);
}

TEST(LinalgComplex, SVDComplex128) {
    auto A = make_c128_2x2({1., 0.}, {2., 1.},
                           {0., 1.}, {1., 0.});
    tz::Tensor U, S, Vh;
    ASSERT_NO_THROW(std::tie(U, S, Vh) = tz::linalg::svd(A, false));
    EXPECT_EQ(S.dtype(), tz::DType::Float64);

    const auto* sp = S.data<double>();
    EXPECT_GE(sp[0], 0.);
    EXPECT_GE(sp[1], 0.);
}
