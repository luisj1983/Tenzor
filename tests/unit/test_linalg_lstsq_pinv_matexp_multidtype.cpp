/**
 * @file test_linalg_lstsq_pinv_matexp_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for lstsq, pinv, and matrix_exp
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <cmath>
#include <complex>
#include <cstring>

using namespace tenzor;
using namespace tenzor::testing;

class LinalgLstsqPinvMatexpMultiDTypeTest : public MultiBackendDTypeTest {};

#define SKIP_IF_FLOAT16() \
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 too imprecise for linalg"

TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, LstsqOverdetermined) {
    SKIP_IF_FLOAT16();
    auto A = zeros({4, 2}, DType::Float64, Device::cpu());
    auto B = zeros({4}, DType::Float64, Device::cpu());
    double A_data[8] = {1,1, 1,2, 1,3, 1,4};
    double B_data[4] = {6, 5, 7, 10};
    std::memcpy(A.data<double>(), A_data, sizeof(A_data));
    std::memcpy(B.data<double>(), B_data, sizeof(B_data));
    A = A.to(device());
    B = B.to(device());

    auto [sol, res] = linalg::lstsq(A, B);
    auto s_cpu = sol.to(Device::cpu()).to(DType::Float32);
    auto* s = s_cpu.data<float>();
    EXPECT_NEAR(s[0], 3.5f, 1e-4f);
    EXPECT_NEAR(s[1], 1.4f, 1e-4f);
}

TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, PinvSquareMatchesInverse) {
    SKIP_IF_FLOAT16();
    auto A = zeros({3, 3}, DType::Float64, Device::cpu());
    double A_data[9] = {2, 1, 0, 0, 3, 1, 1, 0, 2};
    std::memcpy(A.data<double>(), A_data, sizeof(A_data));
    A = A.to(device());

    auto Ainv = linalg::inv(A);
    auto Apinv = linalg::pinv(A);
    auto a_cpu = Ainv.to(Device::cpu()).to(DType::Float32);
    auto b_cpu = Apinv.to(Device::cpu()).to(DType::Float32);
    auto* a = a_cpu.data<float>();
    auto* b = b_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(a[i], b[i], 1e-4f);
    }
}

TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, PinvReconstructsA) {
    SKIP_IF_FLOAT16();
    auto A = zeros({4, 2}, DType::Float64, Device::cpu());
    double A_data[8] = {1,1, 1,2, 1,3, 1,4};
    std::memcpy(A.data<double>(), A_data, sizeof(A_data));
    A = A.to(device());

    auto Ap = linalg::pinv(A);
    auto reconstructed = matmul(matmul(A, Ap), A);
    auto a_cpu = A.to(Device::cpu()).to(DType::Float32);
    auto r_cpu = reconstructed.to(Device::cpu()).to(DType::Float32);
    auto* a = a_cpu.data<float>();
    auto* r = r_cpu.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(r[i], a[i], 1e-4f);
    }
}

// audit-3 T.13: Complex64 pinv coverage. Build identity directly in Complex64
// storage (interleaved real/imag pairs). pinv(I) == I.
TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, PinvIdentity_Complex64) {
    // The dtype() parameter doesn't drive the matrix dtype here — we cover the
    // explicit Complex64 path that the parametrised Float32/Float64/Float16
    // sweep does not reach. Run once per backend (gated on dtype param to
    // avoid 3x redundant work).
    if (dtype() != DType::Float32) GTEST_SKIP() << "Complex64 cover runs in Float32 slot";

    auto A = zeros({3, 3}, DType::Complex64, Device::cpu());
    auto* a = A.data<std::complex<float>>();
    a[0] = {1.0f, 0.0f};
    a[4] = {1.0f, 0.0f};
    a[8] = {1.0f, 0.0f};
    A = A.to(device());

    auto Ap = linalg::pinv(A);
    auto ap_cpu = Ap.to(Device::cpu());
    auto* ap = ap_cpu.data<std::complex<float>>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected_re = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(ap[i * 3 + j].real(), expected_re, 1e-4f);
            EXPECT_NEAR(ap[i * 3 + j].imag(), 0.0f, 1e-4f);
        }
    }
}

// audit-3 T.13: rank-deficient pinv coverage — row 2 = row 0 + row 1, so the
// 3x3 matrix has rank 2. Moore–Penrose property 1: A @ pinv(A) @ A ≈ A.
TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, PinvRankDeficient) {
    SKIP_IF_FLOAT16();

    auto A = zeros({3, 3}, DType::Float64, Device::cpu());
    double A_data[9] = {
        1.0,  2.0, 3.0,
        4.0,  5.0, 6.0,
        5.0,  7.0, 9.0,  // = row0 + row1
    };
    std::memcpy(A.data<double>(), A_data, sizeof(A_data));
    A = A.to(device());

    auto Ap = linalg::pinv(A);
    auto reconstructed = matmul(matmul(A, Ap), A);
    auto a_cpu = A.to(Device::cpu()).to(DType::Float32);
    auto r_cpu = reconstructed.to(Device::cpu()).to(DType::Float32);
    auto* a = a_cpu.data<float>();
    auto* r = r_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(r[i], a[i], 1e-3f) << "Moore–Penrose property 1 fails at " << i;
    }
}

// Regression: pinv of a genuinely complex (non-Hermitian, non-identity) matrix
// must use the conjugate transpose. For a square invertible matrix pinv == inv;
// with a plain (non-conjugated) transpose the result is the elementwise
// conjugate of the correct pinv, so the imaginary parts flip sign and this
// comparison fails.
TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, PinvComplexEqualsInverse_Complex64) {
    if (dtype() != DType::Float32) GTEST_SKIP() << "Complex64 cover runs in Float32 slot";

    auto A = zeros({2, 2}, DType::Complex64, Device::cpu());
    auto* a = A.data<std::complex<float>>();
    a[0] = {1.0f, 1.0f};   // 1+1i
    a[1] = {2.0f, 0.0f};   // 2
    a[2] = {0.0f, 0.0f};   // 0
    a[3] = {1.0f, -1.0f};  // 1-1i  (det = 2, invertible)
    A = A.to(device());

    auto Ainv = linalg::inv(A);
    auto Apinv = linalg::pinv(A);
    auto inv_cpu = Ainv.to(Device::cpu());
    auto pinv_cpu = Apinv.to(Device::cpu());
    auto* inv = inv_cpu.data<std::complex<float>>();
    auto* pin = pinv_cpu.data<std::complex<float>>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(pin[i].real(), inv[i].real(), 1e-4f) << "real mismatch at " << i;
        EXPECT_NEAR(pin[i].imag(), inv[i].imag(), 1e-4f) << "imag mismatch at " << i;
    }
}

// Regression: matrix_rank on a batched (B, N, N) input must return a per-matrix
// rank vector of shape (B,), not a single global scalar. Batch 0 is full rank
// (3), batch 1 is rank 2 (row2 = row0 + row1).
TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, MatrixRankBatched) {
    SKIP_IF_FLOAT16();
    auto A = zeros({2, 3, 3}, DType::Float64, Device::cpu());
    double A_data[18] = {
        1, 0, 0,  0, 1, 0,  0, 0, 1,   // batch 0: rank 3
        1, 2, 3,  4, 5, 6,  5, 7, 9,   // batch 1: rank 2 (row2 = row0+row1)
    };
    std::memcpy(A.data<double>(), A_data, sizeof(A_data));
    A = A.to(device());

    auto rank = linalg::matrix_rank(A);
    auto r_cpu = rank.to(Device::cpu());
    ASSERT_EQ(r_cpu.ndim(), 1);
    ASSERT_EQ(r_cpu.shape()[0], 2);
    auto* r = r_cpu.data<int64_t>();
    EXPECT_EQ(r[0], 3);
    EXPECT_EQ(r[1], 2);
}

// Regression: cond on a batched (B, N, N) input must return a per-matrix
// condition number of shape (B,). Batch 0 = identity (cond 1), batch 1 =
// diag(2,1) (spectral cond = 2).
TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, CondBatched) {
    SKIP_IF_FLOAT16();
    auto A = zeros({2, 2, 2}, DType::Float64, Device::cpu());
    double A_data[8] = {
        1, 0,  0, 1,   // batch 0: identity, cond = 1
        2, 0,  0, 1,   // batch 1: diag(2,1), spectral cond = 2
    };
    std::memcpy(A.data<double>(), A_data, sizeof(A_data));
    A = A.to(device());

    auto c = linalg::cond(A, "2");
    auto c_cpu = c.to(Device::cpu()).to(DType::Float64);
    ASSERT_EQ(c_cpu.ndim(), 1);
    ASSERT_EQ(c_cpu.shape()[0], 2);
    auto* cv = c_cpu.data<double>();
    EXPECT_NEAR(cv[0], 1.0, 1e-5);
    EXPECT_NEAR(cv[1], 2.0, 1e-5);
}

// Regression: vector_norm with a negative multi-dim set {-2,-1} and
// keepdim=false must reduce the last two axes (B, C) of a (A,B,C) tensor.
// Without normalizing negatives before the descending sort, the raw-value
// order [-1,-2] plus keepdim=false shifted the axes and reduced the wrong dims.
TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, VectorNormNegativeMultiDim) {
    SKIP_IF_FLOAT16();
    auto x = zeros({2, 2, 2}, DType::Float64, Device::cpu());
    double x_data[8] = {
        3, 4, 0, 0,   // batch 0: ||.||_2 over last two dims = 5
        0, 0, 6, 8,   // batch 1: ||.||_2 over last two dims = 10
    };
    std::memcpy(x.data<double>(), x_data, sizeof(x_data));
    x = x.to(device());

    auto n = linalg::vector_norm(x, 2.0, {-2, -1}, /*keepdim=*/false);
    auto n_cpu = n.to(Device::cpu()).to(DType::Float64);
    ASSERT_EQ(n_cpu.ndim(), 1);
    ASSERT_EQ(n_cpu.shape()[0], 2);
    auto* nv = n_cpu.data<double>();
    EXPECT_NEAR(nv[0], 5.0, 1e-6);
    EXPECT_NEAR(nv[1], 10.0, 1e-6);
}

TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, MatrixExpOfZero) {
    SKIP_IF_FLOAT16();
    auto Z = zeros({3, 3}, DType::Float64, Device::cpu()).to(device());
    auto E = linalg::matrix_exp(Z);
    auto e_cpu = E.to(Device::cpu()).to(DType::Float32);
    auto* e = e_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(e[i * 3 + j], expected, 1e-5f);
        }
    }
}

TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, MatrixExpOfDiagonal) {
    SKIP_IF_FLOAT16();
    auto D = zeros({2, 2}, DType::Float64, Device::cpu());
    D.data<double>()[0] = 0.5;
    D.data<double>()[3] = -0.3;
    D = D.to(device());

    auto E = linalg::matrix_exp(D);
    auto e_cpu = E.to(Device::cpu()).to(DType::Float32);
    auto* e = e_cpu.data<float>();
    // matrix_exp uses scaling-and-squaring (Padé approximant); the Float32
    // precision floor for the diagonal entries after Float64→Float32 cast
    // is sqrt(eps_F32) ≈ 3.4e-4 — 1e-4f is the tight bound here.
    // reason: Float32 Padé precision floor (~3.4e-4) for matrix_exp diagonal.
    EXPECT_NEAR(e[0], static_cast<float>(std::exp(0.5)), 1e-4f);
    EXPECT_NEAR(e[1], 0.0f, 1e-5f);
    EXPECT_NEAR(e[2], 0.0f, 1e-5f);
    // reason: same Float32 Padé precision floor as e[0] — 1e-4f is tight.
    EXPECT_NEAR(e[3], static_cast<float>(std::exp(-0.3)), 1e-4f);
}

TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, MatrixExpNilpotent) {
    SKIP_IF_FLOAT16();
    auto A = zeros({2, 2}, DType::Float64, Device::cpu());
    A.data<double>()[1] = 1.0;
    A = A.to(device());

    auto E = linalg::matrix_exp(A);
    auto e_cpu = E.to(Device::cpu()).to(DType::Float32);
    auto* e = e_cpu.data<float>();
    EXPECT_NEAR(e[0], 1.0f, 1e-5f);
    EXPECT_NEAR(e[1], 1.0f, 1e-5f);
    EXPECT_NEAR(e[2], 0.0f, 1e-5f);
    EXPECT_NEAR(e[3], 1.0f, 1e-5f);
}

TEST_P(LinalgLstsqPinvMatexpMultiDTypeTest, MatrixExpRotation) {
    SKIP_IF_FLOAT16();
    const double theta = 0.7;
    auto A = zeros({2, 2}, DType::Float64, Device::cpu());
    A.data<double>()[1] = -theta;
    A.data<double>()[2] =  theta;
    A = A.to(device());

    auto E = linalg::matrix_exp(A);
    auto e_cpu = E.to(Device::cpu()).to(DType::Float32);
    auto* e = e_cpu.data<float>();
    EXPECT_NEAR(e[0], static_cast<float>(std::cos(theta)), 1e-4f);
    EXPECT_NEAR(e[1], static_cast<float>(-std::sin(theta)), 1e-4f);
    EXPECT_NEAR(e[2], static_cast<float>(std::sin(theta)), 1e-4f);
    EXPECT_NEAR(e[3], static_cast<float>(std::cos(theta)), 1e-4f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LinalgLstsqPinvMatexpMultiDTypeTest);
