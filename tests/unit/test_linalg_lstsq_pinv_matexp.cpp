// Tests for linalg::lstsq, linalg::pinv, and linalg::matrix_exp.
// Ported to cross-backend via BackendTest fixture. The lstsq/pinv/matrix_exp
// ops themselves take no device argument; inputs are created on `device` and
// host comparisons read back through .cpu().
// Reference values are computed against numpy/scipy on tiny examples.

#include "../backend_test_fixture.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

namespace tenzor {
namespace {

class LinalgAdvancedTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// lstsq
// ============================================================================

TEST_P(LinalgAdvancedTest, LstsqOverdetermined) {
    // A = [[1, 1], [1, 2], [1, 3], [1, 4]], B = [6, 5, 7, 10]
    // Standard line-fitting example. NumPy's lstsq gives
    //   solution ≈ [3.5, 1.4], residual ≈ 4.2
    auto A_cpu = zeros({4, 2}, DType::Float64, Device::cpu());
    auto B_cpu = zeros({4}, DType::Float64, Device::cpu());
    double A_data[8] = {1,1, 1,2, 1,3, 1,4};
    double B_data[4] = {6, 5, 7, 10};
    std::memcpy(A_cpu.data<double>(), A_data, sizeof(A_data));
    std::memcpy(B_cpu.data<double>(), B_data, sizeof(B_data));
    auto A = A_cpu.to(device);
    auto B = B_cpu.to(device);

    auto [sol, res] = linalg::lstsq(A, B);
    auto sol_cpu = sol.cpu();
    const double* s = sol_cpu.data<double>();
    EXPECT_NEAR(s[0], 3.5, 1e-9);
    EXPECT_NEAR(s[1], 1.4, 1e-9);

    ASSERT_EQ(res.numel(), 1);
    auto res_cpu = res.cpu();
    EXPECT_NEAR(res_cpu.data<double>()[0], 4.2, 1e-9);
}

TEST_P(LinalgAdvancedTest, LstsqSquareIsEqualToSolve) {
    // For a square non-singular A, lstsq should match solve.
    auto A_cpu = zeros({3, 3}, DType::Float64, Device::cpu());
    auto B_cpu = zeros({3}, DType::Float64, Device::cpu());
    double A_data[9] = {2, 1, 0,  0, 3, 1,  1, 0, 2};
    double B_data[3] = {1, 2, 3};
    std::memcpy(A_cpu.data<double>(), A_data, sizeof(A_data));
    std::memcpy(B_cpu.data<double>(), B_data, sizeof(B_data));
    auto A = A_cpu.to(device);
    auto B = B_cpu.to(device);

    auto [sol, res] = linalg::lstsq(A, B);
    auto solve_sol = linalg::solve(A, B);
    auto sol_cpu = sol.cpu();
    auto solve_cpu = solve_sol.cpu();
    const double* a = sol_cpu.data<double>();
    const double* b = solve_cpu.data<double>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(a[i], b[i], 1e-10);
    }
}

TEST_P(LinalgAdvancedTest, LstsqMultipleRHS) {
    // Same A as above, two RHS columns stacked as B shape (4, 2).
    auto A_cpu = zeros({4, 2}, DType::Float64, Device::cpu());
    auto B_cpu = zeros({4, 2}, DType::Float64, Device::cpu());
    double A_data[8] = {1,1, 1,2, 1,3, 1,4};
    double B_data[8] = {6,1,  5,2,  7,3,  10,4};
    std::memcpy(A_cpu.data<double>(), A_data, sizeof(A_data));
    std::memcpy(B_cpu.data<double>(), B_data, sizeof(B_data));
    auto A = A_cpu.to(device);
    auto B = B_cpu.to(device);

    auto [sol, res] = linalg::lstsq(A, B);
    ASSERT_EQ(sol.shape().size(), 2u);
    EXPECT_EQ(sol.shape()[0], 2);
    EXPECT_EQ(sol.shape()[1], 2);
    // Second column is an exact fit: [0, 1] (B_col2 = [1, 2, 3, 4]).
    auto sol_cpu = sol.cpu();
    const double* s = sol_cpu.data<double>();
    EXPECT_NEAR(s[0 * 2 + 1], 0.0, 1e-9);  // intercept of col 1
    EXPECT_NEAR(s[1 * 2 + 1], 1.0, 1e-9);  // slope of col 1
}

// ============================================================================
// pinv
// ============================================================================

TEST_P(LinalgAdvancedTest, PinvSquareMatchesInverse) {
    // For a square non-singular matrix, pinv == inv.
    auto A_cpu = zeros({3, 3}, DType::Float64, Device::cpu());
    double A_data[9] = {2, 1, 0,  0, 3, 1,  1, 0, 2};
    std::memcpy(A_cpu.data<double>(), A_data, sizeof(A_data));
    auto A = A_cpu.to(device);

    auto Ainv = linalg::inv(A);
    auto Apinv = linalg::pinv(A);
    auto Ainv_cpu = Ainv.cpu();
    auto Apinv_cpu = Apinv.cpu();
    const double* a = Ainv_cpu.data<double>();
    const double* b = Apinv_cpu.data<double>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(a[i], b[i], 1e-9);
    }
}

TEST_P(LinalgAdvancedTest, PinvReconstructsA) {
    // A @ pinv(A) @ A ≈ A (Moore-Penrose identity 1) for any A.
    auto A_cpu = zeros({4, 2}, DType::Float64, Device::cpu());
    double A_data[8] = {1,1, 1,2, 1,3, 1,4};
    std::memcpy(A_cpu.data<double>(), A_data, sizeof(A_data));
    auto A = A_cpu.to(device);

    auto Ap = linalg::pinv(A);
    EXPECT_EQ(Ap.shape()[0], 2);
    EXPECT_EQ(Ap.shape()[1], 4);

    auto reconstructed = matmul(matmul(A, Ap), A);
    auto A_host = A.cpu();
    auto recon_cpu = reconstructed.cpu();
    const double* a = A_host.data<double>();
    const double* r = recon_cpu.data<double>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(r[i], a[i], 1e-9);
    }
}

TEST_P(LinalgAdvancedTest, PinvFloat32) {
    auto A_cpu = zeros({3, 3}, DType::Float32, Device::cpu());
    float data[9] = {2, 1, 0,  0, 3, 1,  1, 0, 2};
    std::memcpy(A_cpu.data<float>(), data, sizeof(data));
    auto A = A_cpu.to(device);

    auto Ap = linalg::pinv(A);
    EXPECT_EQ(Ap.dtype(), DType::Float32);

    // A @ Ap @ A ≈ A
    auto reconstructed = matmul(matmul(A, Ap), A);
    auto A_host = A.cpu();
    auto recon_cpu = reconstructed.cpu();
    const float* a = A_host.data<float>();
    const float* r = recon_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(r[i], a[i], 1e-4f);
    }
}

// ============================================================================
// matrix_exp
// ============================================================================

TEST_P(LinalgAdvancedTest, MatrixExpOfZero) {
    // exp(0) == I
    auto Z = zeros({3, 3}, DType::Float64, device);
    auto E = linalg::matrix_exp(Z);
    auto E_cpu = E.cpu();
    const double* e = E_cpu.data<double>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(e[i * 3 + j], expected, 1e-12);
        }
    }
}

TEST_P(LinalgAdvancedTest, MatrixExpOfDiagonal) {
    // exp(diag(a, b)) == diag(exp(a), exp(b))
    auto D_cpu = zeros({2, 2}, DType::Float64, Device::cpu());
    D_cpu.data<double>()[0] = 0.5;  // (0,0)
    D_cpu.data<double>()[3] = -0.3; // (1,1)
    auto D = D_cpu.to(device);

    auto E = linalg::matrix_exp(D);
    auto E_cpu = E.cpu();
    const double* e = E_cpu.data<double>();
    EXPECT_NEAR(e[0], std::exp(0.5), 1e-12);
    EXPECT_NEAR(e[1], 0.0, 1e-12);
    EXPECT_NEAR(e[2], 0.0, 1e-12);
    EXPECT_NEAR(e[3], std::exp(-0.3), 1e-12);
}

TEST_P(LinalgAdvancedTest, MatrixExpNilpotent) {
    // A = [[0, 1], [0, 0]] is nilpotent. exp(A) = I + A = [[1, 1], [0, 1]]
    // (exact — the series terminates after two terms).
    auto A_cpu = zeros({2, 2}, DType::Float64, Device::cpu());
    A_cpu.data<double>()[1] = 1.0;  // (0, 1) = 1
    auto A = A_cpu.to(device);

    auto E = linalg::matrix_exp(A);
    auto E_cpu = E.cpu();
    const double* e = E_cpu.data<double>();
    EXPECT_NEAR(e[0], 1.0, 1e-12);
    EXPECT_NEAR(e[1], 1.0, 1e-12);
    EXPECT_NEAR(e[2], 0.0, 1e-12);
    EXPECT_NEAR(e[3], 1.0, 1e-12);
}

TEST_P(LinalgAdvancedTest, MatrixExpRotation) {
    // A = [[0, -theta], [theta, 0]] -> exp(A) = [[cos theta, -sin theta],
    //                                            [sin theta,  cos theta]]
    const double theta = 0.7;
    auto A_cpu = zeros({2, 2}, DType::Float64, Device::cpu());
    A_cpu.data<double>()[1] = -theta;
    A_cpu.data<double>()[2] =  theta;
    auto A = A_cpu.to(device);

    auto E = linalg::matrix_exp(A);
    auto E_cpu = E.cpu();
    const double* e = E_cpu.data<double>();
    EXPECT_NEAR(e[0],  std::cos(theta), 1e-10);
    EXPECT_NEAR(e[1], -std::sin(theta), 1e-10);
    EXPECT_NEAR(e[2],  std::sin(theta), 1e-10);
    EXPECT_NEAR(e[3],  std::cos(theta), 1e-10);
}

TEST_P(LinalgAdvancedTest, MatrixExpScalingBranch) {
    // Matrix with one-norm well above theta_13 ≈ 5.37 forces the
    // scaling-and-squaring branch to run. Verify exp(A) * exp(-A) == I.
    auto A_cpu = zeros({3, 3}, DType::Float64, Device::cpu());
    double data[9] = {3.0, -1.0, 0.5,
                      0.7,  2.0, -1.2,
                     -0.4,  0.8,  1.5};
    std::memcpy(A_cpu.data<double>(), data, sizeof(data));
    auto A = A_cpu.to(device);

    auto E = linalg::matrix_exp(A);

    auto negA_cpu = zeros({3, 3}, DType::Float64, Device::cpu());
    double ndata[9];
    for (int i = 0; i < 9; ++i) ndata[i] = -data[i];
    std::memcpy(negA_cpu.data<double>(), ndata, sizeof(ndata));
    auto negA = negA_cpu.to(device);
    auto Einv = linalg::matrix_exp(negA);

    auto I = matmul(E, Einv);
    auto I_cpu = I.cpu();
    const double* ip = I_cpu.data<double>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(ip[i * 3 + j], expected, 1e-9);
        }
    }
}

INSTANTIATE_BACKEND_TESTS(LinalgAdvancedTest);

} // namespace
} // namespace tenzor
