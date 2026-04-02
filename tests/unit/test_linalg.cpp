#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::testing;

class LinalgTest : public BackendTest {};

namespace {

// Helper: create a tensor from raw float data on CPU, then move to target device
Tensor make_matrix(std::vector<int64_t> shape, const float* data, Device dev) {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    int64_t n = 1;
    for (auto s : shape) n *= s;
    std::memcpy(t.data<float>(), data, n * sizeof(float));
    return t.to(dev);
}

} // anonymous namespace

// ============================================================================
// Determinant
// ============================================================================

TEST_P(LinalgTest, Det2x2) {
    // [[1, 2], [3, 4]] -> det = 1*4 - 2*3 = -2
    float data[] = {1, 2, 3, 4};
    auto A = make_matrix({2, 2}, data, device);

    auto d = linalg::det(A);
    auto d_cpu = d.to(Device::cpu());
    EXPECT_NEAR(d_cpu.data<float>()[0], -2.0f, 1e-4f);
}

TEST_P(LinalgTest, DetIdentity) {
    auto I = eye(4, std::nullopt, DType::Float32, device);
    auto d = linalg::det(I);
    auto d_cpu = d.to(Device::cpu());
    auto* dp = d_cpu.data<float>();
    EXPECT_NEAR(dp[0], 1.0f, 1e-4f);
}

// ============================================================================
// Inverse: A * inv(A) ≈ I
// ============================================================================

TEST_P(LinalgTest, Inv3x3) {
    float data[] = {2, 1, 0, 0, 3, 1, 1, 0, 2};
    auto A = make_matrix({3, 3}, data, device);

    auto Ainv = linalg::inv(A);
    auto I_approx = matmul(A, Ainv);

    auto I_cpu = I_approx.to(Device::cpu());
    auto* p = I_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(p[i * 3 + j], expected, 1e-4f)
                << "i=" << i << " j=" << j;
        }
    }
}

// ============================================================================
// Solve: A x = B
// ============================================================================

TEST_P(LinalgTest, Solve2x2) {
    // A = [[2, 1], [1, 3]], B = [5, 10] -> x = [1, 3]
    float a[] = {2, 1, 1, 3};
    float b[] = {5, 10};
    auto A = make_matrix({2, 2}, a, device);
    auto B = make_matrix({2, 1}, b, device);
    auto x = linalg::solve(A, B);

    auto x_cpu = x.to(Device::cpu());
    auto* xp = x_cpu.data<float>();
    EXPECT_NEAR(xp[0], 1.0f, 1e-4f);
    EXPECT_NEAR(xp[1], 3.0f, 1e-4f);
}

// ============================================================================
// Cholesky: L * L^T = A for SPD matrix
// ============================================================================

TEST_P(LinalgTest, Cholesky3x3) {
    // SPD matrix: [[4, 2, 0], [2, 5, 1], [0, 1, 3]]
    float data[] = {4, 2, 0, 2, 5, 1, 0, 1, 3};
    auto A = make_matrix({3, 3}, data, device);

    auto L = linalg::cholesky(A, false);

    // Verify L * L^T ≈ A
    auto Lt = transpose(L, 0, 1);
    auto A_reconstructed = matmul(L, Lt);

    auto A_rec_cpu = A_reconstructed.to(Device::cpu());
    auto A_orig_cpu = A.to(Device::cpu());
    auto* rp = A_rec_cpu.data<float>();
    auto* op = A_orig_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(rp[i], op[i], 1e-4f) << "index=" << i;
    }
}

// ============================================================================
// QR: A = Q * R, Q^T * Q ≈ I
// ============================================================================

TEST_P(LinalgTest, QR3x3) {
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    auto A = make_matrix({3, 3}, data, device);

    auto [Q, R] = linalg::qr(A);

    // Verify Q * R ≈ A
    auto A_rec = matmul(Q, R);
    auto A_rec_cpu = A_rec.to(Device::cpu());
    auto A_orig_cpu = A.to(Device::cpu());
    auto* rp = A_rec_cpu.data<float>();
    auto* op = A_orig_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(rp[i], op[i], 1e-3f) << "index=" << i;
    }

    // Verify Q^T * Q ≈ I
    auto Qt = transpose(Q, 0, 1);
    auto QtQ = matmul(Qt, Q);
    auto QtQ_cpu = QtQ.to(Device::cpu());
    auto* ip = QtQ_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(ip[i * 3 + j], expected, 1e-3f)
                << "i=" << i << " j=" << j;
        }
    }
}

// ============================================================================
// SVD: singular values of diagonal matrix
// ============================================================================

TEST_P(LinalgTest, SVD3x3) {
    float data[] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    auto A = make_matrix({3, 3}, data, device);

    auto [U, S, Vt] = linalg::svd(A, true);

    // Singular values should be {3, 2, 1} (sorted descending)
    auto S_cpu = S.to(Device::cpu());
    auto* sp = S_cpu.data<float>();
    EXPECT_NEAR(sp[0], 3.0f, 1e-3f);
    EXPECT_NEAR(sp[1], 2.0f, 1e-3f);
    EXPECT_NEAR(sp[2], 1.0f, 1e-3f);
}

// ============================================================================
// Eigh: symmetric eigendecomposition
// ============================================================================

TEST_P(LinalgTest, Eigh2x2) {
    // Symmetric matrix [[2, 1], [1, 2]] -> eigenvalues {1, 3}
    float data[] = {2, 1, 1, 2};
    auto A = make_matrix({2, 2}, data, device);

    auto [W, V] = linalg::eigh(A);

    auto W_cpu = W.to(Device::cpu());
    auto* wp = W_cpu.data<float>();
    // Eigenvalues should be 1 and 3 (ascending order)
    EXPECT_NEAR(wp[0], 1.0f, 1e-3f);
    EXPECT_NEAR(wp[1], 3.0f, 1e-3f);
}

// ============================================================================
// Eig: general eigendecomposition
// ============================================================================

TEST_P(LinalgTest, EigDiagonal) {
    // Diagonal matrix: eigenvalues on the diagonal
    float data[] = {5, 0, 0, 0, 3, 0, 0, 0, 1};
    auto A = make_matrix({3, 3}, data, device);

    auto [WR, WI, V] = linalg::eig(A);

    auto WR_cpu = WR.to(Device::cpu());
    auto WI_cpu = WI.to(Device::cpu());
    auto* wr = WR_cpu.data<float>();
    auto* wi = WI_cpu.data<float>();

    // Sort eigenvalues for comparison (order may vary)
    std::vector<float> evals = {wr[0], wr[1], wr[2]};
    std::sort(evals.begin(), evals.end());

    EXPECT_NEAR(evals[0], 1.0f, 1e-3f);
    EXPECT_NEAR(evals[1], 3.0f, 1e-3f);
    EXPECT_NEAR(evals[2], 5.0f, 1e-3f);

    // Imaginary parts should be zero for real eigenvalues
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(wi[i], 0.0f, 1e-3f);
    }
}

// ============================================================================
// Batched determinant
// ============================================================================

TEST_P(LinalgTest, DetBatched) {
    // Batch of 2 matrices: I and [[1,2],[3,4]]
    float data[] = {
        1, 0, 0, 1,   // I -> det = 1
        1, 2, 3, 4    // -> det = -2
    };
    auto A = make_matrix({2, 2, 2}, data, device);

    auto d = linalg::det(A);
    auto d_cpu = d.to(Device::cpu());
    auto* dp = d_cpu.data<float>();
    EXPECT_NEAR(dp[0], 1.0f, 1e-4f);
    EXPECT_NEAR(dp[1], -2.0f, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(
    CPU, LinalgTest,
    ::testing::Values("cpu"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);
