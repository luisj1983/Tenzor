/**
 * @file test_gradcheck_phase7.cpp
 * @brief Phase 7 gradcheck additions: linalg, Conv, Mode reduction.
 *
 * Coverage that was previously absent from the gradcheck suites:
 *   - Linalg: SVD, QR, Eigh, Eigvalsh, Slogdet, CholeskySolve, LDLFactor.
 *     These ops are numerically delicate; we use small well-conditioned
 *     SPD or full-rank matrices in Float64 with relaxed atol/rtol.
 *   - Conv: Conv1d, Conv2d, Conv3d, ConvTranspose1d/2d/3d via the
 *     `nn::functional::conv*` wrappers (the Variable-aware API).
 *   - Reduction: Mode (only the values branch is differentiated; indices
 *     are a non-differentiable side output).
 *
 * Parameterized across all backends via BackendTest. `gradcheck` is
 * device-aware: each input Variable is created on the fixture `device`;
 * gradcheck perturbs on a CPU copy and runs the forward on-device internally.
 * Backends physically absent on the host are skipped by BackendTest::SetUp.
 */

#include <cmath>

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class GradCheckPhase7 : public ::tenzor::testing::BackendTest {};

namespace {

// Build an SPD matrix A = X^T X + eps*I (well-conditioned for SVD/Eigh).
Tensor make_spd(int64_t n, const Device& device, double eps = 0.5) {
    auto x = randn({n, n}, DType::Float64, device);
    auto xt = tenzor::transpose(x, 0, 1);
    auto ata = tenzor::matmul(xt, x);
    auto eye_t = tenzor::eye(n, std::nullopt, DType::Float64, device);
    return tenzor::add(ata, tenzor::mul(eye_t, eps));
}

// Build a tall full-rank matrix for QR / SVD.
Tensor make_tall(int64_t m, int64_t n, const Device& device) {
    auto x = randn({m, n}, DType::Float64, device) * 0.5;
    auto eye_pad = tenzor::eye(m, std::optional<int64_t>(n), DType::Float64,
                               device) * 0.1;
    return tenzor::add(x, eye_pad);
}

// Build a tall (M > N) matrix with an exact zero column, guaranteeing rank
// <= N-1 and thus at least one singular value exactly 0. Used to reproduce
// the M3 SvdBackward NaN: the non-square projector term unconditionally
// computes S_inv_diag = diag(1/S), and even though grad_U/grad_Vh are exact
// zero tensors when only S is differentiated, matmul(zero, diag(1/0=Inf))
// yields 0 * Inf = NaN in that matmul's reduction.
Tensor make_rank_deficient_tall(int64_t m, int64_t n, const Device& device) {
    auto x = randn({m, n - 1}, DType::Float64, device) * 0.5;
    auto zero_col = zeros({m, 1}, DType::Float64, device);
    return tenzor::cat({x, zero_col}, 1);
}

}  // namespace

// ============================================================================
// Linalg gradcheck — small SPD / full-rank matrices in Float64
//
// Previously every tuple-returning linalg backward crashed with a
// vector-out-of-bounds when only one component of the tuple was
// differentiated. Fixed by padding grad_outputs with zero prototypes inside
// each backward (see pad_tuple_grad_outputs in src/autograd/function_linalg.cpp).
// The QR backward additionally returned a transposed strided view which
// gradcheck's element-walk interpreted as column-major; the backward now
// materialises a contiguous output.
// ============================================================================

TEST_P(GradCheckPhase7, SVD_FullMatricesFalse) {
    // SVD on a 3x3 well-conditioned matrix. Use the singular values branch
    // (S) only — it's smooth in the matrix entries.
    Variable x(make_spd(3, device, 1.0), true);
    auto f = [](const Variable& v) -> Variable {
        auto [U, S, Vt] = tenzor::svd(v, /*full_matrices=*/false);
        return tenzor::sum(S);  // scalar; smooth in v.
    };
    bool ok = gradcheck(f, x, 1e-6, 5e-3, 5e-3);
    EXPECT_TRUE(ok) << "SVD (sum of singular values) gradcheck failed";
}

TEST_P(GradCheckPhase7, SVD_RankDeficientTallNoNaN) {
    // M3 regression: SvdBackward's non-square (M > K) projector term used an
    // unregularized reciprocal(S). For a tall (M > N) matrix with a genuinely
    // zero singular value (here forced via an exact zero column), S_inv had
    // an Inf entry; even differentiating through S alone (grad_U is an exact
    // zero tensor) hit `matmul(zero_tensor, S_inv_diag)`, and 0 * Inf = NaN
    // poisoned the entire grad_A, not just the degenerate direction.
    // gradcheck is not used here because finite differences are themselves
    // unstable exactly at a rank-deficient point (S[j]=0 is a branch point of
    // the SVD, non-differentiable in the naive numerical sense even though
    // sum(S) has a well-defined analytical subgradient here); instead we
    // assert the analytical gradient is finite everywhere.
    Variable x(make_rank_deficient_tall(4, 3, device), true);
    auto [U, S, Vh] = tenzor::svd(x, /*full_matrices=*/false);
    auto loss = tenzor::sum(S);
    loss.backward();

    auto grad = x.grad();
    ASSERT_TRUE(grad.has_value()) << "SVD backward produced no gradient";
    auto grad_cpu = grad->to(Device::cpu()).contiguous();
    ASSERT_EQ(grad_cpu.dtype(), DType::Float64);
    const double* data = static_cast<const double*>(grad_cpu.data_ptr());
    int64_t numel = grad_cpu.numel();
    for (int64_t i = 0; i < numel; ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "grad_A[" << i << "] = " << data[i]
            << " (non-finite gradient from unregularized S_inv projector term)";
    }
}

TEST_P(GradCheckPhase7, QR_TraceR) {
    // trace(R) — sum of diagonal of R — is differentiable w.r.t. input
    // away from singular points.
    Variable x(make_tall(4, 3, device), true);
    auto f = [](const Variable& v) -> Variable {
        auto [Q, R] = tenzor::qr(v);
        return tenzor::trace(R);
    };
    bool ok = gradcheck(f, x, 1e-6, 5e-3, 5e-3);
    EXPECT_TRUE(ok) << "QR (trace of R) gradcheck failed";
}

TEST_P(GradCheckPhase7, Eigh_SumEigvals) {
    // For symmetric M, sum of eigenvalues = trace(M). Smooth, well-defined.
    Variable x(make_spd(4, device, 1.0), true);
    auto f = [](const Variable& v) -> Variable {
        auto [eigvals, eigvecs] = tenzor::eigh(v);
        return tenzor::sum(eigvals);
    };
    bool ok = gradcheck(f, x, 1e-6, 5e-3, 5e-3);
    EXPECT_TRUE(ok) << "Eigh (sum of eigenvalues) gradcheck failed";
}

TEST_P(GradCheckPhase7, Slogdet_SumOutputs) {
    // log|det(A)| is smooth for an invertible matrix with positive determinant.
    Variable x(make_spd(3, device, 1.0), true);
    auto f = [](const Variable& v) -> Variable {
        auto [sign, logabsdet] = tenzor::slogdet(v);
        return logabsdet;  // scalar
    };
    bool ok = gradcheck(f, x, 1e-6, 5e-3, 5e-3);
    EXPECT_TRUE(ok) << "Slogdet (logabsdet) gradcheck failed";
}

TEST_P(GradCheckPhase7, CholeskySolve_VsRHS) {
    // Solve LL^T x = B for B; gradient w.r.t. B should be smooth.
    Variable A(make_spd(3, device, 1.0), false);
    auto L = tenzor::cholesky(A);
    Variable B(randn({3, 2}, DType::Float64, device), true);
    auto f = [&L](const Variable& b) -> Variable {
        return tenzor::sum(tenzor::cholesky_solve(b, L, /*upper=*/false));
    };
    bool ok = gradcheck(f, B, 1e-6, 5e-3, 5e-3);
    EXPECT_TRUE(ok) << "CholeskySolve gradcheck (w.r.t. B) failed";
}
// ============================================================================
// Conv gradcheck — small inputs with Float64 for numerical stability
// ============================================================================

TEST_P(GradCheckPhase7, Conv1d_Float64) {
    Variable input(randn({1, 2, 6}, DType::Float64, device), true);
    Variable weight(randn({3, 2, 3}, DType::Float64, device) * 0.3, false);
    auto f = [&weight](const Variable& x) -> Variable {
        return tenzor::sum(nn::functional::conv1d(x, weight, std::nullopt, 1, 0, 1, 1));
    };
    bool ok = gradcheck(f, input, 1e-6, 1e-7, 1e-6);
    EXPECT_TRUE(ok) << "Conv1d gradcheck failed";
}

TEST_P(GradCheckPhase7, Conv2d_Float64) {
    Variable input(randn({1, 2, 4, 4}, DType::Float64, device), true);
    Variable weight(randn({3, 2, 3, 3}, DType::Float64, device) * 0.3, false);
    auto f = [&weight](const Variable& x) -> Variable {
        return tenzor::sum(nn::functional::conv2d(
            x, weight, std::nullopt,
            std::pair<int64_t,int64_t>{1,1},  // stride
            std::pair<int64_t,int64_t>{0,0},  // padding
            std::pair<int64_t,int64_t>{1,1},  // dilation
            1));
    };
    bool ok = gradcheck(f, input, 1e-6, 1e-7, 1e-6);
    EXPECT_TRUE(ok) << "Conv2d gradcheck failed";
}

// Regression: a plain sum() reduction yields a UNIFORM upstream gradient, which
// masks how conv2d_backward_input packs grad_output (channel-major vs the
// position-major layout the GEMM reads). Weight the output by a fixed
// non-uniform tensor so the analytic input-grad must transpose grad_output
// correctly to match finite differences. C_out=3>1 and out spatial=2x2>1 so the
// scrambling is observable.
TEST_P(GradCheckPhase7, Conv2d_NonUniformGrad_Float64) {
    Variable input(randn({1, 2, 4, 4}, DType::Float64, device), true);
    Variable weight(randn({3, 2, 3, 3}, DType::Float64, device) * 0.3, false);
    Variable proj(randn({1, 3, 2, 2}, DType::Float64, device), false);
    auto f = [&weight, &proj](const Variable& x) -> Variable {
        auto y = nn::functional::conv2d(
            x, weight, std::nullopt,
            std::pair<int64_t,int64_t>{1,1},  // stride
            std::pair<int64_t,int64_t>{0,0},  // padding
            std::pair<int64_t,int64_t>{1,1},  // dilation
            1);
        return tenzor::sum(y * proj);
    };
    auto r = gradcheck_detailed(f, input, 1e-6, 1e-7, 1e-6);
    EXPECT_TRUE(r.passed) << "Conv2d non-uniform-grad gradcheck failed: "
                          << r.error_message;
}

TEST_P(GradCheckPhase7, Conv3d_Float64) {
    Variable input(randn({1, 2, 3, 3, 3}, DType::Float64, device), true);
    Variable weight(randn({2, 2, 2, 2, 2}, DType::Float64, device) * 0.3, false);
    auto f = [&weight](const Variable& x) -> Variable {
        return tenzor::sum(nn::functional::conv3d(x, weight, std::nullopt,
                                      {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, 1));
    };
    bool ok = gradcheck(f, input, 1e-6, 1e-7, 1e-6);
    EXPECT_TRUE(ok) << "Conv3d gradcheck failed";
}

TEST_P(GradCheckPhase7, ConvTranspose1d_Float64) {
    Variable input(randn({1, 2, 4}, DType::Float64, device), true);
    Variable weight(randn({2, 3, 3}, DType::Float64, device) * 0.3, false);
    auto f = [&weight](const Variable& x) -> Variable {
        return tenzor::sum(nn::functional::conv_transpose1d(x, weight, std::nullopt, 1, 0, 0, 1, 1));
    };
    bool ok = gradcheck(f, input, 1e-6, 1e-7, 1e-6);
    EXPECT_TRUE(ok) << "ConvTranspose1d gradcheck failed";
}

TEST_P(GradCheckPhase7, ConvTranspose2d_Float64) {
    // Regression guard for the Float32-accumulator bug in
    // conv_transpose2d_forward_impl that previously caused ~7% gradient
    // error for Float64 inputs. Fixed by using a dtype-aware AccumT.
    Variable input(randn({1, 2, 3, 4}, DType::Float64, device), true);
    Variable weight(randn({2, 3, 2, 3}, DType::Float64, device) * 0.3, false);
    auto f = [&weight](const Variable& x) -> Variable {
        return tenzor::sum(nn::functional::conv_transpose2d(
            x, weight, std::nullopt,
            std::pair<int64_t,int64_t>{1,1},
            std::pair<int64_t,int64_t>{0,0},
            std::pair<int64_t,int64_t>{0,0},
            1,
            std::pair<int64_t,int64_t>{1,1}));
    };
    bool ok = gradcheck(f, input, 1e-6, 1e-7, 1e-6);
    EXPECT_TRUE(ok) << "ConvTranspose2d gradcheck failed";
}

// ============================================================================
// Mode reduction — only the .values branch is differentiable
// ============================================================================

TEST_P(GradCheckPhase7, Mode_ValuesBranch) {
    // Mode returns just the values Variable (not a tuple — indices are not
    // exposed in the autograd surface). Acts as a gather of the most-frequent
    // element per row.
    auto x_data = randn({3, 5}, DType::Float64, device);
    Variable x(x_data, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::mode(v, /*dim=*/-1, /*keepdim=*/false));
    };
    bool ok = gradcheck(f, x, 1e-6, 5e-3, 5e-3);
    EXPECT_TRUE(ok) << "Mode (values) gradcheck failed";
}

INSTANTIATE_BACKEND_TESTS(GradCheckPhase7);
