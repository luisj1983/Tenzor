/**
 * @file test_jvp_rules.cpp
 * @brief Per-op JVP correctness vs central-difference reference.
 *
 * The audit (2026-05-02) flagged jvp_rules.hpp as having only ~4 file
 * references. This file pins each documented JVP rule against a finite-
 * difference baseline:
 *     JVP(f)(x; v) ≈ (f(x + ε·v) - f(x - ε·v)) / (2ε)
 *
 * Tested rules: jvp_add, jvp_sub, jvp_mul, jvp_div, jvp_neg, jvp_matmul,
 * jvp_relu, jvp_sigmoid, jvp_tanh, jvp_exp, jvp_log, jvp_sqrt.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/dual.hpp>
#include <tenzor/autograd/jvp_rules.hpp>
#include <cmath>
#include "../backend_test_fixture.hpp"

using namespace tenzor;

namespace {

// Compute max absolute element-wise difference between two tensors,
// promoting to Float64 for the comparison.
double max_abs_diff_d(const Tensor& a, const Tensor& b) {
    auto ad = a.cpu().to(DType::Float64).contiguous();
    auto bd = b.cpu().to(DType::Float64).contiguous();
    const double* ap = ad.data<double>();
    const double* bp = bd.data<double>();
    double m = 0.0;
    for (int64_t i = 0; i < ad.numel(); ++i) {
        m = std::max(m, std::abs(ap[i] - bp[i]));
    }
    return m;
}

// Central-difference estimate of f'(x) along direction v, using Float64
// internally to keep the FD epsilon out of the dominant error budget.
template <typename Op>
Tensor finite_diff_jvp(Op op, const Tensor& x, const Tensor& v, double eps = 1e-5) {
    auto x64 = x.to(DType::Float64);
    auto v64 = v.to(DType::Float64);
    auto plus  = op(x64 + v64 * eps);
    auto minus = op(x64 - v64 * eps);
    return ((plus - minus) * (1.0 / (2.0 * eps))).to(x.dtype());
}

}  // namespace

class JVPRulesTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ----------------------------------------------------------------------------
// Binary linear ops (exact JVPs — should match FD up to machine epsilon)
// ----------------------------------------------------------------------------

TEST_P(JVPRulesTest, JVP_Add_MatchesFD) {
    auto a = randn({4, 8}, DType::Float32, device);
    auto b = randn({4, 8}, DType::Float32, device);
    auto va = randn({4, 8}, DType::Float32, device);
    auto vb = randn({4, 8}, DType::Float32, device);

    auto out = jvp_add(DualTensor(a, va), DualTensor(b, vb));
    // d/dt (a + t·va) + (b + t·vb) at t=0 = va + vb.
    EXPECT_LT(max_abs_diff_d(out.tangent(), va + vb), 1e-5);
}

TEST_P(JVPRulesTest, JVP_Sub_MatchesFD) {
    auto a = randn({3, 5}, DType::Float32, device);
    auto b = randn({3, 5}, DType::Float32, device);
    auto va = randn({3, 5}, DType::Float32, device);
    auto vb = randn({3, 5}, DType::Float32, device);

    auto out = jvp_sub(DualTensor(a, va), DualTensor(b, vb));
    EXPECT_LT(max_abs_diff_d(out.tangent(), va - vb), 1e-5);
}

TEST_P(JVPRulesTest, JVP_Mul_MatchesProductRule) {
    auto a = randn({4}, DType::Float32, device);
    auto b = randn({4}, DType::Float32, device);
    auto va = randn({4}, DType::Float32, device);
    auto vb = randn({4}, DType::Float32, device);

    auto out = jvp_mul(DualTensor(a, va), DualTensor(b, vb));
    // Product rule: d/dt (a + t·va)(b + t·vb)|t=0 = va*b + a*vb.
    auto expected = va * b + a * vb;
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-5);
}

TEST_P(JVPRulesTest, JVP_Div_MatchesQuotientRule) {
    auto a = randn({4}, DType::Float32, device);
    // Keep b away from 0.
    auto b = abs(randn({4}, DType::Float32, device)) + 0.5f;
    auto va = randn({4}, DType::Float32, device);
    auto vb = randn({4}, DType::Float32, device);

    auto out = jvp_div(DualTensor(a, va), DualTensor(b, vb));
    // Quotient rule: (va*b - a*vb) / b^2
    auto expected = (va * b - a * vb) / (b * b);
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-4);
}

TEST_P(JVPRulesTest, JVP_Neg_FlipsTangent) {
    auto a = randn({3}, DType::Float32, device);
    auto va = randn({3}, DType::Float32, device);

    auto out = jvp_neg(DualTensor(a, va));
    // -(a + t·va) ⇒ tangent = -va.
    auto expected = va * -1.0f;
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-6);
}

// ----------------------------------------------------------------------------
// MatMul: tangent must satisfy d/dt (A + t Va)(B + t Vb)|t=0 = Va·B + A·Vb
// ----------------------------------------------------------------------------

TEST_P(JVPRulesTest, JVP_MatMul_MatchesProductRule) {
    auto A  = randn({3, 4}, DType::Float32, device);
    auto B  = randn({4, 5}, DType::Float32, device);
    auto Va = randn({3, 4}, DType::Float32, device);
    auto Vb = randn({4, 5}, DType::Float32, device);

    auto out = jvp_matmul(DualTensor(A, Va), DualTensor(B, Vb));
    auto expected = matmul(Va, B) + matmul(A, Vb);
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-4);
}

// ----------------------------------------------------------------------------
// Unary nonlinearities — verify tangent ≈ FD approximation.
// ----------------------------------------------------------------------------

TEST_P(JVPRulesTest, JVP_Sigmoid_MatchesFD) {
    auto x = randn({16}, DType::Float32, device);
    auto v = randn({16}, DType::Float32, device);
    auto fd = finite_diff_jvp([](const Tensor& t) { return sigmoid(t); }, x, v);
    auto out = jvp_sigmoid(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}

TEST_P(JVPRulesTest, JVP_Tanh_MatchesFD) {
    auto x = randn({16}, DType::Float32, device);
    auto v = randn({16}, DType::Float32, device);
    auto fd = finite_diff_jvp([](const Tensor& t) { return tanh(t); }, x, v);
    auto out = jvp_tanh(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}

TEST_P(JVPRulesTest, JVP_Exp_MatchesFD) {
    // Restrict input range so exp doesn't overflow Float32 in the FD step.
    auto x = randn({16}, DType::Float32, device) * 0.5f;
    auto v = randn({16}, DType::Float32, device);
    auto fd = finite_diff_jvp([](const Tensor& t) { return exp(t); }, x, v);
    auto out = jvp_exp(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}

TEST_P(JVPRulesTest, JVP_Log_MatchesFD) {
    auto x = abs(randn({16}, DType::Float32, device)) + 0.1f;
    auto v = randn({16}, DType::Float32, device);
    auto fd = finite_diff_jvp([](const Tensor& t) { return log(t); }, x, v);
    auto out = jvp_log(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-2);
}

TEST_P(JVPRulesTest, JVP_Sqrt_MatchesFD) {
    auto x = abs(randn({16}, DType::Float32, device)) + 0.1f;
    auto v = randn({16}, DType::Float32, device);
    auto fd = finite_diff_jvp([](const Tensor& t) { return sqrt(t); }, x, v);
    auto out = jvp_sqrt(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-2);
}

TEST_P(JVPRulesTest, JVP_ReLU_MatchesFD_FarFromZero) {
    // ReLU is non-differentiable at 0; pick inputs strictly away from
    // the kink so finite differences are well-defined.
    auto x = randn({16}, DType::Float32, device) + 2.0f;  // mostly > 0
    auto v = randn({16}, DType::Float32, device);
    auto relu_t = [](const Tensor& t) { return clamp_min(t, 0.0); };
    auto fd = finite_diff_jvp(relu_t, x, v);
    auto out = jvp_relu(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}

// ============================================================================
// S15: JVP rules previously marked NonDifferentiable, now implemented via
// closed-form chain rule / linear-forward / FD-probe.
//
// Each test compares the registered JVP rule (via dispatch_jvp) against a
// central-difference reference computed by running the forward op at
// (x + eps*dx) and (x - eps*dx). The tolerance is ~1e-3 (FD-limited).
// ============================================================================

#include <tenzor/autograd/jvp_dispatch.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

namespace {

// Helper: compute finite-difference JVP from a callable f that maps Tensor -> Tensor.
template <typename Op>
Tensor fd_jvp(Op op, const Tensor& x, const Tensor& dx, double eps = 1e-3) {
    auto plus  = op(x + dx * eps);
    auto minus = op(x - dx * eps);
    return (plus - minus) * (1.0 / (2.0 * eps));
}

// Helper: compute FD JVP for two-arg op f(x, y).
template <typename Op>
Tensor fd_jvp2(Op op, const Tensor& x, const Tensor& dx,
               const Tensor& y, const Tensor& dy, double eps = 1e-3) {
    auto plus  = op(x + dx * eps, y + dy * eps);
    auto minus = op(x - dx * eps, y - dy * eps);
    return (plus - minus) * (1.0 / (2.0 * eps));
}

}  // namespace

// ---- DCT JVP ----
TEST_P(JVPRulesTest, JVP_DCT_S15_LinearMatchesFD) {
    auto x  = randn({4, 16}, DType::Float64, device);
    auto dx = randn({4, 16}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::DCTType, int64_t{2});
    attrs.set(AttrKey::Dim, int64_t{-1});
    attrs.set(AttrKey::Norm, std::string("ortho"));
    std::array<Tensor, 1> p{x}, t{dx};
    auto out = tenzor::dispatch_jvp(OpId::DCT, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& xx) {
        return tenzor::dispatch(OpId::DCT, std::vector<Tensor>{xx}, attrs)[0];
    }, x, dx);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-2);
}

// ---- IDCT JVP ----
TEST_P(JVPRulesTest, JVP_IDCT_S15_LinearMatchesFD) {
    auto x  = randn({4, 16}, DType::Float64, device);
    auto dx = randn({4, 16}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::DCTType, int64_t{2});
    attrs.set(AttrKey::Dim, int64_t{-1});
    attrs.set(AttrKey::Norm, std::string("ortho"));
    std::array<Tensor, 1> p{x}, t{dx};
    auto out = tenzor::dispatch_jvp(OpId::IDCT, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& xx) {
        return tenzor::dispatch(OpId::IDCT, std::vector<Tensor>{xx}, attrs)[0];
    }, x, dx);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-2);
}

// ---- LinalgVectorNorm JVP (p=2) ----
TEST_P(JVPRulesTest, JVP_LinalgVectorNorm_P2_MatchesFD) {
    auto x  = randn({4, 8}, DType::Float64, device) + 0.5;
    auto dx = randn({4, 8}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::P, 2.0);
    attrs.set(AttrKey::Dim, int64_t{-1});
    attrs.set(AttrKey::Keepdim, false);
    std::array<Tensor, 1> p{x}, t{dx};
    auto out = tenzor::dispatch_jvp(OpId::LinalgVectorNorm, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& xx) {
        return tenzor::dispatch(OpId::LinalgVectorNorm,
            std::vector<Tensor>{xx}, attrs)[0];
    }, x, dx);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-3);
}

// ---- LinalgMatrixNorm Frobenius JVP ----
TEST_P(JVPRulesTest, JVP_LinalgMatrixNorm_Fro_MatchesFD) {
    auto x  = randn({4, 6}, DType::Float64, device) + 0.5;
    auto dx = randn({4, 6}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::Order, 0.0);  // Frobenius
    std::array<Tensor, 1> p{x}, t{dx};
    auto out = tenzor::dispatch_jvp(OpId::LinalgMatrixNorm, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& xx) {
        return tenzor::dispatch(OpId::LinalgMatrixNorm,
            std::vector<Tensor>{xx}, attrs)[0];
    }, x, dx);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-3);
}

// ---- CosineSimilarity JVP ----
TEST_P(JVPRulesTest, JVP_CosineSimilarity_MatchesFD) {
    auto a  = randn({4, 8}, DType::Float64, device) + 1.0;
    auto b  = randn({4, 8}, DType::Float64, device) + 1.0;
    auto da = randn({4, 8}, DType::Float64, device);
    auto db = randn({4, 8}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, int64_t{1});
    attrs.set(AttrKey::Eps, 1e-8);
    std::array<Tensor, 2> p{a, b}, t{da, db};
    auto out = tenzor::dispatch_jvp(OpId::CosineSimilarity, p, t, attrs);
    auto fd = fd_jvp2([&](const Tensor& xx, const Tensor& yy) {
        return tenzor::dispatch(OpId::CosineSimilarity,
            std::vector<Tensor>{xx, yy}, attrs)[0];
    }, a, da, b, db);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-3);
}

// ---- CDist JVP (p=2) ----
TEST_P(JVPRulesTest, JVP_CDist_P2_MatchesFD) {
    auto x1  = randn({3, 4}, DType::Float64, device) + 1.0;
    auto x2  = randn({5, 4}, DType::Float64, device) + 1.0;
    auto dx1 = randn({3, 4}, DType::Float64, device);
    auto dx2 = randn({5, 4}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::DistP, 2.0);
    std::array<Tensor, 2> p{x1, x2}, t{dx1, dx2};
    auto out = tenzor::dispatch_jvp(OpId::CDist, p, t, attrs);
    auto fd = fd_jvp2([&](const Tensor& xx, const Tensor& yy) {
        return tenzor::dispatch(OpId::CDist,
            std::vector<Tensor>{xx, yy}, attrs)[0];
    }, x1, dx1, x2, dx2);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-2);
}

// ---- PairwiseDistance JVP (p=2) ----
TEST_P(JVPRulesTest, JVP_PairwiseDistance_P2_MatchesFD) {
    auto x1  = randn({6, 4}, DType::Float64, device) + 1.0;
    auto x2  = randn({6, 4}, DType::Float64, device) + 1.0;
    auto dx1 = randn({6, 4}, DType::Float64, device);
    auto dx2 = randn({6, 4}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::DistP, 2.0);
    std::array<Tensor, 2> p{x1, x2}, t{dx1, dx2};
    auto out = tenzor::dispatch_jvp(OpId::PairwiseDistance, p, t, attrs);
    auto fd = fd_jvp2([&](const Tensor& xx, const Tensor& yy) {
        return tenzor::dispatch(OpId::PairwiseDistance,
            std::vector<Tensor>{xx, yy}, attrs)[0];
    }, x1, dx1, x2, dx2);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-2);
}

// ---- Cov JVP ----
TEST_P(JVPRulesTest, JVP_Cov_MatchesFD) {
    auto X  = randn({4, 8}, DType::Float64, device);
    auto dX = randn({4, 8}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::Correction, int64_t{1});
    std::array<Tensor, 1> p{X}, t{dX};
    auto out = tenzor::dispatch_jvp(OpId::Cov, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& xx) {
        return tenzor::dispatch(OpId::Cov,
            std::vector<Tensor>{xx}, attrs)[0];
    }, X, dX);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-2);
}

// ---- Corrcoef JVP ----
TEST_P(JVPRulesTest, JVP_Corrcoef_MatchesFD) {
    auto X  = randn({4, 16}, DType::Float64, device) + 1.0;
    auto dX = randn({4, 16}, DType::Float64, device) * 0.1;
    OpAttributes attrs;
    std::array<Tensor, 1> p{X}, t{dX};
    auto out = tenzor::dispatch_jvp(OpId::Corrcoef, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& xx) {
        return tenzor::dispatch(OpId::Corrcoef,
            std::vector<Tensor>{xx}, attrs)[0];
    }, X, dX);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 5e-2);
}

// ---- SparseToDense JVP ----
TEST_P(JVPRulesTest, JVP_SparseToDense_S15_LinearMatchesFD) {
    int64_t M = 4, K = 5;
    std::vector<int64_t> crow_d = {0, 2, 3, 5, 6};
    std::vector<int64_t> col_d = {0, 2, 1, 3, 4, 0};
    auto crow = Tensor::from_blob(crow_d.data(), {M + 1}, DType::Int64, Device::cpu())
                    .clone().to(device);
    auto col  = Tensor::from_blob(col_d.data(), {static_cast<int64_t>(col_d.size())},
                                   DType::Int64, Device::cpu()).clone().to(device);
    auto values  = randn({static_cast<int64_t>(col_d.size())}, DType::Float64, device);
    auto dvalues = randn({static_cast<int64_t>(col_d.size())}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::M, M);
    attrs.set(AttrKey::K, K);
    std::array<Tensor, 3> p{crow, col, values};
    std::array<Tensor, 3> t{Tensor(), Tensor(), dvalues};
    auto out = tenzor::dispatch_jvp(OpId::SparseToDense, p, t, attrs);
    // FD reference: change values, observe dense output diff.
    double eps = 1e-3;
    auto plus  = tenzor::dispatch(OpId::SparseToDense,
        std::vector<Tensor>{crow, col, values + dvalues * eps}, attrs)[0];
    auto minus = tenzor::dispatch(OpId::SparseToDense,
        std::vector<Tensor>{crow, col, values - dvalues * eps}, attrs)[0];
    auto fd = (plus - minus) * (1.0 / (2.0 * eps));
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-3);
}

INSTANTIATE_BACKEND_TESTS(JVPRulesTest);

