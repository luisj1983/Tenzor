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

// ---------------------------------------------------------------------------
// Wave-4 multi-output JVP rules: GroupNorm / InstanceNorm / RMSNorm / Chunk /
// Split / TopK / Sort / BatchNorm2dForward. Each compares the analytic
// forward-mode tangent of output `out_idx` against a central finite-difference
// reference, in Float64, summing the contribution of every differentiable
// primal.
// ---------------------------------------------------------------------------
namespace {

Tensor w4_randf64(const std::vector<int64_t>& shape, uint64_t seed) {
    Tensor t(shape, DType::Float64, Device::cpu());
    double* d = t.data<double>();
    int64_t n = t.numel();
    uint64_t s = seed * 2654435761u + 12345u;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        double u = static_cast<double>((s >> 11) & ((1ull << 53) - 1)) /
                   static_cast<double>(1ull << 53);
        d[i] = 2.0 * u - 1.0;
    }
    return t;
}

void w4_verify(OpId op, std::vector<Tensor> primals, std::vector<int> diff_primals,
               int out_idx, const OpAttributes& attrs, double tol) {
    std::vector<Tensor> tangents(primals.size());
    for (size_t i = 0; i < primals.size(); ++i) {
        tangents[i] = zeros(std::vector<int64_t>(primals[i].shape().begin(),
                                                 primals[i].shape().end()),
                            DType::Float64, Device::cpu());
    }
    for (int dp : diff_primals) {
        tangents[dp] = w4_randf64(std::vector<int64_t>(primals[dp].shape().begin(),
                                                       primals[dp].shape().end()),
                                  1000 + dp);
    }
    auto analytic = tenzor::dispatch_jvp_multi(op, primals, tangents, attrs);
    Tensor an = analytic.tangents.at(out_idx);
    Tensor num = zeros(std::vector<int64_t>(an.shape().begin(), an.shape().end()),
                       DType::Float64, Device::cpu());
    const double h = 1e-6;
    for (int dp : diff_primals) {
        auto pp = primals, pm = primals;
        pp[dp] = primals[dp] + tangents[dp] * h;
        pm[dp] = primals[dp] - tangents[dp] * h;
        auto yp = tenzor::dispatch(op, pp, attrs)[out_idx];
        auto ym = tenzor::dispatch(op, pm, attrs)[out_idx];
        num = num + (yp - ym) * (1.0 / (2.0 * h));
    }
    EXPECT_LT(max_abs_diff_d(an, num), tol);
}

// Single-output variant (dispatch_jvp / JvpResult.tangent) for ops whose JVP
// rule is registered via register_jvp_rule rather than register_jvp_rule_multi.
void w4_verify_single(OpId op, std::vector<Tensor> primals, std::vector<int> diff_primals,
                      const OpAttributes& attrs, double tol) {
    std::vector<Tensor> tangents(primals.size());
    for (size_t i = 0; i < primals.size(); ++i) {
        tangents[i] = zeros(std::vector<int64_t>(primals[i].shape().begin(),
                                                 primals[i].shape().end()),
                            DType::Float64, Device::cpu());
    }
    for (int dp : diff_primals) {
        tangents[dp] = w4_randf64(std::vector<int64_t>(primals[dp].shape().begin(),
                                                       primals[dp].shape().end()),
                                  2000 + dp);
    }
    auto analytic = tenzor::dispatch_jvp(op, primals, tangents, attrs);
    Tensor an = analytic.tangent;
    Tensor num = zeros(std::vector<int64_t>(an.shape().begin(), an.shape().end()),
                       DType::Float64, Device::cpu());
    const double h = 1e-6;
    for (int dp : diff_primals) {
        auto pp = primals, pm = primals;
        pp[dp] = primals[dp] + tangents[dp] * h;
        pm[dp] = primals[dp] - tangents[dp] * h;
        auto yp = tenzor::dispatch(op, pp, attrs)[0];
        auto ym = tenzor::dispatch(op, pm, attrs)[0];
        num = num + (yp - ym) * (1.0 / (2.0 * h));
    }
    EXPECT_LT(max_abs_diff_d(an, num), tol);
}

}  // namespace

TEST_P(JVPRulesTest, GroupNorm_JVP_MatchesFD) {
    if (device != Device::cpu()) return;  // Float64 reference is CPU-side.
    OpAttributes a;
    a.set(AttrKey::NumGroups, static_cast<int64_t>(2));
    a.set(AttrKey::Eps, 1e-5);
    w4_verify(OpId::GroupNorm,
              { w4_randf64({2,4,3,3},1), w4_randf64({4},2), w4_randf64({4},3) },
              {0,1,2}, 0, a, 1e-5);
}

TEST_P(JVPRulesTest, InstanceNorm_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Eps, 1e-5);
    w4_verify(OpId::InstanceNorm,
              { w4_randf64({2,4,3,3},4), w4_randf64({4},5), w4_randf64({4},6) },
              {0,1,2}, 0, a, 1e-5);
}

TEST_P(JVPRulesTest, RMSNorm_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Eps, 1e-5);
    w4_verify(OpId::RMSNorm, { w4_randf64({2,8},7), w4_randf64({8},8) }, {0,1}, 0, a, 1e-5);
}

TEST_P(JVPRulesTest, Chunk_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Chunks, static_cast<int64_t>(3));
    a.set(AttrKey::Dim, static_cast<int64_t>(0));
    std::vector<Tensor> p = { w4_randf64({6,4},9) };
    for (int oi = 0; oi < 3; ++oi) w4_verify(OpId::Chunk, p, {0}, oi, a, 1e-7);
}

TEST_P(JVPRulesTest, Split_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::SplitSize, static_cast<int64_t>(2));
    a.set(AttrKey::Dim, static_cast<int64_t>(0));
    std::vector<Tensor> p = { w4_randf64({6,4},10) };
    for (int oi = 0; oi < 3; ++oi) w4_verify(OpId::Split, p, {0}, oi, a, 1e-7);
}

// BatchNorm2dForward (single-output kernel): inputs (x, mean, var), mean/var
// treated as independent per-channel constants (no batch-stat coupling).
TEST_P(JVPRulesTest, BatchNorm2dForward_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Eps, 1e-5);
    std::vector<Tensor> p = { w4_randf64({2,3,4,4}, 51),
                              w4_randf64({3}, 52),
                              w4_randf64({3}, 53) * 0.1 + 1.0 };  // var > 0
    w4_verify_single(OpId::BatchNorm2dForward, p, {0,1,2}, a, 1e-6);
}

// LinalgQR forward-mode JVP (reduced QR, A = Q R). Diagonal-dominant tall
// matrix keeps the factorization smooth (stable signs, no gauge flips).
TEST_P(JVPRulesTest, LinalgQR_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_randf64({5,4}, 61) * 0.3;
    { double* d = A.data<double>(); for (int i=0;i<4;++i) d[i*4+i] += (4.0 + i); }
    w4_verify(OpId::LinalgQR, {A}, {0}, 0, a, 1e-4);  // dQ
    w4_verify(OpId::LinalgQR, {A}, {0}, 1, a, 1e-4);  // dR
}

// TopK / Sort forward-mode JVP. The `values` output is a pure gather of the
// input by the (locally-constant) ranking indices, so its tangent is the same
// gather of the input tangent; the integer `indices` output has zero tangent.
// Distinct random Float64 inputs make the ranking stable under the FD step.
TEST_P(JVPRulesTest, TopK_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::K, static_cast<int64_t>(3));
    a.set(AttrKey::Dim, static_cast<int64_t>(1));
    a.set(AttrKey::Largest, true);
    a.set(AttrKey::Sorted, true);
    w4_verify(OpId::TopK, { w4_randf64({4,8},71) }, {0}, 0, a, 1e-7);  // d(values)
}

TEST_P(JVPRulesTest, Sort_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Dim, static_cast<int64_t>(1));
    a.set(AttrKey::Descending, false);
    w4_verify(OpId::Sort, { w4_randf64({4,8},72) }, {0}, 0, a, 1e-7);  // d(values)
}

// SolveTriangular forward-mode JVP. Upper-triangular, diagonally dominant A
// keeps the system well-conditioned. A X = B => A dX = dB - dA X.
TEST_P(JVPRulesTest, SolveTriangular_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Upper, true);
    a.set(AttrKey::UnitTriangular, false);
    Tensor A = w4_randf64({4,4}, 81) * 0.3;
    {
        double* d = A.data<double>();
        for (int i=0;i<4;++i) for (int j=0;j<i;++j) d[i*4+j] = 0.0;  // upper
        for (int i=0;i<4;++i) d[i*4+i] += (4.0 + i);                 // diag-dominant
    }
    Tensor B = w4_randf64({4,3}, 82);
    w4_verify_single(OpId::SolveTriangular, {A, B}, {0,1}, a, 1e-4);
}

// CholeskySolve / CholeskyInverse forward-mode JVP. Build a lower-triangular
// factor L with positive diagonal directly, so A = L L^T is SPD.
TEST_P(JVPRulesTest, CholeskySolve_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Upper, false);
    Tensor L = w4_randf64({4,4}, 83) * 0.3;
    {
        double* d = L.data<double>();
        for (int i=0;i<4;++i) for (int j=i+1;j<4;++j) d[i*4+j] = 0.0;  // lower
        for (int i=0;i<4;++i) d[i*4+i] += (3.0 + i);                   // pos diag
    }
    Tensor B = w4_randf64({4,3}, 84);
    w4_verify_single(OpId::LinalgCholeskySolve, {B, L}, {0,1}, a, 1e-4);
}

TEST_P(JVPRulesTest, CholeskyInverse_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Upper, false);
    Tensor L = w4_randf64({4,4}, 85) * 0.3;
    {
        double* d = L.data<double>();
        for (int i=0;i<4;++i) for (int j=i+1;j<4;++j) d[i*4+j] = 0.0;
        for (int i=0;i<4;++i) d[i*4+i] += (3.0 + i);
    }
    w4_verify_single(OpId::CholeskyInverse, {L}, {0}, a, 1e-4);
}

// TensorInv forward-mode JVP. A is (2,2,2,2) viewed as a 4x4 matrix (ind=2);
// flat-diagonal dominance keeps the inverse well-conditioned.
TEST_P(JVPRulesTest, TensorInv_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Ind, static_cast<int64_t>(2));
    Tensor A = w4_randf64({2,2,2,2}, 86) * 0.3;
    { double* d = A.data<double>(); for (int i=0;i<4;++i) d[i*4+i] += (4.0 + i); }
    w4_verify_single(OpId::TensorInv, {A}, {0}, a, 1e-4);
}

// TensorSolve forward-mode JVP. A is (2,2,2,2) -> 4x4, B is (2,2) -> rhs of 4.
TEST_P(JVPRulesTest, TensorSolve_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_randf64({2,2,2,2}, 87) * 0.3;
    { double* d = A.data<double>(); for (int i=0;i<4;++i) d[i*4+i] += (4.0 + i); }
    Tensor B = w4_randf64({2,2}, 88);
    w4_verify_single(OpId::TensorSolve, {A, B}, {0,1}, a, 1e-4);
}

// LinalgLUSolve forward-mode JVP. Build a packed LU matrix directly with a
// dominant diagonal (well-conditioned U, small unit-L) and identity pivots
// (no row interchange), then perturb both the packed factors and B.
TEST_P(JVPRulesTest, LUSolve_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor LU = w4_randf64({4,4}, 89) * 0.3;
    { double* d = LU.data<double>(); for (int i=0;i<4;++i) d[i*4+i] += (4.0 + i); }
    Tensor piv(std::vector<int64_t>{4}, DType::Int32, Device::cpu());
    { int32_t* p = piv.data<int32_t>(); for (int i=0;i<4;++i) p[i] = i + 1; }
    Tensor B = w4_randf64({4,3}, 90);
    // pivots (index 1) carry no tangent; verify the LU_data and B paths.
    w4_verify_single(OpId::LinalgLUSolve, {LU, piv, B}, {0,2}, a, 1e-4);
}

// LinalgLU forward-mode JVP, A = P L U.  Diagonally dominant A => no pivoting
// (P = I), so the pivot pattern is stable under the FD perturbation; validates
// the dL / dU differential.
TEST_P(JVPRulesTest, LinalgLU_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_randf64({4,4}, 91) * 0.3;
    { double* d = A.data<double>(); for (int i=0;i<4;++i) d[i*4+i] += (5.0 + i); }
    w4_verify(OpId::LinalgLU, {A}, {0}, 0, a, 1e-4);  // dL
    w4_verify(OpId::LinalgLU, {A}, {0}, 1, a, 1e-4);  // dU
}

// LinalgLU with a real, stable row interchange (column-0 max in row 1) to
// exercise the permutation-reconstruction path. The dominant pivot entry (9)
// keeps the pivot choice constant under the FD step.
TEST_P(JVPRulesTest, LinalgLU_JVP_Permuted_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A(std::vector<int64_t>{3,3}, DType::Float64, Device::cpu());
    {
        double vals[9] = { 1.0, 2.0, 0.5,
                           9.0, 1.0, 0.3,
                           0.4, 0.7, 8.0 };
        double* d = A.data<double>(); for (int i=0;i<9;++i) d[i] = vals[i];
    }
    w4_verify(OpId::LinalgLU, {A}, {0}, 0, a, 1e-4);  // dL
    w4_verify(OpId::LinalgLU, {A}, {0}, 1, a, 1e-4);  // dU
}

// LSTMForward sequence JVP: composes the cell JVP over time. Inputs
// (x, W_ih, W_hh, b_ih, b_hh, h0, c0); small scaled values keep gates smooth.
TEST_P(JVPRulesTest, LSTMForward_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    const int64_t T=3, B=2, I=4, H=2;
    std::vector<Tensor> p = {
        w4_randf64({T,B,I}, 101) * 0.5,
        w4_randf64({4*H,I}, 102) * 0.4,
        w4_randf64({4*H,H}, 103) * 0.4,
        w4_randf64({4*H},   104) * 0.3,
        w4_randf64({4*H},   105) * 0.3,
        w4_randf64({B,H},   106) * 0.3,
        w4_randf64({B,H},   107) * 0.3,
    };
    for (int oi = 0; oi < 3; ++oi)
        w4_verify(OpId::LSTMForward, p, {0,1,2,3,4,5,6}, oi, a, 1e-4);
}

// GRUForward sequence JVP. Inputs (x, W_ih, W_hh, b_ih, h0, b_hh).
TEST_P(JVPRulesTest, GRUForward_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    const int64_t T=3, B=2, I=4, H=2;
    std::vector<Tensor> p = {
        w4_randf64({T,B,I}, 111) * 0.5,
        w4_randf64({3*H,I}, 112) * 0.4,
        w4_randf64({3*H,H}, 113) * 0.4,
        w4_randf64({3*H},   114) * 0.3,
        w4_randf64({B,H},   115) * 0.3,
        w4_randf64({3*H},   116) * 0.3,
    };
    for (int oi = 0; oi < 2; ++oi)
        w4_verify(OpId::GRUForward, p, {0,1,2,3,4,5}, oi, a, 1e-4);
}

// LSTMMultiLayerForward JVP: layer-by-layer composition. Inputs
// {x, h0(L,B,H), c0(L,B,H), per layer (W_ih, W_hh, bias)}.
TEST_P(JVPRulesTest, LSTMMultiLayer_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    const int64_t Lyr=2, T=3, B=2, I=4, H=2;
    a.set(AttrKey::NumLayers, Lyr);
    std::vector<Tensor> p = {
        w4_randf64({T,B,I}, 121) * 0.5,
        w4_randf64({Lyr,B,H}, 122) * 0.3,
        w4_randf64({Lyr,B,H}, 123) * 0.3,
        w4_randf64({4*H,I}, 124) * 0.4, w4_randf64({4*H,H}, 125) * 0.4, w4_randf64({4*H}, 126) * 0.3,
        w4_randf64({4*H,H}, 127) * 0.4, w4_randf64({4*H,H}, 128) * 0.4, w4_randf64({4*H}, 129) * 0.3,
    };
    std::vector<int> diffs; for (int i=0;i<(int)p.size();++i) diffs.push_back(i);
    for (int oi = 0; oi < 3; ++oi) w4_verify(OpId::LSTMMultiLayerForward, p, diffs, oi, a, 1e-4);
}

// GRUMultiLayerForward JVP. Inputs {x, h0(L,B,H), per layer (W_ih, W_hh, bias)}.
TEST_P(JVPRulesTest, GRUMultiLayer_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    const int64_t Lyr=2, T=3, B=2, I=4, H=2;
    a.set(AttrKey::NumLayers, Lyr);
    std::vector<Tensor> p = {
        w4_randf64({T,B,I}, 131) * 0.5,
        w4_randf64({Lyr,B,H}, 132) * 0.3,
        w4_randf64({3*H,I}, 133) * 0.4, w4_randf64({3*H,H}, 134) * 0.4, w4_randf64({3*H}, 135) * 0.3,
        w4_randf64({3*H,H}, 136) * 0.4, w4_randf64({3*H,H}, 137) * 0.4, w4_randf64({3*H}, 138) * 0.3,
    };
    std::vector<int> diffs; for (int i=0;i<(int)p.size();++i) diffs.push_back(i);
    for (int oi = 0; oi < 2; ++oi) w4_verify(OpId::GRUMultiLayerForward, p, diffs, oi, a, 1e-4);
}

// BiLSTMForward JVP: forward + time-reversed backward direction, concatenated.
// Inputs {x, h0(2,B,H), c0(2,B,H), Wih_f,Whh_f,bih_f,bhh_f, Wih_b,Whh_b,bih_b,bhh_b}.
TEST_P(JVPRulesTest, BiLSTM_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    const int64_t T=3, B=2, I=4, H=2;
    std::vector<Tensor> p = {
        w4_randf64({T,B,I}, 141) * 0.5,
        w4_randf64({2,B,H}, 142) * 0.3,
        w4_randf64({2,B,H}, 143) * 0.3,
        w4_randf64({4*H,I},144)*0.4, w4_randf64({4*H,H},145)*0.4, w4_randf64({4*H},146)*0.3, w4_randf64({4*H},147)*0.3,
        w4_randf64({4*H,I},148)*0.4, w4_randf64({4*H,H},149)*0.4, w4_randf64({4*H},150)*0.3, w4_randf64({4*H},151)*0.3,
    };
    std::vector<int> diffs; for (int i=0;i<(int)p.size();++i) diffs.push_back(i);
    for (int oi = 0; oi < 3; ++oi) w4_verify(OpId::BiLSTMForward, p, diffs, oi, a, 1e-4);
}

// NOTE: LinalgLU / LinalgSVD / LinalgEig / LDLFactor / Geqrf / RNN-forward JVP
// adapters are drafted (LU/SVD parked in jvp_rules.cpp) but do not yet pass
// this finite-difference gradcheck, so their OpIds remain registered to the
// NonDifferentiable thrower (forward-mode AD fails loudly rather than returning
// a wrong tangent) until the analytic rule is gradcheck-clean.

INSTANTIATE_BACKEND_TESTS(JVPRulesTest);

