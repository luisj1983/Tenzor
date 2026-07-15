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
#include <tenzor/ops/linalg.hpp>

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

// ---- Embedding JVP: (weight, indices) argument order (H8) ----
// OpId::Embedding's universal convention is (weight, indices) — matching
// every real call site (cpu_kernel_registry.cpp, nn/layers/embedding.cpp,
// nn/functional.cpp). This rule previously assumed (indices, weight).
TEST_P(JVPRulesTest, JVP_Embedding_WeightIndicesOrder) {
    auto weight  = randn({5, 3}, DType::Float64, device);
    auto dweight = randn({5, 3}, DType::Float64, device);
    int64_t idx_data[] = {0, 2, 4, 1};
    auto indices = Tensor::from_blob(idx_data, {4}, DType::Int64, Device::cpu())
                       .clone().to(device);
    OpAttributes attrs;
    std::array<Tensor, 2> p{weight, indices};
    std::array<Tensor, 2> t{dweight, Tensor()};
    auto out = tenzor::dispatch_jvp(OpId::Embedding, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& w) {
        return tenzor::dispatch(OpId::Embedding, std::vector<Tensor>{w, indices}, attrs)[0];
    }, weight, dweight);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-6);
}

// ---- Fmax / Fmin JVP: NaN-ignoring semantics (H7) ----
// fmax/fmin are NaN-IGNORING (IEEE 754-2008 maxNum/minNum), unlike
// maximum/minimum which propagate NaN — confirmed at
// src/backends/cpu/kernels/advanced.cpp:1133-1161.
TEST_P(JVPRulesTest, JVP_Fmax_MatchesFD) {
    auto a  = randn({6}, DType::Float64, device) + 1.0;
    auto b  = randn({6}, DType::Float64, device) + 1.0;
    auto da = randn({6}, DType::Float64, device);
    auto db = randn({6}, DType::Float64, device);
    OpAttributes attrs;
    std::array<Tensor, 2> p{a, b}, t{da, db};
    auto out = tenzor::dispatch_jvp(OpId::Fmax, p, t, attrs);
    auto fd = fd_jvp2([&](const Tensor& xx, const Tensor& yy) {
        return tenzor::dispatch(OpId::Fmax, std::vector<Tensor>{xx, yy}, attrs)[0];
    }, a, da, b, db);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-3);
}

TEST_P(JVPRulesTest, JVP_Fmin_MatchesFD) {
    auto a  = randn({6}, DType::Float64, device) + 1.0;
    auto b  = randn({6}, DType::Float64, device) + 1.0;
    auto da = randn({6}, DType::Float64, device);
    auto db = randn({6}, DType::Float64, device);
    OpAttributes attrs;
    std::array<Tensor, 2> p{a, b}, t{da, db};
    auto out = tenzor::dispatch_jvp(OpId::Fmin, p, t, attrs);
    auto fd = fd_jvp2([&](const Tensor& xx, const Tensor& yy) {
        return tenzor::dispatch(OpId::Fmin, std::vector<Tensor>{xx, yy}, attrs)[0];
    }, a, da, b, db);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-3);
}

TEST_P(JVPRulesTest, JVP_Fmax_NaNIgnoring) {
    double nan = std::nan("");
    double a_data[]  = {1.0, nan};
    double b_data[]  = {nan, 2.0};
    double da_data[] = {3.0, 100.0};
    double db_data[] = {100.0, 5.0};
    auto a  = Tensor::from_blob(a_data,  {2}, DType::Float64, Device::cpu()).clone().to(device);
    auto b  = Tensor::from_blob(b_data,  {2}, DType::Float64, Device::cpu()).clone().to(device);
    auto da = Tensor::from_blob(da_data, {2}, DType::Float64, Device::cpu()).clone().to(device);
    auto db = Tensor::from_blob(db_data, {2}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    std::array<Tensor, 2> p{a, b}, t{da, db};
    auto out = tenzor::dispatch_jvp(OpId::Fmax, p, t, attrs);
    auto primal_cpu  = out.primal.to(Device::cpu());
    auto tangent_cpu = out.tangent.to(Device::cpu());
    auto* prim = primal_cpu.data<double>();
    auto* tan  = tangent_cpu.data<double>();
    EXPECT_NEAR(prim[0], 1.0, 1e-12);   // fmax(1.0, NaN) ignores NaN -> 1.0
    EXPECT_NEAR(prim[1], 2.0, 1e-12);   // fmax(NaN, 2.0) ignores NaN -> 2.0
    EXPECT_NEAR(tan[0], 3.0, 1e-12);    // b NaN -> tangent comes entirely from a
    EXPECT_NEAR(tan[1], 5.0, 1e-12);    // a NaN -> tangent comes entirely from b
}

TEST_P(JVPRulesTest, JVP_Fmin_NaNIgnoring) {
    double nan = std::nan("");
    double a_data[]  = {1.0, nan};
    double b_data[]  = {nan, 2.0};
    double da_data[] = {3.0, 100.0};
    double db_data[] = {100.0, 5.0};
    auto a  = Tensor::from_blob(a_data,  {2}, DType::Float64, Device::cpu()).clone().to(device);
    auto b  = Tensor::from_blob(b_data,  {2}, DType::Float64, Device::cpu()).clone().to(device);
    auto da = Tensor::from_blob(da_data, {2}, DType::Float64, Device::cpu()).clone().to(device);
    auto db = Tensor::from_blob(db_data, {2}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    std::array<Tensor, 2> p{a, b}, t{da, db};
    auto out = tenzor::dispatch_jvp(OpId::Fmin, p, t, attrs);
    auto primal_cpu  = out.primal.to(Device::cpu());
    auto tangent_cpu = out.tangent.to(Device::cpu());
    auto* prim = primal_cpu.data<double>();
    auto* tan  = tangent_cpu.data<double>();
    EXPECT_NEAR(prim[0], 1.0, 1e-12);
    EXPECT_NEAR(prim[1], 2.0, 1e-12);
    EXPECT_NEAR(tan[0], 3.0, 1e-12);
    EXPECT_NEAR(tan[1], 5.0, 1e-12);
}

// ---- CumProd JVP: exact zero-safe tangent (H17) ----
// The old `cumsum(dx/x) * y` closed form divides by x directly, so a single
// zero anywhere in the prefix poisons every SUBSEQUENT position via NaN
// propagation through cumsum, even though positions strictly after the zero
// have a well-defined, finite analytic tangent (matching CumProdBackward's
// own already-fixed exact-prefix-product-excluding-self VJP).
TEST_P(JVPRulesTest, JVP_CumProd_ZeroInPrefix_TangentFinite) {
    double x_data[]  = {1.0, 0.0, 2.0, 3.0};
    double dx_data[] = {1.0, 1.0, 1.0, 1.0};
    auto x  = Tensor::from_blob(x_data,  {4}, DType::Float64, Device::cpu()).clone().to(device);
    auto dx = Tensor::from_blob(dx_data, {4}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, int64_t{0});
    std::array<Tensor, 1> p{x}, t{dx};
    auto out = tenzor::dispatch_jvp(OpId::CumProd, p, t, attrs);

    auto tangent_cpu = out.tangent.to(Device::cpu());
    auto* tan = tangent_cpu.data<double>();
    // primal = [1, 0, 0, 0]; analytic tangent at k=2: sum_i dx_i * prod_{j<=2,j!=i} x_j
    //   i=0: dx0 * (x1*x2) = 1 * (0*2) = 0
    //   i=1: dx1 * (x0*x2) = 1 * (1*2) = 2
    //   i=2: dx2 * (x0*x1) = 1 * (1*0) = 0   => tangent[2] = 2
    // at k=3: i=0: dx0*x1*x2*x3=0; i=1: dx1*x0*x2*x3=1*1*2*3=6; i=2: dx2*x0*x1*x3=0;
    //         i=3: dx3*x0*x1*x2=0 => tangent[3] = 6
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(tan[i])) << "tangent[" << i << "] is not finite: " << tan[i];
    }
    EXPECT_NEAR(tan[2], 2.0, 1e-9);
    EXPECT_NEAR(tan[3], 6.0, 1e-9);
}

TEST_P(JVPRulesTest, JVP_CumProd_MatchesFD_NoZeros) {
    auto x = randn({5}, DType::Float64, device) + 2.0;  // keep away from 0
    auto dx = randn({5}, DType::Float64, device);
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, int64_t{0});
    std::array<Tensor, 1> p{x}, t{dx};
    auto out = tenzor::dispatch_jvp(OpId::CumProd, p, t, attrs);
    auto fd = fd_jvp([&](const Tensor& xx) {
        return tenzor::dispatch(OpId::CumProd, std::vector<Tensor>{xx}, attrs)[0];
    }, x, dx);
    EXPECT_LT(max_abs_diff_d(out.tangent, fd), 1e-3);
}

// ---- M6/M7/M26: zero-safety gaps vs. their VJP siblings ----
TEST_P(JVPRulesTest, JVP_Xlogy_ZeroXFinite) {
    double x_data[]  = {0.0};
    double y_data[]  = {0.0};
    double dx_data[] = {0.0};
    double dy_data[] = {1.0};
    auto x  = Tensor::from_blob(x_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto y  = Tensor::from_blob(y_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto dx = Tensor::from_blob(dx_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto dy = Tensor::from_blob(dy_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    std::array<Tensor, 2> p{x, y}, t{dx, dy};
    auto out = tenzor::dispatch_jvp(OpId::XLogY, p, t, attrs);
    auto tan_cpu = out.tangent.to(Device::cpu());
    EXPECT_TRUE(std::isfinite(tan_cpu.data<double>()[0]));
    EXPECT_NEAR(tan_cpu.data<double>()[0], 0.0, 1e-9);
}

TEST_P(JVPRulesTest, JVP_Addcdiv_ZeroDenominatorFinite) {
    double a_data[] = {0.0};
    double b_data[] = {1.0};
    double c_data[] = {0.0};
    double da_data[] = {0.0};
    double db_data[] = {1.0};
    double dc_data[] = {1.0};
    auto a  = Tensor::from_blob(a_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto b  = Tensor::from_blob(b_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto c  = Tensor::from_blob(c_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto da = Tensor::from_blob(da_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto db = Tensor::from_blob(db_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto dc = Tensor::from_blob(dc_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, 1.0);
    std::array<Tensor, 3> p{a, b, c}, t{da, db, dc};
    auto out = tenzor::dispatch_jvp(OpId::Addcdiv, p, t, attrs);
    auto tan_cpu = out.tangent.to(Device::cpu());
    EXPECT_TRUE(std::isfinite(tan_cpu.data<double>()[0]))
        << "addcdiv JVP tangent not finite at c==0: " << tan_cpu.data<double>()[0];
}

// Discovered while fixing M7: jvp_adapter_addcdiv/addcmul read AttrKey::Value
// instead of AttrKey::Alpha (the key the real dispatcher — ops/math.cpp
// addcdiv()/addcmul() — always sets), silently using the wrong scale factor
// whenever alpha != 1.0. alpha=1.0 alone can't distinguish "read the right
// key" from "read the wrong key and defaulted to 1.0 anyway" — use alpha=3.0
// so a value/alpha mixup is unambiguous.
TEST_P(JVPRulesTest, JVP_Addcdiv_NonUnitAlphaScalesTangent) {
    double a_data[] = {0.0}, b_data[] = {2.0}, c_data[] = {1.0};
    double da_data[] = {0.0}, db_data[] = {1.0}, dc_data[] = {0.0};
    auto a  = Tensor::from_blob(a_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto b  = Tensor::from_blob(b_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto c  = Tensor::from_blob(c_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto da = Tensor::from_blob(da_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto db = Tensor::from_blob(db_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto dc = Tensor::from_blob(dc_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, 3.0);
    std::array<Tensor, 3> p{a, b, c}, t{da, db, dc};
    auto out = tenzor::dispatch_jvp(OpId::Addcdiv, p, t, attrs);
    auto tan_cpu = out.tangent.to(Device::cpu());
    // tangent = da + alpha*(db/c - b*dc/c^2) = 0 + 3*(1/1 - 0) = 3.0
    EXPECT_NEAR(tan_cpu.data<double>()[0], 3.0, 1e-9);
}

TEST_P(JVPRulesTest, JVP_Addcmul_NonUnitAlphaScalesTangent) {
    double a_data[] = {0.0}, b_data[] = {2.0}, c_data[] = {3.0};
    double da_data[] = {0.0}, db_data[] = {1.0}, dc_data[] = {0.0};
    auto a  = Tensor::from_blob(a_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto b  = Tensor::from_blob(b_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto c  = Tensor::from_blob(c_data,  {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto da = Tensor::from_blob(da_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto db = Tensor::from_blob(db_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    auto dc = Tensor::from_blob(dc_data, {1}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, 3.0);
    std::array<Tensor, 3> p{a, b, c}, t{da, db, dc};
    auto out = tenzor::dispatch_jvp(OpId::Addcmul, p, t, attrs);
    auto tan_cpu = out.tangent.to(Device::cpu());
    // tangent = da + alpha*(db*c + b*dc) = 0 + 3*(1*3 + 0) = 9.0
    EXPECT_NEAR(tan_cpu.data<double>()[0], 9.0, 1e-9);
}

// Exact worked example from the audit finding: x=[2.0, 0.0, 3.0], dx=[1,1,1].
// prod(x)=0; correct tangent = 6 (only the zero position contributes,
// weight = product of the other elements: 2*3=6).
TEST_P(JVPRulesTest, JVP_Prod_OneZero_MatchesExactFormula) {
    double x_data[]  = {2.0, 0.0, 3.0};
    double dx_data[] = {1.0, 1.0, 1.0};
    auto x  = Tensor::from_blob(x_data,  {3}, DType::Float64, Device::cpu()).clone().to(device);
    auto dx = Tensor::from_blob(dx_data, {3}, DType::Float64, Device::cpu()).clone().to(device);
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, int64_t{0});
    std::array<Tensor, 1> p{x}, t{dx};
    auto out = tenzor::dispatch_jvp(OpId::Prod, p, t, attrs);
    auto tan_cpu = out.tangent.to(Device::cpu());
    EXPECT_TRUE(std::isfinite(tan_cpu.data<double>()[0]));
    EXPECT_NEAR(tan_cpu.data<double>()[0], 6.0, 1e-9);
}

// M27: near-degenerate (not exactly equal) eigenvalues must not blow up to
// astronomically large / non-finite tangents — the old hard-threshold F
// matrix produced F_ij ~ 1e7 for a 1e-7 gap.
TEST_P(JVPRulesTest, JVP_LinalgEigh_NearDegenerateEigenvaluesFinite) {
    if (device != Device::cpu()) return;
    // Symmetric 2x2 matrix with eigenvalues very close together (but not
    // identical): eigenvalues of [[a,b],[b,a]] are a+b, a-b. Float32 so the
    // gap (1e-7) sits within the dtype's Lorentzian damping band
    // (rel_eps=1e-6 for Float32): diff << eps_tol, so the fixed F is bounded
    // to O(1/eps_tol), while the old hard-threshold F ~ 1/diff is ~100x
    // larger and finite-but-unbounded as the gap shrinks further.
    float eps = 1e-7f;
    float a_data[] = {1.0f + eps / 2.0f, eps / 2.0f, eps / 2.0f, 1.0f - eps / 2.0f};
    float da_data[] = {1.0f, 0.5f, 0.5f, 1.0f};
    auto A  = Tensor::from_blob(a_data,  {2, 2}, DType::Float32, Device::cpu()).clone();
    auto dA = Tensor::from_blob(da_data, {2, 2}, DType::Float32, Device::cpu()).clone();
    OpAttributes attrs;
    std::array<Tensor, 1> p{A}, t{dA};
    auto out = tenzor::dispatch_jvp_multi(OpId::LinalgEigh, p, t, attrs);
    ASSERT_EQ(out.tangents.size(), 2u);
    auto dV_cpu = out.tangents[1].to(Device::cpu());
    auto* dvp = dV_cpu.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < dV_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(dvp[i])) << "dV[" << i << "] not finite: " << dvp[i];
        max_abs = std::max(max_abs, std::abs(dvp[i]));
    }
    // Naive 1/diff would give ~1e7; the Lorentzian-bounded fix keeps this
    // comfortably below 1e6 for this gap/dtype combination.
    EXPECT_LT(max_abs, 1e6f) << "dV unreasonably large (hard-threshold blowup): " << max_abs;
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

// H16: jvp_adapter_layer_norm's mean/rstd (and their tangents) must match
// the real kernel's flat {batch_size} contract (nn_kernels.cpp
// layer_norm_kernel_with_stats), not a keepdim=true [B,T,1] layout — a
// consumer matching saved_tensors()[1]/[2] by data_ptr (e.g. the
// create_graph=true double-backward walker) would otherwise get a
// shape-mismatched tangent for every LayerNorm call.
TEST_P(JVPRulesTest, LayerNorm_JVP_StatsShapeMatchesKernel) {
    if (device != Device::cpu()) return;
    auto x = w4_randf64({2, 3, 4}, 21);
    auto gamma = w4_randf64({4}, 22);
    auto beta = w4_randf64({4}, 23);
    OpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(4));
    attrs.set(AttrKey::Eps, 1e-5);

    // Real forward's mean/rstd shape (ground truth): flat {batch_size} where
    // batch_size = numel(x)/norm_size = 2*3 = 6.
    auto real = tenzor::dispatch(OpId::LayerNorm, std::vector<Tensor>{x, gamma, beta}, attrs);
    std::vector<int64_t> real_mean_shape(real[1].shape().begin(), real[1].shape().end());
    std::vector<int64_t> real_rstd_shape(real[2].shape().begin(), real[2].shape().end());

    std::vector<Tensor> primals{x, gamma, beta};
    std::vector<Tensor> tangents{
        w4_randf64({2, 3, 4}, 24), w4_randf64({4}, 25), w4_randf64({4}, 26)};
    auto out = tenzor::dispatch_jvp_multi(OpId::LayerNorm, primals, tangents, attrs);

    ASSERT_EQ(out.primals.size(), 3u);
    ASSERT_EQ(out.tangents.size(), 3u);
    std::vector<int64_t> mean_primal_shape(out.primals[1].shape().begin(), out.primals[1].shape().end());
    std::vector<int64_t> rstd_primal_shape(out.primals[2].shape().begin(), out.primals[2].shape().end());
    std::vector<int64_t> mean_tangent_shape(out.tangents[1].shape().begin(), out.tangents[1].shape().end());
    std::vector<int64_t> rstd_tangent_shape(out.tangents[2].shape().begin(), out.tangents[2].shape().end());
    EXPECT_EQ(mean_primal_shape, real_mean_shape) << "mean primal shape mismatch";
    EXPECT_EQ(rstd_primal_shape, real_rstd_shape) << "rstd primal shape mismatch";
    EXPECT_EQ(mean_tangent_shape, real_mean_shape) << "mean tangent shape mismatch";
    EXPECT_EQ(rstd_tangent_shape, real_rstd_shape) << "rstd tangent shape mismatch";
}

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

// H4: SparseSpGEMM values-tangent (dC.values = dA.values @ B + A @ dB.values)
// against a central-difference reference, perturbing A.values and B.values
// independently (diff_primals {2,5}) and jointly, checking output slot 2
// (C.values). Structural outputs (crow=0, col=1) are never perturbed —
// integer/non-differentiable, matching every other sparse JVP rule in this
// file.
TEST_P(JVPRulesTest, SparseSpGEMM_JVP_ValuesMatchesFD) {
    if (device != Device::cpu()) return;
    int64_t M = 4, K = 5, N = 3;
    std::vector<int64_t> A_crow_d = {0, 2, 3, 5, 6};
    std::vector<int64_t> A_col_d  = {0, 2, 1, 3, 4, 0};
    std::vector<int64_t> B_crow_d = {0, 1, 2, 3, 3, 4};
    std::vector<int64_t> B_col_d  = {0, 2, 1, 0};

    auto A_crow = Tensor::from_blob(A_crow_d.data(), {M + 1}, DType::Int64, Device::cpu())
                      .clone();
    auto A_col = Tensor::from_blob(A_col_d.data(), {static_cast<int64_t>(A_col_d.size())},
                                    DType::Int64, Device::cpu())
                     .clone();
    auto B_crow = Tensor::from_blob(B_crow_d.data(), {K + 1}, DType::Int64, Device::cpu())
                      .clone();
    auto B_col = Tensor::from_blob(B_col_d.data(), {static_cast<int64_t>(B_col_d.size())},
                                    DType::Int64, Device::cpu())
                     .clone();
    auto A_values = w4_randf64({static_cast<int64_t>(A_col_d.size())}, 11);
    auto B_values = w4_randf64({static_cast<int64_t>(B_col_d.size())}, 12);

    OpAttributes attrs;
    attrs.set(AttrKey::M, M);
    attrs.set(AttrKey::K, K);
    attrs.set(AttrKey::N, N);

    std::vector<Tensor> primals{A_crow, A_col, A_values, B_crow, B_col, B_values};
    w4_verify(OpId::SparseSpGEMM, primals, {2}, 2, attrs, 1e-4);
    w4_verify(OpId::SparseSpGEMM, primals, {5}, 2, attrs, 1e-4);
    w4_verify(OpId::SparseSpGEMM, primals, {2, 5}, 2, attrs, 1e-4);
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

// LinalgSVD JVP (thin SVD A = U S Vh). Well-separated singular values keep the
// singular vectors' sign/gauge stable under the FD step.
TEST_P(JVPRulesTest, LinalgSVD_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::FullMatrices, false);
    Tensor A = w4_randf64({4,4}, 161) * 0.15;
    { double* d = A.data<double>(); double diag[4]={6.0,4.0,2.5,1.0};
      for (int i=0;i<4;++i) d[i*4+i] += diag[i]; }
    w4_verify(OpId::LinalgSVD, {A}, {0}, 1, a, 1e-3);  // dS
    w4_verify(OpId::LinalgSVD, {A}, {0}, 0, a, 1e-3);  // dU
    w4_verify(OpId::LinalgSVD, {A}, {0}, 2, a, 1e-3);  // dVh
}

// LinalgEig JVP (general non-symmetric, real spectrum). Upper-triangular A
// with a distinct, well-separated diagonal => real eigenvalues = diagonal and
// real, non-orthogonal eigenvectors (exercises the Gram normalization). The
// FD step keeps the spectrum real and the ordering stable.
TEST_P(JVPRulesTest, LinalgEig_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_randf64({4,4}, 171) * 0.1;
    {
        double* d = A.data<double>();
        for (int i=0;i<4;++i) for (int j=0;j<i;++j) d[i*4+j] = 0.0;  // upper triangular
        double dg[4] = {8.0, 5.0, 2.0, 0.5};
        for (int i=0;i<4;++i) d[i*4+i] = dg[i];
    }
    // Eigenvalues are gauge-free: FD-verify dRe and dIm directly.
    w4_verify(OpId::LinalgEig, {A}, {0}, 0, a, 1e-3);  // dRe
    w4_verify(OpId::LinalgEig, {A}, {0}, 1, a, 1e-3);  // dIm (== 0)
    // The op's eigenvectors carry a discontinuous sign gauge (a tiny FD step can
    // flip a column's sign), so a raw dV-vs-FD check is meaningless. Validate
    // {dV, dRe} jointly via the gauge-invariant reconstruction d(VΛV⁻¹) == dA:
    //   dA = dV Λ V⁻¹ + V dΛ V⁻¹ − V Λ V⁻¹ dV V⁻¹.
    Tensor dA = w4_randf64({4,4}, 172);
    std::array<Tensor,1> p{A}, t{dA};
    auto out  = tenzor::dispatch_jvp_multi(OpId::LinalgEig, p, t, a);
    auto outs = tenzor::dispatch(OpId::LinalgEig, std::vector<Tensor>{A}, a);
    Tensor Re = outs[0], V = outs[2];
    Tensor dRe = out.tangents[0], dV = out.tangents[2];
    Tensor Vinv = tenzor::linalg::inv(V);
    Tensor Lam  = tenzor::linalg::diag_embed(Re,  0, -2, -1);
    Tensor dLam = tenzor::linalg::diag_embed(dRe, 0, -2, -1);
    Tensor recon = tenzor::matmul(tenzor::matmul(dV, Lam), Vinv);
    recon = recon + tenzor::matmul(tenzor::matmul(V, dLam), Vinv);
    recon = recon - tenzor::matmul(tenzor::matmul(tenzor::matmul(tenzor::matmul(V, Lam), Vinv), dV), Vinv);
    EXPECT_LT(max_abs_diff_d(recon, dA), 1e-3);
}

// LinalgHouseholder (orgqr) JVP: Q from elementary reflectors (smooth in
// (reflectors, tau)). Inputs generated via geqrf so they are a valid pair.
TEST_P(JVPRulesTest, LinalgHouseholder_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_randf64({4,3}, 181) * 0.5;
    { double* d = A.data<double>(); for (int i=0;i<3;++i) d[i*3+i] += 3.0; }
    auto qr = tenzor::dispatch(OpId::Geqrf, std::vector<Tensor>{A}, a);
    w4_verify_single(OpId::LinalgHouseholder, {qr[0], qr[1]}, {0,1}, a, 1e-4);
}

// Ormqr JVP: Y = Q B (left, no transpose), Q from reflectors. Smooth in all
// three inputs.
TEST_P(JVPRulesTest, Ormqr_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Left, true);
    a.set(AttrKey::TransposeQ, false);
    Tensor A = w4_randf64({4,3}, 183) * 0.5;
    { double* d = A.data<double>(); for (int i=0;i<3;++i) d[i*3+i] += 3.0; }
    auto qr = tenzor::dispatch(OpId::Geqrf, std::vector<Tensor>{A}, OpAttributes{});
    Tensor B = w4_randf64({4,2}, 184);
    w4_verify_single(OpId::Ormqr, {qr[0], qr[1], B}, {0,1,2}, a, 1e-4);
}

// L3 regression: the JVP's PRIMAL must come from the real dispatched op
// (LAPACK orgqr/ormqr), not a naive sequential-reflector reconstruction —
// every sibling S15 rule needing a primal (SVD, Eig, LDLFactor, LDLSolve,
// Geqrf) already does this. Use a larger matrix (more reflectors) where the
// naive host-side accumulation visibly drifts from LAPACK's blocked
// compact-WY algorithm, and assert the JVP-returned primal is essentially
// EXACT against the real op's own output (both now literally dispatch the
// same kernel) rather than merely "close" (which the naive reconstruction
// would also satisfy, just not to LAPACK-exact precision).
TEST_P(JVPRulesTest, LinalgHouseholder_JVP_PrimalMatchesRealOp) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    // 200x200: large enough to push LAPACK's dorgqr past its blocked/
    // unblocked crossover (dorg2r), so the naive reconstruction's drift
    // from the blocked compact-WY algorithm is actually visible — a smaller
    // matrix stays under the blocking threshold, where LAPACK itself falls
    // through to the same unblocked recurrence and the two primals agree
    // almost to the bit even with the bug present.
    const int64_t N = 500;
    Tensor A = w4_randf64({N, N}, 281) * 0.3;
    { double* d = A.data<double>(); for (int i = 0; i < N; ++i) d[i*N+i] += 5.0; }
    auto qr = tenzor::dispatch(OpId::Geqrf, std::vector<Tensor>{A}, OpAttributes{});
    Tensor reflectors = qr[0], tau = qr[1];

    Tensor real_Q = tenzor::dispatch(OpId::LinalgHouseholder,
        std::vector<Tensor>{reflectors, tau}, a)[0];

    std::array<Tensor, 2> p{reflectors, tau};
    std::array<Tensor, 2> t{tenzor::zeros_like(reflectors), tenzor::zeros_like(tau)};
    auto out = tenzor::dispatch_jvp(OpId::LinalgHouseholder, p, t, a);

    EXPECT_LT(max_abs_diff_d(out.primal, real_Q), 1e-15)
        << "JVP primal must come from the real dispatched op, not a "
           "numerically-drifted naive reconstruction";
}

// Ormqr shape-safety net (NOT a pre-fix-vs-post-fix regression reproduction
// — see below): reflectors' column count (n) deliberately LESS than `order`
// (the Q dimension Ormqr implicitly multiplies against, = rows(B) for
// left). LAPACK ?ormqr only requires cols(reflectors) >= k (the reflector
// count), not cols(reflectors) == order. The original naive
// householder_q_and_dq reconstruction always builds a full (order,order) Q
// directly (via eye(m)), so it was already shape-safe here and this test
// does NOT fail against it — confirmed empirically during development. It
// WOULD have failed against an intermediate, incorrect version of this fix
// that substituted OpId::LinalgHouseholder (orgqr, which returns an (m,n)
// economy Q) for the primal: that silently returns the WRONG SHAPE whenever
// n < order. The shipped fix instead re-dispatches OpId::Ormqr itself
// against an identity "other" operand to materialize the correctly-shaped
// (order,order) op(Q), which cannot suffer this mismatch since it's the
// exact same kernel/shape contract as the real call. Kept as a permanent
// structural safety net against ever reintroducing that class of mistake.
TEST_P(JVPRulesTest, Ormqr_JVP_PrimalMatchesRealOp_RectangularReflectors) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    a.set(AttrKey::Left, true);
    a.set(AttrKey::TransposeQ, false);
    const int64_t M = 100, N = 20;  // reflectors: (M,N), N < M=order (left, rows(B)=M)
    Tensor A = w4_randf64({M, N}, 283) * 0.3;
    { double* d = A.data<double>(); for (int i = 0; i < N; ++i) d[i*N+i] += 5.0; }
    auto qr = tenzor::dispatch(OpId::Geqrf, std::vector<Tensor>{A}, OpAttributes{});
    Tensor reflectors = qr[0], tau = qr[1];
    ASSERT_EQ(reflectors.shape()[1], N) << "sanity: cols(reflectors) == N < order == M";
    Tensor B = w4_randf64({M, 5}, 284);

    Tensor real_Y = tenzor::dispatch(OpId::Ormqr,
        std::vector<Tensor>{reflectors, tau, B}, a)[0];

    std::array<Tensor, 3> p{reflectors, tau, B};
    std::array<Tensor, 3> t{tenzor::zeros_like(reflectors), tenzor::zeros_like(tau),
                             tenzor::zeros_like(B)};
    auto out = tenzor::dispatch_jvp(OpId::Ormqr, p, t, a);

    ASSERT_EQ(out.primal.shape()[0], M);
    ASSERT_EQ(out.primal.shape()[1], 5);
    EXPECT_LT(max_abs_diff_d(out.primal, real_Y), 1e-9)
        << "JVP primal must come from the real dispatched op, correctly "
           "shaped even when cols(reflectors) != order";
}

// Helper: symmetric diagonally dominant matrix (=> sytrf picks 1x1 pivots in
// order, no interchange) for the LDL tests.
static Tensor w4_sym_dd(uint64_t seed) {
    Tensor base = w4_randf64({4,4}, seed) * 0.2;
    const double* b = base.data<double>();
    Tensor A(std::vector<int64_t>{4,4}, DType::Float64, Device::cpu());
    double* d = A.data<double>();
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) d[i*4+j] = 0.5*(b[i*4+j]+b[j*4+i]);
    for (int i=0;i<4;++i) d[i*4+i] += 5.0 + i;
    return A;
}

// LinalgLDLFactor JVP (symmetric A = L D Lᵀ, no interchange).
TEST_P(JVPRulesTest, LinalgLDLFactor_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_sym_dd(191);
    w4_verify(OpId::LinalgLDLFactor, {A}, {0}, 0, a, 1e-3);  // dLD
}

// LinalgLDLSolve JVP. Inputs (LD, pivots, B) from ldl_factor of a symmetric DD A.
TEST_P(JVPRulesTest, LinalgLDLSolve_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_sym_dd(193);
    auto f = tenzor::dispatch(OpId::LinalgLDLFactor, std::vector<Tensor>{A}, a);
    Tensor B = w4_randf64({4,2}, 194);
    w4_verify_single(OpId::LinalgLDLSolve, {f[0], f[1], B}, {0,2}, a, 1e-3);
}

// Geqrf JVP: packed (R, reflectors) + tau. dlarfg recurrence dual-propagated.
TEST_P(JVPRulesTest, Geqrf_JVP_MatchesFD) {
    if (device != Device::cpu()) return;
    OpAttributes a;
    Tensor A = w4_randf64({4,3}, 201) * 0.5;
    { double* d = A.data<double>(); for (int i=0;i<3;++i) d[i*3+i] += 3.0; }
    w4_verify(OpId::Geqrf, {A}, {0}, 0, a, 1e-4);  // d(packed)
    w4_verify(OpId::Geqrf, {A}, {0}, 1, a, 1e-4);  // d(tau)
}

// NOTE: LinalgLU / LinalgSVD / LinalgEig / LDLFactor / Geqrf / RNN-forward JVP
// adapters are drafted (LU/SVD parked in jvp_rules.cpp) but do not yet pass
// this finite-difference gradcheck, so their OpIds remain registered to the
// NonDifferentiable thrower (forward-mode AD fails loudly rather than returning
// a wrong tangent) until the analytic rule is gradcheck-clean.

// GridSample forward-mode JVP w.r.t. the GRID (release-audit WS15). The grid
// tangent previously threw NonDifferentiable; it is now computed analytically
// for bilinear mode via shifted integer-pixel re-sampling and must match FD.
TEST_P(JVPRulesTest, GridSample_GridJVP_MatchesFD) {
    if (device != Device::cpu()) return;  // Float64 FD reference is CPU-side.
    auto input = w4_randf64({1, 2, 5, 6}, 11);
    auto grid  = w4_randf64({1, 4, 4, 2}, 12) * 0.5;  // mostly-interior coords
    OpAttributes a;
    a.set(AttrKey::Mode, std::string("bilinear"));
    a.set(AttrKey::PaddingMode, std::string("zeros"));
    a.set(AttrKey::AlignCorners, false);
    // diff_primals = {1}: differentiate w.r.t. the grid (the fixed path).
    w4_verify_single(OpId::GridSample, {input, grid}, {1}, a, 1e-5);
}

TEST_P(JVPRulesTest, GridSample_GridJVP_AlignCorners_MatchesFD) {
    if (device != Device::cpu()) return;
    auto input = w4_randf64({1, 3, 4, 4}, 13);
    auto grid  = w4_randf64({1, 5, 5, 2}, 14) * 0.5;
    OpAttributes a;
    a.set(AttrKey::Mode, std::string("bilinear"));
    a.set(AttrKey::PaddingMode, std::string("border"));
    a.set(AttrKey::AlignCorners, true);
    w4_verify_single(OpId::GridSample, {input, grid}, {1}, a, 1e-5);
}

INSTANTIATE_BACKEND_TESTS(JVPRulesTest);

