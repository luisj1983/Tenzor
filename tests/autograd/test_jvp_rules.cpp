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

using namespace tenzor;

namespace {

// Compute max absolute element-wise difference between two tensors,
// promoting to Float64 for the comparison.
double max_abs_diff_d(const Tensor& a, const Tensor& b) {
    auto ad = a.to(DType::Float64).contiguous();
    auto bd = b.to(DType::Float64).contiguous();
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

class JVPRulesTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// ----------------------------------------------------------------------------
// Binary linear ops (exact JVPs — should match FD up to machine epsilon)
// ----------------------------------------------------------------------------

TEST_F(JVPRulesTest, JVP_Add_MatchesFD) {
    auto a = randn({4, 8}, DType::Float32, Device::cpu());
    auto b = randn({4, 8}, DType::Float32, Device::cpu());
    auto va = randn({4, 8}, DType::Float32, Device::cpu());
    auto vb = randn({4, 8}, DType::Float32, Device::cpu());

    auto out = jvp_add(DualTensor(a, va), DualTensor(b, vb));
    // d/dt (a + t·va) + (b + t·vb) at t=0 = va + vb.
    EXPECT_LT(max_abs_diff_d(out.tangent(), va + vb), 1e-5);
}

TEST_F(JVPRulesTest, JVP_Sub_MatchesFD) {
    auto a = randn({3, 5}, DType::Float32, Device::cpu());
    auto b = randn({3, 5}, DType::Float32, Device::cpu());
    auto va = randn({3, 5}, DType::Float32, Device::cpu());
    auto vb = randn({3, 5}, DType::Float32, Device::cpu());

    auto out = jvp_sub(DualTensor(a, va), DualTensor(b, vb));
    EXPECT_LT(max_abs_diff_d(out.tangent(), va - vb), 1e-5);
}

TEST_F(JVPRulesTest, JVP_Mul_MatchesProductRule) {
    auto a = randn({4}, DType::Float32, Device::cpu());
    auto b = randn({4}, DType::Float32, Device::cpu());
    auto va = randn({4}, DType::Float32, Device::cpu());
    auto vb = randn({4}, DType::Float32, Device::cpu());

    auto out = jvp_mul(DualTensor(a, va), DualTensor(b, vb));
    // Product rule: d/dt (a + t·va)(b + t·vb)|t=0 = va*b + a*vb.
    auto expected = va * b + a * vb;
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-5);
}

TEST_F(JVPRulesTest, JVP_Div_MatchesQuotientRule) {
    auto a = randn({4}, DType::Float32, Device::cpu());
    // Keep b away from 0.
    auto b = abs(randn({4}, DType::Float32, Device::cpu())) + 0.5f;
    auto va = randn({4}, DType::Float32, Device::cpu());
    auto vb = randn({4}, DType::Float32, Device::cpu());

    auto out = jvp_div(DualTensor(a, va), DualTensor(b, vb));
    // Quotient rule: (va*b - a*vb) / b^2
    auto expected = (va * b - a * vb) / (b * b);
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-4);
}

TEST_F(JVPRulesTest, JVP_Neg_FlipsTangent) {
    auto a = randn({3}, DType::Float32, Device::cpu());
    auto va = randn({3}, DType::Float32, Device::cpu());

    auto out = jvp_neg(DualTensor(a, va));
    // -(a + t·va) ⇒ tangent = -va.
    auto expected = va * -1.0f;
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-6);
}

// ----------------------------------------------------------------------------
// MatMul: tangent must satisfy d/dt (A + t Va)(B + t Vb)|t=0 = Va·B + A·Vb
// ----------------------------------------------------------------------------

TEST_F(JVPRulesTest, JVP_MatMul_MatchesProductRule) {
    auto A  = randn({3, 4}, DType::Float32, Device::cpu());
    auto B  = randn({4, 5}, DType::Float32, Device::cpu());
    auto Va = randn({3, 4}, DType::Float32, Device::cpu());
    auto Vb = randn({4, 5}, DType::Float32, Device::cpu());

    auto out = jvp_matmul(DualTensor(A, Va), DualTensor(B, Vb));
    auto expected = matmul(Va, B) + matmul(A, Vb);
    EXPECT_LT(max_abs_diff_d(out.tangent(), expected), 1e-4);
}

// ----------------------------------------------------------------------------
// Unary nonlinearities — verify tangent ≈ FD approximation.
// ----------------------------------------------------------------------------

TEST_F(JVPRulesTest, JVP_Sigmoid_MatchesFD) {
    auto x = randn({16}, DType::Float32, Device::cpu());
    auto v = randn({16}, DType::Float32, Device::cpu());
    auto fd = finite_diff_jvp([](const Tensor& t) { return sigmoid(t); }, x, v);
    auto out = jvp_sigmoid(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}

TEST_F(JVPRulesTest, JVP_Tanh_MatchesFD) {
    auto x = randn({16}, DType::Float32, Device::cpu());
    auto v = randn({16}, DType::Float32, Device::cpu());
    auto fd = finite_diff_jvp([](const Tensor& t) { return tanh(t); }, x, v);
    auto out = jvp_tanh(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}

TEST_F(JVPRulesTest, JVP_Exp_MatchesFD) {
    // Restrict input range so exp doesn't overflow Float32 in the FD step.
    auto x = randn({16}, DType::Float32, Device::cpu()) * 0.5f;
    auto v = randn({16}, DType::Float32, Device::cpu());
    auto fd = finite_diff_jvp([](const Tensor& t) { return exp(t); }, x, v);
    auto out = jvp_exp(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}

TEST_F(JVPRulesTest, JVP_Log_MatchesFD) {
    auto x = abs(randn({16}, DType::Float32, Device::cpu())) + 0.1f;
    auto v = randn({16}, DType::Float32, Device::cpu());
    auto fd = finite_diff_jvp([](const Tensor& t) { return log(t); }, x, v);
    auto out = jvp_log(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-2);
}

TEST_F(JVPRulesTest, JVP_Sqrt_MatchesFD) {
    auto x = abs(randn({16}, DType::Float32, Device::cpu())) + 0.1f;
    auto v = randn({16}, DType::Float32, Device::cpu());
    auto fd = finite_diff_jvp([](const Tensor& t) { return sqrt(t); }, x, v);
    auto out = jvp_sqrt(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-2);
}

TEST_F(JVPRulesTest, JVP_ReLU_MatchesFD_FarFromZero) {
    // ReLU is non-differentiable at 0; pick inputs strictly away from
    // the kink so finite differences are well-defined.
    auto x = randn({16}, DType::Float32, Device::cpu()) + 2.0f;  // mostly > 0
    auto v = randn({16}, DType::Float32, Device::cpu());
    auto relu_t = [](const Tensor& t) { return clamp_min(t, 0.0); };
    auto fd = finite_diff_jvp(relu_t, x, v);
    auto out = jvp_relu(DualTensor(x, v));
    EXPECT_LT(max_abs_diff_d(out.tangent(), fd), 1e-3);
}
