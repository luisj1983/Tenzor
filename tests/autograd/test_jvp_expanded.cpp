/**
 * @file test_jvp_expanded.cpp
 * @brief Tests for expanded JVP (forward-mode AD) rules
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/jvp_rules.hpp>
#include <tenzor/autograd/dual.hpp>
#include <cmath>
#include "../backend_test_fixture.hpp"

using namespace tenzor;

// Parameterized over all backends via BackendTest: each TEST_P builds its
// primal/tangent tensors on the fixture's `device`. JVP rule ops carry no
// device argument — they inherit the device of their input DualTensors.
class JVPExpanded : public ::tenzor::testing::BackendTest {};

// Helper: create randn with small scale for numerical stability
static auto small_randn(std::vector<int64_t> shape, const Device& device, float scale = 0.5f) -> Tensor {
    return tenzor::mul(tenzor::randn(shape, DType::Float32, device), scale);
}

// Helper: create positive random values away from zero
static auto pos_rand(std::vector<int64_t> shape, const Device& device, float offset = 0.5f) -> Tensor {
    return tenzor::add(tenzor::abs(tenzor::randn(shape, DType::Float32, device)), offset);
}

// Helper: numerical JVP via finite differences
static auto numerical_jvp(std::function<Tensor(const Tensor&)> fn,
                           const Tensor& primal, const Tensor& tangent,
                           double eps = 1e-4) -> Tensor {
    auto f_plus = fn(tenzor::add(primal, tenzor::mul(tangent, eps)));
    auto f_minus = fn(tenzor::sub(primal, tenzor::mul(tangent, eps)));
    return tenzor::mul(tenzor::sub(f_plus, f_minus), 0.5 / eps);
}

// High-precision variant: compute the finite-difference reference in float64
// on the CPU and return float32 for check_jvp. The analytical JVP under test
// runs in the input's (float32) dtype; a float32 central-difference reference
// accrues ~machine_eps/eps ≈ 1e-3 of roundoff at the default eps=1e-4 — right
// at the 1e-3 check tolerance — so a correct kernel can fail purely from
// vendor-specific exp/sum rounding (e.g. rocm LogSoftmax). float64 drops that
// roundoff to ~1e-11, isolating the analytical kernel's own accuracy. Only
// usable when `fn` depends solely on its (now float64-CPU) input — NOT for
// tests whose lambda captures device tensors (e.g. Linear captures w_p/b_p),
// which would raise a cross-device error.
static auto numerical_jvp_f64(std::function<Tensor(const Tensor&)> fn,
                              const Tensor& primal, const Tensor& tangent,
                              double eps = 1e-4) -> Tensor {
    auto p64 = primal.cpu().to(DType::Float64);
    auto t64 = tangent.cpu().to(DType::Float64);
    auto f_plus = fn(tenzor::add(p64, tenzor::mul(t64, eps)));
    auto f_minus = fn(tenzor::sub(p64, tenzor::mul(t64, eps)));
    return tenzor::mul(tenzor::sub(f_plus, f_minus), 0.5 / eps).to(DType::Float32);
}

// Helper: check JVP against numerical
static void check_jvp(const Tensor& analytical, const Tensor& numerical,
                       const char* name, float tol = 1e-3f) {
    auto a = analytical.cpu().contiguous();
    auto n = numerical.cpu().contiguous();
    auto* ad = a.data<float>();
    auto* nd = n.data<float>();
    int64_t numel = a.numel();
    for (int64_t i = 0; i < numel; ++i) {
        EXPECT_NEAR(ad[i], nd[i], tol)
            << name << " JVP mismatch at index " << i
            << " (analytical=" << ad[i] << " numerical=" << nd[i] << ")";
    }
}

// =========================================================================
// Trig & Hyperbolic
// =========================================================================

TEST_P(JVPExpanded, Tan) {
    // Keep |p| small: tan's derivative sec^2 blows up near +/-pi/2, where the
    // 1e-4 central-difference reference accrues O(eps*tan''') truncation error
    // that exceeds the 1e-3 tolerance. 0.25 scale keeps inputs well-conditioned.
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, device), 0.25f);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_tan(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::tan(x); }, p, t);
    check_jvp(result.tangent(), expected, "tan");
}

TEST_P(JVPExpanded, Asin) {
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, device), 0.3f);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_asin(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::asin(x); }, p, t);
    check_jvp(result.tangent(), expected, "asin");
}

TEST_P(JVPExpanded, Sinh) {
    // Use small values to avoid numerical instability (sinh grows exponentially)
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, device), 0.5f);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_sinh(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::sinh(x); }, p, t);
    check_jvp(result.tangent(), expected, "sinh", 0.01f);
}

TEST_P(JVPExpanded, Cosh) {
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, device), 0.5f);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_cosh(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::cosh(x); }, p, t);
    check_jvp(result.tangent(), expected, "cosh", 0.01f);
}

// =========================================================================
// Extended math
// =========================================================================

TEST_P(JVPExpanded, Log2) {
    auto p = tenzor::add(tenzor::rand({4}, DType::Float32, device), 0.5f);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_log2(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::log2(x); }, p, t);
    check_jvp(result.tangent(), expected, "log2");
}

TEST_P(JVPExpanded, Log1p) {
    auto p = tenzor::add(tenzor::rand({4}, DType::Float32, device), 0.1f);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_log1p(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::log1p(x); }, p, t);
    check_jvp(result.tangent(), expected, "log1p");
}

TEST_P(JVPExpanded, Reciprocal) {
    // Ensure values are away from zero for numerical stability
    auto p = tenzor::add(tenzor::abs(tenzor::randn({4}, DType::Float32, device)), 1.0f);
    auto t = tenzor::mul(tenzor::randn({4}, DType::Float32, device), 0.1f);
    auto result = jvp_reciprocal(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::reciprocal(x); }, p, t);
    check_jvp(result.tangent(), expected, "reciprocal", 0.01f);
}

TEST_P(JVPExpanded, Erf) {
    auto p = tenzor::randn({4}, DType::Float32, device);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_erf(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::erf(x); }, p, t);
    check_jvp(result.tangent(), expected, "erf");
}

// =========================================================================
// Activations
// =========================================================================

TEST_P(JVPExpanded, LeakyReLU) {
    auto p = tenzor::randn({8}, DType::Float32, device);
    auto t = tenzor::randn({8}, DType::Float32, device);
    auto result = jvp_leaky_relu(DualTensor(p, t), 0.1f);
    // Numerical check: leaky_relu(x) = max(0,x) + 0.1*min(0,x)
    auto expected = numerical_jvp([](const Tensor& x) {
        auto zero = tenzor::zeros_like(x);
        auto pos = tenzor::mul(tenzor::gt(x, zero), x);
        auto neg = tenzor::mul(tenzor::sub(tenzor::ones_like(x), tenzor::gt(x, zero)), tenzor::mul(x, 0.1));
        return tenzor::add(pos, neg);
    }, p, t);
    check_jvp(result.tangent(), expected, "leaky_relu", 0.02f);
}

TEST_P(JVPExpanded, Softplus) {
    auto p = small_randn({4}, device);
    auto t = small_randn({4}, device);
    auto result = jvp_softplus(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) {
        auto one = tenzor::ones_like(x);
        return tenzor::log(tenzor::add(one, tenzor::exp(x)));
    }, p, t);
    check_jvp(result.tangent(), expected, "softplus", 0.01f);
}

TEST_P(JVPExpanded, Mish) {
    auto p = tenzor::randn({4}, DType::Float32, device);
    auto t = tenzor::randn({4}, DType::Float32, device);
    auto result = jvp_mish(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) {
        auto one = tenzor::ones_like(x);
        auto sp = tenzor::log(tenzor::add(one, tenzor::exp(x)));
        return tenzor::mul(x, tenzor::tanh(sp));
    }, p, t);
    check_jvp(result.tangent(), expected, "mish");
}

// =========================================================================
// Softmax
// =========================================================================

TEST_P(JVPExpanded, Softmax) {
    auto p = tenzor::randn({2, 4}, DType::Float32, device);
    auto t = tenzor::randn({2, 4}, DType::Float32, device);
    auto result = jvp_softmax(DualTensor(p, t), /*dim=*/1);
    auto expected = numerical_jvp([](const Tensor& x) {
        auto m = tenzor::max(x, 1, true);
        auto e = tenzor::exp(tenzor::sub(x, m));
        return tenzor::div(e, tenzor::sum(e, 1, true));
    }, p, t);
    check_jvp(result.tangent(), expected, "softmax");
}

TEST_P(JVPExpanded, LogSoftmax) {
    // d(log_softmax)_i = dt_i - sum_j(softmax(x)_j * dt_j); the subtracted
    // term is a per-row scalar, not per-i, so summing the tangent over the
    // row does NOT generally vanish (that invariant only holds for uniform
    // softmax / constant dt) — sum_i(dy_i) = sum(dt) - K*sum_j(s_j*dt_j),
    // which is nonzero for a generic random x/dt. Verify against a real
    // numerical (finite-difference) JVP instead, matching the Softmax test
    // immediately above.
    auto p = tenzor::mul(tenzor::randn({2, 4}, DType::Float32, device), 0.5f);
    auto t = tenzor::randn({2, 4}, DType::Float32, device);
    auto result = jvp_log_softmax(DualTensor(p, t), /*dim=*/1);
    EXPECT_EQ(result.tangent().shape()[0], 2);
    EXPECT_EQ(result.tangent().shape()[1], 4);
    auto expected = numerical_jvp_f64([](const Tensor& x) {
        auto m = tenzor::max(x, 1, true);
        auto e = tenzor::exp(tenzor::sub(x, m));
        auto se = tenzor::sum(e, 1, true);
        return tenzor::sub(x, tenzor::add(tenzor::log(se), m));
    }, p, t);
    check_jvp(result.tangent(), expected, "log_softmax");
}

// =========================================================================
// Shape ops
// =========================================================================

TEST_P(JVPExpanded, Permute) {
    auto p = tenzor::randn({2, 3, 4}, DType::Float32, device);
    auto t = tenzor::randn({2, 3, 4}, DType::Float32, device);
    auto result = jvp_permute(DualTensor(p, t), {2, 0, 1});
    EXPECT_EQ(result.primal().shape()[0], 4);
    EXPECT_EQ(result.primal().shape()[1], 2);
    EXPECT_EQ(result.primal().shape()[2], 3);
    EXPECT_EQ(result.tangent().shape()[0], 4);
}

TEST_P(JVPExpanded, Cat) {
    auto p1 = tenzor::randn({2, 3}, DType::Float32, device);
    auto t1 = tenzor::randn({2, 3}, DType::Float32, device);
    auto p2 = tenzor::randn({2, 3}, DType::Float32, device);
    auto t2 = tenzor::randn({2, 3}, DType::Float32, device);
    std::vector<DualTensor> duals = {DualTensor(p1, t1), DualTensor(p2, t2)};
    auto result = jvp_cat(duals, 0);
    EXPECT_EQ(result.primal().shape()[0], 4);
    EXPECT_EQ(result.tangent().shape()[0], 4);
}

TEST_P(JVPExpanded, Linear) {
    auto x_p = tenzor::randn({2, 4}, DType::Float32, device);
    auto x_t = tenzor::randn({2, 4}, DType::Float32, device);
    auto w_p = tenzor::randn({3, 4}, DType::Float32, device);
    auto w_t = tenzor::randn({3, 4}, DType::Float32, device);
    auto b_p = tenzor::randn({3}, DType::Float32, device);
    auto b_t = tenzor::randn({3}, DType::Float32, device);

    auto result = jvp_linear(DualTensor(x_p, x_t), DualTensor(w_p, w_t), DualTensor(b_p, b_t));
    EXPECT_EQ(result.primal().shape()[0], 2);
    EXPECT_EQ(result.primal().shape()[1], 3);

    // Numerical check
    auto expected = numerical_jvp([&](const Tensor& x) {
        return tenzor::add(tenzor::matmul(x, tenzor::transpose(w_p, 0, 1)), b_p);
    }, x_p, x_t);
    // This only checks the input tangent contribution, not weight/bias
    // Full check would need multi-input finite differences
    // Just verify shapes match
    EXPECT_EQ(result.tangent().shape()[0], 2);
    EXPECT_EQ(result.tangent().shape()[1], 3);
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a JVP-rule op throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(JVPExpanded);
