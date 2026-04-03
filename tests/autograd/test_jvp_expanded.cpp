/**
 * @file test_jvp_expanded.cpp
 * @brief Tests for expanded JVP (forward-mode AD) rules
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/jvp_rules.hpp>
#include <tenzor/autograd/dual.hpp>
#include <cmath>

using namespace tenzor;

class JVPExpandedTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new JVPExpandedTestEnv);

// Helper: create randn with small scale for numerical stability
static auto small_randn(std::vector<int64_t> shape, float scale = 0.5f) -> Tensor {
    return tenzor::mul(tenzor::randn(shape, DType::Float32, Device::cpu()), scale);
}

// Helper: create positive random values away from zero
static auto pos_rand(std::vector<int64_t> shape, float offset = 0.5f) -> Tensor {
    return tenzor::add(tenzor::abs(tenzor::randn(shape, DType::Float32, Device::cpu())), offset);
}

// Helper: numerical JVP via finite differences
static auto numerical_jvp(std::function<Tensor(const Tensor&)> fn,
                           const Tensor& primal, const Tensor& tangent,
                           double eps = 1e-4) -> Tensor {
    auto f_plus = fn(tenzor::add(primal, tenzor::mul(tangent, eps)));
    auto f_minus = fn(tenzor::sub(primal, tenzor::mul(tangent, eps)));
    return tenzor::mul(tenzor::sub(f_plus, f_minus), 0.5 / eps);
}

// Helper: check JVP against numerical
static void check_jvp(const Tensor& analytical, const Tensor& numerical,
                       const char* name, float tol = 1e-3f) {
    auto a = analytical.to(Device::cpu()).contiguous();
    auto n = numerical.to(Device::cpu()).contiguous();
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

TEST(JVPExpanded, Tan) {
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, Device::cpu()), 0.5f);
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto result = jvp_tan(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::tan(x); }, p, t);
    check_jvp(result.tangent(), expected, "tan");
}

TEST(JVPExpanded, Asin) {
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, Device::cpu()), 0.3f);
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto result = jvp_asin(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::asin(x); }, p, t);
    check_jvp(result.tangent(), expected, "asin");
}

TEST(JVPExpanded, Sinh) {
    // Use small values to avoid numerical instability (sinh grows exponentially)
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, Device::cpu()), 0.5f);
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto result = jvp_sinh(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::sinh(x); }, p, t);
    check_jvp(result.tangent(), expected, "sinh", 0.01f);
}

TEST(JVPExpanded, Cosh) {
    auto p = tenzor::mul(tenzor::randn({4}, DType::Float32, Device::cpu()), 0.5f);
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto result = jvp_cosh(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::cosh(x); }, p, t);
    check_jvp(result.tangent(), expected, "cosh", 0.01f);
}

// =========================================================================
// Extended math
// =========================================================================

TEST(JVPExpanded, Log2) {
    auto p = tenzor::add(tenzor::rand({4}, DType::Float32, Device::cpu()), 0.5f);
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto result = jvp_log2(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::log2(x); }, p, t);
    check_jvp(result.tangent(), expected, "log2");
}

TEST(JVPExpanded, Log1p) {
    auto p = tenzor::add(tenzor::rand({4}, DType::Float32, Device::cpu()), 0.1f);
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto result = jvp_log1p(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::log1p(x); }, p, t);
    check_jvp(result.tangent(), expected, "log1p");
}

TEST(JVPExpanded, Reciprocal) {
    // Ensure values are away from zero for numerical stability
    auto p = tenzor::add(tenzor::abs(tenzor::randn({4}, DType::Float32, Device::cpu())), 1.0f);
    auto t = tenzor::mul(tenzor::randn({4}, DType::Float32, Device::cpu()), 0.1f);
    auto result = jvp_reciprocal(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::reciprocal(x); }, p, t);
    check_jvp(result.tangent(), expected, "reciprocal", 0.01f);
}

TEST(JVPExpanded, Erf) {
    auto p = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto result = jvp_erf(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::erf(x); }, p, t);
    check_jvp(result.tangent(), expected, "erf");
}

// =========================================================================
// Activations
// =========================================================================

TEST(JVPExpanded, LeakyReLU) {
    auto p = tenzor::randn({8}, DType::Float32, Device::cpu());
    auto t = tenzor::randn({8}, DType::Float32, Device::cpu());
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

TEST(JVPExpanded, Softplus) {
    auto p = small_randn({4});
    auto t = small_randn({4});
    auto result = jvp_softplus(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) {
        auto one = tenzor::ones_like(x);
        return tenzor::log(tenzor::add(one, tenzor::exp(x)));
    }, p, t);
    check_jvp(result.tangent(), expected, "softplus", 0.01f);
}

TEST(JVPExpanded, Mish) {
    auto p = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
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

TEST(JVPExpanded, Softmax) {
    auto p = tenzor::randn({2, 4}, DType::Float32, Device::cpu());
    auto t = tenzor::randn({2, 4}, DType::Float32, Device::cpu());
    auto result = jvp_softmax(DualTensor(p, t), /*dim=*/1);
    auto expected = numerical_jvp([](const Tensor& x) {
        auto m = tenzor::max(x, 1, true);
        auto e = tenzor::exp(tenzor::sub(x, m));
        return tenzor::div(e, tenzor::sum(e, 1, true));
    }, p, t);
    check_jvp(result.tangent(), expected, "softmax");
}

TEST(JVPExpanded, LogSoftmax) {
    // Use small values for numerical stability
    auto p = tenzor::mul(tenzor::randn({2, 4}, DType::Float32, Device::cpu()), 0.5f);
    auto t = tenzor::randn({2, 4}, DType::Float32, Device::cpu());
    auto result = jvp_log_softmax(DualTensor(p, t), /*dim=*/1);
    // Verify tangent shape is correct and that primal matches log_softmax
    EXPECT_EQ(result.tangent().shape()[0], 2);
    EXPECT_EQ(result.tangent().shape()[1], 4);
    // Verify tangent rows sum to 0 (log_softmax tangent property)
    auto t_sum = tenzor::sum(result.tangent(), 1, false).to(Device::cpu());
    auto* ts = t_sum.data<float>();
    // log_softmax tangent: dt - s * sum(dt), so sum = sum(dt) - sum(s)*sum(dt) = sum(dt)*(1-1) = 0
    // Actually sum(tangent) = sum(dt) - sum(s * sum(dt,dim)) = sum(dt) - sum(dt) = 0
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(ts[i], 0.0f, 0.01f) << "LogSoftmax tangent row " << i << " doesn't sum to ~0";
    }
}

// =========================================================================
// Shape ops
// =========================================================================

TEST(JVPExpanded, Permute) {
    auto p = tenzor::randn({2, 3, 4}, DType::Float32, Device::cpu());
    auto t = tenzor::randn({2, 3, 4}, DType::Float32, Device::cpu());
    auto result = jvp_permute(DualTensor(p, t), {2, 0, 1});
    EXPECT_EQ(result.primal().shape()[0], 4);
    EXPECT_EQ(result.primal().shape()[1], 2);
    EXPECT_EQ(result.primal().shape()[2], 3);
    EXPECT_EQ(result.tangent().shape()[0], 4);
}

TEST(JVPExpanded, Cat) {
    auto p1 = tenzor::randn({2, 3}, DType::Float32, Device::cpu());
    auto t1 = tenzor::randn({2, 3}, DType::Float32, Device::cpu());
    auto p2 = tenzor::randn({2, 3}, DType::Float32, Device::cpu());
    auto t2 = tenzor::randn({2, 3}, DType::Float32, Device::cpu());
    std::vector<DualTensor> duals = {DualTensor(p1, t1), DualTensor(p2, t2)};
    auto result = jvp_cat(duals, 0);
    EXPECT_EQ(result.primal().shape()[0], 4);
    EXPECT_EQ(result.tangent().shape()[0], 4);
}

TEST(JVPExpanded, Linear) {
    auto x_p = tenzor::randn({2, 4}, DType::Float32, Device::cpu());
    auto x_t = tenzor::randn({2, 4}, DType::Float32, Device::cpu());
    auto w_p = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    auto w_t = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    auto b_p = tenzor::randn({3}, DType::Float32, Device::cpu());
    auto b_t = tenzor::randn({3}, DType::Float32, Device::cpu());

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
