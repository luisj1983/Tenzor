/**
 * @file test_jvp_expanded_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for expanded JVP (forward-mode AD) rules
 *
 * Multi-backend port of test_jvp_expanded.cpp. Tests JVP rules for trig,
 * hyperbolic, extended math, activations, softmax, and shape operations
 * across all available backends and data types.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/jvp_rules.hpp>
#include <tenzor/autograd/dual.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// Macro (not a method) so that GTEST_SKIP's internal `return`
// statement returns from the TEST_P body rather than from a helper
// method — otherwise the test continues and fails on the first op
// that doesn't support Float16.
#define skipIfHalf() \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, \
                             "JVP finite-difference noise dominates at Float16"); \
        } \
    } while (0)

class JVPExpandedMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Create randn with small scale for numerical stability
    Tensor small_randn(std::vector<int64_t> shape, float scale = 0.5f) {
        return tenzor::mul(tenzor::randn(shape, dtype(), device()), scale);
    }

    // Create positive random values away from zero
    Tensor pos_rand(std::vector<int64_t> shape, float offset = 0.5f) {
        return tenzor::add(tenzor::abs(tenzor::randn(shape, dtype(), device())), offset);
    }

    // Numerical JVP via finite differences
    Tensor numerical_jvp(std::function<Tensor(const Tensor&)> fn,
                         const Tensor& primal, const Tensor& tangent,
                         double eps = 1e-4) {
        // Evaluate f_plus/f_minus in Float64, not just their final
        // subtraction: f_plus and f_minus are O(1) values differing by only
        // O(eps), so each already carries ~1e-7 ABSOLUTE rounding error from
        // its own Float32 internal computation chain (max/sub/exp/sum/log/...
        // for log_softmax) by the time it's produced -- relative to the
        // O(eps)=O(1e-4) true difference, that is already a ~1e-3 relative
        // error, so subtracting in higher precision afterward cannot recover
        // it (confirmed: staging only the final subtraction in Float64 left
        // the observed Vulkan/Float32 log_softmax mismatch byte-identical).
        // The FD reference must be computed end-to-end in Float64 to be
        // meaningfully more accurate than the Float32 analytical JVP it
        // checks, matching gradcheck.cpp's established widen-for-FD-
        // precision pattern. fn's composed Tensor ops are dtype-generic, so
        // this just exercises the same kernels one precision tier up.
        const DType orig_dtype = primal.dtype();
        auto primal64 = primal.to(DType::Float64);
        auto tangent64 = tangent.to(DType::Float64);
        auto f_plus = fn(tenzor::add(primal64, tenzor::mul(tangent64, eps)));
        auto f_minus = fn(tenzor::sub(primal64, tenzor::mul(tangent64, eps)));
        auto diff = tenzor::sub(f_plus.to(DType::Float64), f_minus.to(DType::Float64));
        return tenzor::mul(diff, 0.5 / eps).to(orig_dtype);
    }

    // Check JVP against numerical with dtype-aware tolerance
    void check_jvp(const Tensor& analytical, const Tensor& numerical,
                   const char* name, float tol_scale = 1.0f) {
        float tol = std::max(atol() * 10.0f * tol_scale, 1e-3f * tol_scale);
        auto a = analytical.to(Device::cpu()).to(DType::Float32).contiguous();
        auto n = numerical.to(Device::cpu()).to(DType::Float32).contiguous();
        auto* ad = a.data<float>();
        auto* nd = n.data<float>();
        int64_t numel = a.numel();
        for (int64_t i = 0; i < numel; ++i) {
            EXPECT_NEAR(ad[i], nd[i], tol)
                << name << " JVP mismatch at index " << i
                << " (analytical=" << ad[i] << " numerical=" << nd[i]
                << ") on " << device().to_string();
        }
    }
};

// =========================================================================
// Trig & Hyperbolic
// =========================================================================

TEST_P(JVPExpandedMultiDTypeTest, Tan) {
    skipIfHalf();
    // Scale 0.3 (vs default small_randn 0.5) keeps |p| safely away from
    // pi/2 ~ 1.57. At scale 0.5 a randn 3-sigma sample lands at 1.5, where
    // tan's third derivative explodes (~200) and FD truncation noise
    // dominates the 1e-3 absolute tolerance — flaky on Vulkan/ROCm where
    // tan kernel ULPs differ from CPU's correctly-rounded std::tan.
    auto p = tenzor::mul(tenzor::randn({4}, dtype(), device()), 0.3f);
    auto t = tenzor::randn({4}, dtype(), device());
    auto result = jvp_tan(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::tan(x); }, p, t);
    check_jvp(result.tangent(), expected, "tan");
}

TEST_P(JVPExpandedMultiDTypeTest, Asin) {
    skipIfHalf();
    auto p = tenzor::mul(tenzor::randn({4}, dtype(), device()), 0.3f);
    auto t = tenzor::randn({4}, dtype(), device());
    auto result = jvp_asin(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::asin(x); }, p, t);
    check_jvp(result.tangent(), expected, "asin");
}

TEST_P(JVPExpandedMultiDTypeTest, Sinh) {
    skipIfHalf();
    auto p = small_randn({4});
    auto t = tenzor::randn({4}, dtype(), device());
    auto result = jvp_sinh(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::sinh(x); }, p, t);
    check_jvp(result.tangent(), expected, "sinh", 10.0f);
}

TEST_P(JVPExpandedMultiDTypeTest, Cosh) {
    skipIfHalf();
    auto p = small_randn({4});
    auto t = tenzor::randn({4}, dtype(), device());
    auto result = jvp_cosh(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::cosh(x); }, p, t);
    check_jvp(result.tangent(), expected, "cosh", 10.0f);
}

// =========================================================================
// Extended math
// =========================================================================

TEST_P(JVPExpandedMultiDTypeTest, Log2) {
    skipIfHalf();
    auto p = pos_rand({4});
    auto t = tenzor::randn({4}, dtype(), device());
    auto result = jvp_log2(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::log2(x); }, p, t);
    check_jvp(result.tangent(), expected, "log2");
}

TEST_P(JVPExpandedMultiDTypeTest, Log1p) {
    skipIfHalf();
    auto p = tenzor::add(tenzor::rand({4}, dtype(), device()), 0.1f);
    auto t = tenzor::randn({4}, dtype(), device());
    auto result = jvp_log1p(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::log1p(x); }, p, t);
    check_jvp(result.tangent(), expected, "log1p");
}

TEST_P(JVPExpandedMultiDTypeTest, Reciprocal) {
    skipIfHalf();
    auto p = tenzor::add(tenzor::abs(tenzor::randn({4}, dtype(), device())), 1.0f);
    auto t = tenzor::mul(tenzor::randn({4}, dtype(), device()), 0.1f);
    auto result = jvp_reciprocal(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::reciprocal(x); }, p, t);
    check_jvp(result.tangent(), expected, "reciprocal", 10.0f);
}

TEST_P(JVPExpandedMultiDTypeTest, Erf) {
    skipIfHalf();
    auto p = tenzor::randn({4}, dtype(), device());
    auto t = tenzor::randn({4}, dtype(), device());
    auto result = jvp_erf(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) { return tenzor::erf(x); }, p, t);
    check_jvp(result.tangent(), expected, "erf");
}

// =========================================================================
// Activations
// =========================================================================

TEST_P(JVPExpandedMultiDTypeTest, LeakyReLU) {
    skipIfHalf();
    auto p = tenzor::randn({8}, dtype(), device());
    auto t = tenzor::randn({8}, dtype(), device());
    auto result = jvp_leaky_relu(DualTensor(p, t), 0.1f);
    auto expected = numerical_jvp([](const Tensor& x) {
        auto zero = tenzor::zeros_like(x);
        auto pos = tenzor::mul(tenzor::gt(x, zero), x);
        auto neg = tenzor::mul(tenzor::sub(tenzor::ones_like(x), tenzor::gt(x, zero)), tenzor::mul(x, 0.1));
        return tenzor::add(pos, neg);
    }, p, t);
    check_jvp(result.tangent(), expected, "leaky_relu", 20.0f);
}

TEST_P(JVPExpandedMultiDTypeTest, Softplus) {
    skipIfHalf();
    auto p = small_randn({4});
    auto t = small_randn({4});
    auto result = jvp_softplus(DualTensor(p, t));
    auto expected = numerical_jvp([](const Tensor& x) {
        auto one = tenzor::ones_like(x);
        return tenzor::log(tenzor::add(one, tenzor::exp(x)));
    }, p, t);
    check_jvp(result.tangent(), expected, "softplus", 10.0f);
}

TEST_P(JVPExpandedMultiDTypeTest, Mish) {
    skipIfHalf();
    auto p = tenzor::randn({4}, dtype(), device());
    auto t = tenzor::randn({4}, dtype(), device());
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

TEST_P(JVPExpandedMultiDTypeTest, Softmax) {
    skipIfHalf();
    auto p = tenzor::randn({2, 4}, dtype(), device());
    auto t = tenzor::randn({2, 4}, dtype(), device());
    auto result = jvp_softmax(DualTensor(p, t), /*dim=*/1);
    auto expected = numerical_jvp([](const Tensor& x) {
        auto m = tenzor::max(x, 1, true);
        auto e = tenzor::exp(tenzor::sub(x, m));
        return tenzor::div(e, tenzor::sum(e, 1, true));
    }, p, t);
    check_jvp(result.tangent(), expected, "softmax");
}

TEST_P(JVPExpandedMultiDTypeTest, LogSoftmax) {
    skipIfHalf();
    // d(log_softmax)_i = dt_i - sum_j(softmax(x)_j * dt_j); the subtracted
    // term is a per-row scalar, not per-i, so summing the tangent over the
    // row does NOT generally vanish (that invariant only holds for uniform
    // softmax / constant dt) — sum_i(dy_i) = sum(dt) - K*sum_j(s_j*dt_j),
    // which is nonzero for a generic random x/dt. Verify against a real
    // numerical (finite-difference) JVP instead, matching the Softmax test
    // immediately above.
    auto p = tenzor::mul(tenzor::randn({2, 4}, dtype(), device()), 0.5f);
    auto t = tenzor::randn({2, 4}, dtype(), device());
    auto result = jvp_log_softmax(DualTensor(p, t), /*dim=*/1);
    EXPECT_EQ(result.tangent().shape()[0], 2);
    EXPECT_EQ(result.tangent().shape()[1], 4);
    auto expected = numerical_jvp([](const Tensor& x) {
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

TEST_P(JVPExpandedMultiDTypeTest, Permute) {
    skipIfHalf();
    auto p = tenzor::randn({2, 3, 4}, dtype(), device());
    auto t = tenzor::randn({2, 3, 4}, dtype(), device());
    auto result = jvp_permute(DualTensor(p, t), {2, 0, 1});
    EXPECT_EQ(result.primal().shape()[0], 4);
    EXPECT_EQ(result.primal().shape()[1], 2);
    EXPECT_EQ(result.primal().shape()[2], 3);
    EXPECT_EQ(result.tangent().shape()[0], 4);
}

TEST_P(JVPExpandedMultiDTypeTest, Cat) {
    skipIfHalf();
    auto p1 = tenzor::randn({2, 3}, dtype(), device());
    auto t1 = tenzor::randn({2, 3}, dtype(), device());
    auto p2 = tenzor::randn({2, 3}, dtype(), device());
    auto t2 = tenzor::randn({2, 3}, dtype(), device());
    std::vector<DualTensor> duals = {DualTensor(p1, t1), DualTensor(p2, t2)};
    auto result = jvp_cat(duals, 0);
    EXPECT_EQ(result.primal().shape()[0], 4);
    EXPECT_EQ(result.tangent().shape()[0], 4);
}

TEST_P(JVPExpandedMultiDTypeTest, Linear) {
    skipIfHalf();
    auto x_p = tenzor::randn({2, 4}, dtype(), device());
    auto x_t = tenzor::randn({2, 4}, dtype(), device());
    auto w_p = tenzor::randn({3, 4}, dtype(), device());
    auto w_t = tenzor::randn({3, 4}, dtype(), device());
    auto b_p = tenzor::randn({3}, dtype(), device());
    auto b_t = tenzor::randn({3}, dtype(), device());

    auto result = jvp_linear(DualTensor(x_p, x_t), DualTensor(w_p, w_t), DualTensor(b_p, b_t));
    EXPECT_EQ(result.primal().shape()[0], 2);
    EXPECT_EQ(result.primal().shape()[1], 3);
    EXPECT_EQ(result.tangent().shape()[0], 2);
    EXPECT_EQ(result.tangent().shape()[1], 3);
}

// =========================================================================
// Instantiation
// =========================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(JVPExpandedMultiDTypeTest);
