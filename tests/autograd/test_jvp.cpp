/**
 * @file test_jvp.cpp
 * @brief Tests for forward-mode AD (JVP) rules and functional JVP interface
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/jvp_rules.hpp"
#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/dual.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;

class JVPTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        set_grad_enabled(true);
    }

    /// Helper: verify JVP matches finite differences
    /// (f(x + eps*v) - f(x)) / eps ≈ jvp(f, x, v)
    void verify_jvp_fd(std::function<DualTensor(const DualTensor&)> dual_func,
                       std::function<Tensor(const Tensor&)> primal_func,
                       const Tensor& x, const Tensor& v,
                       float atol = 0.05f) {
        // Compute JVP via dual numbers
        DualTensor dual_x(x, v);
        auto dual_out = dual_func(dual_x);
        auto jvp_tangent = dual_out.tangent();

        // Compute JVP via finite differences
        const float eps = 1e-4f;
        auto x_plus = tenzor::add(x, tenzor::mul(v, eps));
        auto x_minus = tenzor::sub(x, tenzor::mul(v, eps));
        auto fd_tangent = tenzor::mul(
            tenzor::sub(primal_func(x_plus), primal_func(x_minus)),
            1.0f / (2.0f * eps)
        );

        // Compare
        auto diff = tenzor::abs(tenzor::sub(jvp_tangent, fd_tangent));
        auto max_diff = tenzor::max(diff);
        float max_err = *max_diff.data<float>();
        EXPECT_LT(max_err, atol) << "JVP tangent does not match finite differences";
    }
};

// ============================================================================
// Arithmetic JVP tests
// ============================================================================

TEST_F(JVPTest, AddJVP) {
    auto a = tenzor::ones({3, 3}, DType::Float32, Device::cpu());
    auto b = tenzor::mul(tenzor::ones({3, 3}, DType::Float32, Device::cpu()), 2.0);
    auto ta = tenzor::ones({3, 3}, DType::Float32, Device::cpu());
    auto tb = tenzor::mul(tenzor::ones({3, 3}, DType::Float32, Device::cpu()), 0.5);

    DualTensor da(a, ta);
    DualTensor db(b, tb);
    auto result = jvp_add(da, db);

    // primal = a + b = 3.0
    auto primal_val = *tenzor::mean(result.primal()).data<float>();
    EXPECT_NEAR(primal_val, 3.0f, 1e-5f);

    // tangent = ta + tb = 1.5
    auto tangent_val = *tenzor::mean(result.tangent()).data<float>();
    EXPECT_NEAR(tangent_val, 1.5f, 1e-5f);
}

TEST_F(JVPTest, MulJVP) {
    // f(a,b) = a*b, so df = da*b + a*db
    auto x_data = tenzor::zeros({4}, DType::Float32, Device::cpu());
    auto v_data = tenzor::ones({4}, DType::Float32, Device::cpu());

    // Fill x with [1, 2, 3, 4]
    float* xp = x_data.data<float>();
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f; xp[3] = 4.0f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_mul(d, d); },
        [](const Tensor& t) { return tenzor::mul(t, t); },
        x_data, v_data
    );
}

TEST_F(JVPTest, DivJVP) {
    auto a = tenzor::zeros({3}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());
    float* ap = a.data<float>();
    ap[0] = 4.0f; ap[1] = 6.0f; ap[2] = 8.0f;

    auto b = tenzor::mul(tenzor::ones({3}, DType::Float32, Device::cpu()), 2.0);

    DualTensor da(a, v);
    DualTensor db(b, tenzor::zeros_like(b));  // b is constant
    auto result = jvp_div(da, db);

    // tangent = da/b = 1/2 = 0.5
    auto tangent_val = *tenzor::mean(result.tangent()).data<float>();
    EXPECT_NEAR(tangent_val, 0.5f, 1e-5f);
}

// ============================================================================
// Matmul JVP test
// ============================================================================

TEST_F(JVPTest, MatmulJVP) {
    auto a = tenzor::ones({2, 3}, DType::Float32, Device::cpu());
    auto b = tenzor::ones({3, 2}, DType::Float32, Device::cpu());
    auto ta = tenzor::zeros_like(a);
    auto tb = tenzor::ones({3, 2}, DType::Float32, Device::cpu());

    DualTensor da(a, ta);
    DualTensor db(b, tb);
    auto result = jvp_matmul(da, db);

    // primal = ones(2,3) @ ones(3,2) = 3*ones(2,2)
    auto primal_val = *tenzor::mean(result.primal()).data<float>();
    EXPECT_NEAR(primal_val, 3.0f, 1e-5f);

    // tangent = ta @ b + a @ tb = 0 + ones(2,3) @ ones(3,2) = 3*ones(2,2)
    auto tangent_val = *tenzor::mean(result.tangent()).data<float>();
    EXPECT_NEAR(tangent_val, 3.0f, 1e-5f);
}

// ============================================================================
// Activation JVP tests
// ============================================================================

TEST_F(JVPTest, SigmoidJVP) {
    auto x = tenzor::zeros({4}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({4}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = -1.0f; xp[1] = 0.0f; xp[2] = 0.5f; xp[3] = 1.0f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_sigmoid(d); },
        [](const Tensor& t) { return tenzor::sigmoid(t); },
        x, v
    );
}

TEST_F(JVPTest, TanhJVP) {
    auto x = tenzor::zeros({4}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({4}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = -1.0f; xp[1] = 0.0f; xp[2] = 0.5f; xp[3] = 1.0f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_tanh(d); },
        [](const Tensor& t) { return tenzor::tanh(t); },
        x, v
    );
}

// ============================================================================
// Math JVP tests
// ============================================================================

TEST_F(JVPTest, ExpJVP) {
    auto x = tenzor::zeros({4}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({4}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = 0.0f; xp[1] = 0.5f; xp[2] = 1.0f; xp[3] = -1.0f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_exp(d); },
        [](const Tensor& t) { return tenzor::exp(t); },
        x, v
    );
}

TEST_F(JVPTest, LogJVP) {
    auto x = tenzor::zeros({3}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = 0.5f; xp[1] = 1.0f; xp[2] = 2.0f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_log(d); },
        [](const Tensor& t) { return tenzor::log(t); },
        x, v
    );
}

TEST_F(JVPTest, SinJVP) {
    auto x = tenzor::zeros({4}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({4}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = 0.0f; xp[1] = 0.5f; xp[2] = 1.0f; xp[3] = 3.14159f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_sin(d); },
        [](const Tensor& t) { return tenzor::sin(t); },
        x, v
    );
}

TEST_F(JVPTest, SqrtJVP) {
    auto x = tenzor::zeros({3}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = 1.0f; xp[1] = 4.0f; xp[2] = 9.0f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_sqrt(d); },
        [](const Tensor& t) { return tenzor::sqrt(t); },
        x, v
    );
}

TEST_F(JVPTest, PowJVP) {
    auto x = tenzor::zeros({3}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;

    verify_jvp_fd(
        [](const DualTensor& d) { return jvp_pow(d, 3.0); },
        [](const Tensor& t) { return tenzor::pow(t, 3.0f); },
        x, v
    );
}

// ============================================================================
// Composed function JVP test
// ============================================================================

TEST_F(JVPTest, ComposedFunctionJVP) {
    // f(x) = exp(sin(x)), verify via finite differences
    auto x = tenzor::zeros({3}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    xp[0] = 0.5f; xp[1] = 1.0f; xp[2] = 1.5f;

    auto composed_dual = [](const DualTensor& d) -> DualTensor {
        auto sin_d = jvp_sin(d);
        return jvp_exp(sin_d);
    };

    auto composed_primal = [](const Tensor& t) -> Tensor {
        return tenzor::exp(tenzor::sin(t));
    };

    verify_jvp_fd(composed_dual, composed_primal, x, v);
}

// ============================================================================
// Functional JVP interface test
// ============================================================================

TEST_F(JVPTest, FunctionalJVPSquare) {
    // f(x) = x^2, JVP should give 2*x*v
    auto x_data = tenzor::zeros({3}, DType::Float32, Device::cpu());
    float* xp = x_data.data<float>();
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;

    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());

    Variable x(x_data, true);

    auto f = [](const Variable& inp) -> Variable {
        return inp * inp;
    };

    auto [output, tangent_out] = jvp(f, x, v);

    // tangent should be 2*x*v = [2, 4, 6]
    float* tp = tangent_out.data<float>();
    EXPECT_NEAR(tp[0], 2.0f, 0.1f);
    EXPECT_NEAR(tp[1], 4.0f, 0.1f);
    EXPECT_NEAR(tp[2], 6.0f, 0.1f);
}

// M25: jvp_used_fd_fallback() is the programmatic signal for the FD
// degradation previously visible only via an stderr print.
TEST_F(JVPTest, UsedFdFallback_FalseForRegisteredOp) {
    auto x = Variable(tenzor::ones({3}, DType::Float32, Device::cpu()), true);
    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());
    auto f = [](const Variable& inp) -> Variable { return inp * inp; };

    jvp(f, x, v);

    EXPECT_FALSE(jvp_used_fd_fallback())
        << "x*x has a registered JVP rule (Mul) — should take the exact "
           "graph-walk path, not FD";
}

TEST_F(JVPTest, UsedFdFallback_TrueWhenGraphSevered) {
    // A func whose output has no grad_fn (raw-tensor op, requires_grad=false)
    // gives try_traverse_jvp nothing to walk, forcing the FD fallback —
    // deterministic without needing to hunt down a genuinely-unregistered op.
    auto x = Variable(tenzor::ones({3}, DType::Float32, Device::cpu()), true);
    auto v = tenzor::ones({3}, DType::Float32, Device::cpu());
    auto f = [](const Variable& inp) -> Variable {
        return Variable(tenzor::mul(inp.tensor(), inp.tensor()), false);
    };

    jvp(f, x, v);

    EXPECT_TRUE(jvp_used_fd_fallback())
        << "func returns a Variable with no grad_fn — jvp() must fall back "
           "to finite differences and report that it did";
}

// ============================================================================
// Reduction JVP tests
// ============================================================================

TEST_F(JVPTest, SumJVP) {
    auto x = tenzor::ones({3, 4}, DType::Float32, Device::cpu());
    auto v = tenzor::mul(tenzor::ones({3, 4}, DType::Float32, Device::cpu()), 2.0);

    DualTensor dx(x, v);
    auto result = jvp_sum(dx);

    // sum of tangent should be 3*4*2 = 24
    float sum_tangent = *result.tangent().data<float>();
    EXPECT_NEAR(sum_tangent, 24.0f, 1e-5f);
}

TEST_F(JVPTest, MeanJVP) {
    auto x = tenzor::ones({3, 4}, DType::Float32, Device::cpu());
    auto v = tenzor::mul(tenzor::ones({3, 4}, DType::Float32, Device::cpu()), 2.0);

    DualTensor dx(x, v);
    auto result = jvp_mean(dx);

    // mean of tangent should be 2.0
    float mean_tangent = *result.tangent().data<float>();
    EXPECT_NEAR(mean_tangent, 2.0f, 1e-5f);
}

// ============================================================================
// Shape JVP tests
// ============================================================================

TEST_F(JVPTest, ReshapeJVP) {
    auto x = tenzor::ones({2, 6}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({2, 6}, DType::Float32, Device::cpu());

    DualTensor dx(x, v);
    auto result = jvp_reshape(dx, {3, 4});

    auto shape = result.primal().shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);

    auto tshape = result.tangent().shape();
    EXPECT_EQ(tshape[0], 3);
    EXPECT_EQ(tshape[1], 4);
}

TEST_F(JVPTest, TransposeJVP) {
    auto x = tenzor::ones({2, 3}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({2, 3}, DType::Float32, Device::cpu());

    DualTensor dx(x, v);
    auto result = jvp_transpose(dx, 0, 1);

    auto shape = result.primal().shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 2);
}

// ============================================================================
// JVP rules that previously had no test — added as part of the higher-order
// autograd completeness pass (phase 3). Each rule is verified against a
// central finite difference of its primal; the larger tolerance (0.1) is
// inherited from the existing tests which use atol=0.05 with the same eps.
// ============================================================================

TEST_F(JVPTest, SubJVP) {
    auto a = tenzor::ones({3, 3}, DType::Float32, Device::cpu()) * 2.0f;
    auto b = tenzor::ones({3, 3}, DType::Float32, Device::cpu());
    auto ta = tenzor::ones({3, 3}, DType::Float32, Device::cpu());
    auto tb = tenzor::ones({3, 3}, DType::Float32, Device::cpu()) * 0.25f;
    DualTensor da(a, ta), db(b, tb);
    auto r = jvp_sub(da, db);
    EXPECT_NEAR(*tenzor::mean(r.primal()).data<float>(), 1.0f, 1e-5f);
    EXPECT_NEAR(*tenzor::mean(r.tangent()).data<float>(), 0.75f, 1e-5f);
}

TEST_F(JVPTest, NegJVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_neg(d); },
                  [](const Tensor& t) { return tenzor::neg(t); }, x, v);
}

// For activations without a Tensor-level primal in the public API
// (relu/gelu/elu/selu only exist as Variable → Variable), verify the JVP
// produces finite, correctly-shaped tangent output. Numerical correctness
// is already exercised through the Variable-level backward in
// test_gradcheck_*.cpp; this guards against regression of the JVP rule
// signature / shape handling.
TEST_F(JVPTest, ReluJVP) {
    // Deterministic strictly-positive inputs (tenzor::abs(randn) + 1.0 keeps
    // every element in the positive region, where d(relu)/dx = 1 and the
    // tangent must equal v exactly). Previously used `randn + 2.0` which
    // occasionally produced a negative value under specific RNG states and
    // caused a flake when the full test binary was run in sequence.
    auto x = tenzor::abs(tenzor::randn({4}, DType::Float32, Device::cpu())) + 1.0f;
    auto v = tenzor::ones({4}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_relu(dx);
    EXPECT_EQ(r.tangent().shape()[0], 4);
    auto diff = tenzor::max(tenzor::abs(tenzor::sub(r.tangent(), v)));
    EXPECT_LT(*diff.data<float>(), 1e-5f);
}

TEST_F(JVPTest, GeluJVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_gelu(dx);
    EXPECT_EQ(r.primal().numel(), 4);
    EXPECT_EQ(r.tangent().numel(), 4);
}

TEST_F(JVPTest, EluJVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_elu(dx, 1.0f);
    EXPECT_EQ(r.primal().numel(), 4);
    EXPECT_EQ(r.tangent().numel(), 4);
}

TEST_F(JVPTest, SeluJVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_selu(dx);
    EXPECT_EQ(r.primal().numel(), 4);
    EXPECT_EQ(r.tangent().numel(), 4);
}

TEST_F(JVPTest, AbsJVP) {
    // Sample away from zero; abs is non-differentiable at x=0.
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu()) + 2.0f;
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_abs(d); },
                  [](const Tensor& t) { return tenzor::abs(t); }, x, v);
}

TEST_F(JVPTest, AcosJVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu()) * 0.3f;   // keep in (-1, 1)
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_acos(d); },
                  [](const Tensor& t) { return tenzor::acos(t); }, x, v, 0.1f);
}

TEST_F(JVPTest, AtanJVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_atan(d); },
                  [](const Tensor& t) { return tenzor::atan(t); }, x, v);
}

TEST_F(JVPTest, Log10JVP) {
    auto x = tenzor::abs(tenzor::randn({4}, DType::Float32, Device::cpu())) + 0.5f;
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_log10(d); },
                  [](const Tensor& t) { return tenzor::log10(t); }, x, v);
}

TEST_F(JVPTest, Exp2JVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu()) * 0.3f;
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_exp2(d); },
                  [](const Tensor& t) { return tenzor::exp2(t); }, x, v);
}

TEST_F(JVPTest, Expm1JVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu()) * 0.3f;
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_expm1(d); },
                  [](const Tensor& t) { return tenzor::expm1(t); }, x, v);
}

TEST_F(JVPTest, SignJVP) {
    // sign is piecewise-constant → derivative is zero almost everywhere.
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto v = tenzor::ones({4}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_sign(dx);
    auto sum_t = tenzor::sum(tenzor::abs(r.tangent()));
    EXPECT_NEAR(*sum_t.data<float>(), 0.0f, 1e-5f) << "sign has zero derivative";
}

TEST_F(JVPTest, ErfcJVP) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_erfc(d); },
                  [](const Tensor& t) { return tenzor::erfc(t); }, x, v);
}

TEST_F(JVPTest, ClampJVP) {
    // Keep samples inside the clamp range so the derivative is a plain identity.
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu()) * 0.2f;
    auto v = tenzor::randn({4}, DType::Float32, Device::cpu());
    verify_jvp_fd([](const DualTensor& d) { return jvp_clamp(d, -1.0, 1.0); },
                  [](const Tensor& t) { return tenzor::clamp(t, -1.0, 1.0); }, x, v);
}

TEST_F(JVPTest, SqueezeJVP) {
    auto x = tenzor::randn({1, 3, 1}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({1, 3, 1}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_squeeze(dx, std::nullopt);
    EXPECT_EQ(r.primal().ndim(), 1);
    EXPECT_EQ(r.tangent().ndim(), 1);
    EXPECT_EQ(r.primal().shape()[0], 3);
}

TEST_F(JVPTest, UnsqueezeJVP) {
    auto x = tenzor::randn({3}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({3}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_unsqueeze(dx, 0);
    EXPECT_EQ(r.primal().ndim(), 2);
    EXPECT_EQ(r.primal().shape()[0], 1);
    EXPECT_EQ(r.primal().shape()[1], 3);
}

TEST_F(JVPTest, FlattenJVP) {
    auto x = tenzor::randn({2, 3, 4}, DType::Float32, Device::cpu());
    auto v = tenzor::randn({2, 3, 4}, DType::Float32, Device::cpu());
    DualTensor dx(x, v);
    auto r = jvp_flatten(dx, 1, -1);
    EXPECT_EQ(r.primal().ndim(), 2);
    EXPECT_EQ(r.primal().shape()[0], 2);
    EXPECT_EQ(r.primal().shape()[1], 12);
    EXPECT_EQ(r.tangent().numel(), 24);
}

TEST_F(JVPTest, StackJVP) {
    auto a = tenzor::ones({2, 3}, DType::Float32, Device::cpu());
    auto b = tenzor::ones({2, 3}, DType::Float32, Device::cpu()) * 2.0f;
    auto ta = tenzor::ones({2, 3}, DType::Float32, Device::cpu());
    auto tb = tenzor::zeros({2, 3}, DType::Float32, Device::cpu());
    DualTensor da(a, ta), db(b, tb);
    std::vector<DualTensor> tensors = {da, db};
    auto r = jvp_stack(std::span<const DualTensor>(tensors), 0);
    EXPECT_EQ(r.primal().shape()[0], 2);
    EXPECT_EQ(r.primal().shape()[1], 2);
    EXPECT_EQ(r.primal().shape()[2], 3);
    // tangent[0] = ta (all ones), tangent[1] = tb (all zeros)
    EXPECT_NEAR(*tenzor::mean(r.tangent()).data<float>(), 0.5f, 1e-5f);
}

// ============================================================================
// Backend-parameterized JVP test (plan 4.1)
// ============================================================================

class JVPBackendTest : public tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        set_grad_enabled(true);
    }
};

TEST_P(JVPBackendTest, JVP_Runs_CrossBackend) {
    // Lighter contract: jvp() must produce finite outputs of correct shape
    // on every backend. Numerical correctness is verified in the dedicated
    // parity test (tests/backend_parity/test_grad_transform_parity.cpp).
    auto input = tenzor::ones({4}, DType::Float32, device);
    auto tangent = tenzor::ones({4}, DType::Float32, device);
    auto f = [](const Variable& x) -> Variable {
        return Variable(tenzor::mul(x.tensor(), x.tensor()), false);
    };
    // Audit: previously try/catch -> GTEST_SKIP("JVP unsupported on this
    // backend"), which buried a real cross-backend jvp/dispatch break as a
    // clean skip. Let exceptions propagate so the gap surfaces as a failure.
    auto [out, t] = tenzor::jvp(f, Variable(input, false), tangent);
    device.synchronize();
    EXPECT_EQ(out.tensor().numel(), 4);
    EXPECT_EQ(t.numel(), 4);
}

INSTANTIATE_BACKEND_TESTS(JVPBackendTest);
