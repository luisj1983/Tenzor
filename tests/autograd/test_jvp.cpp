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
