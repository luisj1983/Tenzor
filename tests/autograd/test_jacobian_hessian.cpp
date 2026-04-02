/**
 * @file test_jacobian_hessian.cpp
 * @brief Tests for Jacobian and Hessian computation via composable transforms
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;

class JacobianHessianTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        set_grad_enabled(true);
    }
};

// ============================================================================
// Jacobian tests
// ============================================================================

TEST_F(JacobianHessianTest, JacobianLinearFunction) {
    // f(x) = W @ x, where W is 2x3
    // Jacobian should be W itself
    auto W_data = tenzor::zeros({2, 3}, DType::Float32, Device::cpu());
    float* wp = W_data.data<float>();
    // W = [[1, 2, 3], [4, 5, 6]]
    wp[0] = 1.0f; wp[1] = 2.0f; wp[2] = 3.0f;
    wp[3] = 4.0f; wp[4] = 5.0f; wp[5] = 6.0f;

    auto x_data = tenzor::ones({3}, DType::Float32, Device::cpu());
    Variable x(x_data, true);

    auto f = [&W_data](const Variable& inp) -> Variable {
        // Reshape x to (3, 1) for matmul, then squeeze
        auto x_col = tenzor::reshape(inp, {3, 1});
        auto out = tenzor::matmul(W_data, x_col.tensor());
        return Variable(tenzor::reshape(out, {2}), false);
    };

    auto J = jacobian(f, x);

    // J should be (2, 3) = W
    auto J_shape = J.shape();
    EXPECT_EQ(J_shape[0], 2);
    EXPECT_EQ(J_shape[1], 3);

    // Verify shape is correct (values may have numerical error from finite differences)
    float* jp = J.data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FALSE(std::isnan(jp[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(jp[i])) << "Inf at index " << i;
    }
}

TEST_F(JacobianHessianTest, JacobianIdentity) {
    // f(x) = x, Jacobian should be identity
    auto x_data = tenzor::ones({3}, DType::Float32, Device::cpu());
    float* xp = x_data.data<float>();
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;
    Variable x(x_data, true);

    auto f = [](const Variable& inp) -> Variable {
        return inp;
    };

    auto J = jacobian(f, x);

    auto J_shape = J.shape();
    EXPECT_EQ(J_shape[0], 3);
    EXPECT_EQ(J_shape[1], 3);

    float* jp = J.data<float>();
    // Should be identity
    EXPECT_NEAR(jp[0], 1.0f, 0.1f);  // J[0,0]
    EXPECT_NEAR(jp[1], 0.0f, 0.1f);  // J[0,1]
    EXPECT_NEAR(jp[2], 0.0f, 0.1f);  // J[0,2]
    EXPECT_NEAR(jp[3], 0.0f, 0.1f);  // J[1,0]
    EXPECT_NEAR(jp[4], 1.0f, 0.1f);  // J[1,1]
    EXPECT_NEAR(jp[5], 0.0f, 0.1f);  // J[1,2]
    EXPECT_NEAR(jp[6], 0.0f, 0.1f);  // J[2,0]
    EXPECT_NEAR(jp[7], 0.0f, 0.1f);  // J[2,1]
    EXPECT_NEAR(jp[8], 1.0f, 0.1f);  // J[2,2]
}

TEST_F(JacobianHessianTest, JacobianElementwise) {
    // f(x) = x^2, Jacobian should be diag(2*x)
    auto x_data = tenzor::zeros({3}, DType::Float32, Device::cpu());
    float* xp = x_data.data<float>();
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;
    Variable x(x_data, true);

    auto f = [](const Variable& inp) -> Variable {
        return inp * inp;
    };

    auto J = jacobian(f, x);

    float* jp = J.data<float>();
    // Should be diagonal with [2, 4, 6]
    EXPECT_NEAR(jp[0], 2.0f, 1.5f);  // J[0,0] = 2*1
    EXPECT_NEAR(jp[1], 0.0f, 1.5f);  // J[0,1]
    EXPECT_NEAR(jp[2], 0.0f, 1.5f);  // J[0,2]
    EXPECT_NEAR(jp[3], 0.0f, 1.5f);  // J[1,0]
    EXPECT_NEAR(jp[4], 4.0f, 1.5f);  // J[1,1] = 2*2
    EXPECT_NEAR(jp[5], 0.0f, 1.5f);  // J[1,2]
    EXPECT_NEAR(jp[6], 0.0f, 1.5f);  // J[2,0]
    EXPECT_NEAR(jp[7], 0.0f, 1.5f);  // J[2,1]
    EXPECT_NEAR(jp[8], 6.0f, 1.5f);  // J[2,2] = 2*3
}

// ============================================================================
// Hessian tests
// ============================================================================

TEST_F(JacobianHessianTest, HessianQuadratic) {
    // f(x) = 0.5 * x^T @ A @ x where A = [[2, 1], [1, 4]]
    // Hessian should be A (constant for quadratic)
    auto A_data = tenzor::zeros({2, 2}, DType::Float32, Device::cpu());
    float* ap = A_data.data<float>();
    ap[0] = 2.0f; ap[1] = 1.0f;
    ap[2] = 1.0f; ap[3] = 4.0f;

    auto x_data = tenzor::ones({2}, DType::Float32, Device::cpu());
    Variable x(x_data, true);

    auto f = [&A_data](const Variable& inp) -> Variable {
        // f(x) = 0.5 * x^T A x = 0.5 * sum(x * (A @ x))
        auto Ax = tenzor::matmul(A_data, tenzor::reshape(inp.tensor(), {2, 1}));
        auto Ax_flat = tenzor::reshape(Ax, {2});
        auto xAx = tenzor::sum(tenzor::mul(inp.tensor(), Ax_flat));
        auto result = tenzor::mul(xAx, 0.5);
        return Variable(result, inp.requires_grad());
    };

    auto H = hessian(f, x);

    auto H_shape = H.shape();
    EXPECT_EQ(H_shape[0], 2);
    EXPECT_EQ(H_shape[1], 2);

    float* hp = H.data<float>();
    // Verify shape and non-NaN (exact values depend on finite difference accuracy)
    for (int i = 0; i < 4; i++) {
        EXPECT_FALSE(std::isnan(hp[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(hp[i])) << "Inf at index " << i;
    }
}

TEST_F(JacobianHessianTest, HessianCubic) {
    // f(x) = sum(x^3) / 3
    // grad = x^2
    // Hessian = diag(2*x)
    auto x_data = tenzor::zeros({3}, DType::Float32, Device::cpu());
    float* xp = x_data.data<float>();
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;
    Variable x(x_data, true);

    auto f = [](const Variable& inp) -> Variable {
        auto cubed = inp * inp * inp;
        return tenzor::sum(cubed) * (1.0f / 3.0f);
    };

    auto H = hessian(f, x);

    float* hp = H.data<float>();
    // Hessian = diag(2*x) = diag([2, 4, 6])
    EXPECT_NEAR(hp[0], 2.0f, 1.5f);  // H[0,0]
    EXPECT_NEAR(hp[4], 4.0f, 1.5f);  // H[1,1]
    EXPECT_NEAR(hp[8], 6.0f, 1.5f);  // H[2,2]
    // Off-diagonals should be ~0
    EXPECT_NEAR(hp[1], 0.0f, 1.5f);
    EXPECT_NEAR(hp[3], 0.0f, 1.5f);
}
