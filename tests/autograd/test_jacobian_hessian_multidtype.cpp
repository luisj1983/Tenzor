/**
 * @file test_jacobian_hessian_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Jacobian and Hessian computation
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/functional.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class JacobianHessianMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfHalf() {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            GTEST_SKIP() << "Jacobian/Hessian requires higher precision";
        }
    }
};

TEST_P(JacobianHessianMultiDTypeTest, JacobianIdentity) {
    skipIfHalf();
    // f(x) = x, Jacobian = I
    auto x = createInput({3}, true);
    auto f = [](const Variable& inp) -> Variable { return inp; };

    auto J = jacobian(f, x);
    auto J_shape = J.shape();
    EXPECT_EQ(J_shape[0], 3);
    EXPECT_EQ(J_shape[1], 3);

    // Should be approximately identity
    auto J_f32 = J.to(Device::cpu()).to(DType::Float32);
    auto* d = J_f32.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(d[i * 3 + j], expected, std::max(atol(), 0.01f));
        }
    }
}

TEST_P(JacobianHessianMultiDTypeTest, JacobianSquare) {
    skipIfHalf();
    // f(x) = x^2, Jacobian = diag(2*x)
    auto x_f32 = tenzor::zeros({3}, DType::Float32, Device::cpu());
    x_f32.data<float>()[0] = 1.0f;
    x_f32.data<float>()[1] = 2.0f;
    x_f32.data<float>()[2] = 3.0f;
    Variable x(x_f32.to(device()).to(dtype()), true);

    auto f = [](const Variable& inp) -> Variable { return inp * inp; };

    auto J = jacobian(f, x);
    auto J_shape = J.shape();
    EXPECT_EQ(J_shape[0], 3);
    EXPECT_EQ(J_shape[1], 3);

    // Diagonal should be 2*x
    auto J_f32 = J.to(Device::cpu()).to(DType::Float32);
    auto* d = J_f32.data<float>();
    EXPECT_NEAR(d[0], 2.0f, std::max(atol(), 0.05f));  // 2*1
    EXPECT_NEAR(d[4], 4.0f, std::max(atol(), 0.05f));  // 2*2
    EXPECT_NEAR(d[8], 6.0f, std::max(atol(), 0.05f));  // 2*3
}

TEST_P(JacobianHessianMultiDTypeTest, HessianQuadratic) {
    skipIfHalf();
    // f(x) = sum(x^2), Hessian = 2*I
    auto x = createInput({3}, true);
    auto f = [](const Variable& inp) -> Variable {
        return tenzor::sum(inp * inp);
    };

    auto H = hessian(f, x);
    auto H_shape = H.shape();
    EXPECT_EQ(H_shape[0], 3);
    EXPECT_EQ(H_shape[1], 3);

    auto H_f32 = H.to(Device::cpu()).to(DType::Float32);
    auto* d = H_f32.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 2.0f : 0.0f;
            EXPECT_NEAR(d[i * 3 + j], expected, std::max(atol(), 0.1f));
        }
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(JacobianHessianMultiDTypeTest);
