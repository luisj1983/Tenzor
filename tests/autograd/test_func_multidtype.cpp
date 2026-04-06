/**
 * @file test_func_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for composable function transforms
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/func.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class FuncMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(FuncMultiDTypeTest, GradSimpleQuadratic) {
    // f(x) = sum(x^2), grad(f)(x) = 2*x
    auto f = [](const Variable& x) -> Variable {
        auto sq = x * x;
        return tenzor::sum(sq);
    };

    auto grad_f = func::grad(f);

    auto x_f32 = tenzor::zeros({3}, DType::Float32, Device::cpu());
    x_f32.data<float>()[0] = 1.0f;
    x_f32.data<float>()[1] = 2.0f;
    x_f32.data<float>()[2] = 3.0f;
    Variable x(x_f32.to(device()).to(dtype()), true);

    auto grad_val = grad_f(x);
    auto gf = grad_val.tensor().to(Device::cpu()).to(DType::Float32);
    auto* gp = gf.data<float>();

    EXPECT_NEAR(gp[0], 2.0f, std::max(atol(), 1e-3f));
    EXPECT_NEAR(gp[1], 4.0f, std::max(atol(), 1e-3f));
    EXPECT_NEAR(gp[2], 6.0f, std::max(atol(), 1e-3f));
}

TEST_P(FuncMultiDTypeTest, GradCubic) {
    // f(x) = x^3, f'(x) = 3x^2
    auto f = [](const Variable& x) -> Variable {
        return x * x * x;
    };

    auto grad_f = func::grad(f);

    auto x_f32 = tenzor::full({1}, 2.0f, DType::Float32, Device::cpu());
    Variable x(x_f32.to(device()).to(dtype()), true);

    auto grad_val = grad_f(x);
    auto gf = grad_val.tensor().to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(gf.data<float>()[0], 12.0f, std::max(atol(), 0.1f));
}

TEST_P(FuncMultiDTypeTest, GradExpSin) {
    // f(x) = sum(exp(sin(x))), f'(x) = cos(x) * exp(sin(x))
    auto f = [](const Variable& x) -> Variable {
        return tenzor::sum(tenzor::exp(tenzor::sin(x)));
    };

    auto grad_f = func::grad(f);
    auto x = createInput({4}, true);
    auto grad_val = grad_f(x);

    expectShape(grad_val.tensor(), {4});
    expectDevice(grad_val.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FuncMultiDTypeTest);
