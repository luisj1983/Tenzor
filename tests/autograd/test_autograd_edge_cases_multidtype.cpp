/**
 * @file test_autograd_edge_cases_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for autograd edge cases
 *
 * Covers: x*x gradient, detach, gradient accumulation, retain_graph
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class AutogradEdgeCasesMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(AutogradEdgeCasesMultiDTypeTest, SquareGradient) {
    // f(x) = sum(x*x), df/dx = 2*x
    auto x_f32 = tenzor::zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = x_f32.data<float>();
    ptr[0] = 1.0f; ptr[1] = 2.0f; ptr[2] = 3.0f; ptr[3] = -1.0f;
    Variable x(x_f32.to(device()).to(dtype()), true);

    auto y = x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad_f32 = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto* g = grad_f32.data<float>();
    EXPECT_NEAR(g[0], 2.0f, std::max(atol(), 1e-3f));
    EXPECT_NEAR(g[1], 4.0f, std::max(atol(), 1e-3f));
    EXPECT_NEAR(g[2], 6.0f, std::max(atol(), 1e-3f));
    EXPECT_NEAR(g[3], -2.0f, std::max(atol(), 1e-3f));
}

TEST_P(AutogradEdgeCasesMultiDTypeTest, CubeGradient) {
    // f(x) = x^3, f'(x) = 3x^2
    auto x_f32 = tenzor::full({1}, 2.0f, DType::Float32, Device::cpu());
    Variable x(x_f32.to(device()).to(dtype()), true);

    auto y = x * x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto g = x.grad()->to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(g.data<float>()[0], 12.0f, std::max(atol(), 0.1f));
}

TEST_P(AutogradEdgeCasesMultiDTypeTest, DetachStopsGradient) {
    auto x = createInput({4}, true);
    auto y = x * x;
    auto y_detached = y.detach();
    auto z = y_detached * y_detached;
    auto loss = tenzor::sum(z);
    loss.backward();

    // x should NOT have gradient because y was detached
    EXPECT_FALSE(x.has_grad());
}

TEST_P(AutogradEdgeCasesMultiDTypeTest, RetainGraphMultiBackward) {
    auto x = createInput({3}, true);
    auto y = x * x;
    auto loss = tenzor::sum(y);

    loss.backward(std::nullopt, true);
    ASSERT_TRUE(x.has_grad());
    auto grad1 = x.grad()->to(Device::cpu()).to(DType::Float32);

    // Second backward with retain_graph — gradients accumulate
    loss.backward(std::nullopt, true);
}

TEST_P(AutogradEdgeCasesMultiDTypeTest, GradientAccumulation) {
    auto x = createInput({3}, true);

    // Two forward passes, one backward each — gradients should accumulate
    auto y1 = x * x;
    auto loss1 = tenzor::sum(y1);
    loss1.backward(std::nullopt, true);

    ASSERT_TRUE(x.has_grad());
    auto grad1_f32 = x.grad()->to(Device::cpu()).to(DType::Float32);
    float g0_first = grad1_f32.data<float>()[0];

    // Backward again — gradient should be doubled
    auto y2 = x * x;
    auto loss2 = tenzor::sum(y2);
    loss2.backward();

    auto grad2_f32 = x.grad()->to(Device::cpu()).to(DType::Float32);
    float g0_second = grad2_f32.data<float>()[0];
    EXPECT_GT(std::abs(g0_second), std::abs(g0_first) - atol());
}

TEST_P(AutogradEdgeCasesMultiDTypeTest, NoGradContext) {
    auto x = createInput({4}, true);
    {
        auto guard = tenzor::NoGradGuard();
        auto y = x * x;
        EXPECT_EQ(y.grad_fn(), nullptr) << "No grad_fn inside no_grad context";
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AutogradEdgeCasesMultiDTypeTest);
