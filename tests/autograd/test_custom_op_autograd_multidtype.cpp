/**
 * @file test_custom_op_autograd_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for custom autograd operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// Tests use the built-in autograd through Variable operators rather than
// subclassing Function, since Function uses Tensor (not Variable) interface.

class CustomOpAutogradMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(CustomOpAutogradMultiDTypeTest, SquareBackwardGradient) {
    auto x = createInput({4}, true);
    // Use operators for autograd tracking
    auto y = x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    // dy/dx = 2x
    auto x_f32 = x.tensor().to(Device::cpu()).to(DType::Float32);
    auto grad_f32 = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto* xd = x_f32.data<float>();
    auto* gd = grad_f32.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(gd[i], 2.0f * xd[i], std::max(atol(), 1e-3f));
    }
}

TEST_P(CustomOpAutogradMultiDTypeTest, ChainedCustomOps) {
    auto x = createInput({3}, true);
    auto y = x * x;       // x^2
    auto z = y * y;       // x^4
    auto loss = tenzor::sum(z);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    // dz/dx = 4*x^3
    auto x_f32 = x.tensor().to(Device::cpu()).to(DType::Float32);
    auto grad_f32 = x.grad()->to(Device::cpu()).to(DType::Float32);
    auto* xd = x_f32.data<float>();
    auto* gd = grad_f32.data<float>();
    for (int64_t i = 0; i < 3; ++i) {
        float expected = 4.0f * xd[i] * xd[i] * xd[i];
        EXPECT_NEAR(gd[i], expected, std::max(atol() * std::abs(expected), 0.1f));
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(CustomOpAutogradMultiDTypeTest);
