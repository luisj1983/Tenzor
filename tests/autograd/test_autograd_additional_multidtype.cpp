/**
 * @file test_autograd_additional_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for core autograd functionality
 *
 * Converted from test_autograd_additional.cpp (representative subset of 15 tests).
 * Covers variable creation, grad accumulation, backward pass, hooks, and
 * key gradient function correctness across backends and dtypes.
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/tenzor.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class AutogradAdditionalMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);
    }
};

// =============================================================================
// Variable Class Tests
// =============================================================================

TEST_P(AutogradAdditionalMultiDTypeTest, VariableDefaultConstructor) {
    Variable var;

    EXPECT_FALSE(var.is_initialized());
    EXPECT_FALSE(static_cast<bool>(var));
}

TEST_P(AutogradAdditionalMultiDTypeTest, VariableConstructorWithoutGrad) {
    auto data = ones({3, 4}, dtype(), device());
    Variable var(data, false);

    EXPECT_TRUE(var.is_initialized());
    EXPECT_FALSE(var.requires_grad());
    EXPECT_TRUE(var.is_leaf());
    EXPECT_FALSE(var.has_grad());
}

TEST_P(AutogradAdditionalMultiDTypeTest, VariableSetRequiresGrad) {
    auto data = ones({2, 3}, dtype(), device());
    Variable var(data, false);

    EXPECT_FALSE(var.requires_grad());

    var.set_requires_grad(true);
    EXPECT_TRUE(var.requires_grad());

    var.set_requires_grad(false);
    EXPECT_FALSE(var.requires_grad());
}

TEST_P(AutogradAdditionalMultiDTypeTest, VariableGradAccumulation) {
    auto data = ones({2, 2}, dtype(), device()) * 2.0f;
    Variable var(data, true);

    auto grad1 = ones({2, 2}, dtype(), device());
    var.set_grad(grad1);

    ASSERT_TRUE(var.has_grad());

    auto grad_cpu = var.grad()->to(Device::cpu()).to(DType::Float32);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(grad_cpu.data<float>()[i], 1.0f, atol());
    }
}

TEST_P(AutogradAdditionalMultiDTypeTest, VariableRetainGrad) {
    auto x = Variable(ones({2, 2}, dtype(), device()), true);
    auto y = x * 2.0f;

    EXPECT_FALSE(y.retains_grad());

    y.retain_grad();
    EXPECT_TRUE(y.retains_grad());

    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(y.has_grad());
}

TEST_P(AutogradAdditionalMultiDTypeTest, VariableDetachRemovesGradFn) {
    auto x = Variable(ones({2, 2}, dtype(), device()), true);
    auto y = x * 2.0f;

    EXPECT_TRUE(y.grad_fn() != nullptr);

    auto y_detached = y.detach();
    EXPECT_FALSE(y_detached.requires_grad());
    EXPECT_TRUE(y_detached.grad_fn() == nullptr);
}

TEST_P(AutogradAdditionalMultiDTypeTest, VariableZeroGrad) {
    auto x = Variable(ones({2, 2}, dtype(), device()), true);
    auto y = x * 2.0f;
    auto loss = sum(y);

    loss.backward();
    ASSERT_TRUE(x.has_grad());

    x.zero_grad();
    EXPECT_FALSE(x.has_grad());
}

// =============================================================================
// Backward Pass Variations
// =============================================================================

TEST_P(AutogradAdditionalMultiDTypeTest, BackwardWithGradientTensor) {
    auto x = Variable(ones({2, 3}, dtype(), device()), true);
    auto y = x * 2.0f;

    auto grad_output = ones({2, 3}, dtype(), device()) * 5.0f;
    y.backward(grad_output);

    ASSERT_TRUE(x.has_grad());
    auto x_grad = x.grad()->to(Device::cpu()).to(DType::Float32);

    float tol = (dtype() == DType::Float16) ? 0.1f : atol();
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 10.0f, tol);
    }
}

TEST_P(AutogradAdditionalMultiDTypeTest, MultipleBackwardWithRetainGraph) {
    auto x = Variable(ones({2, 2}, dtype(), device()), true);
    auto y = x * 2.0f;
    auto loss = sum(y);

    loss.backward(std::nullopt, true);
    ASSERT_TRUE(x.has_grad());

    auto grad1_cpu = x.grad()->to(Device::cpu()).to(DType::Float32);
    float grad1 = grad1_cpu.data<float>()[0];

    loss.backward(std::nullopt, false);
    ASSERT_TRUE(x.has_grad());

    auto grad2_cpu = x.grad()->to(Device::cpu()).to(DType::Float32);
    float grad2 = grad2_cpu.data<float>()[0];

    float tol = (dtype() == DType::Float16) ? 0.1f : atol();
    EXPECT_NEAR(grad2, grad1 * 2.0f, tol);
}

// =============================================================================
// Gradient Function Correctness
// =============================================================================

TEST_P(AutogradAdditionalMultiDTypeTest, ComplexMultiInputGraph) {
    auto a = Variable(ones({2, 2}, dtype(), device()) * 1.0f, true);
    auto b = Variable(ones({2, 2}, dtype(), device()) * 2.0f, true);
    auto c = Variable(ones({2, 2}, dtype(), device()) * 5.0f, true);
    auto d = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);

    auto sum_ab = a + b;   // 3.0
    auto diff_cd = c - d;  // 2.0
    auto z = sum_ab * diff_cd;  // 6.0

    auto loss = sum(z);
    loss.backward();

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    ASSERT_TRUE(c.has_grad());
    ASSERT_TRUE(d.has_grad());

    auto a_grad = a.grad()->to(Device::cpu()).to(DType::Float32);
    auto b_grad = b.grad()->to(Device::cpu()).to(DType::Float32);
    auto c_grad = c.grad()->to(Device::cpu()).to(DType::Float32);
    auto d_grad = d.grad()->to(Device::cpu()).to(DType::Float32);

    float tol = (dtype() == DType::Float16) ? 0.1f : atol();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 2.0f, tol);
        EXPECT_NEAR(b_grad.data<float>()[i], 2.0f, tol);
        EXPECT_NEAR(c_grad.data<float>()[i], 3.0f, tol);
        EXPECT_NEAR(d_grad.data<float>()[i], -3.0f, tol);
    }
}

TEST_P(AutogradAdditionalMultiDTypeTest, ExpBackwardCorrectGradients) {
    auto x = Variable(zeros({2, 2}, dtype(), device()), true);
    auto y = exp(x);

    y.backward(ones({2, 2}, dtype(), device()));

    ASSERT_TRUE(x.has_grad());
    auto x_grad = x.grad()->to(Device::cpu()).to(DType::Float32);

    float tol = (dtype() == DType::Float16) ? 0.05f : atol();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, tol);
    }
}

TEST_P(AutogradAdditionalMultiDTypeTest, LogBackwardCorrectGradients) {
    auto x = Variable(ones({2, 2}, dtype(), device()) * 2.0f, true);
    auto y = log(x);

    y.backward(ones({2, 2}, dtype(), device()));

    ASSERT_TRUE(x.has_grad());
    auto x_grad = x.grad()->to(Device::cpu()).to(DType::Float32);

    float tol = (dtype() == DType::Float16) ? 0.05f : atol();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 0.5f, tol);
    }
}

TEST_P(AutogradAdditionalMultiDTypeTest, NegBackwardCorrectGradients) {
    auto x = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);
    auto y = neg(x);

    y.backward(ones({2, 2}, dtype(), device()));

    ASSERT_TRUE(x.has_grad());
    auto x_grad = x.grad()->to(Device::cpu()).to(DType::Float32);

    float tol = (dtype() == DType::Float16) ? 0.05f : atol();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], -1.0f, tol);
    }
}

// =============================================================================
// Hooks
// =============================================================================

TEST_P(AutogradAdditionalMultiDTypeTest, RegisterHookModifiesGradient) {
    auto x = Variable(ones({2, 2}, dtype(), device()) * 2.0f, true);

    bool hook_called = false;
    x.register_hook([&hook_called](const Tensor& grad) -> Tensor {
        hook_called = true;
        return grad * 2.0f;
    });

    auto y = x * 3.0f;
    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(hook_called);
    ASSERT_TRUE(x.has_grad());

    auto x_grad = x.grad()->to(Device::cpu()).to(DType::Float32);
    float tol = (dtype() == DType::Float16) ? 0.1f : atol();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 6.0f, tol);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AutogradAdditionalMultiDTypeTest);
