/**
 * @file test_higher_order_gradients_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for higher-order gradient computation
 *
 * Multi-backend port of test_higher_order_gradients.cpp. Tests create_graph=true,
 * double backward, and backward_with_variables across all available backends
 * and data types.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HigherOrderGradMultiDTypeTest : public MultiBackendDTypeTest {};

// Higher-order gradient tests are numerically demanding — skip Float16
#define SKIP_IF_LOW_PRECISION() \
    do { \
        if (dtype() == DType::Float16) \
            GTEST_SKIP() << "Higher-order grads require Float32+ precision"; \
    } while (0)

// Test that create_graph parameter exists and backward works normally
TEST_P(HigherOrderGradMultiDTypeTest, CreateGraphParameterExists) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);
    auto y = x * x;  // y = x^2
    auto loss = tenzor::sum(y);

    // Should work with create_graph=false (default behavior)
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device().to_string();
    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);
    // dy/dx = 2x = 6.0
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 6.0f, atol())
            << "Failed on " << device().to_string();
    }
}

// Test create_graph=true with multiplication
// y = x^2, dy/dx = 2x, d2y/dx2 = 2
TEST_P(HigherOrderGradMultiDTypeTest, DoubleBackwardMul) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);
    auto y = x * x;  // y = x^2
    auto loss = tenzor::sum(y);

    // First backward with create_graph=true
    loss.backward(std::nullopt, /*retain_graph=*/true, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "First backward failed on " << device().to_string();
    auto first_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);
    // dy/dx = 2x = 6.0
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(first_grad.data<float>()[i], 6.0f, atol())
            << "First gradient wrong on " << device().to_string();
    }
}

// Test that is_creating_graph flag is properly set and reset
TEST_P(HigherOrderGradMultiDTypeTest, CreateGraphGuardRAII) {
    SKIP_IF_LOW_PRECISION();

    EXPECT_FALSE(is_creating_graph()) << "Should start false on " << device().to_string();

    {
        CreateGraphGuard guard;
        EXPECT_TRUE(is_creating_graph()) << "Should be true inside guard on " << device().to_string();
    }

    EXPECT_FALSE(is_creating_graph()) << "Should be false after guard on " << device().to_string();
}

// Test that create_graph implies retain_graph
TEST_P(HigherOrderGradMultiDTypeTest, CreateGraphImpliesRetainGraph) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 2}, dtype(), device()) * 2.0f, true);
    auto y = x + x;
    auto loss = tenzor::sum(y);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "Gradient not computed on " << device().to_string();
    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);
    // d(sum(x+x))/dx = 2
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 2.0f, atol())
            << "Failed on " << device().to_string();
    }
}

// Test backward_with_variables for AddBackward
TEST_P(HigherOrderGradMultiDTypeTest, AddBackwardWithVariables) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 2}, dtype(), device()) * 2.0f, true);
    auto y = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);
    auto z = x + y;
    auto loss = tenzor::sum(z);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device().to_string();
    ASSERT_TRUE(y.has_grad()) << "y grad missing on " << device().to_string();

    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);
    auto y_grad = y.grad().value().to(Device::cpu()).to(DType::Float32);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, atol())
            << "x grad wrong on " << device().to_string();
        EXPECT_NEAR(y_grad.data<float>()[i], 1.0f, atol())
            << "y grad wrong on " << device().to_string();
    }
}

// Test backward_with_variables for SubBackward
TEST_P(HigherOrderGradMultiDTypeTest, SubBackwardWithVariables) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 2}, dtype(), device()) * 5.0f, true);
    auto y = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);
    auto z = x - y;
    auto loss = tenzor::sum(z);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device().to_string();
    ASSERT_TRUE(y.has_grad()) << "y grad missing on " << device().to_string();

    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);
    auto y_grad = y.grad().value().to(Device::cpu()).to(DType::Float32);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, atol())
            << "x grad wrong on " << device().to_string();
        EXPECT_NEAR(y_grad.data<float>()[i], -1.0f, atol())
            << "y grad wrong on " << device().to_string();
    }
}

// Test backward_with_variables for NegBackward
TEST_P(HigherOrderGradMultiDTypeTest, NegBackwardWithVariables) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);
    auto y = tenzor::neg(x);
    auto loss = tenzor::sum(y);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device().to_string();
    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], -1.0f, atol())
            << "x grad wrong on " << device().to_string();
    }
}

// Test backward_with_variables for MatMulBackward
TEST_P(HigherOrderGradMultiDTypeTest, MatMulBackwardWithVariables) {
    SKIP_IF_LOW_PRECISION();

    auto a = Variable(ones({2, 3}, dtype(), device()) * 2.0f, true);
    auto b = Variable(ones({3, 2}, dtype(), device()) * 3.0f, true);
    auto c = tenzor::matmul(a, b);
    auto loss = tenzor::sum(c);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(a.has_grad()) << "a grad missing on " << device().to_string();
    ASSERT_TRUE(b.has_grad()) << "b grad missing on " << device().to_string();

    auto a_grad = a.grad().value().to(Device::cpu()).to(DType::Float32);
    auto b_grad = b.grad().value().to(Device::cpu()).to(DType::Float32);

    // dL/dA: each element = inner_dim * val_b = 2 * 3 = 6
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 6.0f, atol())
            << "a grad wrong at " << i << " on " << device().to_string();
    }

    // dL/dB = A^T @ dL/dC: each element = 2 * 2 * 1 = 4
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(b_grad.data<float>()[i], 4.0f, atol())
            << "b grad wrong at " << i << " on " << device().to_string();
    }
}

// Test backward_with_variables for MeanBackward
TEST_P(HigherOrderGradMultiDTypeTest, MeanBackwardWithVariables) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 3}, dtype(), device()) * 2.0f, true);
    auto loss = tenzor::mean(x);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device().to_string();
    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);

    // d(mean(x))/dx = 1/N for each element, N=6
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f / 6.0f, atol())
            << "x grad wrong at " << i << " on " << device().to_string();
    }
}

// Test that normal backward (without create_graph) still works correctly
TEST_P(HigherOrderGradMultiDTypeTest, NormalBackwardStillWorks) {
    SKIP_IF_LOW_PRECISION();

    auto a = Variable(ones({3, 3}, dtype(), device()) * 2.0f, true);
    auto b = Variable(ones({3, 3}, dtype(), device()) * 3.0f, true);
    auto c = a * b;
    auto loss = tenzor::sum(c);

    // Standard backward without create_graph
    loss.backward();

    ASSERT_TRUE(a.has_grad()) << "a grad missing on " << device().to_string();
    ASSERT_TRUE(b.has_grad()) << "b grad missing on " << device().to_string();

    auto a_grad = a.grad().value().to(Device::cpu()).to(DType::Float32);
    auto b_grad = b.grad().value().to(Device::cpu()).to(DType::Float32);

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 3.0f, atol())
            << "a grad wrong on " << device().to_string();
        EXPECT_NEAR(b_grad.data<float>()[i], 2.0f, atol())
            << "b grad wrong on " << device().to_string();
    }
}

// Test backward_with_variables for DivBackward
TEST_P(HigherOrderGradMultiDTypeTest, DivBackwardWithVariables) {
    SKIP_IF_LOW_PRECISION();

    auto x = Variable(ones({2, 2}, dtype(), device()) * 6.0f, true);
    auto y = Variable(ones({2, 2}, dtype(), device()) * 2.0f, true);
    auto z = x / y;
    auto loss = tenzor::sum(z);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device().to_string();
    ASSERT_TRUE(y.has_grad()) << "y grad missing on " << device().to_string();

    // d(x/y)/dx = 1/y = 0.5
    // d(x/y)/dy = -x/y^2 = -6/4 = -1.5
    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);
    auto y_grad = y.grad().value().to(Device::cpu()).to(DType::Float32);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 0.5f, atol())
            << "x grad wrong on " << device().to_string();
        EXPECT_NEAR(y_grad.data<float>()[i], -1.5f, atol())
            << "y grad wrong on " << device().to_string();
    }
}

// Test chained operations with create_graph
TEST_P(HigherOrderGradMultiDTypeTest, ChainedOpsCreateGraph) {
    SKIP_IF_LOW_PRECISION();

    // y = (x + 1) * x = x^2 + x
    // dy/dx = 2x + 1
    auto x = Variable(ones({2, 2}, dtype(), device()) * 3.0f, true);
    auto one = Variable(ones({2, 2}, dtype(), device()), false);
    auto y = (x + one) * x;
    auto loss = tenzor::sum(y);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device().to_string();
    auto x_grad = x.grad().value().to(Device::cpu()).to(DType::Float32);

    // dy/dx = 2x + 1 = 7 when x=3
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 7.0f, atol())
            << "x grad wrong on " << device().to_string();
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HigherOrderGradMultiDTypeTest);
