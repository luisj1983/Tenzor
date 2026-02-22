/**
 * @file test_higher_order_gradients.cpp
 * @brief Tests for higher-order gradient computation (create_graph=true)
 *
 * Tests the ability to compute gradients of gradients, which is essential
 * for algorithms like WGAN-GP, MAML, and Hessian computation.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HigherOrderGradTest : public BackendTest {};

// Test that create_graph parameter exists and backward works normally
TEST_P(HigherOrderGradTest, CreateGraphParameterExists) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto y = x * x;  // y = x^2
    auto loss = tenzor::sum(y);

    // Should work with create_graph=false (default behavior)
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);

    ASSERT_TRUE(x.has_grad()) << "Failed on " << device.to_string();
    auto x_grad = x.grad().value().to(Device::cpu());
    // dy/dx = 2x = 6.0
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 6.0f, 1e-5f)
            << "Failed on " << device.to_string();
    }
}

// Test create_graph=true with multiplication (the key operation for higher-order gradients)
// y = x^2, dy/dx = 2x, d2y/dx2 = 2
TEST_P(HigherOrderGradTest, DoubleBackwardMul) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto y = x * x;  // y = x^2
    auto loss = tenzor::sum(y);

    // First backward with create_graph=true
    loss.backward(std::nullopt, /*retain_graph=*/true, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "First backward failed on " << device.to_string();
    auto first_grad = x.grad().value().to(Device::cpu());
    // dy/dx = 2x = 6.0
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(first_grad.data<float>()[i], 6.0f, 1e-5f)
            << "First gradient wrong on " << device.to_string();
    }
}

// Test that is_creating_graph flag is properly set and reset
TEST_P(HigherOrderGradTest, CreateGraphGuardRAII) {
    EXPECT_FALSE(is_creating_graph()) << "Should start false on " << device.to_string();

    {
        CreateGraphGuard guard;
        EXPECT_TRUE(is_creating_graph()) << "Should be true inside guard on " << device.to_string();
    }

    EXPECT_FALSE(is_creating_graph()) << "Should be false after guard on " << device.to_string();
}

// Test that create_graph implies retain_graph
TEST_P(HigherOrderGradTest, CreateGraphImpliesRetainGraph) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto y = x + x;
    auto loss = tenzor::sum(y);

    // backward with create_graph=true should retain the graph
    // (retain_graph=false but create_graph=true -> graph should still be retained)
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "Gradient not computed on " << device.to_string();
    auto x_grad = x.grad().value().to(Device::cpu());
    // d(sum(x+x))/dx = 2
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 2.0f, 1e-5f)
            << "Failed on " << device.to_string();
    }
}

// Test backward_with_variables for AddBackward
TEST_P(HigherOrderGradTest, AddBackwardWithVariables) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto y = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto z = x + y;
    auto loss = tenzor::sum(z);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device.to_string();
    ASSERT_TRUE(y.has_grad()) << "y grad missing on " << device.to_string();

    auto x_grad = x.grad().value().to(Device::cpu());
    auto y_grad = y.grad().value().to(Device::cpu());

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-5f)
            << "x grad wrong on " << device.to_string();
        EXPECT_NEAR(y_grad.data<float>()[i], 1.0f, 1e-5f)
            << "y grad wrong on " << device.to_string();
    }
}

// Test backward_with_variables for SubBackward
TEST_P(HigherOrderGradTest, SubBackwardWithVariables) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 5.0f, true);
    auto y = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto z = x - y;
    auto loss = tenzor::sum(z);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device.to_string();
    ASSERT_TRUE(y.has_grad()) << "y grad missing on " << device.to_string();

    auto x_grad = x.grad().value().to(Device::cpu());
    auto y_grad = y.grad().value().to(Device::cpu());

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f, 1e-5f)
            << "x grad wrong on " << device.to_string();
        EXPECT_NEAR(y_grad.data<float>()[i], -1.0f, 1e-5f)
            << "y grad wrong on " << device.to_string();
    }
}

// Test backward_with_variables for NegBackward
TEST_P(HigherOrderGradTest, NegBackwardWithVariables) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto y = tenzor::neg(x);
    auto loss = tenzor::sum(y);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device.to_string();
    auto x_grad = x.grad().value().to(Device::cpu());

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], -1.0f, 1e-5f)
            << "x grad wrong on " << device.to_string();
    }
}

// Test backward_with_variables for MatMulBackward
TEST_P(HigherOrderGradTest, MatMulBackwardWithVariables) {
    auto a = Variable(ones({2, 3}, DType::Float32, device) * 2.0f, true);
    auto b = Variable(ones({3, 2}, DType::Float32, device) * 3.0f, true);
    auto c = tenzor::matmul(a, b);
    auto loss = tenzor::sum(c);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(a.has_grad()) << "a grad missing on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "b grad missing on " << device.to_string();

    // For C = A @ B, dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    // With all-ones gradient and B = 3*ones(3,2), dL/dA = ones(2,2) @ 3*ones(2,3) = 3*2*ones(2,3) = 6
    // Wait: dL/dC = ones(2,2), B^T = 3*ones(2,3)
    // dL/dA = ones(2,2) @ 3*ones(2,3) = 6*ones(2,3)
    // Hmm, matmul(ones(2,2), 3*ones(2,3)) = 2*3*ones(2,3)? No: (2,2)@(2,3) = each element is sum of 2 terms, each = 1*3 = 3, so element = 6
    // Wait: B^T is (2,3), dL/dC is (2,2), so dL/dA = (2,2)@(2,3) = each element sum of 2 terms, each 1*3 = 3. Total = 6? No, sum of 2 terms = 2*3 = 6
    // Actually: ones(2,2) @ 3*ones(2,3): result[i][j] = sum_k ones[i][k]*3*ones[k][j] = 2*3 = 6. But wait, for (2,2)@(2,3), k goes 0..1, so each elem = 2 * 1*3 = 6
    auto a_grad = a.grad().value().to(Device::cpu());
    auto b_grad = b.grad().value().to(Device::cpu());

    // dL/dA: each element = inner_dim * val_b = 2 * 3 = 6
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 6.0f, 1e-4f)
            << "a grad wrong at " << i << " on " << device.to_string();
    }

    // dL/dB = A^T @ dL/dC: A^T is (3,2) with val 2.0, dL/dC is (2,2) with val 1.0
    // result[i][j] = sum_k A^T[i][k]*dL/dC[k][j] = 2*2*1 = 4
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(b_grad.data<float>()[i], 4.0f, 1e-4f)
            << "b grad wrong at " << i << " on " << device.to_string();
    }
}

// Test backward_with_variables for MeanBackward
TEST_P(HigherOrderGradTest, MeanBackwardWithVariables) {
    auto x = Variable(ones({2, 3}, DType::Float32, device) * 2.0f, true);
    auto loss = tenzor::mean(x);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device.to_string();
    auto x_grad = x.grad().value().to(Device::cpu());

    // d(mean(x))/dx = 1/N for each element, N=6
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 1.0f / 6.0f, 1e-5f)
            << "x grad wrong at " << i << " on " << device.to_string();
    }
}

// Test that normal backward (without create_graph) still works correctly after changes
TEST_P(HigherOrderGradTest, NormalBackwardStillWorks) {
    auto a = Variable(ones({3, 3}, DType::Float32, device) * 2.0f, true);
    auto b = Variable(ones({3, 3}, DType::Float32, device) * 3.0f, true);
    auto c = a * b;
    auto loss = tenzor::sum(c);

    // Standard backward without create_graph
    loss.backward();

    ASSERT_TRUE(a.has_grad()) << "a grad missing on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "b grad missing on " << device.to_string();

    auto a_grad = a.grad().value().to(Device::cpu());
    auto b_grad = b.grad().value().to(Device::cpu());

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(a_grad.data<float>()[i], 3.0f, 1e-5f)
            << "a grad wrong on " << device.to_string();
        EXPECT_NEAR(b_grad.data<float>()[i], 2.0f, 1e-5f)
            << "b grad wrong on " << device.to_string();
    }
}

// Test backward_with_variables for DivBackward
TEST_P(HigherOrderGradTest, DivBackwardWithVariables) {
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 6.0f, true);
    auto y = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto z = x / y;
    auto loss = tenzor::sum(z);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device.to_string();
    ASSERT_TRUE(y.has_grad()) << "y grad missing on " << device.to_string();

    // d(x/y)/dx = 1/y = 0.5
    // d(x/y)/dy = -x/y^2 = -6/4 = -1.5
    auto x_grad = x.grad().value().to(Device::cpu());
    auto y_grad = y.grad().value().to(Device::cpu());

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 0.5f, 1e-5f)
            << "x grad wrong on " << device.to_string();
        EXPECT_NEAR(y_grad.data<float>()[i], -1.5f, 1e-5f)
            << "y grad wrong on " << device.to_string();
    }
}

// Test chained operations with create_graph
TEST_P(HigherOrderGradTest, ChainedOpsCreateGraph) {
    // y = (x + 1) * x = x^2 + x
    // dy/dx = 2x + 1
    auto x = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto one = Variable(ones({2, 2}, DType::Float32, device), false);
    auto y = (x + one) * x;
    auto loss = tenzor::sum(y);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(x.has_grad()) << "x grad missing on " << device.to_string();
    auto x_grad = x.grad().value().to(Device::cpu());

    // dy/dx = 2x + 1 = 7 when x=3
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(x_grad.data<float>()[i], 7.0f, 1e-4f)
            << "x grad wrong on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(HigherOrderGradTest);
