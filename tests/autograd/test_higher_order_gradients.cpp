/**
 * @file test_higher_order_gradients.cpp
 * @brief Tests for higher-order gradient computation (create_graph=true)
 *
 * Tests the ability to compute gradients of gradients, which is essential
 * for algorithms like WGAN-GP, MAML, and Hessian computation.
 */

#include <gtest/gtest.h>
#include <cstring>
#include "../backend_test_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "../grad_flow_helpers.hpp"

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

// ============================================================================
// M17-M24: bmm/linear/erfinv/det/inv/solve/lu/slogdet forward wrappers used
// to never call save_variables_for_backward(), so backward_with_variables()
// always fell back to a detached Variable(saved_tensors_[...], false). The
// FIRST-order gradient (tested above, e.g. MatMulBackwardWithVariables) was
// still numerically correct, but create_graph=true produced a gradient with
// NO further graph connection back to the input — a second .backward()
// through it silently left the second-order gradient at zero, with no
// error. Mirrors Conv2d_DoubleBackward in test_higher_order_nn.cpp (same bug
// class, found there as H18/H19): read grad_variable() (populated by
// create_graph=true) instead of re-wrapping .grad(), zero_grad() the target
// leaf so the second backward's contribution isn't masked by the
// already-nonzero first-order value, then assert EXPECT_GRAD_FLOWS.
// ============================================================================

namespace {
// Diagonally dominant, symmetric, well-conditioned 3x3 — safe for det/inv/
// solve/lu/slogdet (det ~= 50, condition number small) without needing a
// random-matrix invertibility retry loop.
auto well_conditioned_3x3(Device device) -> Variable {
    Tensor t = zeros({3, 3}, DType::Float64, Device::cpu());
    double vals[9] = {4, 1, 0, 1, 4, 1, 0, 1, 4};
    std::memcpy(t.data_ptr(), vals, sizeof(vals));
    return Variable(t.to(device), true);
}
}  // namespace

// M17-M24's save_variables_for_backward() calls are gated on
// is_creating_graph() || higher_order_graph_retention_enabled(), checked at
// FORWARD time — per higher_order_graph_retention_enabled()'s own doc
// comment, retention is off by default (a bare create_graph=true backward
// after an ordinary forward pass does NOT retroactively enable it) and must
// be opted into for the forward pass via HigherOrderGraphRetentionGuard,
// exactly as hvp()/hessian() do internally (see functional.cpp). Each test
// below wraps forward + the first create_graph=true backward in that guard.

TEST_P(HigherOrderGradTest, Bmm_DoubleBackward) {
    // grad_b = a_t @ grad_out depends on saved `a` — verify d(grad_b)/d(a)
    // is nonzero, i.e. backward_with_variables used the live `a`, not a
    // detached copy.
    auto a = Variable(randn({2, 3, 3}, DType::Float64, device) * 0.5, true);
    auto b = Variable(randn({2, 3, 3}, DType::Float64, device) * 0.5, true);
    Variable grad_b_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = tenzor::bmm(a, b);
        auto loss = tenzor::sum(y);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        ASSERT_TRUE(b.grad_variable().has_value())
            << "create_graph=true must populate grad_variable() for b";
        grad_b_var = b.grad_variable().value();
    }
    a.zero_grad();
    auto grad_norm = tenzor::sum(grad_b_var * grad_b_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(a);
}

TEST_P(HigherOrderGradTest, Linear_DoubleBackward) {
    // grad_w = grad_out.T @ x depends on saved `x` — verify d(grad_w)/d(x)
    // is nonzero.
    auto x = Variable(randn({4, 8}, DType::Float64, device), true);
    auto w = Variable(randn({16, 8}, DType::Float64, device), true);
    auto b = Variable(zeros({16}, DType::Float64, device), false);
    Variable grad_w_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = tenzor::linear(x, w, b);
        auto loss = tenzor::sum(y);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        ASSERT_TRUE(w.grad_variable().has_value())
            << "create_graph=true must populate grad_variable() for w";
        grad_w_var = w.grad_variable().value();
    }
    x.zero_grad();
    auto grad_norm = tenzor::sum(grad_w_var * grad_w_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(x);
}

TEST_P(HigherOrderGradTest, ErfInv_DoubleBackward) {
    Tensor t = zeros({2, 3}, DType::Float64, Device::cpu());
    double vals[6] = {-0.7, -0.3, 0.1, 0.4, 0.6, 0.2};
    std::memcpy(t.data_ptr(), vals, sizeof(vals));
    auto x = Variable(t.to(device), true);
    Variable grad_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = tenzor::erfinv(x);
        auto loss = tenzor::sum(y);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        EXPECT_GRAD_FLOWS(x);
        ASSERT_TRUE(x.grad_variable().has_value())
            << "create_graph=true must populate grad_variable()";
        grad_var = x.grad_variable().value();
    }
    x.zero_grad();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(x);
}

TEST_P(HigherOrderGradTest, Det_DoubleBackward) {
    auto A = well_conditioned_3x3(device);
    Variable grad_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = tenzor::det(A);
        auto loss = tenzor::sum(y);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        EXPECT_GRAD_FLOWS(A);
        ASSERT_TRUE(A.grad_variable().has_value())
            << "create_graph=true must populate grad_variable()";
        grad_var = A.grad_variable().value();
    }
    A.zero_grad();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(A);
}

TEST_P(HigherOrderGradTest, Inv_DoubleBackward) {
    auto A = well_conditioned_3x3(device);
    Variable grad_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = tenzor::inv(A);
        auto loss = tenzor::sum(y);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        EXPECT_GRAD_FLOWS(A);
        ASSERT_TRUE(A.grad_variable().has_value())
            << "create_graph=true must populate grad_variable()";
        grad_var = A.grad_variable().value();
    }
    A.zero_grad();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(A);
}

TEST_P(HigherOrderGradTest, Solve_DoubleBackward) {
    auto A = well_conditioned_3x3(device);
    auto B = Variable(randn({3, 2}, DType::Float64, device), false);
    Variable grad_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = tenzor::solve(A, B);
        auto loss = tenzor::sum(y);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        EXPECT_GRAD_FLOWS(A);
        ASSERT_TRUE(A.grad_variable().has_value())
            << "create_graph=true must populate grad_variable()";
        grad_var = A.grad_variable().value();
    }
    A.zero_grad();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(A);
}

TEST_P(HigherOrderGradTest, Lu_DoubleBackward) {
    auto A = well_conditioned_3x3(device);
    Variable grad_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto [L, U, pivots] = tenzor::lu(A);
        auto loss = tenzor::sum(L) + tenzor::sum(U);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        EXPECT_GRAD_FLOWS(A);
        ASSERT_TRUE(A.grad_variable().has_value())
            << "create_graph=true must populate grad_variable()";
        grad_var = A.grad_variable().value();
    }
    A.zero_grad();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(A);
}

TEST_P(HigherOrderGradTest, Slogdet_DoubleBackward) {
    auto A = well_conditioned_3x3(device);
    Variable grad_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto [sign, logabsdet] = tenzor::slogdet(A);
        auto loss = tenzor::sum(logabsdet);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        EXPECT_GRAD_FLOWS(A);
        ASSERT_TRUE(A.grad_variable().has_value())
            << "create_graph=true must populate grad_variable()";
        grad_var = A.grad_variable().value();
    }
    A.zero_grad();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(A);
}

TEST_P(HigherOrderGradTest, GumbelSoftmax_DoubleBackward) {
    // M32: grad_logits = y_soft * (grad_out - sum(grad_out*y_soft, dim)) / tau
    // depends on y_soft, which itself depends on logits (y_soft =
    // softmax((logits+Gumbel)/tau)) — verify d(grad_logits)/d(logits) is
    // nonzero, i.e. the forward built a genuinely graph-connected y_soft
    // Variable to save (not a pre-detached copy, which silently zeroed this
    // term regardless of has_saved_variables()).
    //
    // loss must be chosen so grad_out (=d(loss)/dy) is a plain CONSTANT with
    // no graph of its own — otherwise grad_out's own chain back to logits
    // (via y's grad_fn) would make the second backward nonzero even with the
    // bug, defeating the test. sum(y) doesn't work either: it makes
    // grad_out a constant ones-vector, but softmax always sums to 1, so its
    // Jacobian applied to a constant vector is identically zero regardless
    // of y_soft's connectivity (grad_logits would be zero even when fixed).
    // sum(y * w) for a fixed, non-differentiable w gives grad_out = w
    // exactly (no graph) while keeping grad_logits generically nonzero —
    // isolating the y_soft_v-connectivity bug precisely.
    auto logits = Variable(randn({4, 5}, DType::Float64, device), true);
    auto w = Variable(randn({4, 5}, DType::Float64, device), false);
    Variable grad_logits_var;
    {
        HigherOrderGraphRetentionGuard guard;
        auto y = tenzor::gumbel_softmax(logits, /*tau=*/1.0, /*hard=*/false, /*dim=*/-1);
        auto loss = tenzor::sum(y * w);
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        ASSERT_TRUE(logits.grad_variable().has_value())
            << "create_graph=true must populate grad_variable() for logits";
        grad_logits_var = logits.grad_variable().value();
    }
    logits.zero_grad();
    auto grad_norm = tenzor::sum(grad_logits_var * grad_logits_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(logits);
}

INSTANTIATE_BACKEND_TESTS(HigherOrderGradTest);
