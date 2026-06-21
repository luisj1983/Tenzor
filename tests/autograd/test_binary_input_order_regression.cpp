/**
 * @file test_binary_input_order_regression.cpp
 * @brief Regression for the conditional-input_variables misalignment in the
 *        binary autograd wrappers (logaddexp / logaddexp2 / xlogy / vecdot /
 *        cosine_similarity / householder_product / tensorsolve).
 *
 * The bug: these wrappers set next_functions = {a.grad_fn(), b.grad_fn()}
 * unconditionally (2 slots) but built input_variables CONDITIONALLY
 * (`if (a.requires_grad()) push a; if (b.requires_grad()) push b;`). Their
 * backward() always returns a fixed 2-element {grad_a, grad_b}. The engine
 * zips input_grads[i] into input_variables[i] positionally, so when ONLY the
 * second operand requires grad, input_variables == {b} (size 1) and the engine
 * accumulates input_grads[0] == grad_a (the FIRST operand's adjoint) onto b —
 * b.grad silently held ∂/∂a instead of ∂/∂b. The fix pushes BOTH operands
 * unconditionally so the positional alignment is preserved (the engine skips
 * non-requires_grad slots itself).
 *
 * These tests exercise the SECOND-operand-only case directly: a is a detached
 * constant, b is the leaf parameter. They gradcheck f(b) = op(a, b) — which
 * compares b.grad against the numerical ∂/∂b — and would FAIL (b.grad ==
 * ∂/∂a) under the old conditional-push code.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;

namespace {

// Centered random values in roughly [-0.5, 0.5] for well-conditioned logs.
Variable make_const(std::vector<int64_t> shape, const Device& device,
                    double scale = 1.0) {
    Tensor t = (randn(shape, DType::Float64, Device::cpu()) * scale)
                   .to(device);
    return Variable(t, /*requires_grad=*/false);  // detached constant (a)
}
Variable make_leaf(std::vector<int64_t> shape, const Device& device,
                   double scale = 1.0) {
    Tensor t = (randn(shape, DType::Float64, Device::cpu()) * scale)
                   .to(device);
    return Variable(t, /*requires_grad=*/true);    // leaf parameter (b)
}

}  // namespace

class BinaryInputOrderTest : public ::tenzor::testing::BackendTest {};

// logaddexp: ∂/∂a = sigmoid(a-b), ∂/∂b = sigmoid(b-a) — different, so routing
// grad_a onto b is observable.
TEST_P(BinaryInputOrderTest, LogAddExpSecondArgOnly) {
    auto a = make_const({4, 5}, device);
    auto b = make_leaf({4, 5}, device);
    auto f = [&a](const Variable& v) { return logaddexp(a, v); };
    EXPECT_TRUE(gradcheck(f, b, 1e-6, 1e-5, 1e-4));
}

TEST_P(BinaryInputOrderTest, LogAddExp2SecondArgOnly) {
    auto a = make_const({4, 5}, device);
    auto b = make_leaf({4, 5}, device);
    auto f = [&a](const Variable& v) { return logaddexp2(a, v); };
    EXPECT_TRUE(gradcheck(f, b, 1e-6, 1e-5, 1e-4));
}

// xlogy: ∂/∂x = log(y), ∂/∂y = x/y — wildly different.
TEST_P(BinaryInputOrderTest, XLogYSecondArgOnly) {
    // x must be a positive constant; y is the positive leaf. rand() is in
    // [0,1); shift to [0.5, 1.5) so both stay strictly positive.
    auto x = Variable((rand({4, 5}, DType::Float64, Device::cpu()) + 0.5)
                          .to(device), false);
    auto y = Variable((rand({4, 5}, DType::Float64, Device::cpu()) + 0.5)
                          .to(device), true);
    auto f = [&x](const Variable& v) { return xlogy(x, v); };
    EXPECT_TRUE(gradcheck(f, y, 1e-6, 1e-5, 1e-4));
}

TEST_P(BinaryInputOrderTest, VecdotSecondArgOnly) {
    auto a = make_const({3, 6}, device);
    auto b = make_leaf({3, 6}, device);
    auto f = [&a](const Variable& v) { return vecdot(a, v, -1); };
    EXPECT_TRUE(gradcheck(f, b, 1e-6, 1e-5, 1e-4));
}

TEST_P(BinaryInputOrderTest, CosineSimilaritySecondArgOnly) {
    auto a = make_const({4, 8}, device);
    auto b = make_leaf({4, 8}, device);
    auto f = [&a](const Variable& v) { return cosine_similarity(a, v, 1); };
    EXPECT_TRUE(gradcheck(f, b, 1e-6, 1e-5, 1e-3));
}

INSTANTIATE_BACKEND_TESTS(BinaryInputOrderTest);
