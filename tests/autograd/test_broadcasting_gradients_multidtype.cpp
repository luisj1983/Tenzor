/**
 * @file test_broadcasting_gradients_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for gradient shapes with broadcasting
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class BroadcastGradientsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void check_grad_shape(const Variable& var, const std::vector<int64_t>& expected_shape) {
        ASSERT_TRUE(var.grad().has_value()) << "Gradient not computed";
        auto grad = var.grad().value();
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        EXPECT_EQ(grad_shape, expected_shape);
    }
};

TEST_P(BroadcastGradientsMultiDTypeTest, AddMatrixPlusRow) {
    Variable a(createOnes({3, 4}), true);
    Variable b(createOnes({4}), true);
    auto c = a + b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 4});
    check_grad_shape(b, {4});
}

TEST_P(BroadcastGradientsMultiDTypeTest, AddMatrixPlusScalar) {
    Variable a(createOnes({3, 4}), true);
    Variable b(createOnes({1}), true);
    auto c = a + b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 4});
    check_grad_shape(b, {1});
}

TEST_P(BroadcastGradientsMultiDTypeTest, MulBroadcast) {
    Variable a(createOnes({2, 3}), true);
    Variable b(createOnes({3}), true);
    auto c = a * b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {2, 3});
    check_grad_shape(b, {3});
}

TEST_P(BroadcastGradientsMultiDTypeTest, SubBroadcast) {
    Variable a(createOnes({4, 1}), true);
    Variable b(createOnes({1, 3}), true);
    auto c = a - b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {4, 1});
    check_grad_shape(b, {1, 3});
}

TEST_P(BroadcastGradientsMultiDTypeTest, DivBroadcast) {
    Variable a(createInput({2, 3}, true));
    Variable b(Variable(tenzor::full({3}, 2.0f, dtype(), device()), true));
    auto c = a / b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {2, 3});
    check_grad_shape(b, {3});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BroadcastGradientsMultiDTypeTest);
