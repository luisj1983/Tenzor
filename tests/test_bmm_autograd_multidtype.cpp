/**
 * @file test_bmm_autograd_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for batched matrix multiply with autograd
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class BmmAutogradMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(BmmAutogradMultiDTypeTest, BmmForwardShape) {
    auto a = createInput({4, 3, 5}, false);
    auto b = createInput({4, 5, 7}, false);
    auto result = tenzor::bmm(a, b);
    expectShape(result.tensor(), {4, 3, 7});
    expectDevice(result.tensor());
    expectDType(result.tensor());
}

TEST_P(BmmAutogradMultiDTypeTest, BmmGradientFlow) {
    auto a = createInput({2, 3, 4}, true);
    auto b = createInput({2, 4, 5}, true);
    auto c = tenzor::bmm(a, b);
    auto loss = tenzor::sum(c);
    loss.backward();

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    expectShape(a.grad().value(), {2, 3, 4});
    expectShape(b.grad().value(), {2, 4, 5});
}

TEST_P(BmmAutogradMultiDTypeTest, MatmulGradient) {
    auto a = createInput({3, 4}, true);
    auto b = createInput({4, 5}, true);
    auto c = tenzor::matmul(a, b);
    auto loss = tenzor::sum(c);
    loss.backward();

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    expectShape(a.grad().value(), {3, 4});
    expectShape(b.grad().value(), {4, 5});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BmmAutogradMultiDTypeTest);
