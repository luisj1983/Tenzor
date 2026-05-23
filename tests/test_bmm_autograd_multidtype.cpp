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
    // Build deterministic input on CPU, compute reference, then move to target.
    auto a_cpu = tenzor::randn({4, 3, 5}, DType::Float32, Device::cpu());
    auto b_cpu = tenzor::randn({4, 5, 7}, DType::Float32, Device::cpu());
    auto ref = tenzor::bmm(Variable(a_cpu, false), Variable(b_cpu, false));

    auto a = Variable(a_cpu.to(dtype_).to(device_), false);
    auto b = Variable(b_cpu.to(dtype_).to(device_), false);
    auto result = tenzor::bmm(a, b);
    expectShape(result.tensor(), {4, 3, 7});
    expectDevice(result.tensor());
    expectDType(result.tensor());
    expectTensorNear(result.tensor(), ref.tensor(),
                     std::max(atol_, 5e-2f));
}

TEST_P(BmmAutogradMultiDTypeTest, BmmGradientFlow) {
    auto a_cpu = tenzor::randn({2, 3, 4}, DType::Float32, Device::cpu());
    auto b_cpu = tenzor::randn({2, 4, 5}, DType::Float32, Device::cpu());

    // CPU reference forward
    auto a_ref = Variable(a_cpu, true);
    auto b_ref = Variable(b_cpu, true);
    auto c_ref = tenzor::bmm(a_ref, b_ref);
    auto loss_ref = tenzor::sum(c_ref);
    loss_ref.backward();

    auto a = Variable(a_cpu.to(dtype_).to(device_), true);
    auto b = Variable(b_cpu.to(dtype_).to(device_), true);
    auto c = tenzor::bmm(a, b);
    auto loss = tenzor::sum(c);
    loss.backward();

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    expectShape(a.grad().value(), {2, 3, 4});
    expectShape(b.grad().value(), {2, 4, 5});
    // Forward output equals CPU reference
    expectTensorNear(c.tensor(), c_ref.tensor(), std::max(atol_, 5e-2f));
    // Gradients match CPU reference
    expectTensorNear(a.grad().value(), a_ref.grad().value(),
                     std::max(atol_, 5e-2f));
    expectTensorNear(b.grad().value(), b_ref.grad().value(),
                     std::max(atol_, 5e-2f));
}

TEST_P(BmmAutogradMultiDTypeTest, MatmulGradient) {
    auto a_cpu = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    auto b_cpu = tenzor::randn({4, 5}, DType::Float32, Device::cpu());

    auto a_ref = Variable(a_cpu, true);
    auto b_ref = Variable(b_cpu, true);
    auto c_ref = tenzor::matmul(a_ref, b_ref);
    auto loss_ref = tenzor::sum(c_ref);
    loss_ref.backward();

    auto a = Variable(a_cpu.to(dtype_).to(device_), true);
    auto b = Variable(b_cpu.to(dtype_).to(device_), true);
    auto c = tenzor::matmul(a, b);
    auto loss = tenzor::sum(c);
    loss.backward();

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    expectShape(a.grad().value(), {3, 4});
    expectShape(b.grad().value(), {4, 5});
    expectTensorNear(c.tensor(), c_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(a.grad().value(), a_ref.grad().value(),
                     std::max(atol_, 5e-2f));
    expectTensorNear(b.grad().value(), b_ref.grad().value(),
                     std::max(atol_, 5e-2f));
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BmmAutogradMultiDTypeTest);
