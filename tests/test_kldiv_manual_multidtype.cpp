/**
 * @file test_kldiv_manual_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for KL divergence loss
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include "multi_backend_dtype_fixture.hpp"
#include "grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class KLDivManualMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(KLDivManualMultiDTypeTest, SimpleSubtract) {
    auto a = Variable(tenzor::full({2, 3}, 2.0f, dtype(), device()), true);
    auto b = Variable(tenzor::full({2, 3}, 1.0f, dtype(), device()), false);
    auto c = a - b;
    auto loss = tenzor::mean(c);
    loss.backward();

    EXPECT_GRAD_FLOWS(a);
    // c = 2-1 = 1, mean(c) = 1, d/da mean(a-b) = 1/numel = 1/6 broadcast
    auto expected_loss = tenzor::full({}, 1.0f, dtype_, device_);
    expectTensorNear(loss.tensor(), expected_loss, std::max(atol_, 1e-3f));
    auto expected_grad = tenzor::full({2, 3}, 1.0f / 6.0f, dtype_, device_);
    expectTensorNear(a.grad().value(), expected_grad, std::max(atol_, 1e-3f));
}

TEST_P(KLDivManualMultiDTypeTest, KLDivForwardShape) {
    nn::KLDivLoss kl_loss(nn::Reduction::Mean);

    // Build CPU reference from same fp32 source data.
    auto log_probs_cpu = tenzor::randn({4, 10}, DType::Float32, Device::cpu());
    auto targets_raw = tenzor::randn({4, 10}, DType::Float32, Device::cpu());
    auto targets_abs = tenzor::abs(targets_raw);
    auto total = tenzor::sum(targets_abs, -1, true);
    auto targets_cpu = tenzor::div(targets_abs, total);

    auto loss_ref = kl_loss.forward(Variable(log_probs_cpu, true),
                                    Variable(targets_cpu, false));

    auto log_probs = Variable(log_probs_cpu.to(dtype_).to(device_), true);
    auto targets = Variable(targets_cpu.to(dtype_).to(device_), false);
    auto loss = kl_loss.forward(log_probs, targets);
    EXPECT_LE(loss.tensor().numel(), 1);
    expectDevice(loss.tensor());
    expectTensorNear(loss.tensor(), loss_ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(KLDivManualMultiDTypeTest, KLDivGradientFlow) {
    nn::KLDivLoss kl_loss(nn::Reduction::Mean);

    auto log_probs_cpu = tenzor::randn({2, 5}, DType::Float32, Device::cpu());
    auto targets_cpu = tenzor::abs(tenzor::randn({2, 5}, DType::Float32, Device::cpu()));

    auto lp_ref = Variable(log_probs_cpu, true);
    auto loss_ref = kl_loss.forward(lp_ref, Variable(targets_cpu, false));
    loss_ref.backward();

    auto log_probs = Variable(log_probs_cpu.to(dtype_).to(device_), true);
    auto targets = Variable(targets_cpu.to(dtype_).to(device_), false);
    auto loss = kl_loss.forward(log_probs, targets);
    loss.backward();

    EXPECT_GRAD_FLOWS(log_probs);
    expectShape(log_probs.grad().value(), {2, 5});
    expectTensorNear(loss.tensor(), loss_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(log_probs.grad().value(), lp_ref.grad().value(),
                     std::max(atol_, 5e-2f));
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(KLDivManualMultiDTypeTest);
