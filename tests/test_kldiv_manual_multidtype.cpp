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

using namespace tenzor;
using namespace tenzor::testing;

class KLDivManualMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(KLDivManualMultiDTypeTest, SimpleSubtract) {
    auto a = Variable(tenzor::full({2, 3}, 2.0f, dtype(), device()), true);
    auto b = Variable(tenzor::full({2, 3}, 1.0f, dtype(), device()), false);
    auto c = a - b;
    auto loss = tenzor::mean(c);
    loss.backward();

    EXPECT_TRUE(a.grad().has_value());
}

TEST_P(KLDivManualMultiDTypeTest, KLDivForwardShape) {
    nn::KLDivLoss kl_loss(nn::Reduction::Mean);

    auto log_probs = createInput({4, 10}, false);
    auto targets = createInput({4, 10}, false);
    // Make targets non-negative (probabilities)
    targets = Variable(tenzor::abs(targets.tensor()), false);
    auto total = tenzor::sum(targets.tensor(), -1, true);
    targets = Variable(tenzor::div(targets.tensor(), total), false);

    auto loss = kl_loss.forward(Variable(log_probs.tensor(), true), targets);
    // Mean reduction should produce scalar-like output
    EXPECT_LE(loss.tensor().numel(), 1);
    expectDevice(loss.tensor());
}

TEST_P(KLDivManualMultiDTypeTest, KLDivGradientFlow) {
    nn::KLDivLoss kl_loss(nn::Reduction::Mean);

    auto log_probs = createInput({2, 5}, true);
    auto targets = Variable(tenzor::abs(createRandn({2, 5})), false);

    auto loss = kl_loss.forward(log_probs, targets);
    loss.backward();

    EXPECT_TRUE(log_probs.grad().has_value());
    expectShape(log_probs.grad().value(), {2, 5});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(KLDivManualMultiDTypeTest);
