/**
 * @file test_schedulers_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for LR schedulers
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <cmath>
#include <numbers>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::optim;

class SchedulersMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(SchedulersMultiDTypeTest, StepLR_BasicStep) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);
    auto scheduler = StepLR(optimizer, 2, 0.1);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-9);
}

TEST_P(SchedulersMultiDTypeTest, ExponentialLR_BasicDecay) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = ExponentialLR(optimizer, 0.9);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.9, 1e-9);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.81, 1e-9);
}

TEST_P(SchedulersMultiDTypeTest, CosineAnnealingLR_BasicCycle) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = CosineAnnealingLR(optimizer, 10, 0.0);

    for (int i = 0; i < 5; i++) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-6);

    for (int i = 5; i < 10; i++) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0, 1e-6);
}

TEST_P(SchedulersMultiDTypeTest, MultiStepLR_DecaysAtMilestones) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = MultiStepLR(optimizer, {3, 6}, 0.5);

    for (int i = 0; i < 2; ++i) scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-9);
    for (int i = 0; i < 3; ++i) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.25, 1e-9);
}

TEST_P(SchedulersMultiDTypeTest, PolynomialLR_LinearDecay) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = PolynomialLR(optimizer, 10, 0.0, 1.0);

    for (int e = 1; e <= 10; ++e) {
        scheduler.step();
        double expected = std::pow(1.0 - static_cast<double>(e) / 10.0, 1.0);
        EXPECT_NEAR(scheduler.get_last_lr(), expected, 1e-9);
    }
}

TEST_P(SchedulersMultiDTypeTest, LinearWarmupScheduler_Ramps) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto base = std::make_shared<StepLR>(optimizer, 10000, 1.0);
    LinearWarmupScheduler scheduler(optimizer, base, 5, 0.1);

    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-9);
    for (int64_t step = 1; step <= 5; ++step) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-9);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SchedulersMultiDTypeTest);
