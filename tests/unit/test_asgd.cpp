/**
 * @file test_asgd.cpp
 * @brief Unit tests for the ASGD (Averaged Stochastic Gradient Descent) optimizer
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/nn/optim/asgd.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::optim;

class ASGDTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        param1_ = std::make_shared<Variable>(ones({2, 3}, DType::Float32, device), true);
        param2_ = std::make_shared<Variable>(ones({4}, DType::Float32, device), true);
        params_ = {param1_, param2_};
    }

    std::shared_ptr<Variable> param1_;
    std::shared_ptr<Variable> param2_;
    std::vector<std::shared_ptr<Variable>> params_;
};

TEST_P(ASGDTest, BasicStep) {
    auto optimizer = ASGD(params_, /*lr=*/0.01);

    // Set gradients
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    auto p1_before_cpu = param1_->tensor().cpu();
    auto p2_before_cpu = param2_->tensor().cpu();
    auto param1_before = p1_before_cpu.data<float>()[0];
    auto param2_before = p2_before_cpu.data<float>()[0];

    optimizer.step();

    auto p1_after_cpu = param1_->tensor().cpu();
    auto p2_after_cpu = param2_->tensor().cpu();
    auto param1_after = p1_after_cpu.data<float>()[0];
    auto param2_after = p2_after_cpu.data<float>()[0];

    // Parameters should have changed
    EXPECT_NE(param1_before, param1_after) << "Failed on " << device.to_string();
    EXPECT_NE(param2_before, param2_after) << "Failed on " << device.to_string();

    // With positive gradients, parameters should decrease
    EXPECT_LT(param1_after, param1_before) << "Failed on " << device.to_string();
    EXPECT_LT(param2_after, param2_before) << "Failed on " << device.to_string();
}

TEST_P(ASGDTest, ConvergenceOnQuadratic) {
    // Minimize f(x) = 0.5 * ||x||^2 (target is 0)
    // Gradient = x
    auto x = std::make_shared<Variable>(ones({3}, DType::Float32, device) * 5.0f, true);
    std::vector<std::shared_ptr<Variable>> params = {x};

    auto optimizer = ASGD(params, /*lr=*/0.1, /*lambd=*/1e-4, /*alpha=*/0.75,
                          /*t0=*/5.0);

    float initial_norm = 0.0f;
    {
        auto init_cpu = x->tensor().cpu();
        auto* init_data = init_cpu.data<float>();
        for (int i = 0; i < 3; ++i) initial_norm += init_data[i] * init_data[i];
    }

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        // Gradient = x
        x->set_grad(x->tensor());
        optimizer.step();
    }

    float final_norm = 0.0f;
    {
        auto result_cpu = x->tensor().cpu();
        auto* result = result_cpu.data<float>();
        for (int i = 0; i < 3; ++i) final_norm += result[i] * result[i];
    }

    EXPECT_LT(final_norm, initial_norm)
        << "ASGD should have reduced the parameter norm toward zero"
        << " - Failed on " << device.to_string();
}

TEST_P(ASGDTest, MultipleStepsDecreaseMonotonically) {
    // With constant positive gradient, parameter should keep decreasing
    auto x = std::make_shared<Variable>(ones({2}, DType::Float32, device) * 10.0f, true);
    auto optimizer = ASGD({x}, /*lr=*/0.1, /*lambd=*/1e-4, /*alpha=*/0.75,
                          /*t0=*/0.0);

    float prev_val;
    {
        auto prev_cpu = x->tensor().cpu();
        prev_val = prev_cpu.data<float>()[0];
    }
    for (int i = 0; i < 10; ++i) {
        x->set_grad(ones({2}, DType::Float32, device));
        optimizer.step();
        auto curr_cpu = x->tensor().cpu();
        float curr_val = curr_cpu.data<float>()[0];
        EXPECT_LT(curr_val, prev_val)
            << "Step " << i << " should decrease parameter"
            << " - Failed on " << device.to_string();
        prev_val = curr_val;
    }
}

TEST_P(ASGDTest, ZeroGrad) {
    auto optimizer = ASGD(params_, 0.01);

    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    optimizer.zero_grad();

    // After zero_grad, gradients should not exist or be zeroed
    // (depending on implementation: either has_grad is false or grad is zeros)
    if (param1_->has_grad()) {
        auto grad_cpu = param1_->grad().value().cpu();
        auto* g = grad_cpu.data<float>();
        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(g[i], 0.0f) << "Failed on " << device.to_string();
        }
    }
}

TEST_P(ASGDTest, LearningRateAccessors) {
    auto optimizer = ASGD(params_, 0.05);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.05) << "Failed on " << device.to_string();

    optimizer.set_lr(0.01);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(ASGDTest);
