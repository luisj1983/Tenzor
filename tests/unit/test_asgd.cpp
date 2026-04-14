/**
 * @file test_asgd.cpp
 * @brief Unit tests for the ASGD (Averaged Stochastic Gradient Descent) optimizer
 */

#include <gtest/gtest.h>
#include "tenzor/nn/optim/asgd.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>
#include <vector>

namespace tenzor {
    void initialize();
}

using namespace tenzor;
using namespace tenzor::optim;

class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

class ASGDTest : public ::testing::Test {
protected:
    void SetUp() override {
        param1_ = std::make_shared<Variable>(ones({2, 3}), true);
        param2_ = std::make_shared<Variable>(ones({4}), true);
        params_ = {param1_, param2_};
    }

    std::shared_ptr<Variable> param1_;
    std::shared_ptr<Variable> param2_;
    std::vector<std::shared_ptr<Variable>> params_;
};

TEST_F(ASGDTest, BasicStep) {
    auto optimizer = ASGD(params_, /*lr=*/0.01);

    // Set gradients
    param1_->set_grad(ones({2, 3}));
    param2_->set_grad(ones({4}));

    auto param1_before = param1_->tensor().data<float>()[0];
    auto param2_before = param2_->tensor().data<float>()[0];

    optimizer.step();

    auto param1_after = param1_->tensor().data<float>()[0];
    auto param2_after = param2_->tensor().data<float>()[0];

    // Parameters should have changed
    EXPECT_NE(param1_before, param1_after);
    EXPECT_NE(param2_before, param2_after);

    // With positive gradients, parameters should decrease
    EXPECT_LT(param1_after, param1_before);
    EXPECT_LT(param2_after, param2_before);
}

TEST_F(ASGDTest, ConvergenceOnQuadratic) {
    // Minimize f(x) = 0.5 * ||x||^2 (target is 0)
    // Gradient = x
    auto x = std::make_shared<Variable>(ones({3}) * 5.0f, true);
    std::vector<std::shared_ptr<Variable>> params = {x};

    auto optimizer = ASGD(params, /*lr=*/0.1, /*lambd=*/1e-4, /*alpha=*/0.75,
                          /*t0=*/5.0);

    float initial_norm = 0.0f;
    auto* init_data = x->tensor().data<float>();
    for (int i = 0; i < 3; ++i) initial_norm += init_data[i] * init_data[i];

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        // Gradient = x
        x->set_grad(x->tensor());
        optimizer.step();
    }

    float final_norm = 0.0f;
    auto* result = x->tensor().data<float>();
    for (int i = 0; i < 3; ++i) final_norm += result[i] * result[i];

    EXPECT_LT(final_norm, initial_norm)
        << "ASGD should have reduced the parameter norm toward zero";
}

TEST_F(ASGDTest, MultipleStepsDecreaseMonotonically) {
    // With constant positive gradient, parameter should keep decreasing
    auto x = std::make_shared<Variable>(ones({2}) * 10.0f, true);
    auto optimizer = ASGD({x}, /*lr=*/0.1, /*lambd=*/1e-4, /*alpha=*/0.75,
                          /*t0=*/0.0);

    float prev_val = x->tensor().data<float>()[0];
    for (int i = 0; i < 10; ++i) {
        x->set_grad(ones({2}));
        optimizer.step();
        float curr_val = x->tensor().data<float>()[0];
        EXPECT_LT(curr_val, prev_val) << "Step " << i << " should decrease parameter";
        prev_val = curr_val;
    }
}

TEST_F(ASGDTest, ZeroGrad) {
    auto optimizer = ASGD(params_, 0.01);

    param1_->set_grad(ones({2, 3}));
    param2_->set_grad(ones({4}));

    optimizer.zero_grad();

    // After zero_grad, gradients should not exist or be zeroed
    // (depending on implementation: either has_grad is false or grad is zeros)
    if (param1_->has_grad()) {
        auto* g = param1_->grad().value().data<float>();
        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(g[i], 0.0f);
        }
    }
}

TEST_F(ASGDTest, LearningRateAccessors) {
    auto optimizer = ASGD(params_, 0.05);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.05);

    optimizer.set_lr(0.01);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);
}
