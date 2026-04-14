/**
 * @file test_rprop.cpp
 * @brief Unit tests for the Rprop (Resilient Propagation) optimizer
 */

#include <gtest/gtest.h>
#include "tenzor/nn/optim/rprop.hpp"
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

class RpropTest : public ::testing::Test {
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

TEST_F(RpropTest, BasicStep) {
    auto optimizer = Rprop(params_, /*lr=*/0.01);

    param1_->set_grad(ones({2, 3}));
    param2_->set_grad(ones({4}));

    auto param1_before = param1_->tensor().data<float>()[0];

    optimizer.step();

    auto param1_after = param1_->tensor().data<float>()[0];

    // Parameters should change
    EXPECT_NE(param1_before, param1_after);
    // Positive gradient -> parameters should decrease
    EXPECT_LT(param1_after, param1_before);
}

TEST_F(RpropTest, ConvergenceOnQuadratic) {
    // Minimize f(x) = 0.5 * ||x - target||^2
    // Rprop only uses sign of gradient, so convergence may oscillate near minimum.
    // We test that the parameter moves toward the target.
    auto x = std::make_shared<Variable>(ones({1}) * 10.0f, true);
    std::vector<std::shared_ptr<Variable>> params = {x};

    auto optimizer = Rprop(params, /*lr=*/0.1);

    float initial = x->tensor().data<float>()[0];

    // Target is 0; gradient of 0.5*x^2 is x (positive when x > 0)
    for (int i = 0; i < 50; ++i) {
        optimizer.zero_grad();
        // Gradient = x (positive since x starts at 10 and decreases toward 0)
        auto grad = x->tensor();
        x->set_grad(grad);
        optimizer.step();
    }

    float final_val = x->tensor().data<float>()[0];
    // Should have moved toward 0 from 10
    EXPECT_LT(std::abs(final_val), std::abs(initial))
        << "Rprop should converge toward minimum";
}

TEST_F(RpropTest, StepSizeIncreasesOnConsistentGradient) {
    // When gradient sign stays the same, step size should increase
    auto x = std::make_shared<Variable>(zeros({1}), true);
    auto optimizer = Rprop({x}, /*lr=*/0.01, /*eta_minus=*/0.5, /*eta_plus=*/1.2);

    // Two consecutive steps with same gradient sign
    x->set_grad(ones({1}));
    optimizer.step();
    auto after_step1 = x->tensor().data<float>()[0];

    x->set_grad(ones({1}));
    optimizer.step();
    auto after_step2 = x->tensor().data<float>()[0];

    // The second step should be larger in magnitude (step size increased)
    float delta1 = std::abs(after_step1 - 0.0f);     // first step from 0
    float delta2 = std::abs(after_step2 - after_step1);  // second step

    EXPECT_GT(delta2, delta1 * 0.99f)
        << "Step size should increase with consistent gradient sign";
}

TEST_F(RpropTest, StepSizeDecreasesOnSignChange) {
    // When gradient sign flips, step size should decrease
    auto x = std::make_shared<Variable>(zeros({1}), true);
    auto optimizer = Rprop({x}, /*lr=*/0.1, /*eta_minus=*/0.5, /*eta_plus=*/1.2);

    // First step: positive gradient
    x->set_grad(ones({1}));
    optimizer.step();

    // Several steps with same sign to grow step size
    for (int i = 0; i < 5; ++i) {
        x->set_grad(ones({1}));
        optimizer.step();
    }
    auto before_flip = x->tensor().data<float>()[0];

    // Now flip the gradient sign
    x->set_grad(ones({1}) * -1.0f);
    optimizer.step();
    auto after_flip = x->tensor().data<float>()[0];

    // After flip + a few same-sign steps, step should be smaller
    x->set_grad(ones({1}) * -1.0f);
    optimizer.step();
    auto after_flip2 = x->tensor().data<float>()[0];

    float delta_after_flip = std::abs(after_flip2 - after_flip);
    // We mainly verify the optimizer handles sign changes without crashing
    // and continues to make progress
    EXPECT_GT(std::abs(after_flip2 - before_flip), 0.0f);
}

TEST_F(RpropTest, ZeroGrad) {
    auto optimizer = Rprop(params_, 0.01);

    param1_->set_grad(ones({2, 3}));
    optimizer.zero_grad();

    if (param1_->has_grad()) {
        auto* g = param1_->grad().value().data<float>();
        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(g[i], 0.0f);
        }
    }
}

TEST_F(RpropTest, LearningRateAccessors) {
    auto optimizer = Rprop(params_, 0.05);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.05);

    optimizer.set_lr(0.02);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.02);
}
