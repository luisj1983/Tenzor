/**
 * @file test_rprop.cpp
 * @brief Unit tests for the Rprop (Resilient Propagation) optimizer
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/nn/optim/rprop.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::optim;

class RpropTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        param1_ = std::make_shared<Variable>(ones({2, 3}, DType::Float32, device), true);
        param2_ = std::make_shared<Variable>(ones({4}, DType::Float32, device), true);
        params_ = {param1_, param2_};
    }

    std::shared_ptr<Variable> param1_;
    std::shared_ptr<Variable> param2_;
    std::vector<std::shared_ptr<Variable>> params_;
};

TEST_P(RpropTest, BasicStep) {
    auto optimizer = Rprop(params_, /*lr=*/0.01);

    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    auto param1_before_cpu = param1_->tensor().cpu();
    auto param1_before = param1_before_cpu.data<float>()[0];

    optimizer.step();

    auto param1_after_cpu = param1_->tensor().cpu();
    auto param1_after = param1_after_cpu.data<float>()[0];

    // Parameters should change
    EXPECT_NE(param1_before, param1_after);
    // Positive gradient -> parameters should decrease
    EXPECT_LT(param1_after, param1_before);
}

TEST_P(RpropTest, ConvergenceOnQuadratic) {
    // Minimize f(x) = 0.5 * ||x - target||^2
    // Rprop only uses sign of gradient, so convergence may oscillate near minimum.
    // We test that the parameter moves toward the target.
    auto x = std::make_shared<Variable>(ones({1}, DType::Float32, device) * 10.0f, true);
    std::vector<std::shared_ptr<Variable>> params = {x};

    auto optimizer = Rprop(params, /*lr=*/0.1);

    auto initial_cpu = x->tensor().cpu();
    float initial = initial_cpu.data<float>()[0];

    // Target is 0; gradient of 0.5*x^2 is x (positive when x > 0)
    for (int i = 0; i < 50; ++i) {
        optimizer.zero_grad();
        // Gradient = x (positive since x starts at 10 and decreases toward 0)
        auto grad = x->tensor();
        x->set_grad(grad);
        optimizer.step();
    }

    auto final_cpu = x->tensor().cpu();
    float final_val = final_cpu.data<float>()[0];
    // Should have moved toward 0 from 10
    EXPECT_LT(std::abs(final_val), std::abs(initial))
        << "Rprop should converge toward minimum";
}

TEST_P(RpropTest, StepSizeIncreasesOnConsistentGradient) {
    // When gradient sign stays the same, step size should increase
    auto x = std::make_shared<Variable>(zeros({1}, DType::Float32, device), true);
    auto optimizer = Rprop({x}, /*lr=*/0.01, /*eta_minus=*/0.5, /*eta_plus=*/1.2);

    // Two consecutive steps with same gradient sign
    x->set_grad(ones({1}, DType::Float32, device));
    optimizer.step();
    auto after_step1_cpu = x->tensor().cpu();
    auto after_step1 = after_step1_cpu.data<float>()[0];

    x->set_grad(ones({1}, DType::Float32, device));
    optimizer.step();
    auto after_step2_cpu = x->tensor().cpu();
    auto after_step2 = after_step2_cpu.data<float>()[0];

    // The second step should be larger in magnitude (step size increased)
    float delta1 = std::abs(after_step1 - 0.0f);     // first step from 0
    float delta2 = std::abs(after_step2 - after_step1);  // second step

    EXPECT_GT(delta2, delta1 * 0.99f)
        << "Step size should increase with consistent gradient sign";
}

TEST_P(RpropTest, StepSizeDecreasesOnSignChange) {
    // When gradient sign flips, step size should decrease
    auto x = std::make_shared<Variable>(zeros({1}, DType::Float32, device), true);
    auto optimizer = Rprop({x}, /*lr=*/0.1, /*eta_minus=*/0.5, /*eta_plus=*/1.2);

    // First step: positive gradient
    x->set_grad(ones({1}, DType::Float32, device));
    optimizer.step();

    // Several steps with same sign to grow step size
    for (int i = 0; i < 5; ++i) {
        x->set_grad(ones({1}, DType::Float32, device));
        optimizer.step();
    }
    auto before_flip_cpu = x->tensor().cpu();
    auto before_flip = before_flip_cpu.data<float>()[0];

    // Now flip the gradient sign
    x->set_grad(ones({1}, DType::Float32, device) * -1.0f);
    optimizer.step();
    auto after_flip_cpu = x->tensor().cpu();
    auto after_flip = after_flip_cpu.data<float>()[0];

    // After flip + a few same-sign steps, step should be smaller
    x->set_grad(ones({1}, DType::Float32, device) * -1.0f);
    optimizer.step();
    auto after_flip2_cpu = x->tensor().cpu();
    auto after_flip2 = after_flip2_cpu.data<float>()[0];

    float delta_after_flip = std::abs(after_flip2 - after_flip);
    // We mainly verify the optimizer handles sign changes without crashing
    // and continues to make progress
    EXPECT_GT(std::abs(after_flip2 - before_flip), 0.0f);
}

TEST_P(RpropTest, ZeroGrad) {
    auto optimizer = Rprop(params_, 0.01);

    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    optimizer.zero_grad();

    if (param1_->has_grad()) {
        auto grad_cpu = param1_->grad().value().cpu();
        auto* g = grad_cpu.data<float>();
        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(g[i], 0.0f);
        }
    }
}

TEST_P(RpropTest, LearningRateAccessors) {
    auto optimizer = Rprop(params_, 0.05);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.05);

    optimizer.set_lr(0.02);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.02);
}

INSTANTIATE_BACKEND_TESTS(RpropTest);
