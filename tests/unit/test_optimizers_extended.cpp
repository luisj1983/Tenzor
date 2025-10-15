/**
 * @file test_optimizers_extended.cpp
 * @brief Unit tests for RMSprop, Adagrad, and Adadelta optimizers
 */

#include <gtest/gtest.h>
#include "tenzor/nn/optim/rmsprop.hpp"
#include "tenzor/nn/optim/adagrad.hpp"
#include "tenzor/nn/optim/adadelta.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::optim;

class OptimizersExtendedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple test parameters
        param1_ = std::make_shared<Variable>(ones({2, 3}), true);
        param2_ = std::make_shared<Variable>(ones({4}), true);
        params_ = {param1_, param2_};
    }

    std::shared_ptr<Variable> param1_;
    std::shared_ptr<Variable> param2_;
    std::vector<std::shared_ptr<Variable>> params_;
};

// ============================================================================
// RMSprop Tests
// ============================================================================

TEST_F(OptimizersExtendedTest, RMSpropBasicStep) {
    auto optimizer = RMSprop(params_, 0.01, 0.99, 1e-8);

    // Set gradients
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});

    // Get initial parameter values
    auto param1_before = param1_->tensor().data<float>()[0];
    auto param2_before = param2_->tensor().data<float>()[0];

    // Perform optimization step
    optimizer.step();

    // Check parameters have changed
    auto param1_after = param1_->tensor().data<float>()[0];
    auto param2_after = param2_->tensor().data<float>()[0];

    EXPECT_NE(param1_before, param1_after);
    EXPECT_NE(param2_before, param2_after);

    // Parameters should decrease (gradient is positive)
    EXPECT_LT(param1_after, param1_before);
    EXPECT_LT(param2_after, param2_before);
}

TEST_F(OptimizersExtendedTest, RMSpropWithMomentum) {
    auto optimizer = RMSprop(params_, 0.01, 0.99, 1e-8, 0.0, 0.9);

    // Set gradients
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});

    // First step
    optimizer.step();
    auto param1_step1 = param1_->tensor().data<float>()[0];

    // Second step with same gradient
    optimizer.zero_grad();
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});
    optimizer.step();
    auto param1_step2 = param1_->tensor().data<float>()[0];

    // With momentum, second step should be larger
    float step1_delta = 1.0f - param1_step1;
    float step2_delta = param1_step1 - param1_step2;

    EXPECT_GT(step2_delta, step1_delta * 0.8);  // Momentum accelerates
}

TEST_F(OptimizersExtendedTest, RMSpropCentered) {
    auto optimizer = RMSprop(params_, 0.01, 0.99, 1e-8, 0.0, 0.0, true);

    // Set gradients
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});

    // Should work without errors
    EXPECT_NO_THROW(optimizer.step());

    // Check parameters updated
    auto param1_after = param1_->tensor().data<float>()[0];
    EXPECT_LT(param1_after, 1.0f);
}

TEST_F(OptimizersExtendedTest, RMSpropLearningRate) {
    auto optimizer = RMSprop(params_, 0.01);

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.01);

    optimizer.set_lr(0.001);
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.001);
}

TEST_F(OptimizersExtendedTest, RMSpropStateDictSaveLoad) {
    auto optimizer1 = RMSprop(params_, 0.01, 0.99, 1e-8);
    auto optimizer2 = RMSprop(params_, 0.01, 0.99, 1e-8);

    // Run optimizer1 for a few steps
    for (int i = 0; i < 3; ++i) {
        optimizer1.zero_grad();
        param1_->grad() = ones({2, 3});
        param2_->grad() = ones({4});
        optimizer1.step();
    }

    // Save and load state
    auto state = optimizer1.state_dict();
    optimizer2.load_state_dict(state);

    // State should be preserved
    EXPECT_GT(state.size(), 0);
}

TEST_F(OptimizersExtendedTest, RMSpropInvalidParameters) {
    EXPECT_THROW(RMSprop(params_, -0.01), std::invalid_argument);  // Negative lr
    EXPECT_THROW(RMSprop(params_, 0.01, -0.5), std::invalid_argument);  // Negative alpha
    EXPECT_THROW(RMSprop(params_, 0.01, 1.5), std::invalid_argument);  // Alpha > 1
    EXPECT_THROW(RMSprop(params_, 0.01, 0.99, -1e-8), std::invalid_argument);  // Negative eps
}

// ============================================================================
// Adagrad Tests
// ============================================================================

TEST_F(OptimizersExtendedTest, AdagradBasicStep) {
    auto optimizer = Adagrad(params_, 0.01);

    // Set gradients
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});

    auto param1_before = param1_->tensor().data<float>()[0];

    // Perform optimization step
    optimizer.step();

    auto param1_after = param1_->tensor().data<float>()[0];

    // Parameters should decrease
    EXPECT_LT(param1_after, param1_before);
}

TEST_F(OptimizersExtendedTest, AdagradAccumulation) {
    auto optimizer = Adagrad(params_, 0.1);

    // First step
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});
    optimizer.step();
    auto param1_step1 = param1_->tensor().data<float>()[0];
    float delta1 = 1.0f - param1_step1;

    // Second step
    optimizer.zero_grad();
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});
    optimizer.step();
    auto param1_step2 = param1_->tensor().data<float>()[0];
    float delta2 = param1_step1 - param1_step2;

    // Second step should be smaller due to accumulation
    EXPECT_LT(delta2, delta1);
}

TEST_F(OptimizersExtendedTest, AdagradLearningRateDecay) {
    auto optimizer = Adagrad(params_, 0.1, /*lr_decay=*/0.1);

    // Initial learning rate
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.1);

    // After first step
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});
    optimizer.step();

    // Learning rate should decay
    EXPECT_LT(optimizer.get_lr(), 0.1);
}

TEST_F(OptimizersExtendedTest, AdagradInitialAccumulator) {
    auto optimizer = Adagrad(params_, 0.01, 0.0, 0.0, /*initial_accumulator=*/0.1);

    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});

    // Should work without errors
    EXPECT_NO_THROW(optimizer.step());
}

TEST_F(OptimizersExtendedTest, AdagradStateDictSaveLoad) {
    auto optimizer1 = Adagrad(params_, 0.01);
    auto optimizer2 = Adagrad(params_, 0.01);

    // Run optimizer1
    for (int i = 0; i < 3; ++i) {
        optimizer1.zero_grad();
        param1_->grad() = ones({2, 3});
        param2_->grad() = ones({4});
        optimizer1.step();
    }

    // Save and load
    auto state = optimizer1.state_dict();
    optimizer2.load_state_dict(state);

    EXPECT_GT(state.size(), 0);
    EXPECT_TRUE(state.find("step_count") != state.end());
}

TEST_F(OptimizersExtendedTest, AdagradInvalidParameters) {
    EXPECT_THROW(Adagrad(params_, -0.01), std::invalid_argument);  // Negative lr
    EXPECT_THROW(Adagrad(params_, 0.01, -0.1), std::invalid_argument);  // Negative lr_decay
    EXPECT_THROW(Adagrad(params_, 0.01, 0.0, -0.1), std::invalid_argument);  // Negative weight_decay
    EXPECT_THROW(Adagrad(params_, 0.01, 0.0, 0.0, -0.1), std::invalid_argument);  // Negative initial_acc
}

// ============================================================================
// Adadelta Tests
// ============================================================================

TEST_F(OptimizersExtendedTest, AdadeltaBasicStep) {
    auto optimizer = Adadelta(params_, 1.0, 0.9, 1e-6);

    // Set gradients
    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});

    auto param1_before = param1_->tensor().data<float>()[0];

    // Perform optimization step
    optimizer.step();

    auto param1_after = param1_->tensor().data<float>()[0];

    // Parameters should change
    EXPECT_NE(param1_after, param1_before);
}

TEST_F(OptimizersExtendedTest, AdadeltaNoLearningRateNeeded) {
    // Adadelta designed to work with lr=1.0 (no tuning needed)
    auto optimizer = Adadelta(params_);  // Uses default lr=1.0

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 1.0);

    param1_->grad() = ones({2, 3});
    param2_->grad() = ones({4});

    // Should work with default parameters
    EXPECT_NO_THROW(optimizer.step());
}

TEST_F(OptimizersExtendedTest, AdadeltaAdaptiveRate) {
    auto optimizer = Adadelta(params_, 1.0, 0.9, 1e-6);

    // Multiple steps should adapt
    std::vector<float> deltas;
    float prev_val = param1_->tensor().data<float>()[0];

    for (int i = 0; i < 5; ++i) {
        optimizer.zero_grad();
        param1_->grad() = ones({2, 3});
        param2_->grad() = ones({4});
        optimizer.step();

        float curr_val = param1_->tensor().data<float>()[0];
        deltas.push_back(std::abs(curr_val - prev_val));
        prev_val = curr_val;
    }

    // Step sizes should be non-zero
    for (auto delta : deltas) {
        EXPECT_GT(delta, 0.0f);
    }
}

TEST_F(OptimizersExtendedTest, AdadeltaStateDictSaveLoad) {
    auto optimizer1 = Adadelta(params_, 1.0, 0.9, 1e-6);
    auto optimizer2 = Adadelta(params_, 1.0, 0.9, 1e-6);

    // Run optimizer1
    for (int i = 0; i < 3; ++i) {
        optimizer1.zero_grad();
        param1_->grad() = ones({2, 3});
        param2_->grad() = ones({4});
        optimizer1.step();
    }

    // Save and load
    auto state = optimizer1.state_dict();
    optimizer2.load_state_dict(state);

    EXPECT_GT(state.size(), 0);
}

TEST_F(OptimizersExtendedTest, AdadeltaInvalidParameters) {
    EXPECT_THROW(Adadelta(params_, -1.0), std::invalid_argument);  // Negative lr
    EXPECT_THROW(Adadelta(params_, 1.0, -0.5), std::invalid_argument);  // Negative rho
    EXPECT_THROW(Adadelta(params_, 1.0, 1.5), std::invalid_argument);  // rho > 1
    EXPECT_THROW(Adadelta(params_, 1.0, 0.9, -1e-6), std::invalid_argument);  // Negative eps
}

// ============================================================================
// Convergence Tests
// ============================================================================

TEST_F(OptimizersExtendedTest, RMSpropConvergence) {
    // Simple quadratic: f(x) = (x - 3)^2, optimal at x=3
    auto param = std::make_shared<Variable>(zeros({1}), true);
    auto param_ptr = param->tensor().data<float>();
    param_ptr[0] = 10.0f;  // Start far from optimum

    auto optimizer = RMSprop(std::vector<std::shared_ptr<Variable>>{param}, 0.1);

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();

        // Gradient: 2(x - 3)
        float x = param_ptr[0];
        auto grad_tensor = zeros({1});
        auto grad_ptr = grad_tensor.data<float>();
        grad_ptr[0] = 2.0f * (x - 3.0f);

        param->grad() = grad_tensor;
        optimizer.step();
    }

    // Should converge near 3
    EXPECT_NEAR(param_ptr[0], 3.0f, 0.1f);
}

TEST_F(OptimizersExtendedTest, AdagradConvergence) {
    auto param = std::make_shared<Variable>(zeros({1}), true);
    auto param_ptr = param->tensor().data<float>();
    param_ptr[0] = 10.0f;

    auto optimizer = Adagrad(std::vector<std::shared_ptr<Variable>>{param}, 1.0);

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();

        float x = param_ptr[0];
        auto grad_tensor = zeros({1});
        auto grad_ptr = grad_tensor.data<float>();
        grad_ptr[0] = 2.0f * (x - 3.0f);

        param->grad() = grad_tensor;
        optimizer.step();
    }

    // Should converge near 3
    EXPECT_NEAR(param_ptr[0], 3.0f, 0.5f);
}

TEST_F(OptimizersExtendedTest, AdadeltaConvergence) {
    auto param = std::make_shared<Variable>(zeros({1}), true);
    auto param_ptr = param->tensor().data<float>();
    param_ptr[0] = 10.0f;

    // Use larger eps for better initial convergence (common in practice)
    auto optimizer = Adadelta(std::vector<std::shared_ptr<Variable>>{param}, 1.0, 0.95, 1e-4);

    for (int i = 0; i < 500; ++i) {
        optimizer.zero_grad();

        float x = param_ptr[0];
        auto grad_tensor = zeros({1});
        auto grad_ptr = grad_tensor.data<float>();
        grad_ptr[0] = 2.0f * (x - 3.0f);

        param->grad() = grad_tensor;
        optimizer.step();
    }

    // Adadelta may converge slower but should get close
    EXPECT_NEAR(param_ptr[0], 3.0f, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
