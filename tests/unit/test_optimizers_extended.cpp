/**
 * @file test_optimizers_extended.cpp
 * @brief Unit tests for RMSprop, Adagrad, and Adadelta optimizers
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
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

class OptimizersExtendedTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        // Create simple test parameters on the target device
        param1_ = std::make_shared<Variable>(ones({2, 3}, DType::Float32, device), true);
        param2_ = std::make_shared<Variable>(ones({4}, DType::Float32, device), true);
        params_ = {param1_, param2_};
    }

    // Read the first scalar element of a tensor on the host.
    static float first(const Tensor& t) {
        return t.cpu().data<float>()[0];
    }

    std::shared_ptr<Variable> param1_;
    std::shared_ptr<Variable> param2_;
    std::vector<std::shared_ptr<Variable>> params_;
};

// ============================================================================
// RMSprop Tests
// ============================================================================

TEST_P(OptimizersExtendedTest, RMSpropBasicStep) {
    auto optimizer = RMSprop(params_, 0.01, 0.99, 1e-8);

    // Set gradients
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    // Get initial parameter values
    auto param1_before = first(param1_->tensor());
    auto param2_before = first(param2_->tensor());

    // Perform optimization step
    optimizer.step();

    // Check parameters have changed
    auto param1_after = first(param1_->tensor());
    auto param2_after = first(param2_->tensor());

    EXPECT_NE(param1_before, param1_after);
    EXPECT_NE(param2_before, param2_after);

    // Parameters should decrease (gradient is positive)
    EXPECT_LT(param1_after, param1_before);
    EXPECT_LT(param2_after, param2_before);
}

TEST_P(OptimizersExtendedTest, RMSpropWithMomentum) {
    auto optimizer = RMSprop(params_, 0.01, 0.99, 1e-8, 0.0, 0.9);

    // Set gradients
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    // First step
    optimizer.step();
    auto param1_step1 = first(param1_->tensor());

    // Second step with same gradient
    optimizer.zero_grad();
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));
    optimizer.step();
    auto param1_step2 = first(param1_->tensor());

    // With momentum, second step should be larger
    float step1_delta = 1.0f - param1_step1;
    float step2_delta = param1_step1 - param1_step2;

    EXPECT_GT(step2_delta, step1_delta * 0.8);  // Momentum accelerates
}

TEST_P(OptimizersExtendedTest, RMSpropCentered) {
    auto optimizer = RMSprop(params_, 0.01, 0.99, 1e-8, 0.0, 0.0, true);

    // Set gradients
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    // Should work without errors
    EXPECT_NO_THROW(optimizer.step());

    // Check parameters updated
    auto param1_after = first(param1_->tensor());
    EXPECT_LT(param1_after, 1.0f);
}

TEST_P(OptimizersExtendedTest, RMSpropLearningRate) {
    auto optimizer = RMSprop(params_, 0.01);

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.01);

    optimizer.set_lr(0.001);
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.001);
}

TEST_P(OptimizersExtendedTest, RMSpropStateDictSaveLoad) {
    auto optimizer1 = RMSprop(params_, 0.01, 0.99, 1e-8);
    auto optimizer2 = RMSprop(params_, 0.01, 0.99, 1e-8);

    // Run optimizer1 for a few steps
    for (int i = 0; i < 3; ++i) {
        optimizer1.zero_grad();
        param1_->set_grad(ones({2, 3}, DType::Float32, device));
        param2_->set_grad(ones({4}, DType::Float32, device));
        optimizer1.step();
    }

    // Save and load state
    auto state = optimizer1.state_dict();
    optimizer2.load_state_dict(state);

    // State should be preserved
    EXPECT_GT(state.size(), 0);
}

TEST_P(OptimizersExtendedTest, RMSpropInvalidParameters) {
    EXPECT_THROW(RMSprop(params_, -0.01), std::invalid_argument);  // Negative lr
    EXPECT_THROW(RMSprop(params_, 0.01, -0.5), std::invalid_argument);  // Negative alpha
    EXPECT_THROW(RMSprop(params_, 0.01, 1.5), std::invalid_argument);  // Alpha > 1
    EXPECT_THROW(RMSprop(params_, 0.01, 0.99, -1e-8), std::invalid_argument);  // Negative eps
}

// Audit item D.5 — RMSprop state_dict must round-trip every hyperparameter
// (lr / alpha / eps / weight_decay / momentum / centered) in addition to the
// per-parameter buffers.  Previously dropped, so a checkpoint→restart of
// a non-default-hyperparam RMSprop silently reverted to constructor
// defaults.
TEST_P(OptimizersExtendedTest, RMSpropStateDictRoundTripsHyperparams) {
    // Use distinctly non-default hyperparams so any default-revert is visible.
    const double lr           = 0.0123;
    const double alpha        = 0.91;
    const double eps          = 1e-6;
    const double weight_decay = 0.0042;
    const double momentum     = 0.07;
    const bool   centered     = true;

    auto src = RMSprop(params_, lr, alpha, eps, weight_decay, momentum, centered);
    // Capture state without running steps so the buffers are zero-initialised
    // (this test focuses on hyperparam round-trip, not buffer reproducibility).
    auto state = src.state_dict();

    // Build a destination optimiser with ALL defaults distinctly different
    // from src so any missing field is detected by the comparison below.
    auto dst = RMSprop(params_, /*lr=*/0.5);
    dst.load_state_dict(state);

    EXPECT_DOUBLE_EQ(dst.get_lr(), lr) << "lr not restored from state_dict";

    // Round-trip through a second state_dict; values should match `state`.
    auto state2 = dst.state_dict();
    EXPECT_DOUBLE_EQ(state2["lr"].cpu().data<double>()[0],            lr);
    EXPECT_DOUBLE_EQ(state2["alpha"].cpu().data<double>()[0],         alpha);
    EXPECT_DOUBLE_EQ(state2["eps"].cpu().data<double>()[0],           eps);
    EXPECT_DOUBLE_EQ(state2["weight_decay"].cpu().data<double>()[0],  weight_decay);
    EXPECT_DOUBLE_EQ(state2["momentum"].cpu().data<double>()[0],      momentum);
    EXPECT_EQ      (state2["centered"].cpu().data<int64_t>()[0],      centered ? 1 : 0);
}

// D.5 (buffer-count guard): load_state_dict must reject a checkpoint whose
// parameter count does not match the current optimiser, mirroring Adam's
// guard (which catches mis-aligned per-parameter buffers).
TEST_P(OptimizersExtendedTest, RMSpropLoadStateDictRejectsParamCountMismatch) {
    auto src = RMSprop(params_, 0.01);
    auto state = src.state_dict();
    // Force a count mismatch by overwriting the stored num_params.
    state["num_params"].data<int64_t>()[0] = static_cast<int64_t>(params_.size() + 1);
    auto dst = RMSprop(params_, 0.01);
    EXPECT_THROW(dst.load_state_dict(state), std::runtime_error);
}

// ============================================================================
// Adagrad Tests
// ============================================================================

TEST_P(OptimizersExtendedTest, AdagradBasicStep) {
    auto optimizer = Adagrad(params_, 0.01);

    // Set gradients
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    auto param1_before = first(param1_->tensor());

    // Perform optimization step
    optimizer.step();

    auto param1_after = first(param1_->tensor());

    // Parameters should decrease
    EXPECT_LT(param1_after, param1_before);
}

TEST_P(OptimizersExtendedTest, AdagradAccumulation) {
    auto optimizer = Adagrad(params_, 0.1);

    // First step
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));
    optimizer.step();
    auto param1_step1 = first(param1_->tensor());
    float delta1 = 1.0f - param1_step1;

    // Second step
    optimizer.zero_grad();
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));
    optimizer.step();
    auto param1_step2 = first(param1_->tensor());
    float delta2 = param1_step1 - param1_step2;

    // Second step should be smaller due to accumulation
    EXPECT_LT(delta2, delta1);
}

TEST_P(OptimizersExtendedTest, AdagradLearningRateDecay) {
    auto optimizer = Adagrad(params_, 0.1, /*lr_decay=*/0.1);

    // get_lr() reports the BASE learning rate (matching set_lr's unit, per the
    // optimizer base-class contract); the lr_decay-adjusted value is exposed
    // separately via effective_lr(). The base lr never changes here.
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.1);
    EXPECT_FLOAT_EQ(optimizer.effective_lr(), 0.1);

    // After the first step the effective lr is lr / (1 + (step-1)*lr_decay)
    // = 0.1 / (1 + 0*0.1) = 0.1 — Adagrad applies no decay on the first step
    // (matches PyTorch; decay begins from the second step).
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));
    optimizer.step();
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.1);          // base lr unchanged
    EXPECT_FLOAT_EQ(optimizer.effective_lr(), 0.1);    // no decay on step 1

    // After the second step the effective lr decays: 0.1 / (1 + 1*0.1) ≈ 0.0909.
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));
    optimizer.step();
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.1);          // base lr still unchanged
    EXPECT_LT(optimizer.effective_lr(), 0.1);          // decay reflected here
}

TEST_P(OptimizersExtendedTest, AdagradInitialAccumulator) {
    auto optimizer = Adagrad(params_, 0.01, 0.0, 0.0, /*initial_accumulator=*/0.1);

    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    // Should work without errors
    EXPECT_NO_THROW(optimizer.step());
}

TEST_P(OptimizersExtendedTest, AdagradStateDictSaveLoad) {
    auto optimizer1 = Adagrad(params_, 0.01);
    auto optimizer2 = Adagrad(params_, 0.01);

    // Run optimizer1
    for (int i = 0; i < 3; ++i) {
        optimizer1.zero_grad();
        param1_->set_grad(ones({2, 3}, DType::Float32, device));
        param2_->set_grad(ones({4}, DType::Float32, device));
        optimizer1.step();
    }

    // Save and load
    auto state = optimizer1.state_dict();
    optimizer2.load_state_dict(state);

    EXPECT_GT(state.size(), 0);
    EXPECT_TRUE(state.find("step_count") != state.end());
}

TEST_P(OptimizersExtendedTest, AdagradInvalidParameters) {
    EXPECT_THROW(Adagrad(params_, -0.01), std::invalid_argument);  // Negative lr
    EXPECT_THROW(Adagrad(params_, 0.01, -0.1), std::invalid_argument);  // Negative lr_decay
    EXPECT_THROW(Adagrad(params_, 0.01, 0.0, -0.1), std::invalid_argument);  // Negative weight_decay
    EXPECT_THROW(Adagrad(params_, 0.01, 0.0, 0.0, -0.1), std::invalid_argument);  // Negative initial_acc
}

// ============================================================================
// Adadelta Tests
// ============================================================================

TEST_P(OptimizersExtendedTest, AdadeltaBasicStep) {
    auto optimizer = Adadelta(params_, 1.0, 0.9, 1e-6);

    // Set gradients
    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    auto param1_before = first(param1_->tensor());

    // Perform optimization step
    optimizer.step();

    auto param1_after = first(param1_->tensor());

    // Parameters should change
    EXPECT_NE(param1_after, param1_before);
}

TEST_P(OptimizersExtendedTest, AdadeltaNoLearningRateNeeded) {
    // Adadelta designed to work with lr=1.0 (no tuning needed)
    auto optimizer = Adadelta(params_);  // Uses default lr=1.0

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 1.0);

    param1_->set_grad(ones({2, 3}, DType::Float32, device));
    param2_->set_grad(ones({4}, DType::Float32, device));

    // Should work with default parameters
    EXPECT_NO_THROW(optimizer.step());
}

TEST_P(OptimizersExtendedTest, AdadeltaAdaptiveRate) {
    auto optimizer = Adadelta(params_, 1.0, 0.9, 1e-6);

    // Multiple steps should adapt
    std::vector<float> deltas;
    float prev_val = first(param1_->tensor());

    for (int i = 0; i < 5; ++i) {
        optimizer.zero_grad();
        param1_->set_grad(ones({2, 3}, DType::Float32, device));
        param2_->set_grad(ones({4}, DType::Float32, device));
        optimizer.step();

        float curr_val = first(param1_->tensor());
        deltas.push_back(std::abs(curr_val - prev_val));
        prev_val = curr_val;
    }

    // Step sizes should be non-zero
    for (auto delta : deltas) {
        EXPECT_GT(delta, 0.0f);
    }
}

TEST_P(OptimizersExtendedTest, AdadeltaStateDictSaveLoad) {
    auto optimizer1 = Adadelta(params_, 1.0, 0.9, 1e-6);
    auto optimizer2 = Adadelta(params_, 1.0, 0.9, 1e-6);

    // Run optimizer1
    for (int i = 0; i < 3; ++i) {
        optimizer1.zero_grad();
        param1_->set_grad(ones({2, 3}, DType::Float32, device));
        param2_->set_grad(ones({4}, DType::Float32, device));
        optimizer1.step();
    }

    // Save and load
    auto state = optimizer1.state_dict();
    optimizer2.load_state_dict(state);

    EXPECT_GT(state.size(), 0);
}

TEST_P(OptimizersExtendedTest, AdadeltaInvalidParameters) {
    EXPECT_THROW(Adadelta(params_, -1.0), std::invalid_argument);  // Negative lr
    EXPECT_THROW(Adadelta(params_, 1.0, -0.5), std::invalid_argument);  // Negative rho
    EXPECT_THROW(Adadelta(params_, 1.0, 1.5), std::invalid_argument);  // rho > 1
    EXPECT_THROW(Adadelta(params_, 1.0, 0.9, -1e-6), std::invalid_argument);  // Negative eps
}

// ============================================================================
// Convergence Tests
// ============================================================================

TEST_P(OptimizersExtendedTest, RMSpropConvergence) {
    // Simple quadratic: f(x) = (x - 3)^2, optimal at x=3
    // Start far from optimum (x = 10) on the target device.
    auto param = std::make_shared<Variable>(
        full({1}, 10.0, DType::Float32, device), true);

    auto optimizer = RMSprop(std::vector<std::shared_ptr<Variable>>{param}, 0.1);

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();

        // Gradient: 2(x - 3)
        float x = first(param->tensor());
        auto grad_tensor = zeros({1}, DType::Float32);
        grad_tensor.data<float>()[0] = 2.0f * (x - 3.0f);

        param->set_grad(grad_tensor.to(device));
        optimizer.step();
    }

    // Should converge near 3
    EXPECT_NEAR(first(param->tensor()), 3.0f, 0.1f);
}

TEST_P(OptimizersExtendedTest, AdagradConvergence) {
    auto param = std::make_shared<Variable>(
        full({1}, 10.0, DType::Float32, device), true);

    auto optimizer = Adagrad(std::vector<std::shared_ptr<Variable>>{param}, 1.0);

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();

        float x = first(param->tensor());
        auto grad_tensor = zeros({1}, DType::Float32);
        grad_tensor.data<float>()[0] = 2.0f * (x - 3.0f);

        param->set_grad(grad_tensor.to(device));
        optimizer.step();
    }

    // Should converge near 3
    EXPECT_NEAR(first(param->tensor()), 3.0f, 0.5f);
}

TEST_P(OptimizersExtendedTest, AdadeltaConvergence) {
    auto param = std::make_shared<Variable>(
        full({1}, 10.0, DType::Float32, device), true);

    // Use larger eps for better initial convergence (common in practice)
    auto optimizer = Adadelta(std::vector<std::shared_ptr<Variable>>{param}, 1.0, 0.95, 1e-4);

    for (int i = 0; i < 500; ++i) {
        optimizer.zero_grad();

        float x = first(param->tensor());
        auto grad_tensor = zeros({1}, DType::Float32);
        grad_tensor.data<float>()[0] = 2.0f * (x - 3.0f);

        param->set_grad(grad_tensor.to(device));
        optimizer.step();
    }

    // Adadelta may converge slower but should get close
    EXPECT_NEAR(first(param->tensor()), 3.0f, 1.0f);
}

INSTANTIATE_BACKEND_TESTS(OptimizersExtendedTest);
