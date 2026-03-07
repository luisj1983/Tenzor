#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <cmath>
#include <numbers>

using namespace tenzor;
using namespace tenzor::optim;

// Global test environment
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

//==============================================================================
// StepLR Tests with SGD
//==============================================================================

TEST(SchedulerTest, StepLR_SGD_BasicStep) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = StepLR(optimizer, 2, 0.1);  // step_size=2, gamma=0.1

    // Initial LR should be 0.1
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.1);
    EXPECT_EQ(scheduler.get_epoch(), 0);

    // After 1 step, LR should stay the same (not a multiple of step_size)
    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.1);
    EXPECT_EQ(scheduler.get_epoch(), 1);

    // After 2 steps, LR should decay to 0.1 * 0.1 = 0.01
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-9);
    EXPECT_NEAR(optimizer.get_lr(), 0.01, 1e-9);
    EXPECT_EQ(scheduler.get_epoch(), 2);
}

TEST(SchedulerTest, StepLR_SGD_MultipleDecays) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = StepLR(optimizer, 3, 0.5);  // step_size=3, gamma=0.5

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);

    // Epoch 1: LR = 1.0
    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);

    // Epoch 2: LR = 1.0
    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);

    // Epoch 3: LR = 1.0 * 0.5 = 0.5
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-9);

    // Epoch 4: LR = 0.5
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-9);

    // Epoch 5: LR = 0.5
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-9);

    // Epoch 6: LR = 1.0 * 0.5^2 = 0.25
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.25, 1e-9);
}

TEST(SchedulerTest, StepLR_SGD_EdgeCaseStepSize1) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = StepLR(optimizer, 1, 0.5);  // Decay every epoch

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-9);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.25, 1e-9);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.125, 1e-9);
}

TEST(SchedulerTest, StepLR_SGD_VerySmallGamma) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = StepLR(optimizer, 1, 0.01);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-9);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0001, 1e-9);
}

//==============================================================================
// StepLR Tests with Adam
//==============================================================================

TEST(SchedulerTest, StepLR_Adam_BasicStep) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    auto scheduler = StepLR(optimizer, 2, 0.1);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.001);

    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0001, 1e-9);
    EXPECT_NEAR(optimizer.get_lr(), 0.0001, 1e-9);
}

TEST(SchedulerTest, StepLR_AdamW_BasicStep) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.001);

    auto scheduler = StepLR(optimizer, 2, 0.1);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);

    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0001, 1e-9);
}

//==============================================================================
// ExponentialLR Tests with SGD
//==============================================================================

TEST(SchedulerTest, ExponentialLR_SGD_BasicDecay) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = ExponentialLR(optimizer, 0.9);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);
    EXPECT_EQ(scheduler.get_epoch(), 0);

    // Epoch 1: LR = 1.0 * 0.9^1 = 0.9
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.9, 1e-9);
    EXPECT_NEAR(optimizer.get_lr(), 0.9, 1e-9);
    EXPECT_EQ(scheduler.get_epoch(), 1);

    // Epoch 2: LR = 1.0 * 0.9^2 = 0.81
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.81, 1e-9);
    EXPECT_EQ(scheduler.get_epoch(), 2);

    // Epoch 3: LR = 1.0 * 0.9^3 = 0.729
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.729, 1e-9);
}

TEST(SchedulerTest, ExponentialLR_SGD_MultipleSteps) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = ExponentialLR(optimizer, 0.95);

    double expected_lr = 0.1;
    for (int i = 0; i < 10; i++) {
        scheduler.step();
        expected_lr *= 0.95;
        EXPECT_NEAR(scheduler.get_last_lr(), expected_lr, 1e-9);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-9);
    }
}

TEST(SchedulerTest, ExponentialLR_SGD_SmallGamma) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = ExponentialLR(optimizer, 0.5);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-9);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.25, 1e-9);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.125, 1e-9);
}

//==============================================================================
// ExponentialLR Tests with Adam
//==============================================================================

TEST(SchedulerTest, ExponentialLR_Adam_BasicDecay) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    auto scheduler = ExponentialLR(optimizer, 0.9);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0009, 1e-9);
    EXPECT_NEAR(optimizer.get_lr(), 0.0009, 1e-9);
}

TEST(SchedulerTest, ExponentialLR_AdamW_BasicDecay) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.001);

    auto scheduler = ExponentialLR(optimizer, 0.9);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0009, 1e-9);
}

//==============================================================================
// CosineAnnealingLR Tests with SGD
//==============================================================================

TEST(SchedulerTest, CosineAnnealingLR_SGD_BasicCycle) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingLR(optimizer, 10, 0.0);  // T_max=10, eta_min=0

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);
    EXPECT_EQ(scheduler.get_epoch(), 0);

    // Epoch 1: cos(pi * 1 / 10) ≈ 0.951
    // lr = 0 + (1 - 0) * (1 + 0.951) / 2 ≈ 0.9755
    scheduler.step();
    double expected_lr_1 = 0.0 + (1.0 - 0.0) * (1.0 + std::cos(std::numbers::pi * 1.0 / 10.0)) / 2.0;
    EXPECT_NEAR(scheduler.get_last_lr(), expected_lr_1, 1e-6);
    EXPECT_NEAR(optimizer.get_lr(), expected_lr_1, 1e-6);

    // Epoch 5: cos(pi * 5 / 10) = cos(pi / 2) = 0
    // lr = 0 + (1 - 0) * (1 + 0) / 2 = 0.5
    for (int i = 1; i < 5; i++) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-6);

    // Epoch 10: cos(pi * 10 / 10) = cos(pi) = -1
    // lr = 0 + (1 - 0) * (1 + (-1)) / 2 = 0
    for (int i = 5; i < 10; i++) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0, 1e-6);
}

TEST(SchedulerTest, CosineAnnealingLR_SGD_WithEtaMin) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingLR(optimizer, 10, 0.1);  // eta_min=0.1

    // Epoch 10: cos(pi) = -1
    // lr = 0.1 + (1 - 0.1) * (1 + (-1)) / 2 = 0.1
    for (int i = 0; i < 10; i++) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);
}

TEST(SchedulerTest, CosineAnnealingLR_SGD_Symmetry) {
    // Test cosine annealing with longer T_max to verify smooth monotonic decrease
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingLR(optimizer, 20, 0.0);

    std::vector<double> lrs;
    for (int i = 0; i < 20; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }

    // Check that learning rate decreases monotonically throughout
    for (size_t i = 1; i < 20; i++) {
        EXPECT_LT(lrs[i], lrs[i-1]) << "LR should decrease monotonically at step " << i;
    }

    // Verify specific points match cosine formula
    // At T_max/2 (epoch 10): cos(pi/2) = 0, lr should be 0.5
    EXPECT_NEAR(lrs[9], 0.5, 0.01) << "LR at T_max/2 should be approximately 0.5";

    // At T_max (epoch 20): cos(pi) = -1, lr should reach eta_min = 0
    EXPECT_NEAR(lrs[19], 0.0, 0.01) << "LR at T_max should reach eta_min";

    // Verify first LR (epoch 1): cos(pi/20) ≈ 0.9877, lr ≈ 0.9938
    double expected_first_lr = 0.0 + (1.0 - 0.0) * (1.0 + std::cos(std::numbers::pi / 20.0)) / 2.0;
    EXPECT_NEAR(lrs[0], expected_first_lr, 0.001) << "First LR after step should match formula";
}

TEST(SchedulerTest, CosineAnnealingLR_SGD_SmallTMax) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingLR(optimizer, 2, 0.0);

    // Epoch 1: cos(pi/2) = 0, lr = 0.5
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 1e-6);

    // Epoch 2: cos(pi) = -1, lr = 0.0
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0, 1e-6);
}

//==============================================================================
// CosineAnnealingLR Tests with Adam
//==============================================================================

TEST(SchedulerTest, CosineAnnealingLR_Adam_BasicCycle) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    auto scheduler = CosineAnnealingLR(optimizer, 10, 0.0);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);

    for (int i = 0; i < 5; i++) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0005, 1e-7);
    EXPECT_NEAR(optimizer.get_lr(), 0.0005, 1e-7);
}

TEST(SchedulerTest, CosineAnnealingLR_AdamW_BasicCycle) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.001);

    auto scheduler = CosineAnnealingLR(optimizer, 10, 0.0);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);

    for (int i = 0; i < 5; i++) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0005, 1e-7);
}

//==============================================================================
// Integration Tests with Optimizer Steps
//==============================================================================

TEST(SchedulerTest, StepLR_SGD_Integration) {
    // Test that scheduler properly updates optimizer during training
    auto param = std::make_shared<Variable>(full({2, 2}, 10.0f, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = StepLR(optimizer, 2, 0.5);

    // Training loop simulation
    for (int epoch = 0; epoch < 6; epoch++) {
        param->set_grad(ones({2, 2}, DType::Float32));
        optimizer.step();
        scheduler.step();

        // Check that optimizer's LR matches scheduler's LR
        EXPECT_NEAR(optimizer.get_lr(), scheduler.get_last_lr(), 1e-9);
    }

    // After 6 epochs with step_size=2, should have decayed 3 times
    // lr = 1.0 * 0.5^3 = 0.125
    EXPECT_NEAR(scheduler.get_last_lr(), 0.125, 1e-9);
}

TEST(SchedulerTest, ExponentialLR_Adam_Integration) {
    auto param = std::make_shared<Variable>(full({2, 2}, 10.0f, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);
    auto scheduler = ExponentialLR(optimizer, 0.9);

    for (int epoch = 0; epoch < 10; epoch++) {
        param->set_grad(ones({2, 2}, DType::Float32));
        optimizer.step();
        scheduler.step();

        EXPECT_NEAR(optimizer.get_lr(), scheduler.get_last_lr(), 1e-9);
    }

    double expected_lr = 0.01 * std::pow(0.9, 10);
    EXPECT_NEAR(scheduler.get_last_lr(), expected_lr, 1e-9);
}

TEST(SchedulerTest, CosineAnnealingLR_SGD_Integration) {
    auto param = std::make_shared<Variable>(full({2, 2}, 10.0f, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = CosineAnnealingLR(optimizer, 10, 0.1);

    for (int epoch = 0; epoch < 10; epoch++) {
        param->set_grad(ones({2, 2}, DType::Float32));
        optimizer.step();
        scheduler.step();

        EXPECT_NEAR(optimizer.get_lr(), scheduler.get_last_lr(), 1e-6);
    }

    // At T_max, should be at eta_min
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);
}

//==============================================================================
// Edge Cases and Validation Tests
//==============================================================================

TEST(SchedulerTest, StepLR_LargeEpoch) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = StepLR(optimizer, 10, 0.5);

    // Run for 100 epochs
    for (int i = 0; i < 100; i++) {
        scheduler.step();
    }

    // After 100 epochs with step_size=10, decayed 10 times
    double expected_lr = 1.0 * std::pow(0.5, 10);
    EXPECT_NEAR(scheduler.get_last_lr(), expected_lr, 1e-9);
}

TEST(SchedulerTest, ExponentialLR_VerySmallGamma) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = ExponentialLR(optimizer, 0.001);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.001, 1e-9);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.000001, 1e-12);
}

TEST(SchedulerTest, CosineAnnealingLR_AfterTMax) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);
    auto scheduler = CosineAnnealingLR(optimizer, 10, 0.0);

    // Go beyond T_max
    for (int i = 0; i < 15; i++) {
        scheduler.step();
    }

    // Cosine continues beyond T_max (wraps around)
    // At epoch 15: cos(pi * 15 / 10) = cos(1.5*pi) = 0
    double expected_lr = 0.0 + (1.0 - 0.0) * (1.0 + std::cos(std::numbers::pi * 15.0 / 10.0)) / 2.0;
    EXPECT_NEAR(scheduler.get_last_lr(), expected_lr, 1e-6);
}

TEST(SchedulerTest, GetLRAlias) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.5);
    auto scheduler = StepLR(optimizer, 1, 0.5);

    // get_lr() should be alias for get_last_lr()
    EXPECT_DOUBLE_EQ(scheduler.get_lr(), scheduler.get_last_lr());
    EXPECT_DOUBLE_EQ(scheduler.get_lr(), 0.5);

    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_lr(), scheduler.get_last_lr());
    EXPECT_NEAR(scheduler.get_lr(), 0.25, 1e-9);
}
