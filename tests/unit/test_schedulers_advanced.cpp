/**
 * @file test_schedulers_advanced.cpp
 * @brief Comprehensive tests for advanced learning rate schedulers
 */

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
// ReduceLROnPlateau Tests
//==============================================================================

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_BasicMinMode) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 2);  // patience=2

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);
    EXPECT_FALSE(scheduler.in_cooldown());

    // Improving metric (decreasing loss)
    scheduler.step(1.0);
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);  // No reduction yet
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);

    scheduler.step(0.9);  // Improvement
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);

    // No improvement within the patience window. ReduceLROnPlateau matches
    // PyTorch: LR is reduced only when num_bad_epochs STRICTLY EXCEEDS
    // patience (patience=2 -> the 3rd consecutive bad epoch triggers the
    // reduction; the 2nd bad epoch is still within the patience window).
    scheduler.step(0.91);  // No improvement, bad_epochs=1
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 1);

    scheduler.step(0.92);  // No improvement, bad_epochs=2 (still <= patience)
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 2);

    scheduler.step(0.93);  // bad_epochs=3 > patience=2, triggers reduction
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-9);  // Reduced by factor 0.1
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);  // Reset counter
}

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_MaxMode) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);

    auto scheduler = ReduceLROnPlateau(optimizer, "max", 0.5, 3);  // mode=max, patience=3

    // Improving metric (increasing accuracy)
    scheduler.step(0.7);
    scheduler.step(0.8);  // Improvement
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.01);

    // No improvement within the patience window. PyTorch ReduceLROnPlateau
    // reduces only when num_bad_epochs STRICTLY EXCEEDS patience (patience=3
    // -> the 4th consecutive non-improving epoch triggers the reduction).
    scheduler.step(0.79);  // No improvement, bad_epochs=1
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 1);
    scheduler.step(0.78);  // No improvement, bad_epochs=2
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 2);
    scheduler.step(0.77);  // No improvement, bad_epochs=3 (still <= patience)
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-9);  // Not reduced yet
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 3);
    scheduler.step(0.76);  // bad_epochs=4 > patience=3, triggers reduction
    EXPECT_NEAR(scheduler.get_last_lr(), 0.005, 1e-9);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);  // Reset after reduction
}

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_Cooldown) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 2, 1e-4, "rel", 2);  // cooldown=2

    // Trigger reduction: patience=2 means the 3rd consecutive bad epoch
    // reduces LR (PyTorch strict-exceed semantics: num_bad > patience).
    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);  // bad_epochs=3 > patience=2, reduce LR
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-9);  // 1.0 * 0.1
    EXPECT_TRUE(scheduler.in_cooldown());

    // During cooldown, bad epochs shouldn't accumulate
    scheduler.step(1.0);
    EXPECT_TRUE(scheduler.in_cooldown());
    scheduler.step(1.0);
    EXPECT_FALSE(scheduler.in_cooldown());  // Cooldown expired
}

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_MinLR) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.1);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 1, 1e-4, "rel", 0, 0.001);  // min_lr=0.001

    // Reduce multiple times
    for (int i = 0; i < 10; i++) {
        scheduler.step(1.0);  // No improvement
        scheduler.step(1.0);  // Trigger reduction
    }

    // Should not go below min_lr
    EXPECT_GE(scheduler.get_last_lr(), 0.001);
}

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_RelativeThreshold) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 2, 0.1, "rel");  // 10% relative threshold

    scheduler.step(1.0);
    scheduler.step(0.95);  // 5% improvement, below 10% threshold - counts as no improvement
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 1);

    scheduler.step(0.85);  // 10.5% improvement from 0.95, significant improvement
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);
}

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_AbsoluteThreshold) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 2, 0.05, "abs");  // absolute threshold

    scheduler.step(1.0);
    scheduler.step(0.98);  // 0.02 improvement, below 0.05 threshold
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 1);

    scheduler.step(0.92);  // 0.06 improvement from 0.98, significant
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);
}

//==============================================================================
// CyclicLR Tests
//==============================================================================

TEST(AdvancedSchedulerTest, CyclicLR_Triangular) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = CyclicLR(optimizer, 0.001, 0.006, 4, -1, "triangular");

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);
    EXPECT_EQ(scheduler.get_iteration(), 0);

    // First half: increasing from base_lr to max_lr
    scheduler.step();
    EXPECT_GT(scheduler.get_last_lr(), 0.001);
    EXPECT_LT(scheduler.get_last_lr(), 0.006);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    // At step 4, should be at max_lr
    EXPECT_NEAR(scheduler.get_last_lr(), 0.006, 1e-9);

    // Second half: decreasing from max_lr to base_lr
    scheduler.step();
    EXPECT_LT(scheduler.get_last_lr(), 0.006);

    for (int i = 0; i < 3; i++) {
        scheduler.step();
    }
    // At step 8, back to base_lr
    EXPECT_NEAR(scheduler.get_last_lr(), 0.001, 1e-6);
}

TEST(AdvancedSchedulerTest, CyclicLR_Triangular2) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    auto scheduler = CyclicLR(optimizer, 0.001, 0.01, 5, -1, "triangular2");

    // First cycle: full amplitude
    for (int i = 0; i < 10; i++) {
        scheduler.step();
    }
    EXPECT_EQ(scheduler.get_cycle(), 1);

    // Second cycle: amplitude should be halved
    double max_lr_cycle2 = 0.0;
    for (int i = 0; i < 10; i++) {
        scheduler.step();
        max_lr_cycle2 = std::max(max_lr_cycle2, scheduler.get_last_lr());
    }
    // Max LR in second cycle should be less than first cycle
    EXPECT_LT(max_lr_cycle2, 0.01);
}

TEST(AdvancedSchedulerTest, CyclicLR_ExpRange) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.001);

    auto scheduler = CyclicLR(optimizer, 0.001, 0.006, 4, -1, "exp_range", 0.99);

    std::vector<double> lrs;
    lrs.push_back(scheduler.get_last_lr());  // Initial LR at base_lr
    for (int i = 0; i < 20; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }

    // With exp_range, the maximum LR in each cycle should decrease
    // Compare max LR in first cycle vs last cycle
    double max_lr_first_cycle = *std::max_element(lrs.begin(), lrs.begin() + 8);
    double max_lr_last_cycle = *std::max_element(lrs.end() - 8, lrs.end());
    EXPECT_LT(max_lr_last_cycle, max_lr_first_cycle);
}

TEST(AdvancedSchedulerTest, CyclicLR_AsymmetricCycle) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = CyclicLR(optimizer, 0.001, 0.006, 2, 6, "triangular");  // step_up=2, step_down=6

    // Cycle size should be 2 + 6 = 8
    for (int i = 0; i < 16; i++) {
        scheduler.step();
    }
    EXPECT_EQ(scheduler.get_cycle(), 2);  // 16 / 8 = 2 cycles
}

//==============================================================================
// OneCycleLR Tests
//==============================================================================

TEST(AdvancedSchedulerTest, OneCycleLR_BasicCycle) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    int total_steps = 100;
    auto scheduler = OneCycleLR(optimizer, 0.1, total_steps, -1, -1, 0.3, "cos", 25.0, 10000.0);

    // Initial LR should be max_lr / div_factor = 0.1 / 25 = 0.004.
    // Per LL.6, OneCycleLR's internal step_count_ starts at -1; the first
    // step() call advances it to 0 and computes the first warmup LR.
    EXPECT_NEAR(scheduler.get_last_lr(), 0.004, 1e-6);
    EXPECT_EQ(scheduler.get_step(), -1);

    // Phase 1: Warmup (30% of steps).
    // Per OneCycleLR semantics, max_lr is hit exactly at step == warmup_steps
    // (the first compute_lr call entering the annealing branch with pct=0).
    // So we step 31 times and the 31st recorded LR (lrs[30]) is the peak.
    std::vector<double> lrs;
    for (int i = 0; i < 31; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }
    // LR should increase during warmup
    EXPECT_GT(lrs[30], lrs[0]);
    EXPECT_NEAR(lrs[30], 0.1, 1e-6);  // Peak at warmup boundary == max_lr

    // Phase 2: Annealing (70% of steps)
    double lr_at_30 = scheduler.get_last_lr();
    for (int i = 31; i < 100; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }
    // LR should decrease during annealing
    EXPECT_LT(scheduler.get_last_lr(), lr_at_30);
}

TEST(AdvancedSchedulerTest, OneCycleLR_LinearAnnealing) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);

    auto scheduler = OneCycleLR(optimizer, 0.01, 50, -1, -1, 0.3, "linear");

    std::vector<double> warmup_lrs;
    for (int i = 0; i < 15; i++) {
        scheduler.step();
        warmup_lrs.push_back(scheduler.get_last_lr());
    }

    // With linear strategy, warmup should be approximately linear
    double diff1 = warmup_lrs[1] - warmup_lrs[0];
    double diff_mid = warmup_lrs[8] - warmup_lrs[7];
    // Differences should be similar (within tolerance for floating point)
    EXPECT_NEAR(diff1, diff_mid, diff1 * 0.5);  // Allow 50% variation
}

TEST(AdvancedSchedulerTest, OneCycleLR_CustomDivFactors) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.001);

    auto scheduler = OneCycleLR(optimizer, 1.0, 100, -1, -1, 0.5, "cos", 10.0, 100.0);

    // Initial LR = max_lr / div_factor = 1.0 / 10.0 = 0.1
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);

    // OneCycleLR's compute_lr() short-circuits to the final value only once
    // step_count_ >= total_steps_; step() reads step_count_ then increments,
    // so we need total_steps_+1 calls for last_lr_ to reflect the final LR.
    for (int i = 0; i < 101; i++) {
        scheduler.step();
    }

    // Final LR = max_lr / final_div_factor = 1.0 / 100.0 = 0.01
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-6);
}

TEST(AdvancedSchedulerTest, OneCycleLR_PctStartVariation) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // Test with pct_start = 0.5 (equal warmup and annealing)
    auto scheduler = OneCycleLR(optimizer, 0.1, 100, -1, -1, 0.5);

    std::vector<double> lrs;
    for (int i = 0; i < 100; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }

    // LR should peak around step 50
    size_t max_idx = std::distance(lrs.begin(), std::max_element(lrs.begin(), lrs.end()));
    EXPECT_NEAR(max_idx, 50, 5);  // Within 5 steps of middle
}

//==============================================================================
// CosineAnnealingWarmRestarts Tests
//==============================================================================

TEST(AdvancedSchedulerTest, CosineAnnealingWarmRestarts_BasicRestart) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1, 0.0);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);
    EXPECT_EQ(scheduler.get_T_cur(), 0);
    EXPECT_EQ(scheduler.get_T_i(), 10);

    // First cycle: step 9 times so T_cur visits 1..9. PyTorch SGDR uses the
    // full period T_i in the denominator: lr = eta_min + (base-eta_min) *
    // (1 + cos(pi * T_cur / T_i)) / 2. So at T_cur == 9 (T_i == 10) the LR is
    // the cycle minimum but NOT exactly eta_min (the exact minimum would occur
    // at T_cur == T_i == 10, which is the restart point). The 10th call
    // triggers the restart roll-over.
    std::vector<double> lrs_cycle1;
    for (int i = 0; i < 9; i++) {
        scheduler.step();
        lrs_cycle1.push_back(scheduler.get_last_lr());
    }

    // LR should decrease monotonically through the cycle toward eta_min.
    EXPECT_GT(lrs_cycle1[0], lrs_cycle1[8]);
    // Exact PyTorch value at T_cur=9, T_i=10: (1 + cos(0.9*pi)) / 2.
    const double expected_min = (1.0 + std::cos(std::numbers::pi * 9.0 / 10.0)) / 2.0;
    EXPECT_NEAR(lrs_cycle1[8], expected_min, 1e-6);

    // 10th call triggers the restart: T_cur rolls 10 -> 0, LR recomputes
    // at T_cur == 0 (which is base_lr == 1.0).
    scheduler.step();

    // After restart, T_cur should be 0.
    EXPECT_EQ(scheduler.get_T_cur(), 0);

    // The restart step itself put us back at base_lr.
    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-6);
}

TEST(AdvancedSchedulerTest, CosineAnnealingWarmRestarts_TMultiplier) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);

    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 5, 2, 0.0);  // T_mult=2

    EXPECT_EQ(scheduler.get_T_i(), 5);

    // First cycle: 5 steps
    for (int i = 0; i < 5; i++) {
        scheduler.step();
    }
    EXPECT_EQ(scheduler.get_T_cur(), 0);
    EXPECT_EQ(scheduler.get_T_i(), 10);  // Period doubled

    // Second cycle: 10 steps
    for (int i = 0; i < 10; i++) {
        scheduler.step();
    }
    EXPECT_EQ(scheduler.get_T_cur(), 0);
    EXPECT_EQ(scheduler.get_T_i(), 20);  // Period doubled again
}

TEST(AdvancedSchedulerTest, CosineAnnealingWarmRestarts_EtaMin) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 1.0);

    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1, 0.1);  // eta_min=0.1

    // 9 steps to land at T_cur == 9 (cycle minimum). With the full-period
    // PyTorch denominator, the LR here is the lowest in-cycle value but not
    // exactly eta_min (that is approached only as T_cur -> T_i).
    for (int i = 0; i < 9; i++) {
        scheduler.step();
    }

    // LR must stay at or above eta_min, and equal the exact PyTorch SGDR value
    // at T_cur=9, T_i=10, eta_min=0.1, base_lr=1.0.
    const double expected = 0.1 + (1.0 - 0.1) * (1.0 + std::cos(std::numbers::pi * 9.0 / 10.0)) / 2.0;
    EXPECT_GE(scheduler.get_last_lr(), 0.1);
    EXPECT_NEAR(scheduler.get_last_lr(), expected, 1e-6);
}

TEST(AdvancedSchedulerTest, CosineAnnealingWarmRestarts_MultipleRestarts) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 5, 1, 0.0);

    std::vector<double> all_lrs;
    for (int i = 0; i < 20; i++) {
        scheduler.step();
        all_lrs.push_back(scheduler.get_last_lr());
    }

    // Check that LR restarts (jumps back up) at cycles.
    // With T_0 == 5 and the LL.6 increment-first semantics, each restart
    // lands on the 5th step of its period: T_cur counts 1..5 then resets,
    // and update_lr at the post-reset T_cur == 0 gives base_lr. So the
    // observed peaks sit at indices 4, 9, 14 (one before what naive
    // 1-indexed counting would suggest).
    EXPECT_GT(all_lrs[4], all_lrs[3]);    // Restart at step 5 (index 4)
    EXPECT_GT(all_lrs[9], all_lrs[8]);    // Restart at step 10 (index 9)
    EXPECT_GT(all_lrs[14], all_lrs[13]);  // Restart at step 15 (index 14)
}

//==============================================================================
// Integration Tests
//==============================================================================

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_Training_Simulation) {
    auto param = std::make_shared<Variable>(full({2, 2}, 10.0f, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);
    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.5, 3);

    // Simulate training with validation loss. With patience=3 and PyTorch's
    // strict-exceed semantics, the LR reduces on the 4th consecutive
    // non-improving epoch (index 7 here; epochs 4-7 hold the loss flat).
    std::vector<double> val_losses = {1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84, 0.84};

    double last_lr = 0.1;
    for (size_t epoch = 0; epoch < val_losses.size(); epoch++) {
        param->set_grad(ones({2, 2}, DType::Float32));
        optimizer.step();
        scheduler.step(val_losses[epoch]);

        if (epoch < 7) {
            EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), last_lr);
        } else {
            // epoch 7: 4th consecutive non-improving epoch -> reduce
            last_lr = 0.05;
            EXPECT_NEAR(scheduler.get_last_lr(), last_lr, 1e-9);
        }
    }
}

TEST(AdvancedSchedulerTest, CyclicLR_Training_Simulation) {
    auto param = std::make_shared<Variable>(full({2, 2}, 10.0f, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);
    auto scheduler = CyclicLR(optimizer, 0.001, 0.01, 10);

    // Simulate 50 batch updates
    for (int batch = 0; batch < 50; batch++) {
        param->set_grad(ones({2, 2}, DType::Float32));
        optimizer.step();
        scheduler.step();

        // Verify LR is within bounds (use NEAR for floating point tolerance)
        EXPECT_GE(scheduler.get_last_lr(), 0.001 - 1e-9);
        EXPECT_LE(scheduler.get_last_lr(), 0.01 + 1e-9);
    }

    // Should have completed 2+ cycles (50 / 20 = 2.5 cycles)
    EXPECT_GE(scheduler.get_cycle(), 2);
}

TEST(AdvancedSchedulerTest, OneCycleLR_Training_Simulation) {
    auto param = std::make_shared<Variable>(full({2, 2}, 10.0f, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    int num_epochs = 10;
    int batches_per_epoch = 10;
    int total_steps = num_epochs * batches_per_epoch;

    auto scheduler = OneCycleLR(optimizer, 0.1, total_steps);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            param->set_grad(ones({2, 2}, DType::Float32));
            optimizer.step();
            scheduler.step();
        }
    }

    // At end of training, LR should be very low
    EXPECT_LT(scheduler.get_last_lr(), 0.001);
}

TEST(AdvancedSchedulerTest, CosineAnnealingWarmRestarts_Training_Simulation) {
    auto param = std::make_shared<Variable>(full({2, 2}, 10.0f, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.01);
    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 5, 2);

    int total_epochs = 30;
    for (int epoch = 0; epoch < total_epochs; epoch++) {
        param->set_grad(ones({2, 2}, DType::Float32));
        optimizer.step();
        scheduler.step();

        // LR should always be within valid range
        EXPECT_GE(scheduler.get_last_lr(), 0.0);
        EXPECT_LE(scheduler.get_last_lr(), 0.01);
    }
}

//==============================================================================
// Edge Cases and Error Handling
//==============================================================================

TEST(AdvancedSchedulerTest, ReduceLROnPlateau_ErrorOnStepWithoutMetric) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);
    auto scheduler = ReduceLROnPlateau(optimizer, "min");

    // Calling step() without metric should throw
    EXPECT_THROW(scheduler.step(), std::runtime_error);
}

TEST(AdvancedSchedulerTest, CyclicLR_ZeroStepSize) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // step_size_up must be positive
    // This would cause division issues, should handle gracefully
    auto scheduler = CyclicLR(optimizer, 0.001, 0.01, 1);
    EXPECT_NO_THROW(scheduler.step());
}

TEST(AdvancedSchedulerTest, OneCycleLR_ZeroTotalSteps) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);

    // Should throw on invalid total_steps
    EXPECT_THROW({
        auto scheduler = OneCycleLR(optimizer, 0.1, 0);
    }, std::invalid_argument);
}

TEST(AdvancedSchedulerTest, CosineAnnealingWarmRestarts_InvalidT0) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // T_0 must be positive
    EXPECT_THROW({
        auto scheduler = CosineAnnealingWarmRestarts(optimizer, 0);
    }, std::invalid_argument);
}

TEST(AdvancedSchedulerTest, CosineAnnealingWarmRestarts_InvalidTMult) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // T_mult must be >= 1
    EXPECT_THROW({
        auto scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 0);
    }, std::invalid_argument);
}
