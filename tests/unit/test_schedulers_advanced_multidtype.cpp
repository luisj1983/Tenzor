/**
 * @file test_schedulers_advanced_multidtype.cpp
 * @brief Multi-dtype tests for advanced learning rate schedulers
 *
 * Advanced LR schedulers tested across multiple dtypes:
 * - Float32: Standard training precision
 * - Float64: High precision for numerical stability in learning rate calculations
 *
 * These specialized schedulers are used in:
 * - Adaptive training (ReduceLROnPlateau)
 * - Cyclic learning rates (CyclicLR)
 * - Super-convergence (OneCycleLR)
 * - Restart strategies (CosineAnnealingWarmRestarts)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include <tenzor/nn/optim/scheduler.hpp>
#include <cmath>
#include <numbers>

using namespace tenzor;
using namespace tenzor::optim;

// ============================================================================
// DType Parameterization Structure
// ============================================================================

struct SchedulerDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;
    double rtol;  // Relative tolerance
    double atol;  // Absolute tolerance

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const SchedulerDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

// ============================================================================
// Test Fixture
// ============================================================================

class SchedulerAdvancedMultiDTypeTest : public ::testing::TestWithParam<SchedulerDTypeParam> {
protected:
    Device device;
    DType dtype;
    double rtol;
    double atol;

    void SetUp() override {
        auto param = GetParam();
        dtype = param.dtype;
        rtol = param.rtol;
        atol = param.atol;

        // Retrofits this hand-rolled fixture with the same TENZOR_SKIP_BACKENDS/
        // TENZOR_REQUIRE_MULTI_BACKEND handling BackendTest/MultiBackendDTypeTest
        // get for free, so a silently-broken GPU driver escalates to a hard
        // failure under TENZOR_REQUIRE_MULTI_BACKEND=1 instead of a quiet skip.
        HONOR_BACKEND_ENV_VARS(param.backend_name);

        // HONOR_BACKEND_ENV_VARS above already returned (skipped or FAILed)
        // if this backend is unavailable, so the per-branch availability
        // re-checks that used to live here were dead code — deleted rather
        // than duplicating what the macro already guarantees.
        if (param.backend_name == "cpu") {
            device = Device::cpu();
            // Explicitly initialize the library for CPU tests
            tenzor::initialize();
        }
        else if (param.backend_name == "cuda") {
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            device = Device::oneapi(0);
        }
        else if (param.backend_name == "rocm") {
            device = Device::rocm(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to create ones tensor
    Tensor createOnes(const std::vector<int64_t>& shape) {
        return ones(shape, dtype, device);
    }

    // Helper to create full tensor
    Tensor createFull(const std::vector<int64_t>& shape, double value) {
        return full(shape, static_cast<float>(value), dtype, device);
    }
};

//==============================================================================
// ReduceLROnPlateau Tests
//==============================================================================

TEST_P(SchedulerAdvancedMultiDTypeTest, ReduceLROnPlateau_BasicMinMode) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 2);  // patience=2

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);
    EXPECT_FALSE(scheduler.in_cooldown());

    // Improving metric (decreasing loss)
    scheduler.step(1.0);
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);

    scheduler.step(0.9);  // Improvement
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);

    // No improvement for patience epochs. Reduction fires when
    // num_bad_epochs strictly EXCEEDS patience (ReduceLROnPlateau::step,
    // scheduler_advanced.cpp — matches PyTorch: patience=2 means the first
    // 2 bad epochs are ignored, reduction happens on the 3rd), so this
    // needs 3 non-improving step() calls, not 2.
    scheduler.step(0.91);  // No improvement, bad_epochs=1
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 1);

    scheduler.step(0.92);  // No improvement, bad_epochs=2 (== patience, not yet > patience)
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 2);
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.1);

    // Trigger LR reduction
    scheduler.step(0.93);  // No improvement, bad_epochs=3 > patience(2), triggers reduction
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-9);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, ReduceLROnPlateau_MaxMode) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);

    auto scheduler = ReduceLROnPlateau(optimizer, "max", 0.5, 3);

    scheduler.step(0.7);
    scheduler.step(0.8);  // Improvement
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.01);

    // No improvement for patience epochs. patience=3 here, and reduction
    // fires only once num_bad_epochs strictly EXCEEDS patience (see the
    // BasicMinMode comment above), so this needs 4 non-improving steps.
    scheduler.step(0.79);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 1);
    scheduler.step(0.78);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 2);
    scheduler.step(0.77);
    EXPECT_EQ(scheduler.get_num_bad_epochs(), 3);
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.01);
    scheduler.step(0.76);  // 4th non-improving step, bad_epochs=4 > patience(3), triggers reduction
    EXPECT_NEAR(scheduler.get_last_lr(), 0.005, 1e-9);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, ReduceLROnPlateau_Cooldown) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 2, 1e-4, "rel", 2);  // cooldown=2

    // Trigger reduction. patience=2 here, and reduction fires only once
    // num_bad_epochs strictly EXCEEDS patience (see BasicMinMode's comment),
    // so this needs 4 step() calls: the first sets the baseline, the next
    // 3 are non-improving (bad_epochs 1, 2, 3 — 3 > patience(2) triggers).
    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-9);
    EXPECT_TRUE(scheduler.in_cooldown());

    // During cooldown
    scheduler.step(1.0);
    EXPECT_TRUE(scheduler.in_cooldown());
    scheduler.step(1.0);
    EXPECT_FALSE(scheduler.in_cooldown());
}

TEST_P(SchedulerAdvancedMultiDTypeTest, ReduceLROnPlateau_MinLR) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.1);

    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 1, 1e-4, "rel", 0, 0.001);  // min_lr=0.001

    // Reduce multiple times
    for (int i = 0; i < 10; i++) {
        scheduler.step(1.0);
        scheduler.step(1.0);
    }

    // Should not go below min_lr
    EXPECT_GE(scheduler.get_last_lr(), 0.001);
}

//==============================================================================
// CyclicLR Tests
//==============================================================================

TEST_P(SchedulerAdvancedMultiDTypeTest, CyclicLR_Triangular) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = CyclicLR(optimizer, 0.001, 0.006, 4, -1, "triangular");

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 0.001);
    EXPECT_EQ(scheduler.get_iteration(), 0);

    // First half: increasing
    scheduler.step();
    EXPECT_GT(scheduler.get_last_lr(), 0.001);
    EXPECT_LT(scheduler.get_last_lr(), 0.006);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    // At step 4, should be at max_lr
    EXPECT_NEAR(scheduler.get_last_lr(), 0.006, 1e-9);

    // Second half: decreasing
    scheduler.step();
    EXPECT_LT(scheduler.get_last_lr(), 0.006);

    for (int i = 0; i < 3; i++) {
        scheduler.step();
    }
    // At step 8, back to base_lr
    EXPECT_NEAR(scheduler.get_last_lr(), 0.001, 1e-6);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, CyclicLR_Triangular2) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    auto scheduler = CyclicLR(optimizer, 0.001, 0.01, 5, -1, "triangular2");

    // First cycle
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
    EXPECT_LT(max_lr_cycle2, 0.01);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, CyclicLR_ExpRange) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.001);

    auto scheduler = CyclicLR(optimizer, 0.001, 0.006, 4, -1, "exp_range", 0.99);

    std::vector<double> lrs;
    lrs.push_back(scheduler.get_last_lr());
    for (int i = 0; i < 20; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }

    // Maximum LR should decrease over cycles
    double max_lr_first = *std::max_element(lrs.begin(), lrs.begin() + 8);
    double max_lr_last = *std::max_element(lrs.end() - 8, lrs.end());
    EXPECT_LT(max_lr_last, max_lr_first);
}

//==============================================================================
// OneCycleLR Tests
//==============================================================================

TEST_P(SchedulerAdvancedMultiDTypeTest, OneCycleLR_BasicCycle) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    int total_steps = 100;
    auto scheduler = OneCycleLR(optimizer, 0.1, total_steps, -1, -1, 0.3, "cos", 25.0, 10000.0);

    // Initial LR should be max_lr / div_factor. Per LL.6, OneCycleLR's
    // internal step_count_ starts at -1; the first step() call advances
    // it to 0 and computes the first warmup LR.
    EXPECT_NEAR(scheduler.get_last_lr(), 0.004, 1e-6);
    EXPECT_EQ(scheduler.get_step(), -1);

    // Phase 1: Warmup (30% of steps).
    // Peak == max_lr is hit at step == warmup_steps (first annealing call,
    // pct=0); step 31 times so lrs[30] reflects the peak.
    std::vector<double> lrs;
    for (int i = 0; i < 31; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }
    // LR should increase during warmup
    EXPECT_GT(lrs[30], lrs[0]);
    EXPECT_NEAR(lrs[30], 0.1, 1e-6);

    // Phase 2: Annealing
    double lr_at_30 = scheduler.get_last_lr();
    for (int i = 31; i < 100; i++) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }
    // LR should decrease
    EXPECT_LT(scheduler.get_last_lr(), lr_at_30);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, OneCycleLR_LinearAnnealing) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);

    auto scheduler = OneCycleLR(optimizer, 0.01, 50, -1, -1, 0.3, "linear");

    std::vector<double> warmup_lrs;
    for (int i = 0; i < 15; i++) {
        scheduler.step();
        warmup_lrs.push_back(scheduler.get_last_lr());
    }

    // Warmup should be approximately linear
    double diff1 = warmup_lrs[1] - warmup_lrs[0];
    double diff_mid = warmup_lrs[8] - warmup_lrs[7];
    EXPECT_NEAR(diff1, diff_mid, diff1 * 0.5);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, OneCycleLR_CustomDivFactors) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 0.001);

    auto scheduler = OneCycleLR(optimizer, 1.0, 100, -1, -1, 0.5, "cos", 10.0, 100.0);

    // Initial LR = max_lr / div_factor
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);

    // Need total_steps+1 calls to land last_lr_ on the final value (see
    // matching comment in test_schedulers_advanced.cpp).
    for (int i = 0; i < 101; i++) {
        scheduler.step();
    }

    // Final LR = max_lr / final_div_factor
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-6);
}

//==============================================================================
// CosineAnnealingWarmRestarts Tests
//==============================================================================

TEST_P(SchedulerAdvancedMultiDTypeTest, CosineAnnealingWarmRestarts_BasicRestart) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1, 0.0);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), 1.0);
    EXPECT_EQ(scheduler.get_T_cur(), 0);
    EXPECT_EQ(scheduler.get_T_i(), 10);

    // First cycle: 9 steps reach the cycle minimum. PyTorch SGDR uses the full
    // period T_i in the denominator, so at T_cur == 9 (T_i == 10) the LR is the
    // lowest in-cycle value but not exactly eta_min. The 10th step triggers the
    // restart (see test_schedulers_advanced.cpp for the matching analysis).
    std::vector<double> lrs_cycle1;
    for (int i = 0; i < 9; i++) {
        scheduler.step();
        lrs_cycle1.push_back(scheduler.get_last_lr());
    }

    // LR should decrease monotonically toward eta_min.
    EXPECT_GT(lrs_cycle1[0], lrs_cycle1[8]);
    const double expected_min = (1.0 + std::cos(std::numbers::pi * 9.0 / 10.0)) / 2.0;
    EXPECT_NEAR(lrs_cycle1[8], expected_min, 1e-6);

    // 10th call triggers the restart roll-over.
    scheduler.step();

    // After restart, T_cur should be 0.
    EXPECT_EQ(scheduler.get_T_cur(), 0);

    // The restart step itself put us back at base_lr.
    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-6);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, CosineAnnealingWarmRestarts_TMultiplier) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
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

TEST_P(SchedulerAdvancedMultiDTypeTest, CosineAnnealingWarmRestarts_EtaMin) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = AdamW(params, 1.0);

    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1, 0.1);  // eta_min=0.1

    // 9 steps reach the cycle minimum (T_cur == 9, T_i == 10). With the full-
    // period PyTorch denominator this is the lowest in-cycle LR but not exactly
    // eta_min.
    for (int i = 0; i < 9; i++) {
        scheduler.step();
    }

    // LR must stay at or above eta_min and equal the exact PyTorch SGDR value.
    const double expected = 0.1 + (1.0 - 0.1) * (1.0 + std::cos(std::numbers::pi * 9.0 / 10.0)) / 2.0;
    EXPECT_GE(scheduler.get_last_lr(), 0.1);
    EXPECT_NEAR(scheduler.get_last_lr(), expected, 1e-6);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, CosineAnnealingWarmRestarts_MultipleRestarts) {
    auto param = std::make_shared<Variable>(createOnes({2, 2}), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 1.0);

    auto scheduler = CosineAnnealingWarmRestarts(optimizer, 5, 1, 0.0);

    std::vector<double> all_lrs;
    for (int i = 0; i < 20; i++) {
        scheduler.step();
        all_lrs.push_back(scheduler.get_last_lr());
    }

    // Check that LR restarts at cycles. With LL.6 increment-first semantics
    // the post-reset peaks sit at indices 4, 9, 14 (see the matching
    // analysis in test_schedulers_advanced.cpp).
    EXPECT_GT(all_lrs[4], all_lrs[3]);    // Restart at step 5 (index 4)
    EXPECT_GT(all_lrs[9], all_lrs[8]);    // Restart at step 10 (index 9)
    EXPECT_GT(all_lrs[14], all_lrs[13]);  // Restart at step 15 (index 14)
}

//==============================================================================
// Integration Tests
//==============================================================================

TEST_P(SchedulerAdvancedMultiDTypeTest, ReduceLROnPlateau_Training_Simulation) {
    auto param = std::make_shared<Variable>(createFull({2, 2}, 10.0), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);
    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.5, 3);

    // patience=3: reduction fires only once num_bad_epochs strictly EXCEEDS
    // patience (ReduceLROnPlateau::step — matches PyTorch), i.e. on the 4th
    // consecutive non-improving epoch. Losses improve through epoch 3
    // (0.84 < 0.84's-predecessor*(1-rel_threshold)), then plateau at 0.84
    // for epochs 4-7 (4 consecutive non-improving epochs: bad_epochs goes
    // 1,2,3,4 — the reduction triggers on epoch 7, not epoch 6), then
    // improves again at the end.
    std::vector<double> val_losses = {1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84, 0.84, 0.83};

    double last_lr = 0.1;
    for (size_t epoch = 0; epoch < val_losses.size(); epoch++) {
        param->set_grad(createOnes({2, 2}));
        optimizer.step();
        scheduler.step(val_losses[epoch]);

        if (epoch < 7) {
            EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), last_lr);
        } else {
            last_lr = 0.05;
            EXPECT_NEAR(scheduler.get_last_lr(), last_lr, 1e-9);
        }
    }
}

TEST_P(SchedulerAdvancedMultiDTypeTest, CyclicLR_Training_Simulation) {
    auto param = std::make_shared<Variable>(createFull({2, 2}, 10.0), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);
    auto scheduler = CyclicLR(optimizer, 0.001, 0.01, 10);

    for (int batch = 0; batch < 50; batch++) {
        param->set_grad(createOnes({2, 2}));
        optimizer.step();
        scheduler.step();

        // LR should be within bounds
        EXPECT_GE(scheduler.get_last_lr(), 0.001 - 1e-9);
        EXPECT_LE(scheduler.get_last_lr(), 0.01 + 1e-9);
    }

    EXPECT_GE(scheduler.get_cycle(), 2);
}

TEST_P(SchedulerAdvancedMultiDTypeTest, OneCycleLR_Training_Simulation) {
    auto param = std::make_shared<Variable>(createFull({2, 2}, 10.0), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    int total_steps = 100;
    auto scheduler = OneCycleLR(optimizer, 0.1, total_steps);

    for (int step = 0; step < total_steps; step++) {
        param->set_grad(createOnes({2, 2}));
        optimizer.step();
        scheduler.step();
    }

    // At end of training, LR should be very low
    EXPECT_LT(scheduler.get_last_lr(), 0.001);
}

//==============================================================================
// Numerical Stability Tests (Critical for Float64)
//==============================================================================

TEST_P(SchedulerAdvancedMultiDTypeTest, ReduceLROnPlateau_PrecisionTest) {
    auto param = std::make_shared<Variable>(createOnes({10}), true);
    std::vector<std::shared_ptr<Variable>> params{param};
    auto optimizer = Adam(params, 0.1);

    // Use threshold of 1e-8 so improvement of 1e-7 (from 1.0 to 0.9999999) is detected
    auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 2, 1e-8, "abs");

    // Test with very small metric differences
    scheduler.step(1.0);
    scheduler.step(0.9999999);  // Improvement of 1e-7, exceeds threshold of 1e-8

    // Float64 should detect this small improvement better than Float32
    if (dtype == DType::Float64) {
        EXPECT_EQ(scheduler.get_num_bad_epochs(), 0);  // Should count as improvement
    }
}

TEST_P(SchedulerAdvancedMultiDTypeTest, CyclicLR_PrecisionTest) {
    auto param = std::make_shared<Variable>(createOnes({10}), true);
    std::vector<std::shared_ptr<Variable>> params{param};
    auto optimizer = SGD(params, 0.1);

    auto scheduler = CyclicLR(optimizer, 1e-8, 1e-6, 100);

    // Test with very small learning rates
    for (int i = 0; i < 200; i++) {
        scheduler.step();
        double lr = scheduler.get_last_lr();

        // Should maintain precision at very small values
        EXPECT_GE(lr, 1e-8 - atol);
        EXPECT_LE(lr, 1e-6 + atol);
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<SchedulerDTypeParam> GenerateSchedulerDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    // Schedulers operate on learning rates (double precision internally)
    // But test with different parameter dtypes
    std::vector<std::tuple<DType, std::string, double, double>> dtypes = {
        {DType::Float32, "float32", 1e-6, 1e-8},   // Standard precision
        {DType::Float64, "float64", 1e-10, 1e-12}, // High precision
    };

    std::vector<SchedulerDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name, rtol, atol] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name, rtol, atol});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsFloatDTypes,
    SchedulerAdvancedMultiDTypeTest,
    ::testing::ValuesIn(GenerateSchedulerDTypeCombinations()),
    [](const ::testing::TestParamInfo<SchedulerDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE IMPACT SUMMARY:
 *
 * Original test_schedulers_advanced.cpp:
 * - 20 tests × 4 backends × 1 dtype (Float32) = 80 test scenarios
 *
 * Refactored test_schedulers_advanced_multidtype.cpp:
 * - 22 tests × 4 backends × 2 dtypes (Float32, Float64) = 176 test scenarios
 *
 * Coverage increase: 2.2x improvement (96 additional test scenarios)
 *
 * Tests covered:
 * - ReduceLROnPlateau: BasicMinMode, MaxMode, Cooldown, MinLR, Training_Simulation, PrecisionTest (6 tests)
 * - CyclicLR: Triangular, Triangular2, ExpRange, Training_Simulation, PrecisionTest (5 tests)
 * - OneCycleLR: BasicCycle, LinearAnnealing, CustomDivFactors, Training_Simulation (4 tests)
 * - CosineAnnealingWarmRestarts: BasicRestart, TMultiplier, EtaMin, MultipleRestarts (4 tests)
 * - Integration tests: 3 realistic training simulations
 * - Numerical stability: 2 precision-focused tests
 *
 * DTypes tested:
 * - Float32: Standard parameter precision
 * - Float64: High precision parameters and LR calculations
 *
 * Key improvements:
 * 1. LR calculation precision verified across dtypes
 * 2. Parameter dtype consistency maintained
 * 3. Numerical stability with very small LR values
 * 4. Threshold detection accuracy (ReduceLROnPlateau)
 * 5. Cyclic pattern precision at different scales
 * 6. Warmup/annealing schedule accuracy
 *
 * Benefits for training:
 * - Better LR scheduling for high-precision models
 * - Accurate metric-based LR reduction
 * - Precise cyclic learning rate patterns
 * - Stable super-convergence schedules
 * - Reliable warm restart strategies
 */
