/**
 * @file test_sam_swa.cpp
 * @brief Tests for SAM optimizer, AveragedModel, and SWALR scheduler
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/sam.hpp>
#include <tenzor/nn/optim/swa.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/ops/creation.hpp>

#include "../../backend_test_fixture.hpp"

namespace tenzor {
namespace {

class SAMSWATest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    std::vector<std::shared_ptr<Variable>> make_params() {
        auto t = tenzor::ones({4, 4}, DType::Float32, device);
        auto param = std::make_shared<Variable>(t, /*requires_grad=*/true);
        return {param};
    }

    void set_unit_grad(std::vector<std::shared_ptr<Variable>>& params) {
        for (auto& p : params) {
            auto shape = p->tensor().shape();
            p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                     p->tensor().dtype(), device));
        }
    }

    float param_sum(const std::vector<std::shared_ptr<Variable>>& params) {
        float sum = 0.0f;
        for (auto& p : params) {
            auto cpu_t = p->tensor().to(Device::cpu());
            const auto* d = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) sum += d[i];
        }
        return sum;
    }
};

// ============================================================================
// SAM Tests
// ============================================================================

TEST_P(SAMSWATest, SAMFirstStepPerturbsWeights) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, /*lr=*/0.1);
    optim::SAM sam(base_opt, /*rho=*/0.05);

    float before = param_sum(params);
    set_unit_grad(params);
    sam.first_step();
    float after = param_sum(params);

    // first_step perturbs weights in the gradient direction
    EXPECT_NE(before, after)
        << "first_step should perturb weights";
}

TEST_P(SAMSWATest, SAMSecondStepRestoresAndSteps) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, /*lr=*/0.1);
    optim::SAM sam(base_opt, /*rho=*/0.05);

    float original = param_sum(params);

    // First forward-backward
    set_unit_grad(params);
    sam.first_step();  // perturb

    float perturbed = param_sum(params);
    EXPECT_NE(original, perturbed);

    // Second forward-backward at perturbed point
    set_unit_grad(params);
    sam.second_step();  // restore + step

    float final_val = param_sum(params);

    // After second_step, weights should differ from both original and perturbed
    // (restored to original, then SGD step applied)
    EXPECT_NE(final_val, perturbed)
        << "second_step should restore weights and step";
}

// Audit item D.7 — SAM must work with the polymorphic Optimizer::step(closure)
// API.  Previously SAM::step_impl threw, so step(closure) (which calls
// step_impl after the first forward+backward) was unusable.  After the
// fix, SAM overrides step(closure) directly to drive the required
// two-pass forward+backward sequence.
TEST_P(SAMSWATest, SAMStepWithClosurePolymorphic) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, /*lr=*/0.1);
    optim::SAM sam(base_opt, /*rho=*/0.05);

    float original = param_sum(params);
    int closure_calls = 0;

    auto closure = [&]() -> tenzor::Variable {
        ++closure_calls;
        sam.zero_grad();
        set_unit_grad(params);
        // Return a Variable representing the loss; value isn't used by
        // SAM (it returns the first-call loss back to the caller).
        return tenzor::Variable(
            tenzor::full({1}, static_cast<float>(closure_calls),
                          tenzor::DType::Float32, device),
            false);
    };

    auto loss = sam.step(closure);
    float final_val = param_sum(params);

    // The closure must be called twice — once at original weights, once
    // at the perturbed weights.
    EXPECT_EQ(closure_calls, 2)
        << "SAM step(closure) must invoke the closure exactly twice "
           "(once before first_step, once after to refresh gradients)";

    // Weights should have moved from their original value (base optimizer
    // stepped with the perturbed-point gradients).
    EXPECT_NE(final_val, original)
        << "step(closure) must update parameters after the SAM two-pass dance";

    // The returned loss should be the first-call loss (per SAM convention).
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_FLOAT_EQ(loss_cpu.data<float>()[0], 1.0f);
}

TEST_P(SAMSWATest, SAMLrGetSet) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, /*lr=*/0.01);
    optim::SAM sam(base_opt, /*rho=*/0.05);

    EXPECT_DOUBLE_EQ(sam.get_lr(), 0.01);
    sam.set_lr(0.05);
    EXPECT_DOUBLE_EQ(sam.get_lr(), 0.05);
}

TEST_P(SAMSWATest, SAMRhoGetSet) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, /*lr=*/0.01);
    optim::SAM sam(base_opt, /*rho=*/0.05);

    EXPECT_DOUBLE_EQ(sam.get_rho(), 0.05);
    sam.set_rho(0.1);
    EXPECT_DOUBLE_EQ(sam.get_rho(), 0.1);
}

// ============================================================================
// AveragedModel Tests
// ============================================================================

TEST_P(SAMSWATest, AveragedModelInitialState) {
    auto params = make_params();
    optim::AveragedModel avg(params);

    // Construction clones the initial params, so n_averaged starts at 1
    EXPECT_EQ(avg.n_averaged(), 1);
    EXPECT_EQ(avg.averaged_params().size(), params.size());
}

TEST_P(SAMSWATest, AveragedModelUpdateProducesRunningAverage) {
    // Start with params = all 1s
    auto params = make_params();
    optim::AveragedModel avg(params);

    // Initial averaged values should be 1.0 (cloned from params)
    auto init_avg_cpu = avg.averaged_params()[0].to(Device::cpu());
    auto* init_avg = init_avg_cpu.data<float>();
    EXPECT_NEAR(init_avg[0], 1.0f, 1e-6f);

    // Update with params = all 3s -> avg = (1*0 + 3) / 1 = 3 after first update
    // or using running mean: (1*0 + 3) / 1 = 3
    auto new_t = tenzor::full({4, 4}, 3.0f, DType::Float32, device);
    params[0] = std::make_shared<Variable>(new_t, /*requires_grad=*/true);
    avg.update_parameters(params);

    EXPECT_EQ(avg.n_averaged(), 2);

    // Update again with params = all 5s
    auto new_t2 = tenzor::full({4, 4}, 5.0f, DType::Float32, device);
    params[0] = std::make_shared<Variable>(new_t2, /*requires_grad=*/true);
    avg.update_parameters(params);

    EXPECT_EQ(avg.n_averaged(), 3);

    // The averaged value should be between the initial and latest
    auto avg_cpu = avg.averaged_params()[0].to(Device::cpu());
    auto* avg_data = avg_cpu.data<float>();
    // Exact value depends on whether initial clone counts as n=1 or n=0
    // Either way, should be in a reasonable range
    EXPECT_GT(avg_data[0], 1.0f);
    EXPECT_LT(avg_data[0], 5.0f);
}

TEST_P(SAMSWATest, AveragedModelApplyTo) {
    auto params = make_params();
    optim::AveragedModel avg(params);

    // Do some updates
    auto new_t = tenzor::full({4, 4}, 3.0f, DType::Float32, device);
    auto new_params = std::vector<std::shared_ptr<Variable>>{
        std::make_shared<Variable>(new_t, true)};
    avg.update_parameters(new_params);

    // Apply averaged params back
    auto target_params = make_params();
    avg.apply_to(target_params);

    // Target params should now hold the averaged values, not 1.0
    auto target_cpu = target_params[0]->tensor().to(Device::cpu());
    auto avg_cpu = avg.averaged_params()[0].to(Device::cpu());
    auto* d = target_cpu.data<float>();
    auto* a = avg_cpu.data<float>();
    EXPECT_FLOAT_EQ(d[0], a[0]);
}

// ============================================================================
// SWALR Tests
// ============================================================================

TEST_P(SAMSWATest, SWALRReturnsCorrectLR) {
    auto params = make_params();
    optim::SGD opt(params, /*lr=*/0.1);
    optim::SWALR swalr(opt, /*swa_lr=*/0.05, /*anneal_epochs=*/5);

    // After enough steps, LR should converge to swa_lr
    for (int i = 0; i < 10; ++i) {
        swalr.step();
    }

    EXPECT_NEAR(swalr.get_last_lr(), 0.05, 1e-6)
        << "After anneal_epochs, LR should be at swa_lr";
}

TEST_P(SAMSWATest, SWALRAnnealingPhase) {
    auto params = make_params();
    optim::SGD opt(params, /*lr=*/0.1);
    optim::SWALR swalr(opt, /*swa_lr=*/0.01, /*anneal_epochs=*/10);

    // During annealing, LR should be between initial and swa_lr
    swalr.step();
    double lr_mid = swalr.get_last_lr();
    EXPECT_GE(lr_mid, 0.01) << "LR should be >= swa_lr during annealing";
    EXPECT_LE(lr_mid, 0.1)  << "LR should be <= initial LR during annealing";
}

TEST_P(SAMSWATest, SWALRImmediateConvergence) {
    auto params = make_params();
    optim::SGD opt(params, /*lr=*/0.1);
    // anneal_epochs = 1 -> immediate jump
    optim::SWALR swalr(opt, /*swa_lr=*/0.05, /*anneal_epochs=*/1);

    swalr.step();
    EXPECT_NEAR(swalr.get_last_lr(), 0.05, 1e-6);
}

INSTANTIATE_BACKEND_TESTS(SAMSWATest);

} // namespace
} // namespace tenzor
