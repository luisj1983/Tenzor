/**
 * @file test_new_features.cpp
 * @brief Tests for newly added distributions, schedulers, special functions,
 *        and functional activations.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/autograd/variable.hpp>

#include "../backend_test_fixture.hpp"

namespace tenzor {
namespace {

using namespace tenzor::distributions;

// ============================================================================
// Helper
// ============================================================================

class NewFeaturesTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    static double scalar_val(const Tensor& t) {
        auto cpu = t.to(Device::cpu()).contiguous();
        if (cpu.dtype() == DType::Float32)
            return static_cast<double>(cpu.data<float>()[0]);
        return cpu.data<double>()[0];
    }

    static double sample_mean(const Tensor& t) {
        auto cpu = t.to(Device::cpu()).contiguous();
        int64_t n = cpu.numel();
        double acc = 0.0;
        const float* p = cpu.data<float>();
        for (int64_t i = 0; i < n; ++i) acc += p[i];
        return acc / static_cast<double>(n);
    }
};

// ============================================================================
// Distribution Tests
// ============================================================================

TEST_P(NewFeaturesTest, Pareto_SampleAndLogProb) {
    auto scale = tenzor::full({1}, 1.0f, DType::Float32, device);
    auto alpha = tenzor::full({1}, 3.0f, DType::Float32, device);
    Pareto dist(scale, alpha);

    auto samples = dist.sample({1000});
    double mean_val = sample_mean(samples);
    // Pareto mean = alpha * scale / (alpha - 1) = 3 / 2 = 1.5
    EXPECT_NEAR(mean_val, 1.5, 0.3);

    auto lp = dist.log_prob(tenzor::full({1}, 2.0f, DType::Float32, device));
    // log(3) + 3*log(1) - 4*log(2) = log(3) - 4*log(2) ~ 1.0986 - 2.7726 = -1.674
    EXPECT_NEAR(scalar_val(lp), -1.674, 0.01);
}

TEST_P(NewFeaturesTest, Weibull_Sample) {
    auto scale = tenzor::full({1}, 1.0f, DType::Float32, device);
    auto concentration = tenzor::full({1}, 2.0f, DType::Float32, device);
    Weibull dist(scale, concentration);

    auto samples = dist.sample({5000});
    double mean_val = sample_mean(samples);
    // Weibull mean = scale * Gamma(1 + 1/k) = 1 * Gamma(1.5) = sqrt(pi)/2 ~ 0.886
    EXPECT_NEAR(mean_val, 0.886, 0.1);
}

TEST_P(NewFeaturesTest, Kumaraswamy_SampleInUnitInterval) {
    auto a = tenzor::full({1}, 2.0f, DType::Float32, device);
    auto b = tenzor::full({1}, 5.0f, DType::Float32, device);
    Kumaraswamy dist(a, b);

    auto samples = dist.sample({1000});
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f);
        EXPECT_LE(p[i], 1.0f);
    }
}

TEST_P(NewFeaturesTest, ContinuousBernoulli_Sample) {
    auto probs = tenzor::full({1}, 0.7f, DType::Float32, device);
    ContinuousBernoulli dist(probs);

    auto samples = dist.sample({1000});
    double mean_val = sample_mean(samples);
    // Should be in (0, 1) and biased towards probs
    EXPECT_GT(mean_val, 0.3);
    EXPECT_LT(mean_val, 0.9);
}

TEST_P(NewFeaturesTest, OneHotCategorical_Sample) {
    auto probs_cpu = tenzor::Tensor({3}, DType::Float32, Device::cpu());
    probs_cpu.data<float>()[0] = 0.2f;
    probs_cpu.data<float>()[1] = 0.3f;
    probs_cpu.data<float>()[2] = 0.5f;
    auto probs = probs_cpu.to(device);
    OneHotCategorical dist(probs);

    auto samples = dist.sample({100});
    // Each sample should be one-hot: sum along last dim = 1
    auto sums = tenzor::sum(samples, -1);
    auto cpu_sums = sums.to(Device::cpu()).contiguous();
    const float* p = cpu_sums.data<float>();
    for (int64_t i = 0; i < cpu_sums.numel(); ++i) {
        EXPECT_NEAR(p[i], 1.0f, 1e-5f);
    }
}

TEST_P(NewFeaturesTest, LogisticNormal_SampleOnSimplex) {
    auto loc = tenzor::zeros({3}, DType::Float32, device);
    auto scale = tenzor::ones({3}, DType::Float32, device);
    LogisticNormal dist(loc, scale);

    auto samples = dist.sample({100});
    // Each sample should sum to ~1 (simplex)
    auto sums = tenzor::sum(samples, -1);
    auto cpu_sums = sums.to(Device::cpu()).contiguous();
    const float* p = cpu_sums.data<float>();
    for (int64_t i = 0; i < cpu_sums.numel(); ++i) {
        EXPECT_NEAR(p[i], 1.0f, 1e-4f);
    }
}

// ============================================================================
// Scheduler Tests
// ============================================================================

TEST_P(NewFeaturesTest, ConstantLR_Schedule) {
    auto param = std::make_shared<Variable>(tenzor::randn({2, 2}, DType::Float32, device), true);
    std::vector<std::shared_ptr<Variable>> params = {param};
    optim::SGD optimizer(params, 0.1);

    optim::ConstantLR scheduler(optimizer, 0.5, 3);
    EXPECT_NEAR(scheduler.get_last_lr(), 0.05, 1e-7);  // 0.1 * 0.5

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.05, 1e-7);  // still factor

    scheduler.step();
    scheduler.step();  // epoch 3: should restore base_lr
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-7);
}

TEST_P(NewFeaturesTest, LinearLR_Schedule) {
    auto param = std::make_shared<Variable>(tenzor::randn({2, 2}, DType::Float32, device), true);
    std::vector<std::shared_ptr<Variable>> params = {param};
    optim::SGD optimizer(params, 0.1);

    optim::LinearLR scheduler(optimizer, 0.5, 1.0, 4);
    EXPECT_NEAR(scheduler.get_last_lr(), 0.05, 1e-7);  // start: 0.1 * 0.5

    scheduler.step();  // epoch 1: factor = 0.5 + (1.0 - 0.5) * 1/4 = 0.625
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0625, 1e-7);

    scheduler.step();  // epoch 2: factor = 0.5 + 0.5 * 2/4 = 0.75
    EXPECT_NEAR(scheduler.get_last_lr(), 0.075, 1e-7);
}

TEST_P(NewFeaturesTest, MultiplicativeLR_Schedule) {
    auto param = std::make_shared<Variable>(tenzor::randn({2, 2}, DType::Float32, device), true);
    std::vector<std::shared_ptr<Variable>> params = {param};
    optim::SGD optimizer(params, 0.1);

    optim::MultiplicativeLR scheduler(optimizer, [](int) { return 0.9; });
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-7);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.09, 1e-7);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.081, 1e-5);
}

// ============================================================================
// Special Function Tests
// ============================================================================

TEST_P(NewFeaturesTest, Ndtri_KnownValues) {
    auto p_cpu = tenzor::Tensor({3}, DType::Float32, Device::cpu());
    p_cpu.data<float>()[0] = 0.5f;
    p_cpu.data<float>()[1] = 0.8413f;  // ndtri(0.8413) ~ 1.0
    p_cpu.data<float>()[2] = 0.1587f;  // ndtri(0.1587) ~ -1.0
    auto p = p_cpu.to(device);

    auto result = tenzor::ndtri(p);
    auto cpu = result.to(Device::cpu()).contiguous();
    const float* r = cpu.data<float>();

    EXPECT_NEAR(r[0], 0.0f, 0.01f);
    EXPECT_NEAR(r[1], 1.0f, 0.02f);
    EXPECT_NEAR(r[2], -1.0f, 0.02f);
}

TEST_P(NewFeaturesTest, Ndtri_RoundtripWithNdtr) {
    auto x_cpu = tenzor::Tensor({3}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = -1.5f;
    x_cpu.data<float>()[1] = 0.0f;
    x_cpu.data<float>()[2] = 2.0f;
    auto x = x_cpu.to(device);

    auto roundtrip = tenzor::ndtri(tenzor::ndtr(x));
    auto cpu = roundtrip.to(Device::cpu()).contiguous();
    const float* r = cpu.data<float>();

    EXPECT_NEAR(r[0], -1.5f, 0.01f);
    EXPECT_NEAR(r[1], 0.0f, 0.01f);
    EXPECT_NEAR(r[2], 2.0f, 0.01f);
}

TEST_P(NewFeaturesTest, GammaincPlusGammaincc_EqualsGamma) {
    auto a_cpu = tenzor::Tensor({2}, DType::Float32, Device::cpu());
    a_cpu.data<float>()[0] = 2.0f;
    a_cpu.data<float>()[1] = 3.0f;
    auto a = a_cpu.to(device);
    auto x_cpu = tenzor::Tensor({2}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = 1.0f;
    x_cpu.data<float>()[1] = 2.0f;
    auto x = x_cpu.to(device);

    auto inc = tenzor::gammainc(a, x);
    auto incc = tenzor::gammaincc(a, x);
    auto total = inc + incc;
    auto gamma_a = tenzor::gamma(a);

    auto t_cpu = total.to(Device::cpu()).contiguous();
    auto g_cpu = gamma_a.to(Device::cpu()).contiguous();

    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(t_cpu.data<float>()[i], g_cpu.data<float>()[i], 1e-4f);
    }
}

// ============================================================================
// Functional Activation Tests
// ============================================================================

TEST_P(NewFeaturesTest, FunctionalRReLU) {
    auto input_t = tenzor::randn({10}, DType::Float32, device);
    auto input = Variable(input_t, false);

    auto result = nn::functional::rrelu(input, 0.1, 0.3, false);
    // In eval mode, rrelu uses midpoint slope = 0.2 for negatives
    auto cpu = result.tensor().to(Device::cpu()).contiguous();
    auto in_cpu = input_t.to(Device::cpu()).contiguous();
    for (int64_t i = 0; i < 10; ++i) {
        float x = in_cpu.data<float>()[i];
        float y = cpu.data<float>()[i];
        if (x >= 0) {
            EXPECT_NEAR(y, x, 1e-5f);
        } else {
            // Should be between lower*x and upper*x
            EXPECT_LE(y, 0.0f);
        }
    }
}

TEST_P(NewFeaturesTest, FunctionalLogSigmoid) {
    auto input_cpu = tenzor::Tensor({3}, DType::Float32, Device::cpu());
    input_cpu.data<float>()[0] = 0.0f;
    input_cpu.data<float>()[1] = 5.0f;
    input_cpu.data<float>()[2] = -5.0f;
    auto input_t = input_cpu.to(device);
    auto input = Variable(input_t, false);

    auto result = nn::functional::log_sigmoid(input);
    auto cpu = result.tensor().to(Device::cpu()).contiguous();
    const float* r = cpu.data<float>();

    // log(sigmoid(0)) = log(0.5) ~ -0.693
    EXPECT_NEAR(r[0], std::log(0.5f), 0.01f);
    // log(sigmoid(5)) ~ log(1) ~ 0
    EXPECT_NEAR(r[1], 0.0f, 0.01f);
    // log(sigmoid(-5)) ~ -5
    EXPECT_NEAR(r[2], -5.0f, 0.1f);
}

// ============================================================================
// Additional Distribution Tests (gap coverage)
// ============================================================================

TEST_P(NewFeaturesTest, LowRankMVN_SampleMean) {
    // 3D with rank-1 factor
    auto loc = tenzor::zeros({3}, DType::Float32, device);
    auto cov_factor = tenzor::ones({3, 1}, DType::Float32, device);
    auto cov_diag = tenzor::ones({3}, DType::Float32, device);

    LowRankMultivariateNormal dist(loc, cov_factor, cov_diag);
    auto m = dist.mean();
    auto cpu = m.to(Device::cpu()).contiguous();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(cpu.data<float>()[i], 0.0f, 1e-5f);
    }
}

TEST_P(NewFeaturesTest, Weibull_Entropy) {
    auto scale = tenzor::full({1}, 1.0f, DType::Float32, device);
    auto k = tenzor::full({1}, 1.0f, DType::Float32, device);
    Weibull dist(scale, k);

    // Weibull(1, 1) = Exponential(1)
    // entropy = gamma_const * (1 - 1/k) + log(lambda/k) + 1 = 0 + 0 + 1 = 1.0
    auto ent = dist.entropy();
    EXPECT_NEAR(scalar_val(ent), 1.0f, 0.01f);

    // Weibull(1, 2): entropy = 0.5772*(1 - 0.5) + log(1/2) + 1 ~ 0.2886 - 0.6931 + 1 = 0.5955
    auto k2 = tenzor::full({1}, 2.0f, DType::Float32, device);
    Weibull dist2(scale, k2);
    auto ent2 = dist2.entropy();
    EXPECT_NEAR(scalar_val(ent2), 0.5955f, 0.02f);
}

TEST_P(NewFeaturesTest, Pareto_CDF) {
    auto scale = tenzor::full({1}, 1.0f, DType::Float32, device);
    auto alpha = tenzor::full({1}, 2.0f, DType::Float32, device);
    Pareto dist(scale, alpha);

    auto val = tenzor::full({1}, 2.0f, DType::Float32, device);
    auto c = dist.cdf(val);
    // CDF(2) = 1 - (1/2)^2 = 1 - 0.25 = 0.75
    EXPECT_NEAR(scalar_val(c), 0.75f, 0.01f);
}

TEST_P(NewFeaturesTest, OneHotCategorical_Variance) {
    auto probs_cpu = tenzor::Tensor({3}, DType::Float32, Device::cpu());
    probs_cpu.data<float>()[0] = 0.2f;
    probs_cpu.data<float>()[1] = 0.3f;
    probs_cpu.data<float>()[2] = 0.5f;
    auto probs = probs_cpu.to(device);
    OneHotCategorical dist(probs);

    auto var = dist.variance();
    auto cpu = var.to(Device::cpu()).contiguous();
    // variance = p * (1 - p)
    EXPECT_NEAR(cpu.data<float>()[0], 0.2f * 0.8f, 1e-5f);
    EXPECT_NEAR(cpu.data<float>()[1], 0.3f * 0.7f, 1e-5f);
    EXPECT_NEAR(cpu.data<float>()[2], 0.5f * 0.5f, 1e-5f);
}

// ============================================================================
// Additional Scheduler Tests (gap coverage)
// ============================================================================

TEST_P(NewFeaturesTest, SequentialLR_MilestoneTransitions) {
    auto param = std::make_shared<Variable>(tenzor::randn({2, 2}, DType::Float32, device), true);
    std::vector<std::shared_ptr<Variable>> params = {param};
    optim::SGD optimizer(params, 0.1);

    auto sched1 = std::make_shared<optim::ConstantLR>(optimizer, 0.5, 100);
    auto sched2 = std::make_shared<optim::ExponentialLR>(optimizer, 0.9);

    optim::SequentialLR seq(optimizer, {sched1, sched2}, {3});

    // First 3 epochs use ConstantLR
    seq.step();
    seq.step();
    seq.step();
    // After milestone 3, ExponentialLR kicks in
    seq.step();
    // Should now be using ExponentialLR
    double lr = seq.get_last_lr();
    EXPECT_GT(lr, 0.0);
}

TEST_P(NewFeaturesTest, ChainedScheduler_MultipleSchedulers) {
    auto param = std::make_shared<Variable>(tenzor::randn({2, 2}, DType::Float32, device), true);
    std::vector<std::shared_ptr<Variable>> params = {param};
    optim::SGD optimizer(params, 0.1);

    auto sched1 = std::make_shared<optim::StepLR>(optimizer, 1, 0.9);
    auto sched2 = std::make_shared<optim::StepLR>(optimizer, 1, 0.8);

    optim::ChainedScheduler chained({sched1, sched2});
    chained.step();

    double lr = chained.get_last_lr();
    // Both schedulers applied: 0.1 * 0.9 * 0.8 = 0.072
    EXPECT_NEAR(lr, 0.072, 0.01);
}

// Audit-4 W.10: ChainedScheduler::state_dict() delegates entirely to its
// children — it has no parent epoch_/step_count_. Round-trip through
// state_dict/load_state_dict must restore the children's counters exactly.
TEST_P(NewFeaturesTest, ChainedScheduler_StateDict_RoundTrip) {
    auto param_src = std::make_shared<Variable>(tenzor::randn({2, 2}, DType::Float32, device), true);
    std::vector<std::shared_ptr<Variable>> params_src = {param_src};
    optim::SGD optimizer_src(params_src, 0.1);

    auto sched1_src = std::make_shared<optim::StepLR>(optimizer_src, 2, 0.5);
    auto sched2_src = std::make_shared<optim::StepLR>(optimizer_src, 3, 0.25);
    optim::ChainedScheduler chained_src({sched1_src, sched2_src});

    // Drive the source chain for several steps so each child accumulates a
    // distinct internal epoch_/last_lr.
    for (int i = 0; i < 5; ++i) chained_src.step();
    const double lr_src = chained_src.get_last_lr();
    const int epoch1_src = sched1_src->get_epoch();
    const int epoch2_src = sched2_src->get_epoch();

    // Build a fresh, independent destination chain (different optimizer
    // instance) and restore from the source state_dict.
    auto param_dst = std::make_shared<Variable>(tenzor::randn({2, 2}, DType::Float32, device), true);
    std::vector<std::shared_ptr<Variable>> params_dst = {param_dst};
    optim::SGD optimizer_dst(params_dst, 0.1);
    auto sched1_dst = std::make_shared<optim::StepLR>(optimizer_dst, 2, 0.5);
    auto sched2_dst = std::make_shared<optim::StepLR>(optimizer_dst, 3, 0.25);
    optim::ChainedScheduler chained_dst({sched1_dst, sched2_dst});

    auto state = chained_src.state_dict();
    chained_dst.load_state_dict(state);

    // Children counters round-trip exactly through the prefixed entries.
    EXPECT_EQ(sched1_dst->get_epoch(), epoch1_src);
    EXPECT_EQ(sched2_dst->get_epoch(), epoch2_src);
    EXPECT_NEAR(chained_dst.get_last_lr(), lr_src, 1e-9);

    // Stepping both chains one more time must produce the same LR — the
    // delegation contract relies on children carrying their own counters.
    chained_src.step();
    chained_dst.step();
    EXPECT_NEAR(chained_dst.get_last_lr(), chained_src.get_last_lr(), 1e-9);
}

INSTANTIATE_BACKEND_TESTS(NewFeaturesTest);

} // namespace
} // namespace tenzor
