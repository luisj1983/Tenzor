// Tests for Gamma / Beta / Dirichlet / Poisson / StudentT / MultivariateNormal.
// Cross-backend. Verifies sample mean/variance against closed-form values at tiny
// sample sizes (statistical tolerance budget), plus a few exact log_prob checks.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>

#include "../backend_test_fixture.hpp"

namespace tenzor {
namespace {

using namespace tenzor::distributions;

class DistributionsAdvancedTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Compute sample mean of a 1D tensor of floats via CPU reduction.
    static double sample_mean(const Tensor& t) {
        auto cpu = t.to(Device::cpu()).contiguous();
        int64_t n = cpu.numel();
        double acc = 0.0;
        if (cpu.dtype() == DType::Float32) {
            const float* p = cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) acc += p[i];
        } else {
            const double* p = cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) acc += p[i];
        }
        return acc / static_cast<double>(n);
    }

    static double sample_var(const Tensor& t) {
        double m = sample_mean(t);
        auto cpu = t.to(Device::cpu()).contiguous();
        int64_t n = cpu.numel();
        double acc = 0.0;
        if (cpu.dtype() == DType::Float32) {
            const float* p = cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                double d = p[i] - m;
                acc += d * d;
            }
        } else {
            const double* p = cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                double d = p[i] - m;
                acc += d * d;
            }
        }
        return acc / static_cast<double>(n);
    }
};

// ============================================================================
// Gamma
// ============================================================================

TEST_P(DistributionsAdvancedTest, GammaSampleMeanApproxConcentrationOverRate) {
    // Gamma(3, 2) has mean 3/2 = 1.5, var 3/4 = 0.75.
    auto conc = full({1}, 3.0, DType::Float32, device);
    auto rate = full({1}, 2.0, DType::Float32, device);
    Gamma g(conc, rate);
    auto samples = g.sample({20000});
    double m = sample_mean(samples);
    double v = sample_var(samples);
    EXPECT_NEAR(m, 1.5, 0.08);
    EXPECT_NEAR(v, 0.75, 0.15);
}

TEST_P(DistributionsAdvancedTest, GammaSampleWithShapeLessThanOne) {
    // Gamma(0.5, 1) mean is 0.5, variance 0.5.
    auto conc = full({1}, 0.5, DType::Float32, device);
    auto rate = full({1}, 1.0, DType::Float32, device);
    Gamma g(conc, rate);
    auto samples = g.sample({20000});
    EXPECT_NEAR(sample_mean(samples), 0.5, 0.06);
    EXPECT_NEAR(sample_var(samples), 0.5, 0.15);
}

// ============================================================================
// Beta
// ============================================================================

TEST_P(DistributionsAdvancedTest, BetaSampleMeanAndRange) {
    // Beta(2, 5) has mean 2/(2+5) = 2/7 ≈ 0.2857.
    auto c1 = full({1}, 2.0, DType::Float32, device);
    auto c0 = full({1}, 5.0, DType::Float32, device);
    Beta b(c1, c0);
    auto samples = b.sample({20000});
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    // Range check.
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f);
        EXPECT_LT(p[i], 1.0f);
    }
    EXPECT_NEAR(sample_mean(samples), 2.0 / 7.0, 0.02);
}

// ============================================================================
// Dirichlet
// ============================================================================

TEST_P(DistributionsAdvancedTest, DirichletSimplex) {
    // Dirichlet([2, 3, 5]) — samples should sum to 1 along the last dim.
    auto alpha_host = zeros({3}, DType::Float32, Device::cpu());
    alpha_host.data<float>()[0] = 2.0f;
    alpha_host.data<float>()[1] = 3.0f;
    alpha_host.data<float>()[2] = 5.0f;
    auto alpha = alpha_host.to(device);
    Dirichlet d(alpha);
    auto samples = d.sample({1000, 3});
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    // Every row should sum to ~1.
    for (int64_t i = 0; i < 1000; ++i) {
        float s = p[i * 3] + p[i * 3 + 1] + p[i * 3 + 2];
        EXPECT_NEAR(s, 1.0f, 1e-4f);
    }
}

// ============================================================================
// StudentT
// ============================================================================

TEST_P(DistributionsAdvancedTest, StudentTMeanApproxZero) {
    // Standard Student-t with df=10 has mean 0 and variance df/(df-2) = 1.25.
    auto df = full({1}, 10.0, DType::Float32, device);
    auto loc = zeros({1}, DType::Float32, device);
    auto scale = full({1}, 1.0, DType::Float32, device);
    StudentT t(df, loc, scale);
    auto samples = t.sample({20000});
    EXPECT_NEAR(sample_mean(samples), 0.0, 0.1);
    EXPECT_NEAR(sample_var(samples),  1.25, 0.3);
}

// ============================================================================
// Poisson
// ============================================================================

TEST_P(DistributionsAdvancedTest, PoissonSmallLambda) {
    // Poisson(3) has mean and variance 3.
    auto rate = full({1}, 3.0, DType::Float32, device);
    Poisson p(rate);
    auto samples = p.sample({20000});
    ASSERT_EQ(samples.dtype(), DType::Int64);
    auto float_samples = samples.to(DType::Float64);
    EXPECT_NEAR(sample_mean(float_samples), 3.0, 0.1);
    EXPECT_NEAR(sample_var(float_samples),  3.0, 0.3);
}

TEST_P(DistributionsAdvancedTest, PoissonLargeLambda) {
    // Poisson(50) should exercise the transformed-rejection branch.
    auto rate = full({1}, 50.0, DType::Float32, device);
    Poisson p(rate);
    auto samples = p.sample({20000});
    auto float_samples = samples.to(DType::Float64);
    EXPECT_NEAR(sample_mean(float_samples), 50.0, 0.6);
    EXPECT_NEAR(sample_var(float_samples),  50.0, 2.5);
}

// ============================================================================
// MultivariateNormal
// ============================================================================

TEST_P(DistributionsAdvancedTest, MVNSampleIdentity) {
    // MVN with loc=0, cov=I should behave like independent standard normals.
    auto loc = zeros({3}, DType::Float32, device);
    auto I_host = zeros({3, 3}, DType::Float32, Device::cpu());
    I_host.data<float>()[0] = 1.0f;
    I_host.data<float>()[4] = 1.0f;
    I_host.data<float>()[8] = 1.0f;
    auto I = I_host.to(device);
    MultivariateNormal mvn(loc, I);
    auto s = mvn.sample();
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 3);
}

TEST_P(DistributionsAdvancedTest, MVNLogProbSanity) {
    // For loc=0, cov=I, x=0, log p(x) = -0.5 * d * log(2*pi).
    auto loc = zeros({2}, DType::Float32, device);
    auto I_host = zeros({2, 2}, DType::Float32, Device::cpu());
    I_host.data<float>()[0] = 1.0f;
    I_host.data<float>()[3] = 1.0f;
    auto I = I_host.to(device);
    MultivariateNormal mvn(loc, I);
    auto zero = zeros({2}, DType::Float32, device);
    auto lp = mvn.log_prob(zero);
    auto lp_cpu = lp.to(Device::cpu()).contiguous();
    float val = lp_cpu.data<float>()[0];
    float expected = -0.5f * 2.0f * static_cast<float>(std::log(2.0 * M_PI));
    EXPECT_NEAR(val, expected, 1e-4f);
}

// ============================================================================
// kl_divergence (closed-form Normal || Normal)
// ============================================================================

TEST_P(DistributionsAdvancedTest, KLNormalNormal) {
    auto loc1 = zeros({1}, DType::Float32, device);
    auto scale1 = full({1}, 1.0, DType::Float32, device);
    auto loc2 = zeros({1}, DType::Float32, device);
    auto scale2 = full({1}, 1.0, DType::Float32, device);
    Normal p(loc1, scale1);
    Normal q(loc2, scale2);
    auto kl = kl_divergence(p, q);
    auto kl_cpu = kl.to(Device::cpu()).contiguous();
    EXPECT_NEAR(kl_cpu.data<float>()[0], 0.0, 1e-5);
}

TEST_P(DistributionsAdvancedTest, KLDifferentMeans) {
    // KL(N(1, 1) || N(0, 1)) = 0.5
    auto loc1 = full({1}, 1.0, DType::Float32, device);
    auto scale1 = full({1}, 1.0, DType::Float32, device);
    auto loc2 = zeros({1}, DType::Float32, device);
    auto scale2 = full({1}, 1.0, DType::Float32, device);
    Normal p(loc1, scale1);
    Normal q(loc2, scale2);
    auto kl = kl_divergence(p, q);
    auto kl_cpu = kl.to(Device::cpu()).contiguous();
    EXPECT_NEAR(kl_cpu.data<float>()[0], 0.5, 1e-5);
}

INSTANTIATE_BACKEND_TESTS(DistributionsAdvancedTest);

} // namespace
} // namespace tenzor
