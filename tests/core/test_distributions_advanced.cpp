// Tests for Gamma / Beta / Dirichlet / Poisson / StudentT / MultivariateNormal.
// CPU-only. Verifies sample mean/variance against closed-form values at tiny
// sample sizes (statistical tolerance budget), plus a few exact log_prob checks.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>

namespace tenzor {
namespace {

using namespace tenzor::distributions;

class DistributionsAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }

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

TEST_F(DistributionsAdvancedTest, GammaSampleMeanApproxConcentrationOverRate) {
    // Gamma(3, 2) has mean 3/2 = 1.5, var 3/4 = 0.75.
    auto conc = full({1}, 3.0, DType::Float32, Device::cpu());
    auto rate = full({1}, 2.0, DType::Float32, Device::cpu());
    Gamma g(conc, rate);
    auto samples = g.sample({20000});
    double m = sample_mean(samples);
    double v = sample_var(samples);
    EXPECT_NEAR(m, 1.5, 0.08);
    EXPECT_NEAR(v, 0.75, 0.15);
}

TEST_F(DistributionsAdvancedTest, GammaSampleWithShapeLessThanOne) {
    // Gamma(0.5, 1) mean is 0.5, variance 0.5.
    auto conc = full({1}, 0.5, DType::Float32, Device::cpu());
    auto rate = full({1}, 1.0, DType::Float32, Device::cpu());
    Gamma g(conc, rate);
    auto samples = g.sample({20000});
    EXPECT_NEAR(sample_mean(samples), 0.5, 0.06);
    EXPECT_NEAR(sample_var(samples), 0.5, 0.15);
}

// ============================================================================
// Beta
// ============================================================================

TEST_F(DistributionsAdvancedTest, BetaSampleMeanAndRange) {
    // Beta(2, 5) has mean 2/(2+5) = 2/7 ≈ 0.2857.
    auto c1 = full({1}, 2.0, DType::Float32, Device::cpu());
    auto c0 = full({1}, 5.0, DType::Float32, Device::cpu());
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

TEST_F(DistributionsAdvancedTest, DirichletSimplex) {
    // Dirichlet([2, 3, 5]) — samples should sum to 1 along the last dim.
    auto alpha = zeros({3}, DType::Float32, Device::cpu());
    alpha.data<float>()[0] = 2.0f;
    alpha.data<float>()[1] = 3.0f;
    alpha.data<float>()[2] = 5.0f;
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

TEST_F(DistributionsAdvancedTest, StudentTMeanApproxZero) {
    // Standard Student-t with df=10 has mean 0 and variance df/(df-2) = 1.25.
    auto df = full({1}, 10.0, DType::Float32, Device::cpu());
    auto loc = zeros({1}, DType::Float32, Device::cpu());
    auto scale = full({1}, 1.0, DType::Float32, Device::cpu());
    StudentT t(df, loc, scale);
    auto samples = t.sample({20000});
    EXPECT_NEAR(sample_mean(samples), 0.0, 0.1);
    EXPECT_NEAR(sample_var(samples),  1.25, 0.3);
}

// ============================================================================
// Poisson
// ============================================================================

TEST_F(DistributionsAdvancedTest, PoissonSmallLambda) {
    // Poisson(3) has mean and variance 3.
    auto rate = full({1}, 3.0, DType::Float32, Device::cpu());
    Poisson p(rate);
    auto samples = p.sample({20000});
    ASSERT_EQ(samples.dtype(), DType::Int64);
    auto float_samples = samples.to(DType::Float64);
    EXPECT_NEAR(sample_mean(float_samples), 3.0, 0.1);
    EXPECT_NEAR(sample_var(float_samples),  3.0, 0.3);
}

TEST_F(DistributionsAdvancedTest, PoissonLargeLambda) {
    // Poisson(50) should exercise the transformed-rejection branch.
    auto rate = full({1}, 50.0, DType::Float32, Device::cpu());
    Poisson p(rate);
    auto samples = p.sample({20000});
    auto float_samples = samples.to(DType::Float64);
    EXPECT_NEAR(sample_mean(float_samples), 50.0, 0.6);
    EXPECT_NEAR(sample_var(float_samples),  50.0, 2.5);
}

// ============================================================================
// MultivariateNormal
// ============================================================================

TEST_F(DistributionsAdvancedTest, MVNSampleIdentity) {
    // MVN with loc=0, cov=I should behave like independent standard normals.
    auto loc = zeros({3}, DType::Float32, Device::cpu());
    auto I = zeros({3, 3}, DType::Float32, Device::cpu());
    I.data<float>()[0] = 1.0f;
    I.data<float>()[4] = 1.0f;
    I.data<float>()[8] = 1.0f;
    MultivariateNormal mvn(loc, I);
    auto s = mvn.sample();
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 3);
}

TEST_F(DistributionsAdvancedTest, MVNLogProbSanity) {
    // For loc=0, cov=I, x=0, log p(x) = -0.5 * d * log(2*pi).
    auto loc = zeros({2}, DType::Float32, Device::cpu());
    auto I = zeros({2, 2}, DType::Float32, Device::cpu());
    I.data<float>()[0] = 1.0f;
    I.data<float>()[3] = 1.0f;
    MultivariateNormal mvn(loc, I);
    auto zero = zeros({2}, DType::Float32, Device::cpu());
    auto lp = mvn.log_prob(zero);
    float val = lp.data<float>()[0];
    float expected = -0.5f * 2.0f * static_cast<float>(std::log(2.0 * M_PI));
    EXPECT_NEAR(val, expected, 1e-4f);
}

// ============================================================================
// kl_divergence (closed-form Normal || Normal)
// ============================================================================

TEST_F(DistributionsAdvancedTest, KLNormalNormal) {
    auto loc1 = zeros({1}, DType::Float32, Device::cpu());
    auto scale1 = full({1}, 1.0, DType::Float32, Device::cpu());
    auto loc2 = zeros({1}, DType::Float32, Device::cpu());
    auto scale2 = full({1}, 1.0, DType::Float32, Device::cpu());
    Normal p(loc1, scale1);
    Normal q(loc2, scale2);
    auto kl = kl_divergence(p, q);
    EXPECT_NEAR(kl.data<float>()[0], 0.0, 1e-5);
}

TEST_F(DistributionsAdvancedTest, KLDifferentMeans) {
    // KL(N(1, 1) || N(0, 1)) = 0.5
    auto loc1 = full({1}, 1.0, DType::Float32, Device::cpu());
    auto scale1 = full({1}, 1.0, DType::Float32, Device::cpu());
    auto loc2 = zeros({1}, DType::Float32, Device::cpu());
    auto scale2 = full({1}, 1.0, DType::Float32, Device::cpu());
    Normal p(loc1, scale1);
    Normal q(loc2, scale2);
    auto kl = kl_divergence(p, q);
    EXPECT_NEAR(kl.data<float>()[0], 0.5, 1e-5);
}

} // namespace
} // namespace tenzor
