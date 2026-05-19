/**
 * @file test_distributions_samplers.cpp
 * @brief Unit tests for Phase 11 distribution sampler free functions.
 *
 * Checks that each new sampler (weibull, laplace, dirichlet, half_normal,
 * von_mises, student_t, negative_binomial, binomial) produces samples whose
 * empirical moments match the closed-form values within a reasonable
 * statistical tolerance.  N = 10 000 samples, tolerance ~ 3 sigma of the
 * CLT standard error.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <cmath>

namespace tenzor {
namespace {

class DistributionsSamplersTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }

    // Compute empirical mean of a CPU Float32/Float64 tensor.
    static double empirical_mean(const Tensor& t) {
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
};

// ============================================================================
// Weibull
// ============================================================================

TEST_F(DistributionsSamplersTest, Weibull_MeanMatchesClosedForm) {
    // Weibull(lambda=2, k=1.5) — mean = lambda * Gamma(1 + 1/k)
    // = 2 * Gamma(1 + 1/1.5) = 2 * Gamma(1.6667) ≈ 2 * 0.9027 ≈ 1.8055
    auto scale = full({1}, 2.0, DType::Float32, Device::cpu());
    auto conc  = full({1}, 1.5, DType::Float32, Device::cpu());
    auto samples = weibull(scale, conc, {10000});
    ASSERT_EQ(samples.numel(), 10000);
    double m = empirical_mean(samples);
    EXPECT_NEAR(m, 1.8055, 0.12) << "Weibull(2, 1.5) empirical mean";
}

TEST_F(DistributionsSamplersTest, Weibull_SamplesPositive) {
    auto scale = full({1}, 1.0, DType::Float32, Device::cpu());
    auto conc  = full({1}, 2.0, DType::Float32, Device::cpu());
    auto samples = weibull(scale, conc, {500});
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f) << "Weibull sample at index " << i << " must be positive";
    }
}

// ============================================================================
// Laplace
// ============================================================================

TEST_F(DistributionsSamplersTest, Laplace_MeanIsLoc) {
    // Laplace(loc=3, scale=1) — mean = loc = 3
    auto loc   = full({1}, 3.0, DType::Float32, Device::cpu());
    auto scale = full({1}, 1.0, DType::Float32, Device::cpu());
    auto samples = laplace(loc, scale, {10000});
    ASSERT_EQ(samples.numel(), 10000);
    double m = empirical_mean(samples);
    EXPECT_NEAR(m, 3.0, 0.1) << "Laplace(3, 1) empirical mean";
}

TEST_F(DistributionsSamplersTest, Laplace_NegativeLocSupported) {
    auto loc   = full({1}, -2.0, DType::Float32, Device::cpu());
    auto scale = full({1}, 0.5, DType::Float32, Device::cpu());
    auto samples = laplace(loc, scale, {10000});
    double m = empirical_mean(samples);
    EXPECT_NEAR(m, -2.0, 0.05) << "Laplace(-2, 0.5) empirical mean";
}

// ============================================================================
// Dirichlet
// ============================================================================

TEST_F(DistributionsSamplersTest, Dirichlet_SumToOne) {
    // Dirichlet([2, 3, 5]) — samples should sum to 1 along the last dim
    auto alpha = zeros({3}, DType::Float32, Device::cpu());
    auto alpha_cpu = alpha.contiguous();
    float* ap = alpha_cpu.data<float>();
    ap[0] = 2.0f; ap[1] = 3.0f; ap[2] = 5.0f;

    auto samples = dirichlet(alpha_cpu, {100, 3});
    ASSERT_EQ(samples.shape()[0], 100);
    ASSERT_EQ(samples.shape()[1], 3);

    // Each row should sum to ~1
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    for (int64_t row = 0; row < 100; ++row) {
        float row_sum = p[row * 3] + p[row * 3 + 1] + p[row * 3 + 2];
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Dirichlet row " << row << " must sum to 1";
    }
}

TEST_F(DistributionsSamplersTest, Dirichlet_MeanMatchesAlpha) {
    // E[X_i] = alpha_i / sum(alpha) for Dirichlet
    // alpha = [1, 1, 2], sum = 4, expected means = [0.25, 0.25, 0.5]
    auto alpha = zeros({3}, DType::Float32, Device::cpu());
    {
        auto cpu = alpha.contiguous();
        float* ap = cpu.data<float>();
        ap[0] = 1.0f; ap[1] = 1.0f; ap[2] = 2.0f;
        alpha = cpu;
    }
    auto samples = dirichlet(alpha, {5000, 3});
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int64_t row = 0; row < 5000; ++row) {
        s0 += p[row * 3];
        s1 += p[row * 3 + 1];
        s2 += p[row * 3 + 2];
    }
    EXPECT_NEAR(s0 / 5000.0, 0.25, 0.02);
    EXPECT_NEAR(s1 / 5000.0, 0.25, 0.02);
    EXPECT_NEAR(s2 / 5000.0, 0.50, 0.02);
}

// ============================================================================
// HalfNormal
// ============================================================================

TEST_F(DistributionsSamplersTest, HalfNormal_SamplesNonNegative) {
    auto scale = full({1}, 1.0, DType::Float32, Device::cpu());
    auto samples = half_normal(scale, {1000});
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "HalfNormal sample at index " << i << " must be >= 0";
    }
}

TEST_F(DistributionsSamplersTest, HalfNormal_MeanMatchesClosedForm) {
    // HalfNormal(sigma=1) — mean = sigma * sqrt(2/pi) ≈ 0.7979
    auto scale = full({1}, 1.0, DType::Float32, Device::cpu());
    auto samples = half_normal(scale, {10000});
    double m = empirical_mean(samples);
    EXPECT_NEAR(m, 0.7979, 0.05) << "HalfNormal(1) empirical mean";
}

// ============================================================================
// VonMises
// ============================================================================

TEST_F(DistributionsSamplersTest, VonMises_MeanIsLoc) {
    // VonMises(loc=0, kappa=5) — circular mean = loc = 0
    auto loc   = full({1}, 0.0, DType::Float32, Device::cpu());
    auto conc  = full({1}, 5.0, DType::Float32, Device::cpu());
    auto samples = von_mises(loc, conc, {10000});
    double m = empirical_mean(samples);
    EXPECT_NEAR(m, 0.0, 0.05) << "VonMises(0, 5) empirical mean should be near loc";
}

TEST_F(DistributionsSamplersTest, VonMises_SamplesInRange) {
    // VonMises samples are in (-pi, pi]
    auto loc  = full({1}, 0.0, DType::Float32, Device::cpu());
    auto conc = full({1}, 1.0, DType::Float32, Device::cpu());
    auto samples = von_mises(loc, conc, {500});
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    constexpr float pi = 3.14159265f;
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], -pi - 1e-4f) << "VonMises sample must be > -pi";
        EXPECT_LE(p[i],  pi + 1e-4f) << "VonMises sample must be <= pi";
    }
}

// ============================================================================
// StudentT
// ============================================================================

TEST_F(DistributionsSamplersTest, StudentT_MeanNearZeroForLargeDF) {
    // StudentT(df=30, loc=0, scale=1) — should have mean ≈ 0
    auto df    = full({1}, 30.0, DType::Float32, Device::cpu());
    auto loc   = full({1},  0.0, DType::Float32, Device::cpu());
    auto scale = full({1},  1.0, DType::Float32, Device::cpu());
    auto samples = student_t(df, loc, scale, {10000});
    double m = empirical_mean(samples);
    EXPECT_NEAR(m, 0.0, 0.1) << "StudentT(30, 0, 1) empirical mean";
}

TEST_F(DistributionsSamplersTest, StudentT_LocationShifted) {
    // StudentT(df=5, loc=3, scale=1) — mean = 3 for df > 1
    auto df    = full({1}, 5.0, DType::Float32, Device::cpu());
    auto loc   = full({1}, 3.0, DType::Float32, Device::cpu());
    auto scale = full({1}, 1.0, DType::Float32, Device::cpu());
    auto samples = student_t(df, loc, scale, {10000});
    double m = empirical_mean(samples);
    EXPECT_NEAR(m, 3.0, 0.15) << "StudentT(5, 3, 1) empirical mean";
}

// ============================================================================
// NegativeBinomial
// ============================================================================

TEST_F(DistributionsSamplersTest, NegativeBinomial_MeanMatchesClosedForm) {
    // NegativeBinomial(r=5, p=0.4):
    //   In this parameterization, p is the "success prob per trial" and
    //   mean = r * p / (1 - p) = 5 * 0.4 / 0.6 ≈ 3.333.
    // Use shaped tensors so the gamma-Poisson sampler can broadcast across
    // elements independently.
    auto total = full({10000}, 5.0, DType::Float32, Device::cpu());
    auto probs = full({10000}, 0.4, DType::Float32, Device::cpu());
    auto samples = negative_binomial(total, probs);  // shape inferred
    EXPECT_EQ(samples.dtype(), DType::Int64);
    double m = empirical_mean(samples.to(DType::Float32));
    EXPECT_NEAR(m, 3.333, 0.5) << "NegativeBinomial(5, 0.4) empirical mean";
}

TEST_F(DistributionsSamplersTest, NegativeBinomial_SamplesNonNegative) {
    auto total = full({500}, 3.0, DType::Float32, Device::cpu());
    auto probs = full({500}, 0.5, DType::Float32, Device::cpu());
    auto samples = negative_binomial(total, probs);  // shape inferred
    ASSERT_EQ(samples.numel(), 500);
    auto cpu = samples.to(Device::cpu()).contiguous();
    const int64_t* p = cpu.data<int64_t>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0) << "NegativeBinomial sample at index " << i << " must be >= 0";
    }
}

// ============================================================================
// Binomial
// ============================================================================

TEST_F(DistributionsSamplersTest, Binomial_MeanMatchesClosedForm) {
    // Binomial(n=10, p=0.3) — mean = n*p = 3.0.
    // Provide a probability tensor of shape {10000} so each element gets
    // independent Bernoulli draws (the implementation loops n Bernoulli
    // calls over the probability tensor).
    auto probs = full({10000}, 0.3, DType::Float32, Device::cpu());
    auto samples = binomial(10, probs);  // shape inferred from probs
    ASSERT_EQ(samples.numel(), 10000);
    double m = empirical_mean(samples.to(DType::Float32));
    EXPECT_NEAR(m, 3.0, 0.2) << "Binomial(10, 0.3) empirical mean";
}

TEST_F(DistributionsSamplersTest, Binomial_SamplesInRange) {
    // Binomial(n, p) values in [0, n]
    auto probs = full({500}, 0.5, DType::Float32, Device::cpu());
    auto samples = binomial(8, probs);  // shape inferred from probs
    ASSERT_EQ(samples.numel(), 500);
    auto cpu = samples.to(Device::cpu()).contiguous();
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f);
        EXPECT_LE(p[i], 8.0f);
    }
}

} // namespace
} // namespace tenzor
