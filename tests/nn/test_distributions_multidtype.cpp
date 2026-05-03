/**
 * @file test_distributions_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for probability distributions
 *
 * Covers: Normal, Uniform distributions
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::distributions;

// ============================================================================
// Fixture
// ============================================================================

class DistributionsMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Normal Distribution Tests
// ============================================================================

TEST_P(DistributionsMultiDTypeTest, NormalSampleShape) {
    auto loc = tenzor::zeros({3}, dtype(), device());
    auto scale = tenzor::ones({3}, dtype(), device());
    Normal dist(loc, scale);

    auto sample = dist.sample({3});
    expectShape(sample, {3});
    expectDevice(sample);
}

TEST_P(DistributionsMultiDTypeTest, NormalSampleDeviceDType) {
    auto loc = tenzor::zeros({4}, dtype(), device());
    auto scale = tenzor::ones({4}, dtype(), device());
    Normal dist(loc, scale);

    auto sample = dist.sample();
    expectDevice(sample);
    expectDType(sample);
}

TEST_P(DistributionsMultiDTypeTest, NormalLogProbAtMean) {
    // log_prob at the mean should be -0.5*log(2*pi) - log(sigma) for N(mu, sigma)
    auto loc = tenzor::full({1}, 0.0f, dtype(), device());
    auto scale = tenzor::full({1}, 1.0f, dtype(), device());
    Normal dist(loc, scale);

    auto lp = dist.log_prob(loc);
    auto lp_f32 = lp.to(Device::cpu()).to(DType::Float32);
    // -0.5*log(2*pi) ≈ -0.9189
    EXPECT_NEAR(lp_f32.data<float>()[0], -0.9189385f, std::max(atol(), 1e-3f));
}

TEST_P(DistributionsMultiDTypeTest, NormalEntropy) {
    // entropy of N(0, sigma) = 0.5*log(2*pi*e) + log(sigma)
    auto loc = tenzor::zeros({1}, dtype(), device());
    auto scale = tenzor::full({1}, 1.0f, dtype(), device());
    Normal dist(loc, scale);

    auto ent = dist.entropy();
    auto ent_f32 = ent.to(Device::cpu()).to(DType::Float32);
    float expected = 0.5f * std::log(2.0f * static_cast<float>(M_PI) * std::exp(1.0f));
    EXPECT_NEAR(ent_f32.data<float>()[0], expected, std::max(atol(), 1e-3f));
}

TEST_P(DistributionsMultiDTypeTest, NormalMean) {
    auto loc = tenzor::full({2}, 3.0f, dtype(), device());
    auto scale = tenzor::full({2}, 1.0f, dtype(), device());
    Normal dist(loc, scale);

    auto m = dist.mean();
    auto m_f32 = m.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(m_f32.data<float>()[0], 3.0f, atol());
    EXPECT_NEAR(m_f32.data<float>()[1], 3.0f, atol());
}

TEST_P(DistributionsMultiDTypeTest, NormalVariance) {
    auto loc = tenzor::zeros({2}, dtype(), device());
    auto scale = tenzor::full({2}, 2.0f, dtype(), device());
    Normal dist(loc, scale);

    auto v = dist.variance();
    auto v_f32 = v.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(v_f32.data<float>()[0], 4.0f, std::max(atol(), 1e-3f));
}

// ============================================================================
// Uniform Distribution Tests
// ============================================================================

TEST_P(DistributionsMultiDTypeTest, UniformSampleRange) {
    auto low = tenzor::full({100}, 0.0f, dtype(), device());
    auto high = tenzor::full({100}, 1.0f, dtype(), device());
    Uniform dist(low, high);

    auto sample = dist.sample();
    float min_val = compute_min(sample);
    float max_val = compute_max(sample);
    EXPECT_GE(min_val, -atol());
    EXPECT_LE(max_val, 1.0f + atol());
}

TEST_P(DistributionsMultiDTypeTest, UniformLogProb) {
    // log_prob = -log(high - low) for values in range
    auto low = tenzor::full({1}, 0.0f, dtype(), device());
    auto high = tenzor::full({1}, 2.0f, dtype(), device());
    Uniform dist(low, high);

    auto value = tenzor::full({1}, 1.0f, dtype(), device());
    auto lp = dist.log_prob(value);
    auto lp_f32 = lp.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(lp_f32.data<float>()[0], -std::log(2.0f), std::max(atol(), 1e-3f));
}

TEST_P(DistributionsMultiDTypeTest, UniformMean) {
    auto low = tenzor::full({1}, 2.0f, dtype(), device());
    auto high = tenzor::full({1}, 6.0f, dtype(), device());
    Uniform dist(low, high);

    auto m = dist.mean();
    auto m_f32 = m.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(m_f32.data<float>()[0], 4.0f, std::max(atol(), 1e-3f));
}

TEST_P(DistributionsMultiDTypeTest, UniformVariance) {
    auto low = tenzor::full({1}, 0.0f, dtype(), device());
    auto high = tenzor::full({1}, 6.0f, dtype(), device());
    Uniform dist(low, high);

    auto v = dist.variance();
    auto v_f32 = v.to(Device::cpu()).to(DType::Float32);
    // var = (high-low)^2 / 12 = 36/12 = 3
    EXPECT_NEAR(v_f32.data<float>()[0], 3.0f, std::max(atol(), 1e-2f));
}

// ============================================================================
// Phase C.3 — Extended distribution backend smoke. Verifies each
// distribution constructs and samples end-to-end on every backend ×
// dtype, catching missing-op-on-backend regressions.
// ============================================================================

TEST_P(DistributionsMultiDTypeTest, ExponentialSampleNonneg) {
    auto rate = tenzor::full({4}, 1.0f, dtype(), device());
    Exponential dist(rate);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "Exponential sample must be non-negative";
    }
}

TEST_P(DistributionsMultiDTypeTest, GammaSamplePositive) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Gamma sampler is Float32/Float64 only";
    }
    auto concentration = tenzor::full({4}, 2.0f, dtype(), device());
    auto rate = tenzor::full({4}, 1.0f, dtype(), device());
    Gamma dist(concentration, rate);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "Gamma samples must be non-negative";
    }
}

TEST_P(DistributionsMultiDTypeTest, BetaSampleInZeroOne) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Beta sampler (uses Gamma) is Float32/Float64 only";
    }
    auto a = tenzor::full({4}, 2.0f, dtype(), device());
    auto b = tenzor::full({4}, 3.0f, dtype(), device());
    Beta dist(a, b);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], -1e-3f);
        EXPECT_LE(p[i], 1.0f + 1e-3f);
    }
}

TEST_P(DistributionsMultiDTypeTest, LogNormalSamplePositive) {
    auto loc = tenzor::full({4}, 0.0f, dtype(), device());
    auto scale = tenzor::full({4}, 1.0f, dtype(), device());
    LogNormal dist(loc, scale);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f) << "LogNormal samples must be positive";
    }
}

TEST_P(DistributionsMultiDTypeTest, CauchySampleFinite) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Cauchy in Float16 has unbounded tails that overflow Float16 range";
    }
    auto loc = tenzor::full({4}, 0.0f, dtype(), device());
    auto scale = tenzor::full({4}, 1.0f, dtype(), device());
    Cauchy dist(loc, scale);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(p[i])) << "Cauchy sample non-finite";
    }
}

TEST_P(DistributionsMultiDTypeTest, HalfNormalSampleNonneg) {
    auto scale = tenzor::full({4}, 1.0f, dtype(), device());
    HalfNormal dist(scale);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f);
    }
}

TEST_P(DistributionsMultiDTypeTest, Chi2SampleNonneg) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Chi2 (uses Gamma) is Float32/Float64 only";
    }
    auto df = tenzor::full({4}, 3.0f, dtype(), device());
    Chi2 dist(df);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f);
    }
}

TEST_P(DistributionsMultiDTypeTest, CategoricalSampleInRange) {
    auto probs = tenzor::full({3}, 1.0f / 3.0f, dtype(), device());
    Categorical dist(probs);
    auto s = dist.sample({16});
    auto cpu = s.to(Device::cpu()).to(DType::Int64).contiguous();
    auto* p = cpu.data<int64_t>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0);
        EXPECT_LT(p[i], 3);
    }
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DistributionsMultiDTypeTest);
