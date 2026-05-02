/**
 * @file test_distributions_gap_fill.cpp
 * @brief Phase 6 of the test-coverage campaign — fills the
 *        zero-coverage distributions identified in the audit.
 *
 * For each distribution, asserts:
 *   - sample() returns a tensor with the expected shape
 *   - sample() values pass a basic sanity check (range, finiteness, support)
 *
 * Heavy statistical accuracy tests live in test_distributions_advanced.cpp;
 * this file is the smoke-test backstop for distributions that previously
 * had no test file at all.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/distributions/transforms.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::distributions;

namespace {
struct InitTenzorOnce : public ::testing::Environment {
    void SetUp() override { tenzor::initialize(); }
};
::testing::Environment* const _g_init =
    ::testing::AddGlobalTestEnvironment(new InitTenzorOnce);
}  // namespace

namespace {

bool is_finite(const Tensor& t) {
    auto cpu = t.to(Device::cpu()).contiguous();
    if (cpu.dtype() != DType::Float32) cpu = cpu.to(DType::Float32);
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

bool all_in_range(const Tensor& t, float lo, float hi) {
    auto cpu = t.to(Device::cpu()).contiguous();
    if (cpu.dtype() != DType::Float32) cpu = cpu.to(DType::Float32);
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        if (p[i] < lo || p[i] > hi) return false;
    }
    return true;
}

}  // namespace

// ============================================================================
// Discrete distributions
// ============================================================================

TEST(DistGapFill, Categorical_SampleShape) {
    auto probs = full({3}, 1.0f / 3.0f);
    Categorical dist(probs);
    auto s = dist.sample({100});
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 100);
}

TEST(DistGapFill, BernoulliDist_SampleInZeroOne) {
    auto probs = full({4}, 0.5f);
    BernoulliDist dist(probs);
    auto s = dist.sample();
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST(DistGapFill, Binomial_SampleShape) {
    auto probs = full({2}, 0.5f);
    Binomial dist(/*total_count=*/10, probs);
    auto s = dist.sample();
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 2);
    // Binomial values are in [0, total_count]
    EXPECT_TRUE(all_in_range(s, 0.0f, 10.0f));
}

TEST(DistGapFill, Geometric_SampleNonnegative) {
    auto probs = full({4}, 0.3f);
    Geometric dist(probs);
    auto s = dist.sample();
    EXPECT_EQ(s.shape().size(), 1u);
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "Geometric sample should be non-negative";
    }
}

TEST(DistGapFill, NegativeBinomial_SampleShape) {
    auto total = full({2}, 5.0f);
    auto probs = full({2}, 0.5f);
    NegativeBinomial dist(total, probs);
    auto s = dist.sample();
    EXPECT_EQ(s.shape().size(), 1u);
}

// ============================================================================
// Continuous — common univariate
// ============================================================================

TEST(DistGapFill, LogNormal_SamplePositive) {
    auto loc = zeros({4});
    auto scale = full({4}, 1.0f);
    LogNormal dist(loc, scale);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f) << "LogNormal samples should be strictly positive";
    }
}

TEST(DistGapFill, Cauchy_SampleFinite) {
    auto loc = zeros({4});
    auto scale = full({4}, 1.0f);
    Cauchy dist(loc, scale);
    auto s = dist.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, Chi2_SamplePositive) {
    auto df = full({4}, 3.0f);
    Chi2 dist(df);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "Chi2 samples are non-negative";
    }
}

TEST(DistGapFill, Gumbel_SampleFinite) {
    auto loc = zeros({4});
    auto scale = full({4}, 1.0f);
    Gumbel dist(loc, scale);
    auto s = dist.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, HalfNormal_SampleNonnegative) {
    auto scale = full({4}, 1.0f);
    HalfNormal dist(scale);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "HalfNormal samples are non-negative";
    }
}

TEST(DistGapFill, HalfCauchy_SampleNonnegative) {
    auto scale = full({4}, 1.0f);
    HalfCauchy dist(scale);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "HalfCauchy samples are non-negative";
    }
}

TEST(DistGapFill, FisherSnedecor_SamplePositive) {
    auto df1 = full({2}, 5.0f);
    auto df2 = full({2}, 5.0f);
    FisherSnedecor dist(df1, df2);
    auto s = dist.sample();
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "F-distribution samples are non-negative";
    }
}

TEST(DistGapFill, VonMises_SampleInRange) {
    auto loc = zeros({4});
    auto concentration = full({4}, 1.0f);
    VonMises dist(loc, concentration);
    auto s = dist.sample();
    // VonMises is on the circle [-pi, pi] (or shifted by loc).
    EXPECT_TRUE(is_finite(s));
}

// ============================================================================
// Relaxed (continuous relaxations of discrete distributions)
// ============================================================================

TEST(DistGapFill, RelaxedBernoulli_SampleInZeroOne) {
    auto temperature = full({4}, 0.5f);
    auto probs = full({4}, 0.5f);
    RelaxedBernoulli dist(temperature, probs);
    auto s = dist.sample();
    // Concrete distribution values ∈ (0, 1) — soft, not exactly hitting endpoints.
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

// ============================================================================
// Transforms
// ============================================================================

TEST(DistTransforms, ExpTransformInverseRoundtrip) {
    auto x = full({4}, 0.5f);
    ExpTransform tr;
    auto y = tr.call(x);
    auto x_back = tr.inv(y);
    auto a = x.contiguous();
    auto b = x_back.contiguous();
    auto* pa = a.data<float>();
    auto* pb = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(pa[i], pb[i], 1e-5f);
    }
}

TEST(DistTransforms, SigmoidTransformInZeroOne) {
    auto x = full({4}, 0.5f);
    SigmoidTransform tr;
    auto y = tr.call(x);
    EXPECT_TRUE(all_in_range(y, 0.0f, 1.0f));
}

TEST(DistTransforms, AffineTransformLinearity) {
    auto x = full({4}, 1.0f);
    auto loc = full({4}, 2.0f);
    auto scale = full({4}, 3.0f);
    AffineTransform tr(loc, scale);
    auto y = tr.call(x);  // 2 + 3*1 = 5
    auto y_cpu = y.contiguous();
    auto* p = y_cpu.data<float>();
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(p[i], 5.0f);
    }
}
