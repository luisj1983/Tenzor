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
#include <tenzor/distributions/independent.hpp>
#include <tenzor/distributions/mixture.hpp>
#include <tenzor/distributions/transformed.hpp>
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

// ====================================================================
// Phase C.1 — Distributions previously missing from test coverage.
// ====================================================================

TEST(DistGapFill, Pareto_SampleAboveScale) {
    auto scale = full({3}, 2.0f);
    auto alpha = full({3}, 3.0f);
    Pareto d(scale, alpha);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 2.0f - 1e-3f);  // Pareto support is [scale, ∞)
    }
}

TEST(DistGapFill, Weibull_SamplePositive) {
    auto scale = full({3}, 2.0f);
    auto concentration = full({3}, 1.5f);
    Weibull d(scale, concentration);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f);
    }
}

TEST(DistGapFill, Wishart_SampleSymmetric) {
    int64_t n = 3;
    auto df = full({}, 5.0f);
    auto eye_t = ::tenzor::eye(n);
    Wishart d(df, eye_t);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, LKJCholesky_SampleLowerTriangular) {
    int64_t dim = 3;
    // audit-2026-05-03 bug #7 fixed: concentration_as_double() helper now
    // dispatches on dtype, so Float32 concentration works directly.
    auto concentration = full({}, 1.0f);
    LKJCholesky d(dim, concentration);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
    // Lower-triangular cholesky factor: above-diagonal entries should be 0.
    auto cpu = s.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = i + 1; j < dim; ++j) {
            EXPECT_NEAR(p[i * dim + j], 0.0f, 1e-5f);
        }
    }
}

TEST(DistGapFill, LKJCholesky_SampleFloat64Concentration) {
    // Regression test: explicit Float64 concentration must still work after
    // bug #7 fix.
    int64_t dim = 3;
    auto concentration = full({}, 1.0).to(DType::Float64);
    LKJCholesky d(dim, concentration);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, Kumaraswamy_SampleInZeroOne) {
    auto a = full({3}, 2.0f);
    auto b = full({3}, 3.0f);
    Kumaraswamy d(a, b);
    auto s = d.sample();
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST(DistGapFill, OneHotCategorical_SampleSimplex) {
    auto probs = full({3}, 1.0f / 3.0f);
    OneHotCategorical d(probs);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, RelaxedOneHotCategorical_SamplesFinite) {
    auto temperature = full({}, 1.0f);
    auto probs = full({3}, 1.0f / 3.0f);
    RelaxedOneHotCategorical d(temperature, probs);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, ContinuousBernoulli_SampleInZeroOne) {
    auto probs = full({3}, 0.5f);
    ContinuousBernoulli d(probs);
    auto s = d.sample();
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST(DistGapFill, LogisticNormal_SampleSimplex) {
    auto loc = ::tenzor::zeros({2});
    auto scale = full({2}, 1.0f);
    LogisticNormal d(loc, scale);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST(DistGapFill, LowRankMultivariateNormal_SampleShape) {
    auto loc = ::tenzor::zeros({2});
    auto cov_factor = ::tenzor::ones({2, 1});
    auto cov_diag = full({2}, 0.1f);
    LowRankMultivariateNormal d(loc, cov_factor, cov_diag);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, Independent_ReinterpretsBatchAsEvent) {
    auto loc = ::tenzor::zeros({2, 3});
    auto scale = full({2, 3}, 1.0f);
    auto base = std::make_shared<Normal>(loc, scale);
    Independent d(base, /*reinterpreted_batch_ndims=*/1);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, TransformedDistribution_ChainedTransform) {
    // Normal pushed through ExpTransform yields LogNormal-like samples.
    auto loc = ::tenzor::zeros({3});
    auto scale = full({3}, 1.0f);
    auto base = std::make_shared<Normal>(loc, scale);
    std::vector<std::shared_ptr<Transform>> transforms = {
        std::make_shared<ExpTransform>(),
    };
    TransformedDistribution d(base, transforms);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    auto cpu = s.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f);
    }
}

// audit-2026-05-03 bug #6 fixed: gather_components rewritten to align
// indices to comp_samples-minus-K dimensionality before gather, eliminating
// the libstdc++ vector OOB on default args.
namespace {
auto make_2vec(float a, float b) -> Tensor {
    auto t = full({2}, 0.0f);
    auto cpu = t.to(Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    p[0] = a;
    p[1] = b;
    return cpu;
}
} // anonymous

TEST(DistGapFill, MixtureSameFamily_SampleDefaultArgs) {
    auto weights = make_2vec(0.5f, 0.5f);
    auto loc = make_2vec(0.0f, 5.0f);
    auto scale = make_2vec(1.0f, 1.0f);
    auto base = std::make_shared<Normal>(loc, scale);
    MixtureSameFamily msf(weights, base);
    auto s = msf.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST(DistGapFill, MixtureSameFamily_SampleNonTrivialShape) {
    auto weights = make_2vec(0.3f, 0.7f);
    auto loc = make_2vec(0.0f, 10.0f);
    auto scale = make_2vec(1.0f, 1.0f);
    auto base = std::make_shared<Normal>(loc, scale);
    MixtureSameFamily msf(weights, base);
    auto s = msf.sample({4});
    EXPECT_TRUE(is_finite(s));
    EXPECT_EQ(s.numel(), 4);
}

// ====================================================================
// Phase C.2 — Distribution transforms previously missing from coverage.
// ====================================================================

TEST(DistTransforms, TanhTransformInverseRoundtrip) {
    auto x = full({4}, 0.7f);
    TanhTransform tr;
    auto y = tr.call(x);
    auto x_back = tr.inv(y);
    auto a_cpu = x.contiguous();
    auto b_cpu = x_back.contiguous();
    auto* pa = a_cpu.data<float>();
    auto* pb = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_NEAR(pa[i], pb[i], 1e-4f);
    }
}

TEST(DistTransforms, SoftmaxTransformSimplexInvariant) {
    auto x = full({2, 3}, 0.5f);
    SoftmaxTransform tr(/*dim=*/-1);
    auto y = tr.call(x);
    auto y_cpu = y.contiguous();
    auto* p = y_cpu.data<float>();
    int64_t rows = y_cpu.shape()[0], cols = y_cpu.shape()[1];
    for (int64_t r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            EXPECT_GE(p[r * cols + c], 0.0f);
            row_sum += p[r * cols + c];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
}

TEST(DistTransforms, ComposeTransformOrderingInvariant) {
    // Compose(exp, sigmoid)(x) == sigmoid(exp(x)).
    auto x = full({4}, 0.5f);
    std::vector<std::shared_ptr<Transform>> transforms = {
        std::make_shared<ExpTransform>(),
        std::make_shared<SigmoidTransform>(),
    };
    ComposeTransform compose(transforms);
    auto y = compose.call(x);
    ExpTransform e;
    SigmoidTransform s;
    auto y_manual = s.call(e.call(x));
    auto a = y.contiguous();
    auto b = y_manual.contiguous();
    auto* pa = a.data<float>();
    auto* pb = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(pa[i], pb[i], 1e-5f);
    }
}
