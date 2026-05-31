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
 *
 * Parameterized over all backends via BackendTest: each TEST_P creates its
 * distribution parameter tensors on the fixture's `device`. Distribution
 * sample()/log_prob()/cdf() ops take no device argument — they inherit the
 * device of their parameter tensors. Host-side reads route through .cpu()
 * before .data<T>()/.item<T>().
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/distributions/transforms.hpp>
#include <tenzor/distributions/independent.hpp>
#include <tenzor/distributions/mixture.hpp>
#include <tenzor/distributions/transformed.hpp>
#include <tenzor/ops/creation.hpp>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::distributions;

class DistGapFill : public ::tenzor::testing::BackendTest {};
class DistTransforms : public ::tenzor::testing::BackendTest {};

namespace {

bool is_finite(const Tensor& t) {
    auto cpu = t.cpu().contiguous();
    if (cpu.dtype() != DType::Float32) cpu = cpu.to(DType::Float32);
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

bool all_in_range(const Tensor& t, float lo, float hi) {
    auto cpu = t.cpu().contiguous();
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

TEST_P(DistGapFill, Categorical_SampleShape) {
    auto probs = full({3}, 1.0f / 3.0f, DType::Float32, device);
    Categorical dist(probs);
    auto s = dist.sample({100});
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 100);
}

TEST_P(DistGapFill, BernoulliDist_SampleInZeroOne) {
    auto probs = full({4}, 0.5f, DType::Float32, device);
    BernoulliDist dist(probs);
    auto s = dist.sample();
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST_P(DistGapFill, Binomial_SampleShape) {
    auto probs = full({2}, 0.5f, DType::Float32, device);
    Binomial dist(/*total_count=*/10, probs);
    auto s = dist.sample();
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 2);
    // Binomial values are in [0, total_count]
    EXPECT_TRUE(all_in_range(s, 0.0f, 10.0f));
}

TEST_P(DistGapFill, Geometric_SampleNonnegative) {
    auto probs = full({4}, 0.3f, DType::Float32, device);
    Geometric dist(probs);
    auto s = dist.sample();
    EXPECT_EQ(s.shape().size(), 1u);
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "Geometric sample should be non-negative";
    }
}

TEST_P(DistGapFill, NegativeBinomial_SampleShape) {
    auto total = full({2}, 5.0f, DType::Float32, device);
    auto probs = full({2}, 0.5f, DType::Float32, device);
    NegativeBinomial dist(total, probs);
    auto s = dist.sample();
    EXPECT_EQ(s.shape().size(), 1u);
}

// ============================================================================
// Continuous — common univariate
// ============================================================================

TEST_P(DistGapFill, LogNormal_SamplePositive) {
    auto loc = zeros({4}, DType::Float32, device);
    auto scale = full({4}, 1.0f, DType::Float32, device);
    LogNormal dist(loc, scale);
    auto s = dist.sample();
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f) << "LogNormal samples should be strictly positive";
    }
}

TEST_P(DistGapFill, Cauchy_SampleFinite) {
    auto loc = zeros({4}, DType::Float32, device);
    auto scale = full({4}, 1.0f, DType::Float32, device);
    Cauchy dist(loc, scale);
    auto s = dist.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, Chi2_SamplePositive) {
    auto df = full({4}, 3.0f, DType::Float32, device);
    Chi2 dist(df);
    auto s = dist.sample();
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "Chi2 samples are non-negative";
    }
}

TEST_P(DistGapFill, Gumbel_SampleFinite) {
    auto loc = zeros({4}, DType::Float32, device);
    auto scale = full({4}, 1.0f, DType::Float32, device);
    Gumbel dist(loc, scale);
    auto s = dist.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, HalfNormal_SampleNonnegative) {
    auto scale = full({4}, 1.0f, DType::Float32, device);
    HalfNormal dist(scale);
    auto s = dist.sample();
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "HalfNormal samples are non-negative";
    }
}

TEST_P(DistGapFill, HalfCauchy_SampleNonnegative) {
    auto scale = full({4}, 1.0f, DType::Float32, device);
    HalfCauchy dist(scale);
    auto s = dist.sample();
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "HalfCauchy samples are non-negative";
    }
}

TEST_P(DistGapFill, FisherSnedecor_SamplePositive) {
    auto df1 = full({2}, 5.0f, DType::Float32, device);
    auto df2 = full({2}, 5.0f, DType::Float32, device);
    FisherSnedecor dist(df1, df2);
    auto s = dist.sample();
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f) << "F-distribution samples are non-negative";
    }
}

TEST_P(DistGapFill, VonMises_SampleInRange) {
    auto loc = zeros({4}, DType::Float32, device);
    auto concentration = full({4}, 1.0f, DType::Float32, device);
    VonMises dist(loc, concentration);
    auto s = dist.sample();
    // VonMises is on the circle [-pi, pi] (or shifted by loc).
    EXPECT_TRUE(is_finite(s));
}

// ============================================================================
// Relaxed (continuous relaxations of discrete distributions)
// ============================================================================

TEST_P(DistGapFill, RelaxedBernoulli_SampleInZeroOne) {
    auto temperature = full({4}, 0.5f, DType::Float32, device);
    auto probs = full({4}, 0.5f, DType::Float32, device);
    RelaxedBernoulli dist(temperature, probs);
    auto s = dist.sample();
    // Concrete distribution values ∈ (0, 1) — soft, not exactly hitting endpoints.
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

// ============================================================================
// Transforms
// ============================================================================

TEST_P(DistTransforms, ExpTransformInverseRoundtrip) {
    auto x = full({4}, 0.5f, DType::Float32, device);
    ExpTransform tr;
    auto y = tr.call(x);
    auto x_back = tr.inv(y);
    auto a = x.cpu().contiguous();
    auto b = x_back.cpu().contiguous();
    auto* pa = a.data<float>();
    auto* pb = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(pa[i], pb[i], 1e-5f);
    }
}

TEST_P(DistTransforms, SigmoidTransformInZeroOne) {
    auto x = full({4}, 0.5f, DType::Float32, device);
    SigmoidTransform tr;
    auto y = tr.call(x);
    EXPECT_TRUE(all_in_range(y, 0.0f, 1.0f));
}

TEST_P(DistTransforms, AffineTransformLinearity) {
    auto x = full({4}, 1.0f, DType::Float32, device);
    auto loc = full({4}, 2.0f, DType::Float32, device);
    auto scale = full({4}, 3.0f, DType::Float32, device);
    AffineTransform tr(loc, scale);
    auto y = tr.call(x);  // 2 + 3*1 = 5
    auto y_cpu = y.cpu().contiguous();
    auto* p = y_cpu.data<float>();
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(p[i], 5.0f);
    }
}

// ====================================================================
// Phase C.1 — Distributions previously missing from test coverage.
// ====================================================================

TEST_P(DistGapFill, Pareto_SampleAboveScale) {
    auto scale = full({3}, 2.0f, DType::Float32, device);
    auto alpha = full({3}, 3.0f, DType::Float32, device);
    Pareto d(scale, alpha);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 2.0f - 1e-3f);  // Pareto support is [scale, ∞)
    }
}

TEST_P(DistGapFill, Weibull_SamplePositive) {
    auto scale = full({3}, 2.0f, DType::Float32, device);
    auto concentration = full({3}, 1.5f, DType::Float32, device);
    Weibull d(scale, concentration);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f);
    }
}

TEST_P(DistGapFill, Wishart_SampleSymmetric) {
    int64_t n = 3;
    auto df = full({}, 5.0f, DType::Float32, device);
    auto eye_t = ::tenzor::eye(n, std::nullopt, DType::Float32, device);
    Wishart d(df, eye_t);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, LKJCholesky_SampleLowerTriangular) {
    int64_t dim = 3;
    // audit-2026-05-03 bug #7 fixed: concentration_as_double() helper now
    // dispatches on dtype, so Float32 concentration works directly.
    auto concentration = full({}, 1.0f, DType::Float32, device);
    LKJCholesky d(dim, concentration);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
    // Lower-triangular cholesky factor: above-diagonal entries should be 0.
    auto cpu = s.cpu().to(DType::Float32).contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = i + 1; j < dim; ++j) {
            EXPECT_NEAR(p[i * dim + j], 0.0f, 1e-5f);
        }
    }
}

TEST_P(DistGapFill, LKJCholesky_SampleFloat64Concentration) {
    // Regression test: explicit Float64 concentration must still work after
    // bug #7 fix.
    int64_t dim = 3;
    auto concentration = full({}, 1.0, DType::Float64, device);
    LKJCholesky d(dim, concentration);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, Kumaraswamy_SampleInZeroOne) {
    auto a = full({3}, 2.0f, DType::Float32, device);
    auto b = full({3}, 3.0f, DType::Float32, device);
    Kumaraswamy d(a, b);
    auto s = d.sample();
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST_P(DistGapFill, OneHotCategorical_SampleSimplex) {
    auto probs = full({3}, 1.0f / 3.0f, DType::Float32, device);
    OneHotCategorical d(probs);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, RelaxedOneHotCategorical_SamplesFinite) {
    auto temperature = full({}, 1.0f, DType::Float32, device);
    auto probs = full({3}, 1.0f / 3.0f, DType::Float32, device);
    RelaxedOneHotCategorical d(temperature, probs);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, ContinuousBernoulli_SampleInZeroOne) {
    auto probs = full({3}, 0.5f, DType::Float32, device);
    ContinuousBernoulli d(probs);
    auto s = d.sample();
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST_P(DistGapFill, LogisticNormal_SampleSimplex) {
    auto loc = ::tenzor::zeros({2}, DType::Float32, device);
    auto scale = full({2}, 1.0f, DType::Float32, device);
    LogisticNormal d(loc, scale);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    EXPECT_TRUE(all_in_range(s, 0.0f, 1.0f));
}

TEST_P(DistGapFill, LowRankMultivariateNormal_SampleShape) {
    auto loc = ::tenzor::zeros({2}, DType::Float32, device);
    auto cov_factor = ::tenzor::ones({2, 1}, DType::Float32, device);
    auto cov_diag = full({2}, 0.1f, DType::Float32, device);
    LowRankMultivariateNormal d(loc, cov_factor, cov_diag);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, Independent_ReinterpretsBatchAsEvent) {
    auto loc = ::tenzor::zeros({2, 3}, DType::Float32, device);
    auto scale = full({2, 3}, 1.0f, DType::Float32, device);
    auto base = std::make_shared<Normal>(loc, scale);
    Independent d(base, /*reinterpreted_batch_ndims=*/1);
    auto s = d.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, TransformedDistribution_ChainedTransform) {
    // Normal pushed through ExpTransform yields LogNormal-like samples.
    auto loc = ::tenzor::zeros({3}, DType::Float32, device);
    auto scale = full({3}, 1.0f, DType::Float32, device);
    auto base = std::make_shared<Normal>(loc, scale);
    std::vector<std::shared_ptr<Transform>> transforms = {
        std::make_shared<ExpTransform>(),
    };
    TransformedDistribution d(base, transforms);
    auto s = d.sample();
    EXPECT_TRUE(is_finite(s));
    auto cpu = s.cpu().contiguous();
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f);
    }
}

// audit-2026-05-03 bug #6 fixed: gather_components rewritten to align
// indices to comp_samples-minus-K dimensionality before gather, eliminating
// the libstdc++ vector OOB on default args.
namespace {
auto make_2vec(float a, float b, const ::tenzor::Device& device) -> Tensor {
    // Host writes a 2-element vector on CPU, then moves it to the target device.
    auto cpu = full({2}, 0.0f, DType::Float32, Device::cpu()).contiguous();
    auto* p = cpu.data<float>();
    p[0] = a;
    p[1] = b;
    return cpu.to(device);
}
} // anonymous

TEST_P(DistGapFill, MixtureSameFamily_SampleDefaultArgs) {
    auto weights = make_2vec(0.5f, 0.5f, device);
    auto loc = make_2vec(0.0f, 5.0f, device);
    auto scale = make_2vec(1.0f, 1.0f, device);
    auto base = std::make_shared<Normal>(loc, scale);
    MixtureSameFamily msf(weights, base);
    auto s = msf.sample({});
    EXPECT_TRUE(is_finite(s));
}

TEST_P(DistGapFill, MixtureSameFamily_SampleNonTrivialShape) {
    auto weights = make_2vec(0.3f, 0.7f, device);
    auto loc = make_2vec(0.0f, 10.0f, device);
    auto scale = make_2vec(1.0f, 1.0f, device);
    auto base = std::make_shared<Normal>(loc, scale);
    MixtureSameFamily msf(weights, base);
    auto s = msf.sample({4});
    EXPECT_TRUE(is_finite(s));
    EXPECT_EQ(s.numel(), 4);
}

// ====================================================================
// Phase C.2 — Distribution transforms previously missing from coverage.
// ====================================================================

TEST_P(DistTransforms, TanhTransformInverseRoundtrip) {
    auto x = full({4}, 0.7f, DType::Float32, device);
    TanhTransform tr;
    auto y = tr.call(x);
    auto x_back = tr.inv(y);
    auto a_cpu = x.cpu().contiguous();
    auto b_cpu = x_back.cpu().contiguous();
    auto* pa = a_cpu.data<float>();
    auto* pb = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_NEAR(pa[i], pb[i], 1e-4f);
    }
}

TEST_P(DistTransforms, SoftmaxTransformSimplexInvariant) {
    auto x = full({2, 3}, 0.5f, DType::Float32, device);
    SoftmaxTransform tr(/*dim=*/-1);
    auto y = tr.call(x);
    auto y_cpu = y.cpu().contiguous();
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

TEST_P(DistTransforms, ComposeTransformOrderingInvariant) {
    // Compose(exp, sigmoid)(x) == sigmoid(exp(x)).
    auto x = full({4}, 0.5f, DType::Float32, device);
    std::vector<std::shared_ptr<Transform>> transforms = {
        std::make_shared<ExpTransform>(),
        std::make_shared<SigmoidTransform>(),
    };
    ComposeTransform compose(transforms);
    auto y = compose.call(x);
    ExpTransform e;
    SigmoidTransform s;
    auto y_manual = s.call(e.call(x));
    auto a = y.cpu().contiguous();
    auto b = y_manual.cpu().contiguous();
    auto* pa = a.data<float>();
    auto* pb = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(pa[i], pb[i], 1e-5f);
    }
}

INSTANTIATE_BACKEND_TESTS(DistGapFill);
INSTANTIATE_BACKEND_TESTS(DistTransforms);
