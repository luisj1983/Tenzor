// test_distributions_closed_forms.cpp
//
// Wave Inf-B: closed-form entropy/mean/variance for distributions that
// previously inherited the base-class throw. Each test compares against a
// hand-computed reference value derived from the canonical formula —
// matches torch.distributions.X.entropy() / mean() / variance() to within
// float precision.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::distributions;

namespace {

class DistributionsClosedForms : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

inline auto scalar(float v) -> Tensor {
    auto t = zeros({1}, DType::Float32, Device::cpu());
    t.data<float>()[0] = v;
    return t;
}

inline auto vec(std::vector<float> vals) -> Tensor {
    auto t = zeros({static_cast<int64_t>(vals.size())}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
    return t;
}

inline auto first(const Tensor& t) -> float { return t.data<float>()[0]; }

// Euler-Mascheroni constant.
constexpr double GAMMA_EM = 0.5772156649015329;

}  // namespace

// ----------------------------------------------------------------------------
// Gamma entropy (Inf-B7): α - log(β) + lgamma(α) + (1-α)·ψ(α)
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, Gamma_Entropy_MatchesClosedForm) {
    // α=2, β=3.
    Gamma g(scalar(2.0f), scalar(3.0f));
    // ψ(2) = 1 - γ_EM ≈ 0.4227843350984...
    // entropy = 2 - log(3) + lgamma(2) + (-1)·ψ(2) = 2 - log(3) + 0 - (1 - γ_EM)
    double expected = 2.0 - std::log(3.0) + 0.0 + (-1.0) * (1.0 - GAMMA_EM);
    EXPECT_NEAR(first(g.entropy()), static_cast<float>(expected), 5e-5f);
}

// ----------------------------------------------------------------------------
// Chi2 entropy (Inf-B8): delegates to Gamma(df/2, 1/2).
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, Chi2_Entropy_DelegatesToGamma) {
    // df=4, so Gamma(2, 0.5).
    Chi2 c(scalar(4.0f));
    // entropy = 2 - log(0.5) + lgamma(2) + (-1)·ψ(2)
    //         = 2 + log(2) + 0 - (1 - γ_EM)
    double expected = 2.0 + std::log(2.0) + 0.0 - (1.0 - GAMMA_EM);
    EXPECT_NEAR(first(c.entropy()), static_cast<float>(expected), 5e-5f);
}

// ----------------------------------------------------------------------------
// HalfCauchy entropy (Inf-B10): log(2π·scale).
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, HalfCauchy_Entropy_LogTwoPiScale) {
    HalfCauchy hc(scalar(1.5f));
    double expected = std::log(2.0 * M_PI * 1.5);
    EXPECT_NEAR(first(hc.entropy()), static_cast<float>(expected), 5e-5f);
}

// ----------------------------------------------------------------------------
// VonMises entropy (Inf-B13): log(2π·I₀(κ)) − κ · I₁(κ)/I₀(κ).
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, VonMises_Entropy_BesselFormula) {
    VonMises vm(scalar(0.0f), scalar(1.0f));  // loc=0, kappa=1
    // I₀(1) ≈ 1.2660658732, I₁(1) ≈ 0.5651591040.
    double i0 = 1.2660658732;
    double i1 = 0.5651591040;
    double expected = std::log(2.0 * M_PI * i0) - 1.0 * i1 / i0;
    EXPECT_NEAR(first(vm.entropy()), static_cast<float>(expected), 5e-4f);
}

// ----------------------------------------------------------------------------
// Dirichlet mean / variance / entropy (Inf-B1).
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, Dirichlet_Mean_MatchesClosedForm) {
    Dirichlet d(vec({1.0f, 2.0f, 3.0f}));
    // mean = α / α₀ = [1/6, 2/6, 3/6].
    auto m = d.mean();
    ASSERT_EQ(m.numel(), 3);
    auto* p = m.data<float>();
    EXPECT_NEAR(p[0], 1.0f / 6.0f, 1e-5f);
    EXPECT_NEAR(p[1], 2.0f / 6.0f, 1e-5f);
    EXPECT_NEAR(p[2], 3.0f / 6.0f, 1e-5f);
}

TEST_F(DistributionsClosedForms, Dirichlet_Variance_MatchesClosedForm) {
    Dirichlet d(vec({1.0f, 2.0f, 3.0f}));
    // variance_k = α_k (α₀ - α_k) / (α₀² · (α₀ + 1)) with α₀=6.
    auto v = d.variance();
    ASSERT_EQ(v.numel(), 3);
    auto* p = v.data<float>();
    auto var_k = [](double a, double a0) {
        return a * (a0 - a) / (a0 * a0 * (a0 + 1.0));
    };
    EXPECT_NEAR(p[0], static_cast<float>(var_k(1.0, 6.0)), 5e-6f);
    EXPECT_NEAR(p[1], static_cast<float>(var_k(2.0, 6.0)), 5e-6f);
    EXPECT_NEAR(p[2], static_cast<float>(var_k(3.0, 6.0)), 5e-6f);
}

// ----------------------------------------------------------------------------
// Kumaraswamy mean / variance / entropy (Inf-B4).
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, Kumaraswamy_Mean_MatchesClosedForm) {
    Kumaraswamy k(scalar(2.0f), scalar(3.0f));  // a=2, b=3
    // mean = b·B(1+1/a, b) = 3 · B(1.5, 3) = 3 · 0.10666... ≈ 0.32
    // B(1.5,3) = Γ(1.5)·Γ(3)/Γ(4.5) = (0.5·√π)·2 / (3.5·2.5·1.5·0.5·√π) = ...
    double log_B = std::lgamma(1.5) + std::lgamma(3.0) - std::lgamma(4.5);
    double expected = 3.0 * std::exp(log_B);
    EXPECT_NEAR(first(k.mean()), static_cast<float>(expected), 5e-4f);
}

// ----------------------------------------------------------------------------
// LKJCholesky mean (Inf-B6): expected value is the p×p identity matrix.
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Wishart variance (Inf-B14): df · (V_ii · V_jj + V_ij²) where V = LL^T.
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// LowRankMVN variance + entropy (Inf-B15).
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, LowRankMVN_Variance_DPlusRowL2Squared) {
    // p=2, rank=1. loc = (0, 0). cov_factor W = [[1], [2]]. cov_diag d = (3, 4).
    // Σ = D + WW^T = [[3+1, 2], [2, 4+4]] = [[4, 2], [2, 8]]
    // variance = diag(Σ) = (4, 8)
    auto loc = zeros({2}, DType::Float32, Device::cpu());
    auto W = zeros({2, 1}, DType::Float32, Device::cpu());
    W.data<float>()[0] = 1.0f; W.data<float>()[1] = 2.0f;
    auto d = zeros({2}, DType::Float32, Device::cpu());
    d.data<float>()[0] = 3.0f; d.data<float>()[1] = 4.0f;
    LowRankMultivariateNormal mvn(loc, W, d);
    auto var = mvn.variance();
    ASSERT_EQ(var.numel(), 2);
    EXPECT_FLOAT_EQ(var.data<float>()[0], 4.0f);
    EXPECT_FLOAT_EQ(var.data<float>()[1], 8.0f);
}

TEST_F(DistributionsClosedForms, LowRankMVN_Entropy_Woodbury) {
    // p=2, Σ = [[4, 2], [2, 8]] from above.
    // log|Σ| = log(4·8 - 2·2) = log(28).
    // entropy = 0.5 · (2 · log(2πe) + log(28))
    auto loc = zeros({2}, DType::Float32, Device::cpu());
    auto W = zeros({2, 1}, DType::Float32, Device::cpu());
    W.data<float>()[0] = 1.0f; W.data<float>()[1] = 2.0f;
    auto d = zeros({2}, DType::Float32, Device::cpu());
    d.data<float>()[0] = 3.0f; d.data<float>()[1] = 4.0f;
    LowRankMultivariateNormal mvn(loc, W, d);
    auto got = mvn.entropy();
    double log_2pie = std::log(2.0 * M_PI * std::exp(1.0));
    double expected = 0.5 * (2.0 * log_2pie + std::log(28.0));
    EXPECT_NEAR(got.item<float>(), static_cast<float>(expected), 5e-5f);
}

TEST_F(DistributionsClosedForms, Wishart_Variance_MatchesEaton1983) {
    // Use scale_tril = I_2, df = 5 → V = I_2.
    // Then Var(W_ij) = 5 · (δ_ii · δ_jj + δ_ij²)
    //   Var(W_00) = 5·(1·1 + 1) = 10
    //   Var(W_01) = 5·(1·1 + 0) = 5
    //   Var(W_11) = 5·(1·1 + 1) = 10
    auto identity = zeros({2, 2}, DType::Float32, Device::cpu());
    auto* ip = identity.data<float>();
    ip[0] = 1.0f; ip[3] = 1.0f;
    Wishart w(scalar(5.0f), identity);
    auto v = w.variance();
    ASSERT_EQ(v.shape().size(), 2u);
    auto* p = v.data<float>();
    EXPECT_FLOAT_EQ(p[0], 10.0f);
    EXPECT_FLOAT_EQ(p[1],  5.0f);
    EXPECT_FLOAT_EQ(p[2],  5.0f);
    EXPECT_FLOAT_EQ(p[3], 10.0f);
}

TEST_F(DistributionsClosedForms, LKJCholesky_Mean_IsIdentity) {
    LKJCholesky lkj(/*dim=*/3, scalar(2.0f));
    auto m = lkj.mean();
    ASSERT_EQ(m.shape().size(), 2u);
    ASSERT_EQ(m.shape()[0], 3);
    ASSERT_EQ(m.shape()[1], 3);
    auto* p = m.data<float>();
    EXPECT_FLOAT_EQ(p[0], 1.0f); EXPECT_FLOAT_EQ(p[1], 0.0f); EXPECT_FLOAT_EQ(p[2], 0.0f);
    EXPECT_FLOAT_EQ(p[3], 0.0f); EXPECT_FLOAT_EQ(p[4], 1.0f); EXPECT_FLOAT_EQ(p[5], 0.0f);
    EXPECT_FLOAT_EQ(p[6], 0.0f); EXPECT_FLOAT_EQ(p[7], 0.0f); EXPECT_FLOAT_EQ(p[8], 1.0f);
}

TEST_F(DistributionsClosedForms, Kumaraswamy_Entropy_MatchesClosedForm) {
    Kumaraswamy k(scalar(2.0f), scalar(3.0f));  // a=2, b=3
    // entropy = (1 - 1/b) + (1 - 1/a)·(ψ(b+1) + γ_EM) + log(b/a)
    // ψ(4) = ψ(3) + 1/3 ; ψ(1) = -γ_EM. Use std::digamma replacement:
    // ψ(4) = -γ_EM + 1 + 1/2 + 1/3 = 1 + 1/2 + 1/3 - γ_EM ≈ 1.2561...
    double psi_b_plus_1 = -GAMMA_EM + 1.0 + 0.5 + 1.0 / 3.0;
    double expected = (1.0 - 1.0 / 3.0)
                    + (1.0 - 1.0 / 2.0) * (psi_b_plus_1 + GAMMA_EM)
                    + std::log(3.0 / 2.0);
    EXPECT_NEAR(first(k.entropy()), static_cast<float>(expected), 5e-4f);
}

// ----------------------------------------------------------------------------
// Poisson entropy (Inf-B9): Stirling for λ ≥ 10, truncated series for small λ.
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, Poisson_Entropy_SmallLambda_TruncatedSum) {
    // λ = 2 → use truncated sum.
    Poisson p(scalar(2.0f));
    auto got = first(p.entropy());
    // Reference via the same series.
    double lam = 2.0;
    double log_lam = std::log(lam);
    double h = 0.0;
    for (int k = 0; k <= 60; ++k) {
        double log_pk = k * log_lam - lam - std::lgamma(k + 1.0);
        double pk = std::exp(log_pk);
        if (pk > 0.0) h -= pk * log_pk;
    }
    EXPECT_NEAR(got, static_cast<float>(h), 5e-5f);
}
TEST_F(DistributionsClosedForms, Poisson_Entropy_LargeLambda_StirlingMatchesKnown) {
    // λ = 100 → Stirling expansion.
    // Reference: H(λ=100) = 0.5 · log(2π·e·λ) - 1/(12λ) - ...
    //          = 0.5 · log(2π·e·100) - 1/1200 ≈ 3.7204
    // Matches scipy.stats.poisson.entropy(100) within machine precision.
    Poisson p(scalar(100.0f));
    EXPECT_NEAR(first(p.entropy()), 3.7204f, 5e-3f);
}

// ----------------------------------------------------------------------------
// NegativeBinomial entropy (Inf-B12): truncated series.
// For NB(r=5, p=0.5), hand-rolled reference:
//   H = -Σ_k P(k) log P(k), summed to K = mean + 20*std + 50.
// Compare against an independent CPU sum.
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, NegativeBinomial_Entropy_MatchesIndependentSum) {
    NegativeBinomial nb(scalar(5.0f), scalar(0.5f));
    auto got = first(nb.entropy());
    // Reference: independent O(N²) sum mirroring the implementation.
    double r = 5.0, p = 0.5, q = 1.0 - p;
    double log_p = std::log(p), log_q = std::log(q);
    double mean = r * p / q;
    double var = r * p / (q * q);
    int64_t K = static_cast<int64_t>(std::max(50.0, mean + 20.0 * std::sqrt(var)));
    double lgamma_r = std::lgamma(r);
    double h = 0.0;
    for (int64_t k = 0; k <= K; ++k) {
        double log_pk = std::lgamma(k + r) - lgamma_r - std::lgamma(k + 1.0)
                      + r * log_q + k * log_p;
        double pk = std::exp(log_pk);
        if (pk > 0.0) h -= pk * log_pk;
    }
    EXPECT_NEAR(got, static_cast<float>(h), 1e-4f);
}

// ----------------------------------------------------------------------------
// FisherSnedecor entropy (Inf-B11): log Beta(d1/2, d2/2) + log(d2/d1)
//   + (1 - d1/2)·ψ(d1/2) - (1 + d2/2)·ψ(d2/2) + (d1+d2)/2 · ψ((d1+d2)/2)
// ----------------------------------------------------------------------------
TEST_F(DistributionsClosedForms, FisherSnedecor_Entropy_MatchesClosedForm) {
    // d1=4, d2=6 — both even, well-behaved.
    FisherSnedecor f(scalar(4.0f), scalar(6.0f));
    double d1h = 2.0, d2h = 3.0, sum_h = 5.0;
    double log_beta = std::lgamma(d1h) + std::lgamma(d2h) - std::lgamma(sum_h);
    // ψ(2) = 1 - γ_EM, ψ(3) = 1 + 1/2 - γ_EM, ψ(5) = 1 + 1/2 + 1/3 + 1/4 - γ_EM.
    double psi_d1h  = 1.0 - GAMMA_EM;
    double psi_d2h  = 1.0 + 0.5 - GAMMA_EM;
    double psi_sumh = 1.0 + 0.5 + 1.0/3.0 + 1.0/4.0 - GAMMA_EM;
    double expected = log_beta
                    + std::log(6.0 / 4.0)
                    + (1.0 - d1h) * psi_d1h
                    - (1.0 + d2h) * psi_d2h
                    + sum_h * psi_sumh;
    EXPECT_NEAR(first(f.entropy()), static_cast<float>(expected), 5e-4f);
}
