"""Pytest tests for the Python tenzor.distributions namespace (Phase 11).

Tests the pure-Python distribution classes backed by numpy/scipy.
Does NOT require tenzor_core to be built — imports directly from
python/tenzor/distributions/*.

Each test:
  - Constructs the distribution with typical parameters.
  - Calls sample(), verifies shape, dtype, and domain constraints.
  - Checks log_prob() against known values or numerical sanity.
  - Checks closed-form properties (mean, variance, entropy) where available.
  - Checks cdf/icdf roundtrip where both are implemented.
"""
import math
import sys
import os
import pytest
import numpy as np

# Import the distributions package directly from source, bypassing the
# top-level tenzor __init__.py which requires tenzor_core (C++ extension).
# Insert the tenzor/distributions directory as a package root.
_src_tenzor = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'tenzor')
_src_tenzor = os.path.abspath(_src_tenzor)
# Make tenzor.distributions importable as a standalone package by adding
# the distributions directory itself and synthesising the parent package.
import types
if 'tenzor' not in sys.modules:
    _tenzor_pkg = types.ModuleType('tenzor')
    _tenzor_pkg.__path__ = [_src_tenzor]
    _tenzor_pkg.__package__ = 'tenzor'
    sys.modules['tenzor'] = _tenzor_pkg
if _src_tenzor not in sys.path:
    sys.path.insert(0, os.path.dirname(_src_tenzor))  # add python/ dir

from tenzor.distributions import (
    Normal, Uniform, Bernoulli, Categorical, Multinomial,
    Poisson, Exponential, Gamma, Beta,
    Cauchy, LogNormal, Chi2, Geometric,
    Dirichlet, Binomial, Weibull, HalfNormal,
    VonMises, Laplace, StudentT, NegativeBinomial,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allclose(a, b, atol=1e-4):
    return np.allclose(_f(a), _f(b), atol=atol)


def _f(x):
    """Convert to float array."""
    return np.asarray(x, dtype=np.float64)


# ===========================================================================
# Normal
# ===========================================================================

class TestNormal:
    def test_log_prob_standard(self):
        n = Normal(0.0, 1.0)
        lp = n.log_prob(0.0)
        assert allclose(lp, -0.5 * math.log(2 * math.pi)), f"got {lp}"

    def test_log_prob_at_mean(self):
        n = Normal(2.0, 1.5)
        # At x=mean, log_prob = -0.5*log(2*pi) - log(scale)
        expected = -0.5 * math.log(2 * math.pi) - math.log(1.5)
        assert allclose(n.log_prob(2.0), expected)

    def test_cdf_at_mean(self):
        n = Normal(0.0, 1.0)
        assert allclose(n.cdf(0.0), 0.5)

    def test_icdf_roundtrip(self):
        n = Normal(1.0, 2.0)
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert np.allclose(n.cdf(n.icdf(p)), p, atol=1e-5)

    def test_entropy(self):
        n = Normal(0.0, 1.0)
        # H = 0.5 * log(2*pi*e) ≈ 1.4189
        expected = 0.5 * math.log(2 * math.pi * math.e)
        assert allclose(n.entropy(), expected)

    def test_mean_variance(self):
        n = Normal(3.0, 2.0)
        assert allclose(n.mean, 3.0)
        assert allclose(n.variance, 4.0)

    def test_sample_shape(self):
        n = Normal(0.0, 1.0)
        s = n.sample((100,))
        assert s.shape == (100,)
        assert np.isfinite(s).all()

    def test_rsample(self):
        n = Normal(0.0, 1.0)
        s = n.rsample((50,))
        assert s.shape == (50,)


# ===========================================================================
# Uniform
# ===========================================================================

class TestUniform:
    def test_sample_in_range(self):
        u = Uniform(1.0, 3.0)
        s = u.sample((500,))
        assert np.all(s >= 1.0) and np.all(s <= 3.0)

    def test_log_prob_in_support(self):
        u = Uniform(0.0, 4.0)
        # log(1/(4-0)) = -log(4) ≈ -1.386
        assert allclose(u.log_prob(2.0), -math.log(4.0))

    def test_log_prob_outside_support(self):
        u = Uniform(0.0, 1.0)
        assert u.log_prob(-0.1) == -np.inf
        assert u.log_prob(1.1) == -np.inf

    def test_mean_variance(self):
        u = Uniform(2.0, 6.0)
        assert allclose(u.mean, 4.0)
        assert allclose(u.variance, (6.0 - 2.0) ** 2 / 12.0)

    def test_cdf_icdf(self):
        u = Uniform(0.0, 1.0)
        p = np.array([0.0, 0.5, 1.0])
        assert np.allclose(u.cdf(p), p, atol=1e-6)
        assert np.allclose(u.icdf(p), p, atol=1e-6)


# ===========================================================================
# Bernoulli
# ===========================================================================

class TestBernoulli:
    def test_log_prob(self):
        b = Bernoulli(0.7)
        # log P(1) = log(0.7)
        assert allclose(b.log_prob(1.0), math.log(0.7))
        # log P(0) = log(0.3)
        assert allclose(b.log_prob(0.0), math.log(0.3))

    def test_mean_variance(self):
        b = Bernoulli(0.4)
        assert allclose(b.mean, 0.4)
        assert allclose(b.variance, 0.4 * 0.6)

    def test_sample_binary(self):
        b = Bernoulli(0.5)
        s = b.sample((200,))
        assert set(np.unique(s)).issubset({0.0, 1.0})

    def test_entropy(self):
        b = Bernoulli(0.5)
        # Maximum entropy = log(2) ≈ 0.693
        assert allclose(b.entropy(), math.log(2.0))


# ===========================================================================
# Categorical
# ===========================================================================

class TestCategorical:
    def test_entropy_uniform(self):
        # Uniform over 4 classes: entropy = log(4)
        c = Categorical(np.array([0.25, 0.25, 0.25, 0.25]))
        assert allclose(c.entropy(), math.log(4.0))

    def test_sample_valid_class(self):
        c = Categorical(np.array([0.1, 0.3, 0.6]))
        s = c.sample((200,))
        assert np.all((s >= 0) & (s <= 2))

    def test_probs_normalized(self):
        c = Categorical(np.array([2.0, 3.0, 5.0]))
        assert allclose(c.probs.sum(), 1.0)


# ===========================================================================
# Multinomial
# ===========================================================================

class TestMultinomial:
    def test_mean(self):
        m = Multinomial(10, np.array([0.2, 0.3, 0.5]))
        expected = np.array([2.0, 3.0, 5.0])
        assert np.allclose(m.mean, expected)

    def test_log_prob(self):
        m = Multinomial(3, np.array([1 / 3, 1 / 3, 1 / 3]))
        lp = m.log_prob(np.array([1.0, 1.0, 1.0]))
        # log(3!/(1!1!1!) * (1/3)^3) = log(6) + 3*log(1/3) = log(6) - 3*log(3)
        expected = math.log(6) - 3 * math.log(3)
        assert allclose(lp, expected)


# ===========================================================================
# Poisson
# ===========================================================================

class TestPoisson:
    def test_mean_equals_rate(self):
        p = Poisson(5.0)
        assert allclose(p.mean, 5.0)
        assert allclose(p.variance, 5.0)

    def test_log_prob(self):
        p = Poisson(2.0)
        # P(k=1) = e^-2 * 2^1 / 1! = 2 * e^-2
        lp = p.log_prob(1.0)
        assert allclose(lp, math.log(2.0) - 2.0)

    def test_sample_nonneg(self):
        p = Poisson(3.0)
        s = p.sample((100,))
        assert np.all(s >= 0)


# ===========================================================================
# Exponential
# ===========================================================================

class TestExponential:
    def test_mean_variance(self):
        e = Exponential(2.0)
        assert allclose(e.mean, 0.5)
        assert allclose(e.variance, 0.25)

    def test_log_prob(self):
        e = Exponential(1.0)
        # log p(1) = log(1) - 1 = -1
        assert allclose(e.log_prob(1.0), -1.0)

    def test_cdf_icdf(self):
        e = Exponential(1.0)
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(e.cdf(e.icdf(p)), p, atol=1e-5)

    def test_entropy(self):
        e = Exponential(1.0)
        # H = 1 - log(1) = 1
        assert allclose(e.entropy(), 1.0)

    def test_sample_positive(self):
        e = Exponential(0.5)
        s = e.sample((100,))
        assert np.all(s > 0)


# ===========================================================================
# Gamma
# ===========================================================================

class TestGamma:
    def test_mean_variance(self):
        g = Gamma(3.0, 2.0)
        assert allclose(g.mean, 1.5)
        assert allclose(g.variance, 0.75)

    def test_log_prob(self):
        g = Gamma(1.0, 1.0)
        # Gamma(1, 1) = Exponential(1): log p(x) = -x
        assert allclose(g.log_prob(1.0), -1.0)

    def test_sample_positive(self):
        g = Gamma(2.0, 1.0)
        s = g.sample((100,))
        assert np.all(s > 0)


# ===========================================================================
# Beta
# ===========================================================================

class TestBeta:
    def test_mean(self):
        # Beta(2, 3): mean = 2/(2+3) = 0.4
        b = Beta(2.0, 3.0)
        assert allclose(b.mean, 0.4)

    def test_variance(self):
        b = Beta(2.0, 3.0)
        # Var = 2*3 / (5^2 * 6) = 6/150 = 0.04
        assert allclose(b.variance, 6.0 / 150.0)

    def test_sample_in_range(self):
        b = Beta(1.0, 1.0)  # Uniform(0, 1)
        s = b.sample((200,))
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_cdf_at_mean(self):
        b = Beta(1.0, 1.0)
        assert allclose(b.cdf(0.5), 0.5)


# ===========================================================================
# Cauchy
# ===========================================================================

class TestCauchy:
    def test_log_prob(self):
        c = Cauchy(0.0, 1.0)
        # log p(0) = -log(pi)
        assert allclose(c.log_prob(0.0), -math.log(math.pi))

    def test_cdf_at_loc(self):
        c = Cauchy(0.0, 1.0)
        assert allclose(c.cdf(0.0), 0.5)

    def test_icdf_roundtrip(self):
        c = Cauchy(1.0, 2.0)
        p = np.array([0.1, 0.3, 0.7, 0.9])
        assert np.allclose(c.cdf(c.icdf(p)), p, atol=1e-5)

    def test_mean_raises(self):
        c = Cauchy(0.0, 1.0)
        with pytest.raises(NotImplementedError):
            _ = c.mean

    def test_sample_finite(self):
        c = Cauchy(0.0, 1.0)
        s = c.sample((100,))
        assert s.shape == (100,)
        # Most samples finite (Cauchy has heavy tails)
        assert np.sum(np.isfinite(s)) > 90


# ===========================================================================
# LogNormal
# ===========================================================================

class TestLogNormal:
    def test_mean(self):
        # LogNormal(0, 1): mean = exp(0 + 0.5) = sqrt(e)
        ln = LogNormal(0.0, 1.0)
        assert allclose(ln.mean, math.exp(0.5))

    def test_sample_positive(self):
        ln = LogNormal(0.0, 1.0)
        s = ln.sample((100,))
        assert np.all(s > 0)

    def test_cdf_at_exp_loc(self):
        ln = LogNormal(0.0, 1.0)
        # P(X <= exp(0)) = P(X <= 1) = 0.5
        assert allclose(ln.cdf(1.0), 0.5)


# ===========================================================================
# Chi2
# ===========================================================================

class TestChi2:
    def test_mean_variance(self):
        c = Chi2(4.0)
        assert allclose(c.mean, 4.0)
        assert allclose(c.variance, 8.0)

    def test_sample_positive(self):
        c = Chi2(2.0)
        s = c.sample((100,))
        assert np.all(s > 0)

    def test_log_prob(self):
        c = Chi2(2.0)
        # Chi2(2) = Exponential(1/2): p(x) = 0.5*exp(-x/2)
        # log p(1) = log(0.5) - 0.5 = -log(2) - 0.5
        lp = c.log_prob(1.0)
        assert allclose(lp, -math.log(2.0) - 0.5, atol=1e-4)


# ===========================================================================
# Geometric
# ===========================================================================

class TestGeometric:
    def test_mean(self):
        g = Geometric(0.5)
        assert allclose(g.mean, 1.0)  # (1-0.5)/0.5 = 1.0

    def test_log_prob(self):
        g = Geometric(0.5)
        # P(k=0) = p = 0.5
        assert allclose(g.log_prob(0.0), math.log(0.5))

    def test_sample_nonneg(self):
        g = Geometric(0.3)
        s = g.sample((100,))
        assert np.all(s >= 0)


# ===========================================================================
# Dirichlet  (new)
# ===========================================================================

class TestDirichlet:
    def test_mean(self):
        d = Dirichlet(np.array([1.0, 2.0, 3.0]))
        expected = np.array([1 / 6, 2 / 6, 3 / 6])
        assert np.allclose(d.mean, expected, atol=1e-6)

    def test_sample_sum_to_one(self):
        d = Dirichlet(np.array([2.0, 3.0, 5.0]))
        s = d.sample((100,))
        assert s.shape == (100, 3)
        assert np.allclose(s.sum(axis=1), 1.0, atol=1e-6)

    def test_sample_non_negative(self):
        d = Dirichlet(np.array([1.0, 1.0, 1.0]))
        s = d.sample((50,))
        assert np.all(s >= 0)

    def test_log_prob(self):
        d = Dirichlet(np.array([1.0, 1.0, 1.0]))
        # Symmetric Dirichlet([1,1,1]) is Uniform on simplex
        # log p = lgamma(3) - 3*lgamma(1) + 0*log(x) = log(2) - 0 = log(2)
        x = np.array([1 / 3, 1 / 3, 1 / 3])
        lp = d.log_prob(x)
        assert allclose(lp, math.log(2.0))


# ===========================================================================
# Binomial  (new)
# ===========================================================================

class TestBinomial:
    def test_mean_variance(self):
        b = Binomial(10, 0.4)
        assert allclose(b.mean, 4.0)
        assert allclose(b.variance, 2.4)

    def test_sample_in_range(self):
        b = Binomial(8, 0.5)
        s = b.sample((200,))
        assert np.all(s >= 0) and np.all(s <= 8)

    def test_log_prob(self):
        # P(k=0; n=1, p=0.3) = 0.7
        b = Binomial(1, 0.3)
        lp = b.log_prob(0.0)
        assert allclose(lp, math.log(0.7))


# ===========================================================================
# Weibull  (new)
# ===========================================================================

class TestWeibull:
    def test_sample_positive(self):
        w = Weibull(scale=2.0, concentration=1.5)
        s = w.sample((100,))
        assert np.all(s > 0)

    def test_mean(self):
        # Weibull(lambda=2, k=1): mean = lambda * Gamma(2) = 2.0 * 1.0 = 2.0
        from scipy.special import gamma as gamma_fn
        w = Weibull(scale=2.0, concentration=1.0)
        assert allclose(w.mean, 2.0 * gamma_fn(2.0))

    def test_empirical_mean(self):
        # Weibull(lambda=2, k=1.5): mean ≈ 2 * Gamma(1 + 1/1.5) ≈ 1.8055
        np.random.seed(123)
        w = Weibull(scale=2.0, concentration=1.5)
        s = w.sample((10000,))
        assert abs(s.mean() - 1.8055) < 0.15

    def test_cdf_icdf(self):
        w = Weibull(scale=1.0, concentration=2.0)
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(w.cdf(w.icdf(p)), p, atol=1e-5)


# ===========================================================================
# HalfNormal  (new)
# ===========================================================================

class TestHalfNormal:
    def test_sample_nonneg(self):
        h = HalfNormal(scale=1.0)
        s = h.sample((200,))
        assert np.all(s >= 0)

    def test_mean(self):
        h = HalfNormal(scale=1.0)
        assert allclose(h.mean, math.sqrt(2 / math.pi))

    def test_empirical_mean(self):
        np.random.seed(42)
        h = HalfNormal(scale=2.0)
        s = h.sample((10000,))
        expected = 2.0 * math.sqrt(2 / math.pi)
        assert abs(s.mean() - expected) < 0.1

    def test_log_prob_negative_is_neginf(self):
        h = HalfNormal(scale=1.0)
        assert h.log_prob(-1.0) == -np.inf

    def test_cdf_at_zero(self):
        h = HalfNormal(scale=1.0)
        assert allclose(h.cdf(0.0), 0.0)


# ===========================================================================
# VonMises  (new)
# ===========================================================================

class TestVonMises:
    def test_sample_in_range(self):
        v = VonMises(loc=0.0, concentration=2.0)
        s = v.sample((100,))
        assert np.all(s >= -math.pi - 1e-6) and np.all(s <= math.pi + 1e-6)

    def test_empirical_mean(self):
        np.random.seed(42)
        v = VonMises(loc=0.0, concentration=5.0)
        s = v.sample((10000,))
        assert abs(s.mean()) < 0.05

    def test_entropy_uniform(self):
        # kappa=0 → uniform on circle, H = log(2*pi)
        v = VonMises(loc=0.0, concentration=1e-7)
        assert allclose(v.entropy(), math.log(2 * math.pi), atol=1e-3)

    def test_cdf_at_loc(self):
        # Audit E.5: cdf(mu; mu, kappa) == 0.5 by symmetry.
        v = VonMises(loc=0.5, concentration=2.0)
        assert allclose(v.cdf(0.5), 0.5, atol=1e-6)

    def test_cdf_icdf_roundtrip(self):
        v = VonMises(loc=0.0, concentration=3.0)
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert np.allclose(v.cdf(v.icdf(p)), p, atol=1e-5)


# ===========================================================================
# Laplace  (new)
# ===========================================================================

class TestLaplace:
    def test_mean_is_loc(self):
        la = Laplace(loc=3.0, scale=1.0)
        assert allclose(la.mean, 3.0)

    def test_empirical_mean(self):
        np.random.seed(42)
        la = Laplace(loc=3.0, scale=1.0)
        s = la.sample((10000,))
        assert abs(s.mean() - 3.0) < 0.1

    def test_log_prob_at_loc(self):
        la = Laplace(loc=0.0, scale=1.0)
        # log p(0) = -log(2)
        assert allclose(la.log_prob(0.0), -math.log(2.0))

    def test_cdf_at_loc(self):
        la = Laplace(loc=0.0, scale=1.0)
        assert allclose(la.cdf(0.0), 0.5)

    def test_icdf_roundtrip(self):
        la = Laplace(loc=1.0, scale=2.0)
        p = np.array([0.1, 0.3, 0.7, 0.9])
        assert np.allclose(la.cdf(la.icdf(p)), p, atol=1e-5)

    def test_entropy(self):
        la = Laplace(loc=0.0, scale=2.0)
        # H = 1 + log(2*scale)
        assert allclose(la.entropy(), 1.0 + math.log(4.0))


# ===========================================================================
# StudentT  (new)
# ===========================================================================

class TestStudentT:
    def test_mean_for_large_df(self):
        t = StudentT(df=30.0, loc=0.0, scale=1.0)
        np.random.seed(42)
        s = t.sample((10000,))
        assert abs(s.mean()) < 0.1

    def test_location_shifted(self):
        t = StudentT(df=5.0, loc=3.0, scale=1.0)
        np.random.seed(42)
        s = t.sample((10000,))
        assert abs(s.mean() - 3.0) < 0.15

    def test_log_prob(self):
        t = StudentT(df=1.0)  # = Cauchy(0, 1)
        # log p(0) = log(Gamma(1)) - 0.5*log(pi) - log(Gamma(0.5))
        #          = -log(pi) ≈ -1.1447
        lp = float(t.log_prob(0.0))
        assert allclose(lp, -math.log(math.pi))

    def test_cdf_icdf(self):
        t = StudentT(df=10.0)
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(t.cdf(t.icdf(p)), p, atol=1e-5)


# ===========================================================================
# NegativeBinomial  (new)
# ===========================================================================

class TestNegativeBinomial:
    def test_mean(self):
        nb = NegativeBinomial(total_count=5.0, probs=0.4)
        # mean = r * p / (1 - p) = 5 * 0.4 / 0.6 ≈ 3.333
        assert allclose(nb.mean, 5.0 * 0.4 / 0.6)

    def test_sample_nonneg(self):
        nb = NegativeBinomial(total_count=3.0, probs=0.5)
        s = nb.sample((100,))
        assert np.all(s >= 0)

    def test_empirical_mean(self):
        np.random.seed(42)
        nb = NegativeBinomial(total_count=5.0, probs=0.4)
        s = nb.sample((10000,))
        assert abs(s.mean() - 3.333) < 0.5

    def test_cdf_monotone_and_in_unit_interval(self):
        # Audit E.5: cdf must be monotone non-decreasing and in [0, 1].
        nb = NegativeBinomial(total_count=5.0, probs=0.4)
        ks = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
        c = nb.cdf(ks)
        assert np.all(c >= 0.0) and np.all(c <= 1.0)
        assert np.all(np.diff(c) >= -1e-12)
        # As k grows, cdf -> 1.
        assert nb.cdf(1000.0) > 0.999

    def test_cdf_below_zero(self):
        # k < 0 has zero probability mass.
        nb = NegativeBinomial(total_count=3.0, probs=0.5)
        assert allclose(nb.cdf(-1.0), 0.0)

    def test_icdf_matches_cdf(self):
        # cdf(icdf(q)) >= q for the smallest integer support point.
        nb = NegativeBinomial(total_count=4.0, probs=0.3)
        qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        ks = nb.icdf(qs)
        # icdf returns integers (failure counts).
        assert np.allclose(ks, np.round(ks))
        # cdf(k) >= q and cdf(k-1) < q.
        assert np.all(nb.cdf(ks) >= qs - 1e-9)
        assert np.all(nb.cdf(ks - 1.0) < qs + 1e-9)


# ===========================================================================
# A.9 — Audit regressions: batched math bugs in pure-Python distributions.
# ===========================================================================

class TestCategoricalBatchedLogProb:
    """Audit item A.9.a — Categorical.log_prob with batched probs+value
    returned the wrong shape because `lp[..., value]` advanced-indexes the
    last axis with the full `value` array instead of one-per-batch.
    Expected: with probs (B, K) and value (B,), result has shape (B,)."""

    def test_batched_shape(self):
        # B=4 batch, K=3 classes
        probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.3, 0.3, 0.4],
                [0.2, 0.5, 0.3],
            ]
        )
        value = np.array([0, 1, 2, 1], dtype=np.int64)
        cat = Categorical(probs)
        lp = cat.log_prob(value)
        assert lp.shape == (4,), f"expected (4,), got {lp.shape}"

    def test_batched_values(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        value = np.array([0, 1], dtype=np.int64)
        cat = Categorical(probs)
        lp = cat.log_prob(value)
        expected = np.log(np.array([0.5, 0.6]))
        assert np.allclose(lp, expected, atol=1e-6), (
            f"expected {expected}, got {lp}"
        )

    def test_unbatched_still_works(self):
        # Sanity: the unbatched K-only path used by existing tests.
        cat = Categorical(np.array([0.5, 0.3, 0.2]))
        lp = cat.log_prob(np.array(2, dtype=np.int64))
        assert allclose(lp, math.log(0.2))

    def test_unbatched_with_1d_value(self):
        # Audit H.3 regression: Categorical(probs=(K,)).log_prob(value=(N,))
        # previously threw "indices and arr must have the same number of
        # dimensions" because the take_along_axis path required
        # value.shape == lp.shape[:-1] which is () for an unbatched K-only
        # distribution.  Fix: special-case `lp.ndim == 1` to direct-index.
        cat = Categorical(np.array([0.4, 0.3, 0.3]))
        lp = cat.log_prob(np.array([0, 1, 2], dtype=np.int64))
        expected = np.log(np.array([0.4, 0.3, 0.3]))
        assert np.allclose(lp, expected, atol=1e-6), (
            f"expected {expected}, got {lp}"
        )

        # Single-element 1-D array.
        lp = cat.log_prob(np.array([1], dtype=np.int64))
        assert lp.shape == (1,)
        assert np.allclose(lp, [math.log(0.3)], atol=1e-6)


class TestDirichletBatchedSample:
    """Audit item A.9.b — Dirichlet.sample(()) with batched concentration
    raised because np.random.dirichlet accepts only 1-D alpha.  Build
    samples via the Gamma-normalisation trick so any leading batch shape
    is supported."""

    def test_batched_concentration_sample_shape(self):
        # concentration shape (B=2, K=3) ⇒ sample default shape (2, 3)
        conc = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        d = Dirichlet(conc)
        s = d.sample()
        assert s.shape == (2, 3), f"expected (2, 3), got {s.shape}"
        # Every row must sum to 1 and be non-negative.
        assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-6)
        assert np.all(s >= 0)

    def test_batched_concentration_sample_with_extra_leading(self):
        # Caller adds a leading sample dim ⇒ shape (N=5, B=2, K=3).
        conc = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        d = Dirichlet(conc)
        s = d.sample((5,))
        assert s.shape == (5, 2, 3), f"expected (5, 2, 3), got {s.shape}"
        assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-6)


class TestBinomialEntropyExact:
    """Audit item A.9.c — Binomial.entropy used a Gaussian approximation
    `0.5*log(2πenp(1-p))`, which is wildly wrong for small n.
    Compare against the exact sum -Σ p(k)log p(k)."""

    @staticmethod
    def _exact_entropy(n, p):
        from scipy.special import gammaln
        if p <= 0.0 or p >= 1.0:
            return 0.0
        ks = np.arange(0, int(n) + 1)
        log_binom = (
            gammaln(n + 1.0)
            - gammaln(ks + 1.0)
            - gammaln(n - ks + 1.0)
        )
        log_p = log_binom + ks * math.log(p) + (n - ks) * math.log(1.0 - p)
        probs = np.exp(log_p)
        # Numerical safety
        probs = probs / probs.sum()
        return float(-(probs * np.log(np.clip(probs, 1e-300, 1.0))).sum())

    def test_small_n(self):
        b = Binomial(5, 0.3)
        exact = self._exact_entropy(5, 0.3)
        assert allclose(b.entropy(), exact, atol=1e-6), (
            f"expected {exact}, got {b.entropy()}"
        )

    def test_n_equals_1_is_bernoulli(self):
        # Binomial(1, p) entropy equals Bernoulli entropy.
        b = Binomial(1, 0.4)
        p = 0.4
        expected = -(p * math.log(p) + (1 - p) * math.log(1 - p))
        assert allclose(b.entropy(), expected, atol=1e-6)

    def test_large_n_close_to_gaussian(self):
        # For large n, the exact entropy approaches the Gaussian approx.
        n = 200
        p = 0.5
        b = Binomial(n, p)
        gaussian = 0.5 * math.log(2 * math.pi * math.e * n * p * (1 - p))
        # ≤ 1% relative error.
        assert abs(b.entropy() - gaussian) / gaussian < 0.01


# ===========================================================================
# A.5 / E.5 — cdf / icdf coverage added for several previously-missing
# distributions.
# ===========================================================================

class TestBernoulliCdfIcdf:
    def test_cdf_below_support(self):
        b = Bernoulli(0.3)
        assert b.cdf(-0.5) == 0.0

    def test_cdf_at_zero(self):
        b = Bernoulli(0.3)
        assert allclose(b.cdf(0.0), 0.7)  # P(X<=0) = 1-p

    def test_cdf_at_one(self):
        b = Bernoulli(0.3)
        assert b.cdf(1.0) == 1.0

    def test_icdf_roundtrip(self):
        b = Bernoulli(0.3)
        # Below threshold ⇒ 0; above ⇒ 1.
        assert b.icdf(0.5) == 0.0   # 0.5 < 0.7 (= 1-p)
        assert b.icdf(0.8) == 1.0   # 0.8 > 0.7


class TestPoissonCdfIcdf:
    def test_cdf_below_zero(self):
        p = Poisson(2.5)
        assert p.cdf(-1.0) == 0.0

    def test_cdf_zero_known(self):
        # P(X<=0; lambda) = exp(-lambda)
        p = Poisson(1.5)
        assert allclose(p.cdf(0.0), math.exp(-1.5))

    def test_icdf_roundtrip(self):
        p = Poisson(3.0)
        # ppf(0.999) should land at some integer k for lambda=3.
        k = p.icdf(0.999)
        assert k >= 5 and k <= 12, f"ppf(0.999) gave {k}"


class TestGammaCdfIcdf:
    def test_cdf_below_zero(self):
        g = Gamma(2.0, 1.0)
        assert g.cdf(-1.0) == 0.0

    def test_icdf_roundtrip(self):
        g = Gamma(2.0, 3.0)
        q = np.array([0.1, 0.5, 0.9])
        x = g.icdf(q)
        assert np.allclose(g.cdf(x), q, atol=1e-6)


class TestChi2CdfIcdf:
    def test_cdf_below_zero(self):
        c = Chi2(df=3)
        assert c.cdf(-1.0) == 0.0

    def test_icdf_roundtrip(self):
        c = Chi2(df=5)
        q = np.array([0.05, 0.5, 0.95])
        x = c.icdf(q)
        assert np.allclose(c.cdf(x), q, atol=1e-6)


class TestBetaIcdf:
    def test_icdf_roundtrip(self):
        b = Beta(2.0, 3.0)
        q = np.array([0.1, 0.5, 0.9])
        x = b.icdf(q)
        assert np.allclose(b.cdf(x), q, atol=1e-6)


class TestLogNormalIcdf:
    def test_icdf_roundtrip(self):
        ln = LogNormal(loc=0.0, scale=1.0)
        q = np.array([0.1, 0.5, 0.9])
        x = ln.icdf(q)
        assert np.allclose(ln.cdf(x), q, atol=1e-6)


class TestGeometricCdfIcdf:
    def test_cdf_below_zero(self):
        g = Geometric(0.5)
        assert g.cdf(-1.0) == 0.0

    def test_cdf_known(self):
        g = Geometric(0.5)
        # P(X<=0; p=0.5) = p = 0.5
        assert allclose(g.cdf(0.0), 0.5)

    def test_icdf_roundtrip(self):
        g = Geometric(0.4)
        # cdf(icdf(q)) should be >= q (CDF is a step function).
        for q in [0.1, 0.5, 0.8]:
            k = g.icdf(q)
            assert g.cdf(k) >= q - 1e-9


class TestBinomialCdf:
    def test_cdf_below(self):
        b = Binomial(5, 0.5)
        assert b.cdf(-1.0) == 0.0

    def test_cdf_above(self):
        b = Binomial(5, 0.5)
        assert b.cdf(5.0) == 1.0

    def test_cdf_known(self):
        # Binomial(2, 0.5): P(X<=0) = 0.25, P(X<=1) = 0.75
        b = Binomial(2, 0.5)
        assert allclose(b.cdf(0.0), 0.25)
        assert allclose(b.cdf(1.0), 0.75)
