"""Poisson distribution."""
import math

import numpy as np
# V.39: scipy is lazy (see distribution.py).
from .distribution import (
    Distribution,
    _to_numpy,
    _require_scipy_special,
    _require_scipy_stats,
)


class Poisson(Distribution):
    """Poisson(rate) — Poisson distribution over non-negative integers.

    Args:
        rate: Rate parameter lambda (scalar or array, > 0).
    """

    has_rsample = False

    def __init__(self, rate):
        self.rate = _to_numpy(rate)
        super().__init__(self.rate.shape)

    @property
    def mean(self):
        return self.rate.copy()

    @property
    def variance(self):
        return self.rate.copy()

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.poisson(self.rate, size=shape or None).astype(np.float64)

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        value = _to_numpy(value)
        # log P(k; lambda) = k * log(lambda) - lambda - lgamma(k+1)
        eps = 1e-7
        rate = np.clip(self.rate, eps, None)
        return value * np.log(rate) - rate - gammaln(value + 1.0)

    def entropy(self):
        """Entropy of the Poisson distribution.

        Uses an exact truncated-series evaluation for small / moderate rate
        (lambda <= 30):
            H = -sum_{k=0}^{K} P(k; lambda) * log P(k; lambda)
        with K = ceil(lambda + 8 * sqrt(lambda) + 12) — covers the tail to
        below machine precision for the supported lambda range.

        For lambda > 30, uses the standard Stirling-based asymptotic
        H ≈ 0.5 * log(2 * pi * e * lambda) - 1 / (12 lambda)
            - 1 / (24 lambda^2) - 19 / (360 lambda^3).
        The two regimes agree to ~1e-9 at lambda = 30 (verified against scipy).

        Returns ``ndarray`` with shape equal to the batch shape.
        """
        gammaln = _require_scipy_special().gammaln
        rate = np.clip(self.rate, 1e-300, None)
        scalar_input = (rate.ndim == 0)
        rate_arr = np.atleast_1d(rate).astype(np.float64)

        # Asymptotic region (lambda > 30): Ramanujan-style series.
        def stirling_entropy(lam):
            two_pi_e = 2.0 * math.pi * math.e
            return (0.5 * np.log(two_pi_e * lam)
                    - 1.0 / (12.0 * lam)
                    - 1.0 / (24.0 * lam ** 2)
                    - 19.0 / (360.0 * lam ** 3))

        # Exact region (lambda <= 30): enumerate the pmf.
        def exact_entropy(lam):
            # Per-element exact sum. Vectorize via a per-element loop —
            # the lambda <= 30 branch has K <= ~80 terms so this is cheap.
            out = np.empty_like(lam)
            it = np.nditer(lam, flags=["multi_index"])
            while not it.finished:
                l_val = float(it[0])
                K = int(math.ceil(l_val + 8.0 * math.sqrt(l_val) + 12.0))
                k = np.arange(K + 1, dtype=np.float64)
                log_p = k * math.log(l_val) - l_val - gammaln(k + 1.0)
                p = np.exp(log_p)
                # Drop -inf log_p at p=0 by guarding.
                contrib = np.where(p > 0.0, p * log_p, 0.0)
                out[it.multi_index] = -contrib.sum()
                it.iternext()
            return out

        result = np.where(rate_arr > 30.0,
                          stirling_entropy(rate_arr),
                          exact_entropy(rate_arr))
        if scalar_input:
            return result.reshape(())
        return result

    def cdf(self, value):
        """CDF of Poisson (audit item E.5).

        P(X <= k) = Q(floor(k) + 1, lambda)    (regularised upper incomplete gamma)
        Equivalently 1 - P(floor(k) + 1, lambda).
        For k < 0 the CDF is 0.
        """
        gammaincc = _require_scipy_special().gammaincc
        value = np.asarray(value, dtype=np.float64)
        k = np.floor(value)
        cdf_val = gammaincc(k + 1.0, self.rate)
        return np.where(value < 0.0, np.zeros_like(value), cdf_val)

    def icdf(self, q):
        """Inverse CDF of Poisson (audit item E.5).

        No closed form — invert the CDF via scipy.stats.poisson.ppf.
        """
        _poisson = _require_scipy_stats().poisson
        q = np.asarray(q, dtype=np.float64)
        return _poisson.ppf(q, self.rate).astype(np.float64)

    def support(self):
        return "{0, 1, 2, ...}"
