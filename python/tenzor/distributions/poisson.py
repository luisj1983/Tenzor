"""Poisson distribution."""
import numpy as np
from scipy.special import gammaln
from .distribution import Distribution, _to_numpy


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
        value = _to_numpy(value)
        # log P(k; lambda) = k * log(lambda) - lambda - lgamma(k+1)
        eps = 1e-7
        rate = np.clip(self.rate, eps, None)
        return value * np.log(rate) - rate - gammaln(value + 1.0)

    def entropy(self):
        # No simple closed form; approximate via continuous approximation
        # H ≈ 0.5 * log(2*pi*e*lambda) for large lambda
        # Exact: -sum_k P(k) log P(k) (not computed here)
        raise NotImplementedError(
            "Poisson entropy has no simple closed form. "
            "Use Monte Carlo estimation for exact entropy."
        )

    def support(self):
        return "{0, 1, 2, ...}"
