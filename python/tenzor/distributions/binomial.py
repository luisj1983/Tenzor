"""Binomial distribution."""
import math
import numpy as np
from scipy.special import gammaln
from .distribution import Distribution, _to_numpy


class Binomial(Distribution):
    """Binomial(total_count, probs) distribution.

    Args:
        total_count: Number of trials n (int or scalar, >= 0).
        probs: Success probability (scalar or array, in [0, 1]).
    """

    has_rsample = False

    def __init__(self, total_count, probs):
        self.total_count = int(total_count)
        self.probs = _to_numpy(probs)
        super().__init__(self.probs.shape)

    @property
    def mean(self):
        return self.probs * float(self.total_count)

    @property
    def variance(self):
        return self.probs * (1.0 - self.probs) * float(self.total_count)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.binomial(self.total_count, self.probs,
                                  size=shape or None).astype(np.float64)

    def log_prob(self, value):
        value = _to_numpy(value)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        n = float(self.total_count)
        log_binom = gammaln(n + 1.0) - gammaln(value + 1.0) - gammaln(n - value + 1.0)
        return log_binom + value * np.log(p) + (n - value) * np.log(1.0 - p)

    def entropy(self):
        # No simple closed form; use the Gaussian approximation
        # H ≈ 0.5 * log(2*pi*e*n*p*(1-p)) for large n
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        n = float(self.total_count)
        return 0.5 * np.log(2 * math.pi * math.e * n * p * (1.0 - p))

    def support(self):
        return f"{{0, 1, ..., {self.total_count}}}"
