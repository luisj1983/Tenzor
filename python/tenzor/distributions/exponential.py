"""Exponential distribution."""
import math
import numpy as np
from .distribution import Distribution, _to_numpy


class Exponential(Distribution):
    """Exponential(rate) — exponential distribution.

    Args:
        rate: Rate parameter lambda (scalar or array, > 0).
              scale = 1/rate.
    """

    has_rsample = True

    def __init__(self, rate):
        self.rate = _to_numpy(rate)
        super().__init__(self.rate.shape)

    @property
    def mean(self):
        return 1.0 / self.rate

    @property
    def variance(self):
        return 1.0 / (self.rate ** 2)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        # numpy exponential takes scale = 1/rate
        return np.random.exponential(scale=1.0 / self.rate, size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        in_support = value >= 0
        lp = np.log(self.rate) - self.rate * value
        return np.where(in_support, lp, -np.inf)

    def cdf(self, value):
        value = _to_numpy(value)
        return np.where(value >= 0, 1.0 - np.exp(-self.rate * value), 0.0)

    def icdf(self, p):
        p = _to_numpy(p)
        return -np.log(1.0 - p) / self.rate

    def entropy(self):
        # H = 1 - log(lambda)
        return 1.0 - np.log(self.rate)

    def support(self):
        return "[0, inf)"
