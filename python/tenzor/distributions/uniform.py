"""Uniform distribution."""
import numpy as np
from .distribution import Distribution, _to_numpy


class Uniform(Distribution):
    """Uniform(low, high) — uniform distribution on [low, high).

    Args:
        low: Lower boundary (scalar or array).
        high: Upper boundary (scalar or array, > low).
    """

    has_rsample = True

    def __init__(self, low, high):
        self.low = _to_numpy(low)
        self.high = _to_numpy(high)
        super().__init__(np.broadcast_shapes(self.low.shape, self.high.shape))

    @property
    def mean(self):
        return (self.low + self.high) / 2.0

    @property
    def variance(self):
        d = self.high - self.low
        return d * d / 12.0

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.uniform(self.low, self.high, size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        in_support = (value >= self.low) & (value <= self.high)
        lp = -np.log(self.high - self.low)
        return np.where(in_support, lp, -np.inf)

    def cdf(self, value):
        value = _to_numpy(value)
        return np.clip((value - self.low) / (self.high - self.low), 0.0, 1.0)

    def icdf(self, p):
        p = _to_numpy(p)
        return self.low + p * (self.high - self.low)

    def entropy(self):
        return np.log(self.high - self.low)

    def support(self):
        return "[low, high)"
