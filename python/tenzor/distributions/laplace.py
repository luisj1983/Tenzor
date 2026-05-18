"""Laplace distribution."""
import math
import numpy as np
from .distribution import Distribution, _to_numpy


class Laplace(Distribution):
    """Laplace(loc, scale) — double exponential distribution.

    Args:
        loc: Location parameter (scalar or array).
        scale: Scale parameter (scalar or array, > 0).
    """

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_numpy(loc)
        self.scale = _to_numpy(scale)
        super().__init__(np.broadcast_shapes(self.loc.shape, self.scale.shape))

    @property
    def mean(self):
        return self.loc.copy()

    @property
    def variance(self):
        return 2.0 * self.scale ** 2

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.laplace(self.loc, self.scale, size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        return -math.log(2.0) - np.log(self.scale) - np.abs(value - self.loc) / self.scale

    def cdf(self, value):
        value = _to_numpy(value)
        z = (value - self.loc) / self.scale
        return 0.5 + 0.5 * np.sign(z) * (1.0 - np.exp(-np.abs(z)))

    def icdf(self, p):
        p = _to_numpy(p)
        u = p - 0.5
        return self.loc - self.scale * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))

    def entropy(self):
        # H = log(2 * e * scale)
        return 1.0 + np.log(2.0 * self.scale)

    def support(self):
        return "(-inf, inf)"
