"""Cauchy distribution."""
import math
import numpy as np
from .distribution import Distribution, _to_numpy


class Cauchy(Distribution):
    """Cauchy(loc, scale) distribution.

    Args:
        loc: Location (median) parameter (scalar or array).
        scale: Scale parameter (scalar or array, > 0).
    """

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_numpy(loc)
        self.scale = _to_numpy(scale)
        super().__init__(np.broadcast_shapes(self.loc.shape, self.scale.shape))

    @property
    def mean(self):
        raise NotImplementedError("Cauchy distribution has no defined mean")

    @property
    def variance(self):
        raise NotImplementedError("Cauchy distribution has no defined variance")

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.standard_cauchy(size=shape or None) * self.scale + self.loc

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        z = (value - self.loc) / self.scale
        return -math.log(math.pi) - np.log(self.scale) - np.log(1.0 + z * z)

    def cdf(self, value):
        value = _to_numpy(value)
        return 0.5 + np.arctan((value - self.loc) / self.scale) / math.pi

    def icdf(self, p):
        p = _to_numpy(p)
        return self.loc + self.scale * np.tan(math.pi * (p - 0.5))

    def entropy(self):
        # H = log(4*pi*scale)
        return np.log(4.0 * math.pi * self.scale)

    def support(self):
        return "(-inf, inf)"
