"""Normal (Gaussian) distribution."""
import math
import numpy as np
from scipy import special as scipy_special
from .distribution import Distribution, _to_numpy


class Normal(Distribution):
    """Normal(loc, scale) — Gaussian distribution.

    Args:
        loc: Mean (scalar or array).
        scale: Standard deviation (scalar or array, > 0).
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
        return self.scale ** 2

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.normal(self.loc, self.scale, size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        z = (value - self.loc) / self.scale
        return -0.5 * (math.log(2 * math.pi) + 2 * np.log(self.scale) + z * z)

    def cdf(self, value):
        value = _to_numpy(value)
        return 0.5 * (1.0 + scipy_special.erf((value - self.loc) / (self.scale * math.sqrt(2.0))))

    def icdf(self, p):
        p = _to_numpy(p)
        return self.loc + self.scale * math.sqrt(2.0) * scipy_special.erfinv(2.0 * p - 1.0)

    def entropy(self):
        return 0.5 * math.log(2 * math.pi * math.e) + np.log(self.scale)

    def support(self):
        return "(-inf, inf)"
