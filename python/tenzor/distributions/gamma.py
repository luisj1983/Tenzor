"""Gamma distribution."""
import numpy as np
from scipy.special import gammaln, digamma
from .distribution import Distribution, _to_numpy


class Gamma(Distribution):
    """Gamma(concentration, rate) distribution.

    Args:
        concentration: Shape parameter alpha (scalar or array, > 0).
        rate: Rate parameter beta (scalar or array, > 0).
              scale = 1/rate.
    """

    has_rsample = True

    def __init__(self, concentration, rate):
        self.concentration = _to_numpy(concentration)
        self.rate = _to_numpy(rate)
        super().__init__(np.broadcast_shapes(self.concentration.shape, self.rate.shape))

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate ** 2)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        # numpy gamma uses shape (alpha) and scale (1/beta)
        return np.random.gamma(shape=self.concentration, scale=1.0 / self.rate,
                               size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        a = self.concentration
        b = self.rate
        return (a * np.log(b) - gammaln(a)
                + (a - 1.0) * np.log(value)
                - b * value)

    def entropy(self):
        a = self.concentration
        b = self.rate
        # H = a - log(b) + lgamma(a) + (1-a)*digamma(a)
        return a - np.log(b) + gammaln(a) + (1.0 - a) * digamma(a)

    def support(self):
        return "(0, inf)"
