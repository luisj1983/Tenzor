"""Weibull distribution."""
import math
import numpy as np
from scipy.special import gamma as gamma_fn
from .distribution import Distribution, _to_numpy


class Weibull(Distribution):
    """Weibull(scale, concentration) distribution.

    Args:
        scale: Scale parameter lambda (scalar or array, > 0).
        concentration: Shape parameter k (scalar or array, > 0).
    """

    has_rsample = True

    def __init__(self, scale, concentration):
        self.scale = _to_numpy(scale)
        self.concentration = _to_numpy(concentration)
        super().__init__(np.broadcast_shapes(self.scale.shape, self.concentration.shape))

    @property
    def mean(self):
        # E[X] = lambda * Gamma(1 + 1/k)
        return self.scale * gamma_fn(1.0 + 1.0 / self.concentration)

    @property
    def variance(self):
        k = self.concentration
        lam = self.scale
        g1 = gamma_fn(1.0 + 1.0 / k)
        g2 = gamma_fn(1.0 + 2.0 / k)
        return lam ** 2 * (g2 - g1 ** 2)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        # Inverse-CDF: lambda * (-log(1 - U))^(1/k)
        u = np.random.uniform(size=shape or None)
        u = np.clip(u, 1e-7, 1.0 - 1e-7)
        return self.scale * (-np.log(1.0 - u)) ** (1.0 / self.concentration)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        k = self.concentration
        lam = self.scale
        x_over_l = value / lam
        return (np.log(k / lam)
                + (k - 1.0) * np.log(x_over_l)
                - x_over_l ** k)

    def cdf(self, value):
        value = _to_numpy(value)
        return 1.0 - np.exp(-(value / self.scale) ** self.concentration)

    def icdf(self, p):
        p = _to_numpy(p)
        return self.scale * (-np.log(1.0 - p)) ** (1.0 / self.concentration)

    def entropy(self):
        # H = euler_mascheroni * (1 - 1/k) + log(lambda/k) + 1
        euler_mascheroni = 0.5772156649015328
        k = self.concentration
        return (euler_mascheroni * (1.0 - 1.0 / k)
                + np.log(self.scale / k) + 1.0)

    def support(self):
        return "(0, inf)"
