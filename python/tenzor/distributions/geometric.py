"""Geometric distribution."""
import math
import numpy as np
from .distribution import Distribution, _to_numpy


class Geometric(Distribution):
    """Geometric(probs) — geometric distribution: number of failures before first success.

    P(X = k) = (1 - p)^k * p   for k = 0, 1, 2, ...

    Args:
        probs: Success probability (scalar or array, 0 < p <= 1).
    """

    has_rsample = False

    def __init__(self, probs):
        self.probs = _to_numpy(probs)
        super().__init__(self.probs.shape)

    @property
    def mean(self):
        return (1.0 - self.probs) / self.probs

    @property
    def variance(self):
        return (1.0 - self.probs) / (self.probs ** 2)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        # numpy geometric returns number of trials until first success (starts at 1)
        # we want number of failures before first success (starts at 0)
        return (np.random.geometric(p=self.probs, size=shape or None) - 1).astype(np.float64)

    def log_prob(self, value):
        value = _to_numpy(value)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        return value * np.log(1.0 - p) + np.log(p)

    def entropy(self):
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        return -(np.log(p) + (1.0 - p) * np.log(1.0 - p) / p)

    def support(self):
        return "{0, 1, 2, ...}"
