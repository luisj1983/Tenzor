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

    def cdf(self, value):
        """CDF of Geometric (audit item E.5).

        P(X <= k) = 1 - (1 - p)^(floor(k) + 1)   for k >= 0
                  = 0                            for k <  0
        """
        value = np.asarray(value, dtype=np.float64)
        k = np.floor(value)
        cdf_val = 1.0 - np.power(1.0 - self.probs, k + 1.0)
        return np.where(value < 0.0, np.zeros_like(value), cdf_val)

    def icdf(self, q):
        """Inverse CDF of Geometric (audit item E.5).

        For Geometric counting failures-before-success:
            q = P(X <= k) = 1 - (1 - p)^(k + 1)
            k = ceil(log(1 - q) / log(1 - p)) - 1
        """
        q = np.asarray(q, dtype=np.float64)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        # Avoid log(0) at q == 1.0 — pin to a finite large value.
        q_clip = np.clip(q, 0.0, 1.0 - eps)
        return np.ceil(np.log(1.0 - q_clip) / np.log(1.0 - p)) - 1.0

    def support(self):
        return "{0, 1, 2, ...}"
