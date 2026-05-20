"""Bernoulli distribution."""
import numpy as np
from .distribution import Distribution, _to_numpy


class Bernoulli(Distribution):
    """Bernoulli(probs) — Bernoulli distribution over {0, 1}.

    Args:
        probs: Probability of success (scalar or array, in [0, 1]).
    """

    has_rsample = False

    def __init__(self, probs):
        self.probs = _to_numpy(probs)
        super().__init__(self.probs.shape)

    @property
    def mean(self):
        return self.probs.copy()

    @property
    def variance(self):
        return self.probs * (1.0 - self.probs)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return (np.random.uniform(size=shape or None) < self.probs).astype(np.float64)

    def log_prob(self, value):
        value = _to_numpy(value)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        return value * np.log(p) + (1.0 - value) * np.log(1.0 - p)

    def entropy(self):
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    def cdf(self, value):
        """CDF of Bernoulli (audit item E.5).

        P(X <= k) = 0       if k < 0
                  = 1 - p   if 0 <= k < 1
                  = 1       if k >= 1
        """
        value = np.asarray(value, dtype=np.float64)
        below = value < 0.0
        between = (value >= 0.0) & (value < 1.0)
        at_or_above = value >= 1.0
        out = np.zeros_like(value)
        out = np.where(between, 1.0 - self.probs, out)
        out = np.where(at_or_above, np.ones_like(out), out)
        out = np.where(below, np.zeros_like(out), out)
        return out

    def icdf(self, q):
        """Inverse CDF of Bernoulli (audit item E.5).

        Q(q) = 0  if q <= 1 - p
             = 1  if q >  1 - p
        """
        q = np.asarray(q, dtype=np.float64)
        return (q > (1.0 - self.probs)).astype(np.float64)

    def support(self):
        return "{0, 1}"
