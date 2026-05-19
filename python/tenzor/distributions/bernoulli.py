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

    def support(self):
        return "{0, 1}"
