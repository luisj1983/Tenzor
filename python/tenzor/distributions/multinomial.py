"""Multinomial distribution."""
import math
import numpy as np
from scipy.special import gammaln
from .distribution import Distribution, _to_numpy


class Multinomial(Distribution):
    """Multinomial(total_count, probs) — multinomial distribution.

    Args:
        total_count: Number of trials (int, > 0).
        probs: Unnormalized event probabilities (array of shape (..., K)).
    """

    has_rsample = False

    def __init__(self, total_count, probs):
        probs = _to_numpy(probs)
        self.total_count = int(total_count)
        total = probs.sum(axis=-1, keepdims=True)
        self.probs = probs / total
        self._num_events = probs.shape[-1]
        super().__init__(probs.shape[:-1])

    @property
    def mean(self):
        return self.probs * float(self.total_count)

    @property
    def variance(self):
        p = self.probs
        return p * (1.0 - p) * float(self.total_count)

    def sample(self, sample_shape=()):
        batch_size = int(np.prod(self._batch_shape)) if self._batch_shape else 1
        probs_2d = self.probs.reshape(batch_size, self._num_events)
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        out = np.array([
            np.random.multinomial(self.total_count, probs_2d[b], size=n_samples)
            for b in range(batch_size)
        ], dtype=np.float64)
        out_shape = tuple(sample_shape) + self._batch_shape + (self._num_events,)
        return out.reshape(out_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0)
        n = float(self.total_count)
        log_factorial_n = math.lgamma(n + 1.0)
        log_fac_x = gammaln(value + 1.0).sum(axis=-1)
        log_p_term = (value * np.log(p)).sum(axis=-1)
        return log_factorial_n - log_fac_x + log_p_term

    def entropy(self):
        raise NotImplementedError(
            "Multinomial entropy has no simple closed form; "
            "use Monte Carlo estimation."
        )

    def support(self):
        return f"{{x in Z^K : sum(x) = {self.total_count}, x_i >= 0}}"
