"""Dirichlet distribution."""
import numpy as np
from scipy.special import gammaln, digamma
from .distribution import Distribution, _to_numpy


class Dirichlet(Distribution):
    """Dirichlet(concentration) distribution over the simplex.

    Args:
        concentration: Concentration parameters alpha_i (array of shape (..., K),
                       all entries > 0).
    """

    has_rsample = True

    def __init__(self, concentration):
        self.concentration = _to_numpy(concentration)
        super().__init__(self.concentration.shape[:-1])

    @property
    def mean(self):
        # E[X_i] = alpha_i / sum(alpha)
        return self.concentration / self.concentration.sum(axis=-1, keepdims=True)

    @property
    def variance(self):
        a = self.concentration
        a0 = a.sum(axis=-1, keepdims=True)
        return a * (a0 - a) / (a0 ** 2 * (a0 + 1.0))

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.dirichlet(self.concentration, size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        a = self.concentration
        # log p(x) = lgamma(sum_a) - sum(lgamma(a_i)) + sum((a_i-1)*log(x_i))
        log_norm = gammaln(a.sum(axis=-1)) - gammaln(a).sum(axis=-1)
        return log_norm + ((a - 1.0) * np.log(value)).sum(axis=-1)

    def entropy(self):
        a = self.concentration
        k = a.shape[-1]
        a0 = a.sum(axis=-1)
        # H = log B(alpha) + (a0 - k)*psi(a0) - sum((a_i-1)*psi(a_i))
        log_b = gammaln(a).sum(axis=-1) - gammaln(a0)
        return log_b + (a0 - k) * digamma(a0) - ((a - 1.0) * digamma(a)).sum(axis=-1)

    def support(self):
        return "simplex (sum to 1, all >= 0)"
