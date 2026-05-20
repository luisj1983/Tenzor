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
        # np.random.dirichlet rejects batched alpha; build samples via the
        # Gamma(α_i, 1)/Σ Gamma(α_j, 1) construction so any leading batch
        # shape of `concentration` is supported (audit item A.9.b).
        sample_shape = tuple(sample_shape)
        output_shape = sample_shape + self.concentration.shape
        # Broadcast the alpha array to (sample_shape, batch_shape, K).
        alpha = np.broadcast_to(self.concentration, output_shape)
        # Gamma(α_i, 1) iid per-element; normalise across the last axis.
        g = np.random.gamma(shape=alpha, scale=1.0, size=output_shape)
        return g / g.sum(axis=-1, keepdims=True)

    def rsample(self, sample_shape=()):
        # Same construction as sample() but routed through a path that
        # would be reparameterised if/when we wire pathwise gradients
        # through Gamma (audit item A.9 reparameterisation list).
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
