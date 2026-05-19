"""NegativeBinomial distribution."""
import numpy as np
from scipy.special import gammaln
from .distribution import Distribution, _to_numpy


class NegativeBinomial(Distribution):
    """NegativeBinomial(total_count, probs) distribution.

    Number of failures before achieving total_count successes.
    Parameterization: probs = p (success probability per trial).
    Mean = r * p / (1 - p).

    Args:
        total_count: Number of successes r (scalar or array, > 0).
        probs: Success probability p (scalar or array, 0 < p < 1).
    """

    has_rsample = False

    def __init__(self, total_count, probs):
        self.total_count = _to_numpy(total_count)
        self.probs = _to_numpy(probs)
        super().__init__(np.broadcast_shapes(self.total_count.shape, self.probs.shape))

    @property
    def mean(self):
        # E[X] = r * p / (1 - p)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        return self.total_count * p / (1.0 - p)

    @property
    def variance(self):
        # Var[X] = r * p / (1 - p)^2
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        return self.total_count * p / (1.0 - p) ** 2

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        # Our probs is the *failure* probability (PyTorch convention):
        #   mean = r * p / (1 - p) where p = probs.
        # numpy.negative_binomial(n, p) uses p as *success* probability with
        #   mean = n * (1 - p) / p.
        # To reconcile: pass (1 - self.probs) as numpy's success probability,
        #   so numpy mean = r * self.probs / (1 - self.probs) = our mean.
        r = np.round(self.total_count).astype(int)
        success_prob = np.clip(1.0 - self.probs, 1e-7, 1.0 - 1e-7)
        return np.random.negative_binomial(r, success_prob, size=shape or None).astype(np.float64)

    def log_prob(self, value):
        value = _to_numpy(value)
        r = self.total_count
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        # log P(k) = lgamma(k+r) - lgamma(r) - lgamma(k+1)
        #          + r*log(1-p) + k*log(p)
        return (gammaln(value + r) - gammaln(r) - gammaln(value + 1.0)
                + r * np.log(1.0 - p) + value * np.log(p))

    def entropy(self):
        raise NotImplementedError(
            "NegativeBinomial entropy has no simple closed form. "
            "Use Monte Carlo estimation."
        )

    def support(self):
        return "{0, 1, 2, ...}"
