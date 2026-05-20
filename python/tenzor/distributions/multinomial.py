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
        """Entropy of the multinomial distribution.

        Uses the exact closed-form expression:
            H(Multinomial(n, p)) = -log(n!)
                                   + n * sum_k -p_k log(p_k)
                                   + sum_k E[log(x_k!)]
                                 = log(n!)  (constant)
                                   - n * sum_k p_k * log(p_k)  (entropy term)
                                   - sum_k E_x[log(x_k!)]      (log-factorial term)

        Equivalently the entropy of `n` independent categorical draws is
        `n * H(Categorical(p))` plus the multinomial's log-binomial-coefficient
        correction; numerically we evaluate via E[log(x_k!)] using the per-cell
        marginal Binomial(n, p_k) under which
        x_k ~ Binomial(n, p_k) so E[log(x_k!)] = sum_{i=0}^{n} P(x_k = i) * log(i!).

        Returns ``ndarray`` with shape equal to the batch shape.
        """
        n = float(self.total_count)
        p = np.clip(self.probs, 1e-300, 1.0)

        # log(n!) — constant per batch element
        log_n_factorial = math.lgamma(n + 1.0)

        # n * sum_k p_k * log(p_k)
        cat_entropy_term = n * (-(p * np.log(p)).sum(axis=-1))

        # E[log(x_k!)] for x_k ~ Binomial(n, p_k), enumerated.
        # Shape: (..., K) where ... is batch.
        n_int = int(self.total_count)
        i_arr = np.arange(n_int + 1, dtype=np.float64)
        log_fact = gammaln(i_arr + 1.0)  # log(i!) for i in 0..n
        log_choose = (math.lgamma(n + 1.0)
                      - gammaln(i_arr + 1.0)
                      - gammaln(n - i_arr + 1.0))   # log C(n, i)

        # broadcast over batch
        p_exp = p[..., np.newaxis]                 # (..., K, 1)
        log_p = np.log(p_exp)
        log_1mp = np.log(np.clip(1.0 - p_exp, 1e-300, 1.0))
        # log P(x_k = i) = log C(n, i) + i log p_k + (n - i) log(1 - p_k)
        log_pmf = (log_choose                       # (n+1,)
                   + i_arr * log_p                   # broadcasts to (..., K, n+1)
                   + (n - i_arr) * log_1mp)
        pmf = np.exp(log_pmf)                       # (..., K, n+1)
        # Per-cell expected log-factorial: sum_i pmf[i] * log(i!)
        e_log_fact_per_k = (pmf * log_fact).sum(axis=-1)   # (..., K)
        log_fact_term = e_log_fact_per_k.sum(axis=-1)      # (...,)

        return -log_n_factorial + cat_entropy_term + log_fact_term

    def support(self):
        return f"{{x in Z^K : sum(x) = {self.total_count}, x_i >= 0}}"
