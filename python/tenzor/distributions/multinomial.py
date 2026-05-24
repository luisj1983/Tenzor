"""Multinomial distribution."""
import math
import numpy as np
# V.39: scipy is lazy (see distribution.py).
from .distribution import Distribution, _to_numpy, _require_scipy_special


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
        """Sample from Multinomial via the C++ tz.multinomial op.

        Audit item I.9: previously did an O(B) Python-level loop calling
        ``rng.multinomial`` once per batch element. Now delegates to
        ``tenzor.multinomial(probs, total_count, replacement=True)`` which
        draws ``total_count`` category indices per batch row inside the C++
        kernel, then we bincount those indices to recover the multinomial
        counts.  Falls back to ``numpy.random.Generator.multinomial`` only
        if the C++ binding is unavailable (e.g. a stripped build).
        """
        batch_size = int(np.prod(self._batch_shape)) if self._batch_shape else 1
        probs_2d = self.probs.reshape(batch_size, self._num_events)
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        K = self._num_events
        N = self.total_count

        out = np.zeros((batch_size, n_samples, K), dtype=np.float64)
        try:
            import tenzor as _tz
            # tz.multinomial expects (N, C) float probs, returns (N, S) Int64
            # indices.  We draw n_samples * total_count indices per row, then
            # split into n_samples chunks of total_count and bincount each.
            probs_t = _tz.tensor(probs_2d.astype(np.float32))
            total_draws = n_samples * N
            idx = _tz.multinomial(probs_t, total_draws, True)
            # Convert back to numpy.  Use __array__ via numpy() if exposed,
            # else fall back to constructing via numpy.
            idx_np = np.asarray(idx.numpy() if hasattr(idx, "numpy") else idx,
                                dtype=np.int64)
            # idx_np shape: (batch_size, n_samples * total_count) for batched
            # input, or (n_samples * total_count,) for 1-D probs.
            if idx_np.ndim == 1:
                idx_np = idx_np.reshape(1, -1)
            # Reshape to (batch_size * n_samples, N) so we can vectorise
            # the per-row bincount with np.add.at.
            flat = idx_np.reshape(batch_size * n_samples, N)
            counts = np.zeros((batch_size * n_samples, K), dtype=np.float64)
            row_idx = np.arange(batch_size * n_samples)[:, None]
            np.add.at(counts, (row_idx, flat), 1)
            out = counts.reshape(batch_size, n_samples, K)
        except (ImportError, AttributeError, RuntimeError):
            # Fallback path: numpy's own multinomial (vectorised in C).
            rng = np.random.default_rng()
            for b in range(batch_size):
                out[b] = rng.multinomial(N, probs_2d[b], size=n_samples)

        # Reshape to (sample_shape..., batch_shape..., num_events).
        out = np.transpose(out, (1, 0, 2))  # (n_samples, batch, K)
        out_shape = tuple(sample_shape) + self._batch_shape + (K,)
        return out.reshape(out_shape)

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
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
        gammaln = _require_scipy_special().gammaln
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
