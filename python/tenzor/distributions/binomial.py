"""Binomial distribution."""
import math
import numpy as np
# V.39: scipy is lazy (see distribution.py).
from .distribution import (
    Distribution,
    _to_numpy,
    _require_scipy_special,
    _require_scipy_stats,
)


class Binomial(Distribution):
    """Binomial(total_count, probs) distribution.

    Args:
        total_count: Number of trials n (int or scalar, >= 0).
        probs: Success probability (scalar or array, in [0, 1]).
    """

    has_rsample = False

    def __init__(self, total_count, probs):
        self.total_count = int(total_count)
        self.probs = _to_numpy(probs)
        super().__init__(self.probs.shape)

    @property
    def mean(self):
        return self.probs * float(self.total_count)

    @property
    def variance(self):
        return self.probs * (1.0 - self.probs) * float(self.total_count)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.binomial(self.total_count, self.probs,
                                  size=shape or None).astype(np.float64)

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        value = _to_numpy(value)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        n = float(self.total_count)
        log_binom = gammaln(n + 1.0) - gammaln(value + 1.0) - gammaln(n - value + 1.0)
        return log_binom + value * np.log(p) + (n - value) * np.log(1.0 - p)

    def entropy(self):
        # Exact entropy: H = -Σ p(k) log p(k) for k=0..n.
        # Previously used the Gaussian large-n approximation which is wildly
        # wrong for small n (audit item A.9.c).  We compute the PMF in
        # log-space for numerical stability, exponentiate, renormalise, then
        # sum.  Vectorised over arbitrary batched `probs`.
        gammaln = _require_scipy_special().gammaln
        p = np.asarray(self.probs, dtype=np.float64)
        n_int = int(self.total_count)
        ks = np.arange(0, n_int + 1, dtype=np.float64)            # (n+1,)
        log_binom = gammaln(n_int + 1.0) - gammaln(ks + 1.0) - gammaln(n_int - ks + 1.0)

        # Broadcast: result is p.shape + (n+1,).
        # Handle p ∈ {0, 1} explicitly so log_p does not produce -inf*0 = NaN.
        out = np.zeros_like(p)
        nontrivial = (p > 0.0) & (p < 1.0)
        if np.any(nontrivial):
            p_nt = p[nontrivial]
            # log p(k|n,p) — broadcast (M, n+1).
            log_p = (
                log_binom
                + ks * np.log(p_nt)[..., None]
                + (n_int - ks) * np.log(1.0 - p_nt)[..., None]
            )
            # Numerical safety: subtract max, exponentiate, renormalise.
            log_p = log_p - log_p.max(axis=-1, keepdims=True)
            probs_k = np.exp(log_p)
            probs_k = probs_k / probs_k.sum(axis=-1, keepdims=True)
            ent = -(probs_k * np.log(np.clip(probs_k, 1e-300, 1.0))).sum(axis=-1)
            out[nontrivial] = ent
        # For p ∈ {0, 1} the distribution is a point mass ⇒ entropy 0.
        return out

    def cdf(self, value):
        """CDF of Binomial(n, p) (audit item E.5).

        P(X <= k) = I_{1-p}(n-k, k+1)   (regularised incomplete beta)
        For k < 0 the CDF is 0; for k >= n it is 1.
        """
        betainc = _require_scipy_special().betainc
        value = np.asarray(value, dtype=np.float64)
        k = np.floor(value)
        n = float(self.total_count)
        eps = 1e-12
        p = np.clip(self.probs, eps, 1.0 - eps)
        # betainc(n-k, k+1, 1-p) = P(X <= k) for Binomial(n, p).
        # Clip k to [0, n] before passing in; we re-apply the
        # below-/above-range checks afterwards.
        k_clip = np.clip(k, 0.0, n)
        cdf_val = betainc(n - k_clip, k_clip + 1.0, 1.0 - p)
        out = np.where(value < 0.0, np.zeros_like(value), cdf_val)
        out = np.where(value >= n, np.ones_like(value), out)
        return out

    def icdf(self, q):
        """Inverse CDF of Binomial (audit item E.5).

        No closed form — invert via scipy.stats.binom.ppf.
        """
        _binom = _require_scipy_stats().binom
        q = np.asarray(q, dtype=np.float64)
        return _binom.ppf(q, self.total_count, self.probs).astype(np.float64)

    def support(self):
        return f"{{0, 1, ..., {self.total_count}}}"
