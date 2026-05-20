"""NegativeBinomial distribution."""
import numpy as np
from scipy.special import gammaln, betainc
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

    def entropy(self, n_samples: int = 10000, rng: np.random.Generator | None = None):
        """Entropy of the negative-binomial distribution via Monte Carlo.

        NegativeBinomial entropy has no compact closed form. This estimator
        draws `n_samples` per batch element and returns
            H ≈ -mean_i log P(x_i; r, p).

        Args:
            n_samples: Number of Monte Carlo draws per batch element. Default
                10000 — empirically gives ~1% relative error on moderate-r
                distributions.
            rng: Optional `numpy.random.Generator` for deterministic sampling.

        Returns:
            ``ndarray`` with shape equal to the batch shape.
        """
        if rng is None:
            rng = np.random.default_rng()

        eps = 1e-7
        r = self.total_count
        p = np.clip(self.probs, eps, 1.0 - eps)
        success_prob = 1.0 - p  # See sample(): numpy uses success-prob convention.

        # broadcast r and p to the batch shape so we know the per-element size.
        r_b, p_b, sp_b = np.broadcast_arrays(r, p, success_prob)
        batch_shape = r_b.shape

        # Generate samples: shape (n_samples,) + batch_shape.
        r_int = np.maximum(np.round(r_b).astype(int), 1)
        samples = rng.negative_binomial(r_int, sp_b, size=(n_samples,) + batch_shape).astype(np.float64)

        # Evaluate log P(x; r, p) at each draw.
        log_p_x = (gammaln(samples + r_b) - gammaln(r_b) - gammaln(samples + 1.0)
                   + r_b * np.log(1.0 - p_b) + samples * np.log(p_b))
        # H ≈ -E[log P]
        return -log_p_x.mean(axis=0)

    def cdf(self, value):
        """CDF of NegativeBinomial via the regularised incomplete beta
        identity (audit item E.5).

        For X ~ NegativeBinomial(r, p) counting failures-before-r-successes
        with `probs` = p the failure prob (PyTorch convention),
            P(X <= k) = I_{1-p}(r, floor(k)+1)
        where I_x(a,b) is the regularised incomplete beta. For k < 0 the
        CDF is 0.
        """
        value = _to_numpy(value)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        r = self.total_count
        k = np.floor(value)
        # I_{1-p}(r, k+1).  betainc is the regularised lower-incomplete beta.
        cdf_val = betainc(r, k + 1.0, 1.0 - p)
        return np.where(value < 0.0, np.zeros_like(value, dtype=np.float64), cdf_val)

    def icdf(self, q):
        """Inverse CDF of NegativeBinomial (audit item E.5).

        We invert the regularised incomplete beta identity numerically by
        finding the smallest non-negative integer k with `cdf(k) >= q`. The
        continuous inverse via `betaincinv` gives a real-valued seed; we
        then ceil-and-clamp.
        """
        q = _to_numpy(q)
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0 - eps)
        r = self.total_count
        q_clip = np.clip(q, 0.0, 1.0 - eps)
        # betaincinv(a, b, y) = x s.t. I_x(a, b) = y.  Solve for k:
        #   I_{1-p}(r, k+1) = q   ⇒  k+1 follows from inverse w.r.t. b.
        # SciPy's betaincinv inverts in the x-argument only, but here we
        # need to invert in the b-argument (k+1), so fall back to a
        # bracketed binary search on k using the cdf to find the smallest
        # k with cdf(k) >= q. Use an upper bound from mean + 50*sqrt(var) to keep
        # the search short for non-degenerate parameters.
        mean = r * p / (1.0 - p)
        var = r * p / (1.0 - p) ** 2
        upper = (mean + 50.0 * np.sqrt(var + 1.0)).astype(np.float64)

        # Vectorise the search via a binary search on integer k.
        k_lo = np.zeros_like(q_clip, dtype=np.float64)
        k_hi = np.broadcast_to(upper, q_clip.shape).copy()
        # Expand k_hi until cdf(k_hi) >= q_clip everywhere.
        for _ in range(50):
            insufficient = self.cdf(k_hi) < q_clip
            if not np.any(insufficient):
                break
            k_hi = np.where(insufficient, k_hi * 2.0 + 1.0, k_hi)
        # Binary search.
        for _ in range(60):
            mid = np.floor((k_lo + k_hi) / 2.0)
            cdf_mid = self.cdf(mid)
            k_lo = np.where(cdf_mid < q_clip, mid + 1.0, k_lo)
            k_hi = np.where(cdf_mid >= q_clip, mid, k_hi)
        return k_lo

    def support(self):
        return "{0, 1, 2, ...}"
