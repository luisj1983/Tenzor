"""Binomial distribution — Tensor-native (sample only)."""
from __future__ import annotations

import math

import numpy as np

from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _shape_of,
    _to_variable,
    _wrap_numpy,
    _require_scipy_special,
    _require_scipy_stats,
)


class Binomial(Distribution):
    """``Binomial(total_count, probs)`` distribution.

    ``has_rsample = False`` (discrete distribution).
    """

    has_rsample = False

    def __init__(self, total_count, probs):
        self.total_count = int(total_count)
        self.probs = _to_variable(probs)
        super().__init__(_shape_of(self.probs))

    @property
    def mean(self):
        return self.probs * float(self.total_count)

    @property
    def variance(self):
        return self.probs * (1.0 - self.probs) * float(self.total_count)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        return _wrap_numpy(np.asarray(
            np.random.binomial(self.total_count, p_np, size=out_shape or None),
            dtype=np.float32,
        ))

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        eps = 1e-7
        p_np = np.clip(np.asarray(self.probs.tensor(), dtype=np.float64),
                       eps, 1.0 - eps)
        n = float(self.total_count)
        log_binom = gammaln(n + 1.0) - gammaln(v_np + 1.0) - gammaln(n - v_np + 1.0)
        out = log_binom + v_np * np.log(p_np) + (n - v_np) * np.log(1.0 - p_np)
        return _wrap_numpy(out)

    def entropy(self):
        # Exact entropy via the PMF; see audit note A.9.c.
        gammaln = _require_scipy_special().gammaln
        p = np.asarray(self.probs.tensor(), dtype=np.float64)
        n_int = int(self.total_count)
        ks = np.arange(0, n_int + 1, dtype=np.float64)
        log_binom = (gammaln(n_int + 1.0) - gammaln(ks + 1.0)
                     - gammaln(n_int - ks + 1.0))
        out = np.zeros_like(p)
        nontrivial = (p > 0.0) & (p < 1.0)
        # Vectorise over batched p — expand last axis with the (n+1,) k-axis.
        # Per-element scalar fall-back keeps memory bounded.
        if not np.any(nontrivial):
            return _wrap_numpy(out)
        # General path.
        log_p = np.log(np.where(nontrivial, p, 0.5))
        log_1mp = np.log(np.where(nontrivial, 1.0 - p, 0.5))
        # broadcast over shape p.shape + (n+1,)
        log_pmf = (log_binom + log_p[..., None] * ks + log_1mp[..., None] * (n_int - ks))
        pmf = np.exp(log_pmf)
        pmf = pmf / pmf.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(pmf > 0.0, -pmf * np.log(pmf), 0.0)
        entropy = term.sum(axis=-1)
        out = np.where(nontrivial, entropy, 0.0)
        return _wrap_numpy(out)

    def cdf(self, value):
        scipy_t = _require_scipy_stats().binom
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        return _wrap_numpy(scipy_t.cdf(v_np, n=self.total_count, p=p_np))

    def support(self):
        return f"{{0, 1, ..., {self.total_count}}}"
