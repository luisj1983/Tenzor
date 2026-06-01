"""NegativeBinomial distribution — Tensor-native (sample only)."""
from __future__ import annotations

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _broadcast_shape,
    _shape_of,
    _to_variable,
    _wrap_numpy,
)


class NegativeBinomial(Distribution):
    """``NegativeBinomial(total_count, probs)`` distribution.

    Counts the number of failures before ``total_count`` successes.
    ``probs`` follows the PyTorch convention (failure probability):
    ``mean = total_count · p / (1 - p)``.
    """

    has_rsample = False

    def __init__(self, total_count, probs):
        self.total_count = _to_variable(total_count)
        self.probs = _to_variable(probs)
        super().__init__(_broadcast_shape(_shape_of(self.total_count),
                                          _shape_of(self.probs)))

    @property
    def mean(self):
        eps = 1e-7
        r_np = np.asarray(self.total_count.tensor(), dtype=np.float64)
        p_np = np.clip(np.asarray(self.probs.tensor(), dtype=np.float64),
                       eps, 1.0 - eps)
        return _wrap_numpy(r_np * p_np / (1.0 - p_np))

    @property
    def variance(self):
        eps = 1e-7
        r_np = np.asarray(self.total_count.tensor(), dtype=np.float64)
        p_np = np.clip(np.asarray(self.probs.tensor(), dtype=np.float64),
                       eps, 1.0 - eps)
        return _wrap_numpy(r_np * p_np / (1.0 - p_np) ** 2)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        r_np = np.round(np.asarray(self.total_count.tensor(), dtype=np.float64)).astype(int)
        success_prob = np.clip(1.0 - np.asarray(self.probs.tensor(), dtype=np.float64),
                               1e-7, 1.0 - 1e-7)
        s = np.random.negative_binomial(r_np, success_prob,
                                        size=out_shape or None)
        return _wrap_numpy(np.asarray(s, dtype=np.float32))

    def log_prob(self, value):
        # log p(x) = lgamma(x+r) - lgamma(r) - lgamma(x+1)
        #            + r·log(1-p) + x·log(p).
        # Built entirely with autograd-aware Variable ops so the gradient flows
        # to BOTH the learnable total_count (r) and probs (p) — a numpy
        # round-trip would detach both.  lgamma(x+1) is computed on-device from
        # the (detached) value and contributes no gradient to either param.
        value = _to_variable(value)
        r = self.total_count
        eps = 1e-7
        log_p = _tz.log(self.probs + eps)
        log_1mp = _tz.log((1.0 - self.probs) + eps)
        return (_tz.lgamma(value + r) - _tz.lgamma(r) - _tz.lgamma(value + 1.0)
                + r * log_1mp + value * log_p)

    def entropy(self, n_samples: int = 10000, rng=None):
        """Monte-Carlo entropy estimate (no closed form)."""
        if rng is None:
            rng = np.random.default_rng()
        # Sample n_samples and compute empirical entropy via -E[log p(x)].
        s = self.sample((n_samples,))
        v_np = np.asarray(s, dtype=np.float64)
        lp_np = np.asarray(self.log_prob(s), dtype=np.float64)
        return _wrap_numpy(np.asarray([-lp_np.mean()], dtype=np.float32).reshape(()))

    def support(self):
        return "{0, 1, 2, ...}"
