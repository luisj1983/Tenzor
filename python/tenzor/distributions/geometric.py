"""Geometric distribution — Tensor-native (sample only)."""
from __future__ import annotations

import math

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _shape_of,
    _to_variable,
    _wrap_numpy,
)


class Geometric(Distribution):
    """``Geometric(probs)`` — number of failures before the first success.

    ``P(X = k) = (1 - p)^k · p`` for ``k = 0, 1, 2, …``.
    """

    has_rsample = False

    def __init__(self, probs):
        self.probs = _to_variable(probs)
        super().__init__(_shape_of(self.probs))

    @property
    def mean(self):
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        return _wrap_numpy((1.0 - p_np) / p_np)

    @property
    def variance(self):
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        return _wrap_numpy((1.0 - p_np) / (p_np * p_np))

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        # numpy geometric: 1-indexed (trials until first success);
        # subtract one to get failure count.
        out = np.random.geometric(p=p_np, size=out_shape or None) - 1
        return _wrap_numpy(np.asarray(out, dtype=np.float32))

    def log_prob(self, value):
        # log p(x) = x·log(1-p) + log(p).  Every term depends on the learnable
        # probs, so the whole expression is built with autograd-aware Variable
        # ops (a numpy round-trip would detach the gradient to probs).
        value = _to_variable(value)
        eps = 1e-7
        log_p = _tz.log(self.probs + eps)
        log_1mp = _tz.log((1.0 - self.probs) + eps)
        return value * log_1mp + log_p

    def entropy(self):
        # H = -(log(p) + (1-p)*log(1-p)/p), autograd-aware in probs via _tz.log
        # (same eps-regularised logs as log_prob).
        eps = 1e-7
        log_p = _tz.log(self.probs + eps)
        log_1mp = _tz.log((1.0 - self.probs) + eps)
        # Divide by an eps-regularised probs to match the eps-guarded logs;
        # without this the term blows up as p->0 (or is a division by zero
        # at p == 0), yielding inf/NaN where the logs were deliberately
        # protected.
        return -1.0 * (log_p + (1.0 - self.probs) * log_1mp / (self.probs + eps))

    def cdf(self, value):
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        k = np.floor(v_np)
        cdf_val = 1.0 - np.power(1.0 - p_np, k + 1.0)
        return _wrap_numpy(np.where(v_np < 0.0, np.zeros_like(v_np), cdf_val))

    def icdf(self, q):
        q_np = np.asarray(_to_variable(q).tensor(), dtype=np.float64)
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        with np.errstate(divide="ignore"):
            k = np.ceil(np.log(1.0 - q_np) / np.log(1.0 - p_np)) - 1.0
        return _wrap_numpy(np.where(q_np <= 0.0, np.zeros_like(q_np), k))

    def support(self):
        return "{0, 1, 2, ...}"
