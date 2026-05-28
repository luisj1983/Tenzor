"""Geometric distribution — Tensor-native (sample only)."""
from __future__ import annotations

import math

import numpy as np

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
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        eps = 1e-7
        p_np = np.clip(np.asarray(self.probs.tensor(), dtype=np.float64),
                       eps, 1.0 - eps)
        out = v_np * np.log(1.0 - p_np) + np.log(p_np)
        return _wrap_numpy(out)

    def entropy(self):
        eps = 1e-7
        p_np = np.clip(np.asarray(self.probs.tensor(), dtype=np.float64),
                       eps, 1.0 - eps)
        out = -(np.log(p_np) + (1.0 - p_np) * np.log(1.0 - p_np) / p_np)
        return _wrap_numpy(out)

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
