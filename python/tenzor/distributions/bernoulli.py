"""Bernoulli distribution — Tensor/Variable native."""
from __future__ import annotations

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore
from tenzor.tenzor_core import nn as _cpp_nn  # type: ignore

from .distribution import (
    Distribution,
    _shape_of,
    _to_variable,
    _wrap_numpy,
)
from ._reparam import standard_uniform


class Bernoulli(Distribution):
    """``Bernoulli(probs)`` — Bernoulli distribution over ``{0, 1}``.

    Discrete distribution: ``rsample`` is not implemented (no
    continuous reparameterisation).  ``sample`` returns a Tenzor Tensor
    of 0/1 values (float32).

    Args:
        probs: success probability (scalar / array / Tensor / Variable),
            values in ``[0, 1]``.
    """

    has_rsample = False

    def __init__(self, probs):
        self.probs = _to_variable(probs)
        super().__init__(_shape_of(self.probs))

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1.0 - self.probs)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        u = standard_uniform(out_shape if out_shape else [])
        # u < probs → {0, 1} as float32.  Do the compare on numpy buffers
        # to avoid Tensor-comparison shape rules; the result is detached.
        u_np = np.asarray(u, dtype=np.float32)
        p_np = np.asarray(self.probs.tensor(), dtype=np.float32)
        out = (u_np < p_np).astype(np.float32)
        return _wrap_numpy(out)

    @staticmethod
    def _clamp_probs(probs, eps):
        # Autograd-aware clamp of probs into [eps, 1 - eps] via hardtanh,
        # matching PyTorch (probs.clamp(eps, 1 - eps)).  This keeps the
        # score d/dp log(p) unbiased for p in the interior and prevents
        # log of a non-positive number when p drifts to <= 0 or >= 1.
        return _cpp_nn.hardtanh(probs, eps, 1.0 - eps)

    def log_prob(self, value):
        # log p(x) = x*log(p) + (1-x)*log(1-p)
        value = _to_variable(value)
        eps = 1e-7
        probs = self._clamp_probs(self.probs, eps)
        log_p = _tz.log(probs)
        log_1mp = _tz.log(1.0 - probs)
        return value * log_p + (1.0 - value) * log_1mp

    def entropy(self):
        eps = 1e-7
        probs = self._clamp_probs(self.probs, eps)
        log_p = _tz.log(probs)
        log_1mp = _tz.log(1.0 - probs)
        return -(probs * log_p + (1.0 - probs) * log_1mp)

    def cdf(self, value):
        value_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        below = value_np < 0.0
        between = (value_np >= 0.0) & (value_np < 1.0)
        at_or_above = value_np >= 1.0
        out = np.zeros_like(value_np)
        out = np.where(between, 1.0 - p_np, out)
        out = np.where(at_or_above, np.ones_like(out), out)
        out = np.where(below, np.zeros_like(out), out)
        return _wrap_numpy(out)

    def icdf(self, q):
        q_np = np.asarray(_to_variable(q).tensor(), dtype=np.float64)
        p_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        return _wrap_numpy((q_np > (1.0 - p_np)).astype(np.float32))

    def support(self):
        return "{0, 1}"
