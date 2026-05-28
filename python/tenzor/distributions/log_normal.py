"""LogNormal distribution — Tensor-native.

Reparameterisation: if ``X ~ Normal(loc, scale)`` then ``exp(X) ~
LogNormal(loc, scale)``.  ``tz.exp`` lacks a Variable overload, so the
reparam path cannot be expressed in Variable arithmetic alone.  We
sample at Tensor level only and disable ``rsample``.
"""
from __future__ import annotations

import math

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _broadcast_shape,
    _shape_of,
    _to_variable,
    _wrap_numpy,
    _require_scipy_special,
)
from ._reparam import standard_normal


class LogNormal(Distribution):
    """``LogNormal(loc, scale)``."""

    has_rsample = False

    def __init__(self, loc, scale):
        self.loc = _to_variable(loc)
        self.scale = _to_variable(scale)
        super().__init__(_broadcast_shape(_shape_of(self.loc), _shape_of(self.scale)))

    @property
    def mean(self):
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        return _wrap_numpy(np.exp(loc_np + 0.5 * scale_np ** 2))

    @property
    def variance(self):
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        s2 = scale_np ** 2
        return _wrap_numpy((np.exp(s2) - 1.0) * np.exp(2.0 * loc_np + s2))

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        z = np.asarray(standard_normal(out_shape if out_shape else []), dtype=np.float32)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float32)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float32)
        return _wrap_numpy(np.exp(loc_np + scale_np * z))

    def log_prob(self, value):
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        z = (np.log(v_np) - loc_np) / scale_np
        out = (-0.5 * (math.log(2.0 * math.pi) + z * z)
               - np.log(scale_np) - np.log(v_np))
        return _wrap_numpy(out)

    def cdf(self, value):
        scipy_special = _require_scipy_special()
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = 0.5 * (1.0 + scipy_special.erf((np.log(v_np) - loc_np) /
                                             (scale_np * math.sqrt(2.0))))
        return _wrap_numpy(out)

    def icdf(self, q):
        ndtri = _require_scipy_special().ndtri
        q_np = np.asarray(_to_variable(q).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        eps = 1e-12
        q_clip = np.clip(q_np, eps, 1.0 - eps)
        return _wrap_numpy(np.exp(loc_np + scale_np * ndtri(q_clip)))

    def entropy(self):
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        return _wrap_numpy(loc_np + np.log(scale_np)
                           + 0.5 * (1.0 + math.log(2.0 * math.pi)))

    def support(self):
        return "(0, inf)"
