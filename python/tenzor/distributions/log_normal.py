"""LogNormal distribution — Tensor/Variable native.

Reparameterised: if ``X ~ Normal(loc, scale)`` then ``exp(X) ~
LogNormal(loc, scale)``. With the autograd-aware ``exp`` Variable overload
this is fully differentiable, so ``rsample`` flows gradients to loc/scale.
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

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_variable(loc)
        self.scale = _to_variable(scale)
        super().__init__(_broadcast_shape(_shape_of(self.loc), _shape_of(self.scale)))

    @property
    def mean(self):
        # E[X] = exp(loc + scale^2/2); autograd-aware in loc and scale.
        s2 = self.scale * self.scale
        return _tz.exp(self.loc + 0.5 * s2)

    @property
    def variance(self):
        # Var[X] = (exp(scale^2) - 1) * exp(2*loc + scale^2); autograd-aware.
        s2 = self.scale * self.scale
        return (_tz.exp(s2) - 1.0) * _tz.exp(2.0 * self.loc + s2)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        z = np.asarray(standard_normal(out_shape if out_shape else []), dtype=np.float32)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float32)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float32)
        return _wrap_numpy(np.exp(loc_np + scale_np * z))

    def rsample(self, sample_shape=()):
        # Reparameterised: exp(loc + scale * z), z ~ N(0,1). Autograd flows to
        # loc and scale through exp's Variable overload.
        out_shape = tuple(sample_shape) + self._batch_shape
        z = Variable(standard_normal(out_shape if out_shape else []), False)
        return _tz.exp(self.loc + self.scale * z)

    def log_prob(self, value):
        # Autograd-aware (log Variable overload): grad flows to loc, scale, value.
        value = _to_variable(value)
        log_v = _tz.log(value)
        z = (log_v - self.loc) / self.scale
        return (-0.5 * (math.log(2.0 * math.pi) + z * z)
                - _tz.log(self.scale) - log_v)

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
        # H = loc + log(scale) + 0.5(1 + log 2π) — autograd-aware in loc, scale.
        return (self.loc + _tz.log(self.scale)
                + 0.5 * (1.0 + math.log(2.0 * math.pi)))

    def support(self):
        return "(0, inf)"
