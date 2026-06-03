"""HalfNormal distribution — Tensor/Variable native.

Reparameterised: ``|σ z|`` with ``z ~ N(0,1)``. The ``|z|`` factor is
detached noise, so multiplying by the scale Variable yields a correct,
differentiable rsample (non-smoothness at 0 is measure-zero).
"""
from __future__ import annotations

import math

import numpy as np

import tenzor as _tz  # noqa: F401  (kept for parity; scale*noise uses Variable mul)
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _shape_of,
    _to_variable,
    _wrap_numpy,
    _require_scipy_special,
)
from ._reparam import standard_normal


class HalfNormal(Distribution):
    """``HalfNormal(scale)``."""

    has_rsample = True

    def __init__(self, scale):
        self.scale = _to_variable(scale)
        super().__init__(_shape_of(self.scale))

    @property
    def mean(self):
        # E[X] = scale * sqrt(2/pi); autograd-aware in scale.
        return self.scale * math.sqrt(2.0 / math.pi)

    @property
    def variance(self):
        # Var[X] = scale^2 * (1 - 2/pi); autograd-aware in scale.
        return (self.scale * self.scale) * (1.0 - 2.0 / math.pi)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        z = np.asarray(standard_normal(out_shape if out_shape else []), dtype=np.float32)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float32)
        return _wrap_numpy(np.abs(scale_np * z))

    def rsample(self, sample_shape=()):
        # scale * |z|, z ~ N(0,1). |z| is detached noise; the scale Variable
        # multiply carries the gradient.
        out_shape = tuple(sample_shape) + self._batch_shape
        z = np.asarray(standard_normal(out_shape if out_shape else []), dtype=np.float32)
        abs_z = Variable(Tensor.from_numpy(np.ascontiguousarray(np.abs(z))), False)
        return self.scale * abs_z

    def log_prob(self, value):
        # log p(v) = log2 - 0.5 log(2pi) - log(scale) - 0.5 (v/scale)^2 for v>=0,
        # -inf otherwise. Built from Variable ops so the gradient flows to
        # scale; the support mask uses the autograd-aware where (gradient only
        # on the selected in-support branch, never the constant -inf branch).
        value = _to_variable(value)
        z = value / self.scale
        smooth = (math.log(2.0) - 0.5 * math.log(2.0 * math.pi)
                  - _tz.log(self.scale) - 0.5 * z * z)
        cond = Variable(value >= 0, False)  # bool mask Tensor -> Variable
        sm_shape = tuple(smooth.shape)
        neg_inf = Variable(
            Tensor.from_numpy(np.full(sm_shape if sm_shape else (1,),
                                      -np.inf, dtype=np.float32)),
            False)
        return _tz.where(cond, smooth, neg_inf)

    def cdf(self, value):
        scipy_special = _require_scipy_special()
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = np.where(v_np >= 0,
                       scipy_special.erf(v_np / (scale_np * math.sqrt(2.0))),
                       0.0)
        return _wrap_numpy(out)

    def icdf(self, p):
        scipy_special = _require_scipy_special()
        p_np = np.asarray(_to_variable(p).tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        return _wrap_numpy(scale_np * math.sqrt(2.0) * scipy_special.erfinv(p_np))

    def entropy(self):
        # H = 0.5*log(pi*scale^2/2) + 0.5 = log(scale) + 0.5*log(pi/2) + 0.5;
        # autograd-aware in scale via _tz.log.
        return _tz.log(self.scale) + (0.5 * math.log(math.pi / 2.0) + 0.5)

    def support(self):
        return "[0, inf)"
