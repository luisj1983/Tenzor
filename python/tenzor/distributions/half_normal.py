"""HalfNormal distribution — Tensor-native (sample only).

Reparameterisation ``|σ z|`` is well-defined in principle, but ``abs``
has no Variable overload and the underlying operator is non-smooth at
0, so we expose ``has_rsample = False``.
"""
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
)
from ._reparam import standard_normal


class HalfNormal(Distribution):
    """``HalfNormal(scale)``."""

    has_rsample = False

    def __init__(self, scale):
        self.scale = _to_variable(scale)
        super().__init__(_shape_of(self.scale))

    @property
    def mean(self):
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float32)
        return _wrap_numpy(scale_np * math.sqrt(2.0 / math.pi))

    @property
    def variance(self):
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float32)
        return _wrap_numpy(scale_np ** 2 * (1.0 - 2.0 / math.pi))

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        z = np.asarray(standard_normal(out_shape if out_shape else []), dtype=np.float32)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float32)
        return _wrap_numpy(np.abs(scale_np * z))

    def log_prob(self, value):
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        in_support = v_np >= 0
        z = v_np / scale_np
        lp = (math.log(2.0) - 0.5 * math.log(2.0 * math.pi)
              - np.log(scale_np) - 0.5 * z * z)
        return _wrap_numpy(np.where(in_support, lp, -np.inf))

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
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        return _wrap_numpy(0.5 * np.log(math.pi * scale_np ** 2 / 2.0) + 0.5)

    def support(self):
        return "[0, inf)"
