"""Weibull distribution — Tensor-native (sample only)."""
from __future__ import annotations

import numpy as np

from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _broadcast_shape,
    _shape_of,
    _to_variable,
    _wrap_numpy,
    _require_scipy_special,
)
from ._reparam import standard_uniform


class Weibull(Distribution):
    """``Weibull(scale, concentration)`` distribution.

    Reparameterisation ``scale * (-log(1 - U))^(1/k)`` is plausible but
    requires generic ``pow`` with a Variable exponent (not supported
    here in the autograd path), so ``has_rsample = False``.
    """

    has_rsample = False

    def __init__(self, scale, concentration):
        self.scale = _to_variable(scale)
        self.concentration = _to_variable(concentration)
        super().__init__(_broadcast_shape(_shape_of(self.scale),
                                          _shape_of(self.concentration)))

    @property
    def mean(self):
        gamma_fn = _require_scipy_special().gamma
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        return _wrap_numpy(scale_np * gamma_fn(1.0 + 1.0 / k_np))

    @property
    def variance(self):
        gamma_fn = _require_scipy_special().gamma
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        g1 = gamma_fn(1.0 + 1.0 / k_np)
        g2 = gamma_fn(1.0 + 2.0 / k_np)
        return _wrap_numpy(scale_np ** 2 * (g2 - g1 ** 2))

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        u = np.asarray(standard_uniform(out_shape if out_shape else [], eps=1e-7),
                       dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        out = scale_np * (-np.log(1.0 - u)) ** (1.0 / k_np)
        return _wrap_numpy(np.asarray(out, dtype=np.float32))

    def log_prob(self, value):
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        x_over_l = v_np / scale_np
        out = (np.log(k_np / scale_np)
               + (k_np - 1.0) * np.log(x_over_l)
               - x_over_l ** k_np)
        return _wrap_numpy(out)

    def cdf(self, value):
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        return _wrap_numpy(1.0 - np.exp(-(v_np / scale_np) ** k_np))

    def icdf(self, p):
        p_np = np.asarray(_to_variable(p).tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        return _wrap_numpy(scale_np * (-np.log(1.0 - p_np)) ** (1.0 / k_np))

    def entropy(self):
        euler_mascheroni = 0.5772156649015328
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        out = (euler_mascheroni * (1.0 - 1.0 / k_np)
               + np.log(scale_np / k_np) + 1.0)
        return _wrap_numpy(out)

    def support(self):
        return "(0, inf)"
