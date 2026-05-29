"""Weibull distribution — Tensor-native (sample only)."""
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
    _require_scipy_special,
)
from ._reparam import standard_uniform


class Weibull(Distribution):
    """``Weibull(scale, concentration)`` distribution.

    Reparameterised: ``scale * (-log(1-U))^(1/k)``. The Variable-exponent
    power is expressed as ``exp(log(E)/k)`` (E ~ Exp(1)), so the gradient
    flows to scale and concentration.
    """

    has_rsample = True

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

    def rsample(self, sample_shape=()):
        # X = scale * E^(1/k), E ~ Exp(1) = -log(1-U). The E^(1/k) term is
        # exp(log(E)/k) so a Variable exponent works; E is detached noise.
        out_shape = tuple(sample_shape) + self._batch_shape
        u = np.asarray(standard_uniform(out_shape if out_shape else [], eps=1e-7),
                       dtype=np.float32)
        log_e = np.log(-np.log(1.0 - u)).astype(np.float32)
        log_e_v = Variable(Tensor.from_numpy(np.ascontiguousarray(log_e)), False)
        return self.scale * _tz.exp(log_e_v / self.concentration)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        u = np.asarray(standard_uniform(out_shape if out_shape else [], eps=1e-7),
                       dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        out = scale_np * (-np.log(1.0 - u)) ** (1.0 / k_np)
        return _wrap_numpy(np.asarray(out, dtype=np.float32))

    def log_prob(self, value):
        # log p = log(k/λ) + (k-1) log(x/λ) - (x/λ)^k. The power uses
        # exp(k·log(x/λ)) so it stays autograd-aware with a Variable exponent
        # (Variable ** Variable is unsupported). Gradients flow to k, λ, value.
        value = _to_variable(value)
        k = self.concentration
        lam = self.scale
        log_xol = _tz.log(value / lam)
        return (_tz.log(k / lam) + (k - 1.0) * log_xol
                - _tz.exp(k * log_xol))

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
