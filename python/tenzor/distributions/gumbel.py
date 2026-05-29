"""Gumbel distribution — Tensor/Variable native.

Standard Gumbel(0, 1) sample: ``-log(-log(U))`` with ``U ~ U(0, 1)``.
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
)
from ._reparam import standard_gumbel


_EULER_MASCHERONI = 0.5772156649015328606


class Gumbel(Distribution):
    """``Gumbel(loc, scale)`` — Type-I extreme value distribution.

    Reparameterised: ``rsample = loc - scale * log(-log(U))``,
    ``U ~ Uniform(0, 1)``.
    """

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_variable(loc)
        self.scale = _to_variable(scale)
        super().__init__(_broadcast_shape(_shape_of(self.loc), _shape_of(self.scale)))

    @property
    def mean(self):
        return self.loc + self.scale * _EULER_MASCHERONI

    @property
    def variance(self):
        # σ² = (π² / 6) · scale²
        return (math.pi * math.pi / 6.0) * (self.scale * self.scale)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        g = standard_gumbel(out_shape if out_shape else [])
        loc_t = self.loc.tensor() if isinstance(self.loc, Variable) else self.loc
        scale_t = self.scale.tensor() if isinstance(self.scale, Variable) else self.scale
        return loc_t + scale_t * g

    def rsample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        g = Variable(standard_gumbel(out_shape if out_shape else []), False)
        return self.loc + self.scale * g

    def log_prob(self, value):
        # log p(x) = -log(scale) - (z + exp(-z))   where z = (x-loc)/scale
        # exp() now has an autograd-aware Variable overload, so the gradient
        # flows correctly through value, loc and scale (no detachment).
        value = _to_variable(value)
        z = (value - self.loc) / self.scale
        exp_neg_z = _tz.exp(0.0 - z)
        return -_tz.log(self.scale) - z - exp_neg_z

    def cdf(self, value):
        # F(x) = exp(-exp(-z))
        value_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        z = (value_np - loc_np) / scale_np
        return Tensor.from_numpy(np.exp(-np.exp(-z)).astype(np.float32, copy=False))

    def icdf(self, p):
        p_np = np.asarray(_to_variable(p).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = loc_np - scale_np * np.log(-np.log(np.clip(p_np, 1e-12, 1.0 - 1e-12)))
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def entropy(self):
        # H = log(scale) + γ + 1
        return _tz.log(self.scale) + (_EULER_MASCHERONI + 1.0)

    def support(self):
        return "(-inf, inf)"
