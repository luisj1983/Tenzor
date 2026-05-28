"""Cauchy distribution — Tensor/Variable native."""
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
from ._reparam import standard_cauchy


class Cauchy(Distribution):
    """``Cauchy(loc, scale)`` distribution.

    Reparameterised: ``rsample = loc + scale * tan(pi * (U - 0.5))`` with
    ``U ~ Uniform(0, 1)``.
    """

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_variable(loc)
        self.scale = _to_variable(scale)
        super().__init__(_broadcast_shape(_shape_of(self.loc), _shape_of(self.scale)))

    @property
    def mean(self):
        raise NotImplementedError("Cauchy distribution has no defined mean")

    @property
    def variance(self):
        raise NotImplementedError("Cauchy distribution has no defined variance")

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        c = standard_cauchy(out_shape if out_shape else [])
        loc_t = self.loc.tensor() if isinstance(self.loc, Variable) else self.loc
        scale_t = self.scale.tensor() if isinstance(self.scale, Variable) else self.scale
        return loc_t + scale_t * c

    def rsample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        c = Variable(standard_cauchy(out_shape if out_shape else []), False)
        return self.loc + self.scale * c

    def log_prob(self, value):
        # log p(x) = -log(pi) - log(scale) - log(1 + ((x-loc)/scale)^2)
        value = _to_variable(value)
        z = (value - self.loc) / self.scale
        return (-math.log(math.pi)) - _tz.log(self.scale) - _tz.log(1.0 + z * z)

    def cdf(self, value):
        value_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = 0.5 + np.arctan((value_np - loc_np) / scale_np) / math.pi
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def icdf(self, p):
        p_np = np.asarray(_to_variable(p).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = loc_np + scale_np * np.tan(math.pi * (p_np - 0.5))
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def entropy(self):
        # H = log(4 * pi * scale)
        return _tz.log(4.0 * math.pi * self.scale)

    def support(self):
        return "(-inf, inf)"
