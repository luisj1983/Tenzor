"""Laplace distribution — Tensor/Variable native."""
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
from ._reparam import standard_uniform


class Laplace(Distribution):
    """``Laplace(loc, scale)`` — double-exponential distribution.

    Reparameterised via the inverse-CDF route: for ``U ~ Uniform(0, 1)``
    we form ``u = U - 0.5`` and return ``loc - scale * sign(u) * log(1 - 2|u|)``.

    The random part (``sign(u) * log(1 - 2|u|)``) is computed at the
    Tensor level and detached; only ``loc`` and ``scale`` carry gradient.
    """

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_variable(loc)
        self.scale = _to_variable(scale)
        super().__init__(_broadcast_shape(_shape_of(self.loc), _shape_of(self.scale)))

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return 2.0 * (self.scale * self.scale)

    def _detached_noise(self, shape):
        """Compute ``-sign(u) * log(1 - 2|u|)`` as a detached Tensor."""
        # Use numpy for sign/abs to sidestep missing Variable overloads;
        # this branch never propagates gradient so the autograd path is
        # unaffected.
        u_np = np.asarray(standard_uniform(shape, eps=1e-7), dtype=np.float32) - 0.5
        noise = -np.sign(u_np) * np.log(1.0 - 2.0 * np.abs(u_np))
        return Tensor.from_numpy(np.ascontiguousarray(noise))

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        noise = self._detached_noise(out_shape if out_shape else [])
        loc_t = self.loc.tensor() if isinstance(self.loc, Variable) else self.loc
        scale_t = self.scale.tensor() if isinstance(self.scale, Variable) else self.scale
        return loc_t + scale_t * noise

    def rsample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        noise = Variable(self._detached_noise(out_shape if out_shape else []), False)
        return self.loc + self.scale * noise

    def log_prob(self, value):
        # log p(x) = -log(2) - log(scale) - |x - loc| / scale
        # Use sqrt((x-loc)^2) as an autograd-aware |.| since `abs` lacks
        # a Variable overload at module level.
        value = _to_variable(value)
        diff = value - self.loc
        abs_diff = _tz.pow(diff * diff, 0.5)
        return -math.log(2.0) - _tz.log(self.scale) - abs_diff / self.scale

    def cdf(self, value):
        value = _to_variable(value)
        z = (value - self.loc) / self.scale
        # 0.5 + 0.5 * sign(z) * (1 - exp(-|z|))  — compute via numpy.
        z_np = np.asarray(z.tensor() if isinstance(z, Variable) else z, dtype=np.float64)
        out = 0.5 + 0.5 * np.sign(z_np) * (1.0 - np.exp(-np.abs(z_np)))
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def icdf(self, p):
        p_np = np.asarray(p.tensor() if isinstance(p, Variable)
                          else _to_variable(p).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        u = p_np - 0.5
        out = loc_np - scale_np * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def entropy(self):
        # H = 1 + log(2 * scale)
        return 1.0 + _tz.log(2.0 * self.scale)

    def support(self):
        return "(-inf, inf)"
