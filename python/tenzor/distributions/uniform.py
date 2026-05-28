"""Uniform distribution — Tensor/Variable native."""
from __future__ import annotations

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _broadcast_shape,
    _shape_of,
    _to_tensor,
    _to_variable,
)
from ._reparam import standard_uniform


class Uniform(Distribution):
    """``Uniform(low, high)`` — uniform distribution on ``[low, high)``.

    Reparameterised: ``rsample = low + (high - low) * U`` with
    ``U ~ U(0, 1)``.
    """

    has_rsample = True

    def __init__(self, low, high):
        self.low = _to_variable(low)
        self.high = _to_variable(high)
        super().__init__(_broadcast_shape(_shape_of(self.low), _shape_of(self.high)))

    @property
    def mean(self):
        return (self.low + self.high) * 0.5

    @property
    def variance(self):
        d = self.high - self.low
        return (d * d) * (1.0 / 12.0)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        u = standard_uniform(out_shape if out_shape else [])
        low_t = self.low.tensor() if isinstance(self.low, Variable) else self.low
        high_t = self.high.tensor() if isinstance(self.high, Variable) else self.high
        return low_t + (high_t - low_t) * u

    def rsample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        u = standard_uniform(out_shape if out_shape else [])
        u_v = Variable(u, False)
        return self.low + (self.high - self.low) * u_v

    def log_prob(self, value):
        # log p(x) = -log(high - low) inside support.  No mask — callers
        # needing strict validation check `value` themselves (mirrors
        # PyTorch validate_args=False).
        return -_tz.log(self.high - self.low)

    def cdf(self, value):
        value = _to_variable(value)
        d = self.high - self.low
        return (value - self.low) / d

    def icdf(self, p):
        p = _to_variable(p)
        return self.low + p * (self.high - self.low)

    def entropy(self):
        return _tz.log(self.high - self.low)

    def support(self):
        return "[low, high)"
