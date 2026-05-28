"""Exponential distribution — Tensor/Variable native."""
from __future__ import annotations

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import Distribution, _shape_of, _to_variable
from ._reparam import standard_exponential


class Exponential(Distribution):
    """``Exponential(rate)`` — exponential distribution.

    Reparameterised: ``rsample = standard_exponential / rate`` where the
    standard variate is ``-log(U)`` for ``U ~ Uniform(0, 1)`` and is
    detached.  Gradient flows into ``rate``.
    """

    has_rsample = True

    def __init__(self, rate):
        self.rate = _to_variable(rate)
        super().__init__(_shape_of(self.rate))

    @property
    def mean(self):
        return 1.0 / self.rate

    @property
    def variance(self):
        return 1.0 / (self.rate * self.rate)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        e = standard_exponential(out_shape if out_shape else [])
        rate_t = self.rate.tensor() if isinstance(self.rate, Variable) else self.rate
        return e / rate_t

    def rsample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        e = Variable(standard_exponential(out_shape if out_shape else []), False)
        return e / self.rate

    def log_prob(self, value):
        # log p(x) = log(rate) - rate * x
        value = _to_variable(value)
        return _tz.log(self.rate) - self.rate * value

    def cdf(self, value):
        value_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        rate_np = np.asarray(self.rate.tensor(), dtype=np.float64)
        out = np.where(value_np >= 0, 1.0 - np.exp(-rate_np * value_np), 0.0)
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def icdf(self, p):
        p_np = np.asarray(_to_variable(p).tensor(), dtype=np.float64)
        rate_np = np.asarray(self.rate.tensor(), dtype=np.float64)
        out = -np.log(1.0 - p_np) / rate_np
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def entropy(self):
        # H = 1 - log(rate)
        return 1.0 - _tz.log(self.rate)

    def support(self):
        return "[0, inf)"
