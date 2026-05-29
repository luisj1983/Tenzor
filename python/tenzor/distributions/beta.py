"""Beta distribution — Tensor-native (sample only)."""
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


class Beta(Distribution):
    """``Beta(concentration1, concentration0)`` distribution on ``(0, 1)``.

    Gamma-derived reparameterisation is possible in principle but
    relies on a reparameterisable Gamma sampler, which Tenzor does not
    yet provide.  ``has_rsample = False``.
    """

    has_rsample = True

    def __init__(self, concentration1, concentration0):
        self.concentration1 = _to_variable(concentration1)
        self.concentration0 = _to_variable(concentration0)
        super().__init__(_broadcast_shape(_shape_of(self.concentration1),
                                          _shape_of(self.concentration0)))

    @property
    def mean(self):
        a = self.concentration1
        b = self.concentration0
        return a / (a + b)

    @property
    def variance(self):
        a = self.concentration1
        b = self.concentration0
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        a_np = np.asarray(self.concentration1.tensor(), dtype=np.float64)
        b_np = np.asarray(self.concentration0.tensor(), dtype=np.float64)
        s = np.random.beta(a_np, b_np, size=out_shape or None)
        return _wrap_numpy(np.asarray(s, dtype=np.float32))

    def rsample(self, sample_shape=()):
        # Beta(c1, c0) = X/(X+Y), X~Gamma(c1,1), Y~Gamma(c0,1); both Gammas are
        # reparameterised, so the gradient flows to both concentrations.
        from .gamma import Gamma
        x = Gamma(self.concentration1, 1.0).rsample(sample_shape)
        y = Gamma(self.concentration0, 1.0).rsample(sample_shape)
        return x / (x + y)

    def log_prob(self, value):
        # log p(x) = (a-1)log x + (b-1)log(1-x) - [lgamma(a)+lgamma(b)-lgamma(a+b)]
        # Autograd-aware (lgamma/log Variable overloads): gradients flow to both
        # concentrations and value, no scipy detachment.
        value = _to_variable(value)
        a = self.concentration1
        b = self.concentration0
        log_beta = _tz.lgamma(a) + _tz.lgamma(b) - _tz.lgamma(a + b)
        return ((a - 1.0) * _tz.log(value)
                + (b - 1.0) * _tz.log(1.0 - value)
                - log_beta)

    def cdf(self, value):
        scipy_special = _require_scipy_special()
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        a_np = np.asarray(self.concentration1.tensor(), dtype=np.float64)
        b_np = np.asarray(self.concentration0.tensor(), dtype=np.float64)
        return _wrap_numpy(scipy_special.betainc(a_np, b_np, v_np))

    def icdf(self, q):
        scipy_special = _require_scipy_special()
        q_np = np.asarray(_to_variable(q).tensor(), dtype=np.float64)
        a_np = np.asarray(self.concentration1.tensor(), dtype=np.float64)
        b_np = np.asarray(self.concentration0.tensor(), dtype=np.float64)
        return _wrap_numpy(scipy_special.betaincinv(a_np, b_np, q_np))

    def entropy(self):
        # H = log B(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b) — autograd-aware.
        a = self.concentration1
        b = self.concentration0
        log_beta = _tz.lgamma(a) + _tz.lgamma(b) - _tz.lgamma(a + b)
        return (log_beta
                - (a - 1.0) * _tz.digamma(a)
                - (b - 1.0) * _tz.digamma(b)
                + (a + b - 2.0) * _tz.digamma(a + b))

    def support(self):
        return "(0, 1)"
