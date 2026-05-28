"""Beta distribution — Tensor-native (sample only)."""
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


class Beta(Distribution):
    """``Beta(concentration1, concentration0)`` distribution on ``(0, 1)``.

    Gamma-derived reparameterisation is possible in principle but
    relies on a reparameterisable Gamma sampler, which Tenzor does not
    yet provide.  ``has_rsample = False``.
    """

    has_rsample = False

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

    def log_prob(self, value):
        scipy_special = _require_scipy_special()
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        a_np = np.asarray(self.concentration1.tensor(), dtype=np.float64)
        b_np = np.asarray(self.concentration0.tensor(), dtype=np.float64)
        out = ((a_np - 1.0) * np.log(v_np)
               + (b_np - 1.0) * np.log(1.0 - v_np)
               - scipy_special.betaln(a_np, b_np))
        return _wrap_numpy(out)

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
        scipy_special = _require_scipy_special()
        a_np = np.asarray(self.concentration1.tensor(), dtype=np.float64)
        b_np = np.asarray(self.concentration0.tensor(), dtype=np.float64)
        out = (scipy_special.betaln(a_np, b_np)
               - (a_np - 1.0) * scipy_special.digamma(a_np)
               - (b_np - 1.0) * scipy_special.digamma(b_np)
               + (a_np + b_np - 2.0) * scipy_special.digamma(a_np + b_np))
        return _wrap_numpy(out)

    def support(self):
        return "(0, 1)"
