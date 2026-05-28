"""VonMises distribution — Tensor-native (sample only)."""
from __future__ import annotations

import math

import numpy as np

from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _broadcast_shape,
    _shape_of,
    _to_variable,
    _wrap_numpy,
    _require_scipy_special,
    _require_scipy_stats,
)


class VonMises(Distribution):
    """``VonMises(loc, concentration)`` distribution on the circle."""

    has_rsample = False  # No straightforward reparameterisation.

    def __init__(self, loc, concentration):
        self.loc = _to_variable(loc)
        self.concentration = _to_variable(concentration)
        super().__init__(_broadcast_shape(_shape_of(self.loc),
                                          _shape_of(self.concentration)))

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        scipy_special = _require_scipy_special()
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        return _wrap_numpy(1.0 - scipy_special.i1(k_np) / scipy_special.i0(k_np))

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        return _wrap_numpy(np.asarray(
            np.random.vonmises(loc_np, k_np, size=out_shape or None),
            dtype=np.float32,
        ))

    def log_prob(self, value):
        scipy_special = _require_scipy_special()
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        # log p(x) = k cos(x - loc) - log(2π I0(k))
        out = (k_np * np.cos(v_np - loc_np)
               - math.log(2.0 * math.pi)
               - np.log(scipy_special.i0(k_np)))
        return _wrap_numpy(out)

    def entropy(self):
        scipy_special = _require_scipy_special()
        k_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        # H = -k I1(k)/I0(k) + log(2π I0(k))
        out = (-k_np * scipy_special.i1(k_np) / scipy_special.i0(k_np)
               + math.log(2.0 * math.pi) + np.log(scipy_special.i0(k_np)))
        return _wrap_numpy(out)

    def support(self):
        return "[-pi, pi)"
