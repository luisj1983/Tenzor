"""Poisson distribution — Tensor-native (sample only)."""
from __future__ import annotations

import math

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _shape_of,
    _to_variable,
    _wrap_numpy,
    _require_scipy_special,
)


class Poisson(Distribution):
    """``Poisson(rate)`` distribution over non-negative integers."""

    has_rsample = False

    def __init__(self, rate):
        self.rate = _to_variable(rate)
        super().__init__(_shape_of(self.rate))

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def sample(self, sample_shape=()):
        # Native device-side Poisson sampler (tz.poisson / OpId::PoissonSample),
        # no NumPy round-trip. Rates broadcast to the full draw shape.
        out_shape = list(tuple(sample_shape) + self._batch_shape)
        rate_t = self.rate.tensor() if hasattr(self.rate, "tensor") else self.rate
        if out_shape:
            rate_t = rate_t * _tz.ones(out_shape)
        return _tz.poisson(rate_t)

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        eps = 1e-7
        rate_np = np.clip(np.asarray(self.rate.tensor(), dtype=np.float64), eps, None)
        out = v_np * np.log(rate_np) - rate_np - gammaln(v_np + 1.0)
        return _wrap_numpy(out)

    def entropy(self):
        gammaln = _require_scipy_special().gammaln
        rate = np.clip(np.asarray(self.rate.tensor(), dtype=np.float64), 1e-300, None)
        scalar_input = rate.ndim == 0
        rate_flat = np.atleast_1d(rate)
        out = np.empty_like(rate_flat)
        for i, lam in enumerate(rate_flat):
            if lam <= 30.0:
                K = int(np.ceil(lam + 8.0 * math.sqrt(lam) + 12))
                ks = np.arange(0, K + 1, dtype=np.float64)
                log_pmf = ks * math.log(lam) - lam - gammaln(ks + 1.0)
                pmf = np.exp(log_pmf)
                pmf = pmf / pmf.sum()
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.where(pmf > 0.0, -pmf * np.log(pmf), 0.0)
                out[i] = term.sum()
            else:
                out[i] = (0.5 * math.log(2.0 * math.pi * math.e * lam)
                          - 1.0 / (12.0 * lam)
                          - 1.0 / (24.0 * lam * lam)
                          - 19.0 / (360.0 * lam ** 3))
        out = out.reshape(rate.shape) if not scalar_input else out[0]
        return _wrap_numpy(out)

    def support(self):
        return "{0, 1, 2, ...}"
