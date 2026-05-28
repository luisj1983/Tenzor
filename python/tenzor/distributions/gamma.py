"""Gamma distribution — Tensor-native (sample only).

``rsample`` for Gamma requires either Marsaglia-Tsang rejection sampling
with implicit-reparameterisation gradients (Figurnov et al. 2018) or a
custom autograd Function — both are non-trivial.  This implementation
therefore exposes ``has_rsample = False`` and falls back to a numpy
Marsaglia-Tsang sample wrapped as a Tensor.  Adding proper
reparameterisation is left as a follow-up.
"""
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


class Gamma(Distribution):
    """``Gamma(concentration, rate)`` distribution.

    Args:
        concentration: shape parameter α (> 0).
        rate: rate parameter β (> 0); scale = 1/β.
    """

    has_rsample = False

    def __init__(self, concentration, rate):
        self.concentration = _to_variable(concentration)
        self.rate = _to_variable(rate)
        super().__init__(_broadcast_shape(_shape_of(self.concentration),
                                          _shape_of(self.rate)))

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate * self.rate)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        alpha_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        beta_np = np.asarray(self.rate.tensor(), dtype=np.float64)
        # numpy gamma uses shape α and scale 1/β.
        samples = np.random.gamma(shape=alpha_np, scale=1.0 / beta_np,
                                  size=out_shape or None)
        return _wrap_numpy(np.asarray(samples, dtype=np.float32))

    def log_prob(self, value):
        # log p(x) = α·log(β) - lgamma(α) + (α - 1)·log(x) - β·x
        # tz.special.lgamma lacks a Variable overload (audit gap) so we
        # compute log_prob via scipy on numpy buffers and return a Tensor.
        gammaln = _require_scipy_special().gammaln
        value_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        a_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        b_np = np.asarray(self.rate.tensor(), dtype=np.float64)
        out = (a_np * np.log(b_np) - gammaln(a_np)
               + (a_np - 1.0) * np.log(value_np)
               - b_np * value_np)
        return _wrap_numpy(out)

    def entropy(self):
        # H = α - log(β) + lgamma(α) + (1-α)·ψ(α)
        scipy_special = _require_scipy_special()
        a_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        b_np = np.asarray(self.rate.tensor(), dtype=np.float64)
        out = (a_np - np.log(b_np)
               + scipy_special.gammaln(a_np)
               + (1.0 - a_np) * scipy_special.digamma(a_np))
        return _wrap_numpy(out)

    def cdf(self, value):
        gammainc = _require_scipy_special().gammainc
        value_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        a_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        b_np = np.asarray(self.rate.tensor(), dtype=np.float64)
        out = np.where(value_np <= 0.0,
                       np.zeros_like(value_np),
                       gammainc(a_np, b_np * value_np))
        return _wrap_numpy(out)

    def icdf(self, q):
        gammaincinv = _require_scipy_special().gammaincinv
        q_np = np.asarray(_to_variable(q).tensor(), dtype=np.float64)
        a_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        b_np = np.asarray(self.rate.tensor(), dtype=np.float64)
        return _wrap_numpy(gammaincinv(a_np, q_np) / b_np)

    def support(self):
        return "(0, inf)"
