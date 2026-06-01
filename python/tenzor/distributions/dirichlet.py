"""Dirichlet distribution — Tensor-native (sample only)."""
from __future__ import annotations

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _to_variable,
    _wrap_numpy,
    _require_scipy_special,
)


class Dirichlet(Distribution):
    """``Dirichlet(concentration)`` distribution over the simplex.

    The trailing dimension of ``concentration`` is the event axis.
    ``rsample`` is reparameterised via the Gamma sampler
    (``Dirichlet(a) = G / sum(G)``, ``G_i ~ Gamma(a_i, 1)``), so gradients
    flow to the concentration. ``has_rsample = True``.
    """

    has_rsample = True

    def __init__(self, concentration):
        self.concentration = _to_variable(concentration)
        conc_shape = tuple(int(s) for s in self.concentration.shape)
        super().__init__(conc_shape[:-1])

    @property
    def mean(self):
        c_np = np.asarray(self.concentration.tensor(), dtype=np.float32)
        out = c_np / c_np.sum(axis=-1, keepdims=True)
        return _wrap_numpy(out)

    @property
    def variance(self):
        c_np = np.asarray(self.concentration.tensor(), dtype=np.float32)
        a0 = c_np.sum(axis=-1, keepdims=True)
        out = c_np * (a0 - c_np) / (a0 * a0 * (a0 + 1.0))
        return _wrap_numpy(out)

    def sample(self, sample_shape=()):
        # Dirichlet(a) = G / sum_last(G), G_i ~ Gamma(a_i, 1). Native device
        # Gamma draw (no NumPy), normalised over the last (event) axis.
        sample_shape = tuple(int(s) for s in sample_shape)
        alpha_t = self.concentration.tensor() if hasattr(self.concentration, "tensor") \
            else self.concentration
        # Tensor.shape is a property (list), not a method.
        alpha_shape = list(alpha_t.shape) if hasattr(alpha_t, "shape") else []
        out_shape = list(sample_shape) + alpha_shape
        if out_shape:
            ones = _tz.ones(out_shape)
            alpha_b = alpha_t * ones
        else:
            alpha_b = alpha_t
        beta_b = alpha_b * 0.0 + 1.0  # rate = 1
        g = _tz.gamma_sample(alpha_b, beta_b)
        return g / _tz.sum(g, -1, True)

    def rsample(self, sample_shape=()):
        # Dirichlet(a) = G / sum(G), G_i ~ Gamma(a_i, 1); reparameterised, so the
        # gradient flows to the concentration.
        from .gamma import Gamma
        g = Gamma(self.concentration, 1.0).rsample(sample_shape)
        return g / _tz.sum(g, -1, True)

    def log_prob(self, value):
        # log p = lgamma(sum a) - sum lgamma(a) + sum (a-1) log x  (over event axis)
        # Autograd-aware (lgamma/log/sum Variable overloads): grad flows to the
        # concentration and value.
        value = _to_variable(value)
        a = self.concentration
        log_norm = _tz.lgamma(_tz.sum(a, -1)) - _tz.sum(_tz.lgamma(a), -1)
        return log_norm + _tz.sum((a - 1.0) * _tz.log(value), -1)

    def entropy(self):
        # H = log B(a) + (a0 - K) ψ(a0) - sum (a_i - 1) ψ(a_i), a0 = sum a_i,
        # log B(a) = sum lgamma(a_i) - lgamma(a0). Autograd-aware in concentration.
        a = self.concentration
        k = float(a.shape[-1])
        a0 = _tz.sum(a, -1)
        log_b = _tz.sum(_tz.lgamma(a), -1) - _tz.lgamma(a0)
        return (log_b
                + (a0 - k) * _tz.digamma(a0)
                - _tz.sum((a - 1.0) * _tz.digamma(a), -1))

    def support(self):
        return "simplex (sum to 1, all >= 0)"
