"""Dirichlet distribution — Tensor-native (sample only)."""
from __future__ import annotations

import numpy as np

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
    Dirichlet reparameterisation routes through Gamma, which Tenzor's
    sampler does not yet expose — ``has_rsample = False``.
    """

    has_rsample = False

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
        sample_shape = tuple(int(s) for s in sample_shape)
        alpha_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        output_shape = sample_shape + alpha_np.shape
        alpha_b = np.broadcast_to(alpha_np, output_shape)
        g = np.random.gamma(shape=alpha_b, scale=1.0, size=output_shape)
        s = g / g.sum(axis=-1, keepdims=True)
        return _wrap_numpy(np.asarray(s, dtype=np.float32))

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        a_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        log_norm = gammaln(a_np.sum(axis=-1)) - gammaln(a_np).sum(axis=-1)
        out = log_norm + ((a_np - 1.0) * np.log(v_np)).sum(axis=-1)
        return _wrap_numpy(out)

    def entropy(self):
        scipy_special = _require_scipy_special()
        a_np = np.asarray(self.concentration.tensor(), dtype=np.float64)
        k = a_np.shape[-1]
        a0 = a_np.sum(axis=-1)
        log_b = scipy_special.gammaln(a_np).sum(axis=-1) - scipy_special.gammaln(a0)
        out = (log_b
               + (a0 - k) * scipy_special.digamma(a0)
               - ((a_np - 1.0) * scipy_special.digamma(a_np)).sum(axis=-1))
        return _wrap_numpy(out)

    def support(self):
        return "simplex (sum to 1, all >= 0)"
