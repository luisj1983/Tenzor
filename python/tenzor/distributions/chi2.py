"""Chi-squared distribution — Tensor-native (sample only)."""
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


class Chi2(Distribution):
    """``Chi2(df)`` — chi-squared distribution.

    ``Chi2(df) = Gamma(df/2, 1/2)``.  ``rsample`` is reparameterised via the
    Gamma sampler, so gradients flow to ``df``. ``has_rsample = True``.
    """

    has_rsample = True

    def __init__(self, df):
        self.df = _to_variable(df)
        super().__init__(_shape_of(self.df))

    @property
    def mean(self):
        return self.df

    @property
    def variance(self):
        return 2.0 * self.df

    def sample(self, sample_shape=()):
        # Chi2(df) = Gamma(df/2, rate=1/2). Draw via the native Gamma sampler
        # (device-side, no NumPy) with params broadcast to the full shape.
        out_shape = list(tuple(sample_shape) + self._batch_shape)
        df_t = self.df.tensor() if hasattr(self.df, "tensor") else self.df
        alpha = df_t * 0.5
        if out_shape:
            ones = _tz.ones(out_shape)
            alpha = alpha * ones
        beta = alpha * 0.0 + 0.5  # rate = 1/2, matching alpha's shape/device
        return _tz.gamma_sample(alpha, beta)

    def rsample(self, sample_shape=()):
        # Chi2(df) = Gamma(df/2, 1/2); reparameterised via Gamma.rsample so the
        # gradient flows to df.
        from .gamma import Gamma
        return Gamma(self.df * 0.5, 0.5).rsample(sample_shape)

    def log_prob(self, value):
        # Chi2(k) = Gamma(k/2, 1/2). Autograd-aware (lgamma/log Variable
        # overloads): gradient flows to df and value.
        value = _to_variable(value)
        half_k = self.df * 0.5
        return ((half_k - 1.0) * _tz.log(value)
                - value * 0.5
                - half_k * math.log(2.0)
                - _tz.lgamma(half_k))

    def entropy(self):
        # H = k/2 + log 2 + lgamma(k/2) + (1 - k/2) ψ(k/2) — autograd-aware in df.
        half_k = self.df * 0.5
        return (half_k + math.log(2.0) + _tz.lgamma(half_k)
                + (1.0 - half_k) * _tz.digamma(half_k))

    def cdf(self, value):
        gammainc = _require_scipy_special().gammainc
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        k = np.asarray(self.df.tensor(), dtype=np.float64)
        out = np.where(v_np <= 0.0,
                       np.zeros_like(v_np),
                       gammainc(k / 2.0, v_np / 2.0))
        return _wrap_numpy(out)

    def icdf(self, q):
        gammaincinv = _require_scipy_special().gammaincinv
        q_np = np.asarray(_to_variable(q).tensor(), dtype=np.float64)
        k = np.asarray(self.df.tensor(), dtype=np.float64)
        return _wrap_numpy(2.0 * gammaincinv(k / 2.0, q_np))

    def support(self):
        return "(0, inf)"
