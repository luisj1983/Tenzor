"""Chi-squared distribution — Tensor-native (sample only)."""
from __future__ import annotations

import math

import numpy as np

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

    ``Chi2(df) = Gamma(df/2, 1/2)``.  Reparameterisation routes through
    Gamma, which Tenzor does not yet expose with autograd; therefore
    ``has_rsample = False``.
    """

    has_rsample = False

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
        out_shape = tuple(sample_shape) + self._batch_shape
        df_np = np.asarray(self.df.tensor(), dtype=np.float64)
        return _wrap_numpy(np.asarray(
            np.random.chisquare(df=df_np, size=out_shape or None),
            dtype=np.float32,
        ))

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        k = np.asarray(self.df.tensor(), dtype=np.float64)
        out = ((k / 2.0 - 1.0) * np.log(v_np)
               - v_np / 2.0
               - (k / 2.0) * math.log(2.0)
               - gammaln(k / 2.0))
        return _wrap_numpy(out)

    def entropy(self):
        scipy_special = _require_scipy_special()
        k = np.asarray(self.df.tensor(), dtype=np.float64)
        out = (k / 2.0 + math.log(2.0) + scipy_special.gammaln(k / 2.0)
               + (1.0 - k / 2.0) * scipy_special.digamma(k / 2.0))
        return _wrap_numpy(out)

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
