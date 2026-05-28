"""Student's t distribution — Tensor-native (sample only).

Proper reparameterisation for StudentT(df) routes through a Chi-square
sample (``z / sqrt(chi2/df)``); since Chi2/Gamma have no reparam path
in this implementation, StudentT.rsample is also unavailable.
"""
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


class StudentT(Distribution):
    """``StudentT(df, loc=0, scale=1)`` — Student's t distribution."""

    has_rsample = False

    def __init__(self, df, loc=0.0, scale=1.0):
        self.df = _to_variable(df)
        self.loc = _to_variable(loc)
        self.scale = _to_variable(scale)
        super().__init__(_broadcast_shape(_shape_of(self.df),
                                          _shape_of(self.loc),
                                          _shape_of(self.scale)))

    @property
    def mean(self):
        df_np = np.asarray(self.df.tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        return _wrap_numpy(np.where(df_np > 1.0, loc_np, np.nan))

    @property
    def variance(self):
        df_np = np.asarray(self.df.tensor(), dtype=np.float64)
        s_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        s2 = s_np ** 2
        var = s2 * df_np / (df_np - 2.0)
        out = np.where(df_np > 2.0, var, np.where(df_np > 1.0, np.inf, np.nan))
        return _wrap_numpy(out)

    def sample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        df_np = np.asarray(self.df.tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        z = np.random.standard_t(df_np, size=out_shape or None)
        return _wrap_numpy(np.asarray(loc_np + scale_np * z, dtype=np.float32))

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        nu = np.asarray(self.df.tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        y = (v_np - loc_np) / scale_np
        log_unnorm = gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(nu * math.pi) - gammaln(nu / 2.0)
        log_kernel = -0.5 * (nu + 1.0) * np.log(1.0 + y * y / nu)
        return _wrap_numpy(log_unnorm + log_kernel - np.log(scale_np))

    def cdf(self, value):
        scipy_t = _require_scipy_stats().t
        v_np = np.asarray(_to_variable(value).tensor(), dtype=np.float64)
        df_np = np.asarray(self.df.tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        return _wrap_numpy(scipy_t.cdf((v_np - loc_np) / scale_np, df=df_np))

    def icdf(self, p):
        scipy_t = _require_scipy_stats().t
        p_np = np.asarray(_to_variable(p).tensor(), dtype=np.float64)
        df_np = np.asarray(self.df.tensor(), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        return _wrap_numpy(loc_np + scale_np * scipy_t.ppf(p_np, df=df_np))

    def entropy(self):
        scipy_special = _require_scipy_special()
        digamma = scipy_special.digamma
        betaln = scipy_special.betaln
        nu = np.asarray(self.df.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = (np.log(scale_np)
               + 0.5 * (nu + 1.0) * (digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0))
               + 0.5 * np.log(nu) + betaln(nu / 2.0, 0.5))
        return _wrap_numpy(out)

    def support(self):
        return "(-inf, inf)"
