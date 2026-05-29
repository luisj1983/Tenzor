"""Student's t distribution — Tensor-native (sample only).

Proper reparameterisation for StudentT(df) routes through a Chi-square
sample (``z / sqrt(chi2/df)``); since Chi2/Gamma have no reparam path
in this implementation, StudentT.rsample is also unavailable.
"""
from __future__ import annotations

import math

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
    _require_scipy_stats,
)


class StudentT(Distribution):
    """``StudentT(df, loc=0, scale=1)`` — Student's t distribution."""

    has_rsample = True

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

    def rsample(self, sample_shape=()):
        # T = loc + scale * Z / sqrt(V/df), Z ~ N(0,1) (noise), V ~ Chi2(df).
        # 1/sqrt(x) is written exp(-0.5 log x) to stay autograd-aware; df flows
        # through V (reparameterised Chi2) and the explicit /df.
        from .chi2 import Chi2
        out_shape = tuple(sample_shape) + self._batch_shape
        z_np = np.random.standard_normal(out_shape if out_shape else 1).astype(np.float32)
        z = Variable(Tensor.from_numpy(np.ascontiguousarray(z_np)), False)
        v = Chi2(self.df).rsample(sample_shape)
        inv_sqrt = _tz.exp(-0.5 * _tz.log(v / self.df))
        return self.loc + self.scale * z * inv_sqrt

    def log_prob(self, value):
        # Autograd-aware (lgamma/log Variable overloads): gradient flows to df,
        # loc, scale and value.
        value = _to_variable(value)
        nu = self.df
        y = (value - self.loc) / self.scale
        log_unnorm = (_tz.lgamma((nu + 1.0) * 0.5)
                      - 0.5 * _tz.log(nu * math.pi)
                      - _tz.lgamma(nu * 0.5))
        log_kernel = -0.5 * (nu + 1.0) * _tz.log(1.0 + (y * y) / nu)
        return log_unnorm + log_kernel - _tz.log(self.scale)

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
        # H = log(scale) + 0.5(ν+1)[ψ((ν+1)/2) - ψ(ν/2)] + 0.5 log ν + B(ν/2, 1/2)
        # with B(ν/2,1/2) = lgamma(ν/2) + lgamma(1/2) - lgamma(ν/2 + 1/2).
        # Autograd-aware in df and scale.
        nu = self.df
        half_nu = nu * 0.5
        betaln = (_tz.lgamma(half_nu) + math.lgamma(0.5)
                  - _tz.lgamma(half_nu + 0.5))
        return (_tz.log(self.scale)
                + 0.5 * (nu + 1.0) * (_tz.digamma((nu + 1.0) * 0.5) - _tz.digamma(half_nu))
                + 0.5 * _tz.log(nu) + betaln)

    def support(self):
        return "(-inf, inf)"
