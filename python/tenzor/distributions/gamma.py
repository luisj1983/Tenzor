"""Gamma distribution — Tensor/Variable native, with reparameterised rsample.

``rsample`` uses implicit reparameterisation gradients (Figurnov et al. 2018):
the sample is drawn with fixed randomness and the pathwise gradient is
``dz/dalpha = -dP(alpha,z_u)/dalpha / pdf(z_u; alpha)`` (z_u ~ Gamma(alpha,1)),
with ``dP/dalpha`` computed by central difference of the regularised lower
incomplete gamma ``tz.gammainc`` (exact to O(h^2)). ``has_rsample = True``.
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


# Implicit-reparameterisation sampler for Gamma, wrapped as a custom autograd
# Function. Built lazily so the class body does not reference tenzor.autograd at
# import time (circular-import safe during package initialisation).
_GAMMA_REPARAM = None


def _gamma_reparam():
    global _GAMMA_REPARAM
    if _GAMMA_REPARAM is not None:
        return _GAMMA_REPARAM

    class _GammaReparam(_tz.autograd.Function):
        @staticmethod
        def forward(ctx, alpha, rate):
            a64 = np.asarray(alpha.tensor() if hasattr(alpha, "tensor") else alpha,
                             dtype=np.float64)
            r64 = np.asarray(rate.tensor() if hasattr(rate, "tensor") else rate,
                             dtype=np.float64)
            # z_unit ~ Gamma(alpha, 1); z = z_unit / rate. Randomness is held
            # fixed (the gradient comes from the implicit-reparam formula).
            z_unit = np.random.gamma(shape=np.maximum(a64, 1e-300), size=a64.shape)
            z = (z_unit / r64).astype(np.float32)
            ctx.save_for_backward(
                Tensor.from_numpy(np.ascontiguousarray(a64)),
                Tensor.from_numpy(np.ascontiguousarray(z_unit)),
                Tensor.from_numpy(np.ascontiguousarray(r64)),
            )
            # forward must return a Tensor (apply wraps it into a Variable).
            return Tensor.from_numpy(np.ascontiguousarray(z))

        @staticmethod
        def backward(ctx, grad_z):
            a_t, zu_t, r_t = ctx.saved_tensors
            a = np.asarray(a_t, dtype=np.float64)
            zu = np.asarray(zu_t, dtype=np.float64)
            r = np.asarray(r_t, dtype=np.float64)
            g = np.asarray(grad_z.tensor() if hasattr(grad_z, "tensor") else grad_z,
                           dtype=np.float64)

            def _P(aa):  # regularised lower incomplete gamma P(aa, zu) = γ/Γ
                # tz.gammainc returns the UNREGULARISED lower incomplete gamma
                # γ(a,x); divide by Γ(a)=exp(lgamma(a)) to get the CDF P(a,x).
                g = np.asarray(
                    _tz.gammainc(Tensor.from_numpy(np.ascontiguousarray(aa)),
                                 Tensor.from_numpy(np.ascontiguousarray(zu))),
                    dtype=np.float64)
                lga = np.asarray(
                    _tz.lgamma(Tensor.from_numpy(np.ascontiguousarray(aa))),
                    dtype=np.float64)
                return g * np.exp(-lga)

            h = 1e-4
            dP_da = (_P(a + h) - _P(a - h)) / (2.0 * h)
            lg = np.asarray(_tz.lgamma(Tensor.from_numpy(np.ascontiguousarray(a))),
                            dtype=np.float64)
            pdf = np.exp((a - 1.0) * np.log(zu) - zu - lg)
            pdf = np.where(pdf < 1e-300, 1e-300, pdf)
            dzu_da = -dP_da / pdf

            grad_alpha = (g * (dzu_da / r)).astype(np.float32)
            grad_rate = (g * (-zu / (r * r))).astype(np.float32)
            return (Tensor.from_numpy(np.ascontiguousarray(grad_alpha)),
                    Tensor.from_numpy(np.ascontiguousarray(grad_rate)))

    _GAMMA_REPARAM = _GammaReparam
    return _GAMMA_REPARAM


class Gamma(Distribution):
    """``Gamma(concentration, rate)`` distribution.

    Args:
        concentration: shape parameter α (> 0).
        rate: rate parameter β (> 0); scale = 1/β.
    """

    has_rsample = True

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

    def rsample(self, sample_shape=()):
        # Reparameterised sample with implicit-reparam gradient (see module
        # docstring / _gamma_reparam). Broadcast alpha/rate to the full sample
        # shape so each draw carries its own (alpha, rate) for the gradient.
        out_shape = list(tuple(sample_shape) + self._batch_shape)
        if not out_shape:
            out_shape = [1]
        ones = Variable(_tz.ones(out_shape), False)
        a_exp = self.concentration * ones
        r_exp = self.rate * ones
        return _gamma_reparam().apply(a_exp, r_exp)

    def log_prob(self, value):
        # log p(x) = α·log(β) - lgamma(α) + (α - 1)·log(x) - β·x
        # Computed with autograd-aware Variable ops (lgamma/log have Variable
        # overloads), so gradients flow to concentration, rate and value —
        # no scipy/numpy detachment.
        value = _to_variable(value)
        a = self.concentration
        b = self.rate
        return (a * _tz.log(b) - _tz.lgamma(a)
                + (a - 1.0) * _tz.log(value)
                - b * value)

    def entropy(self):
        # H = α - log(β) + lgamma(α) + (1-α)·ψ(α) — autograd-aware in α, β.
        a = self.concentration
        b = self.rate
        return (a - _tz.log(b) + _tz.lgamma(a)
                + (1.0 - a) * _tz.digamma(a))

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
