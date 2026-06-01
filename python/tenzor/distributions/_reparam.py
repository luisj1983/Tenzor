"""Shared reparameterisation primitives for ``tenzor.distributions``.

The location-scale families (Normal, Uniform, Laplace, Exponential,
Cauchy, Gumbel, ...) all build their reparameterised sample by
combining the distribution parameters (Variables, possibly with
``requires_grad=True``) with a *detached* standard variate drawn from a
fixed reference distribution.

Keeping the standard variate detached is correct:

* the user wants gradients only through the *parameters*;
* the reference distribution has no parameters of its own to
  differentiate against;
* numpy / tenzor RNG ops have no gradient anyway.

This module centralises the standard-variate helpers so each
``Distribution`` subclass stays small.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore


def _shape_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(s) for s in shape)


def standard_normal(shape: Sequence[int]) -> Tensor:
    """Draw a standard normal sample with the given shape.

    Returns a *Tensor* (no autograd attached).  Callers wrap it into a
    Variable manually when composing with parameter Variables.
    """
    return _tz.randn(list(_shape_tuple(shape)))


def standard_uniform(shape: Sequence[int], eps: float = 0.0) -> Tensor:
    """Draw a uniform sample on ``[eps, 1 - eps)``.

    ``eps > 0`` is useful for distributions that need ``log(U)`` /
    ``log(1 - U)`` and want to avoid ``-inf``.
    """
    u = _tz.rand(list(_shape_tuple(shape)))
    if eps > 0.0:
        # u in [0, 1)  →  u in [eps, 1 - eps)
        scale = 1.0 - 2.0 * eps
        # Tensor * float → Tensor; addition broadcasts the scalar offset.
        return u * scale + eps
    return u


def standard_exponential(shape: Sequence[int], eps: float = 1e-7) -> Tensor:
    """Draw an Exponential(rate=1) sample.

    Uses the inverse-CDF route ``-log(U)`` with ``U`` clamped away from
    0 to keep ``log`` finite.
    """
    u = standard_uniform(shape, eps=eps)
    # Tensor-level log + neg → Tensor (no Variable involved).
    return _tz.neg(_tz.log(u)) if hasattr(_tz, "neg") else (0.0 - _tz.log(u))


def standard_gumbel(shape: Sequence[int], eps: float = 1e-7) -> Tensor:
    """Draw a Gumbel(0, 1) sample via ``-log(-log(U))``."""
    u = standard_uniform(shape, eps=eps)
    log_u = _tz.log(u)
    # -log(-log_u) = -log(neg(log_u))
    inner = 0.0 - log_u
    return 0.0 - _tz.log(inner)


def standard_cauchy(shape: Sequence[int], eps: float = 1e-7) -> Tensor:
    """Draw a standard Cauchy sample via ``tan(pi * (U - 0.5))``.

    Computed at the Tensor level so the result is detached from the
    autograd graph — the caller wraps as needed.
    """
    import math
    u = standard_uniform(shape, eps=eps)
    # tan(pi * (u - 0.5)) — pi*(u-0.5) is a Tensor; tan(Tensor) is fine.
    return _tz.tan((u - 0.5) * math.pi)


def gamma_sample(concentration, rate=None, shape: Sequence[int] = ()) -> Tensor:
    """Draw a Gamma(α, β) sample via the native Marsaglia-Tsang kernel.

    Uses ``tz.gamma_sample`` (device-side, no NumPy round-trip). concentration
    and rate are materialised to the requested output ``shape`` (or their
    broadcast shape) so every draw carries its own (α, β). Returns a detached
    Tensor on the parameter tensor's device.
    """
    def _as_tensor(v):
        if isinstance(v, Variable):
            return v.tensor()
        return v  # Tensor (or scalar handled below)

    a_t = _as_tensor(concentration)
    b_t = _as_tensor(rate) if rate is not None else None

    # Determine the explicit output shape (caller-supplied wins; else the
    # broadcast of the two parameter shapes).
    if shape:
        out_shape = list(_shape_tuple(shape))
    else:
        # Tensor.shape is a property (list), not a method.
        a_shape = list(a_t.shape) if hasattr(a_t, "shape") else []
        b_shape = list(b_t.shape) if (b_t is not None and hasattr(b_t, "shape")) else []
        out_shape = a_shape if len(a_shape) >= len(b_shape) else b_shape

    if out_shape:
        ones = _tz.ones(out_shape)
        a_t = a_t * ones
        b_t = (b_t * ones) if b_t is not None else ones  # rate=None -> β=1
    elif b_t is None:
        # Scalar α with no shape and rate=None: β = 1 of matching shape.
        b_t = _tz.ones(list(a_t.shape) if hasattr(a_t, "shape") else [])

    return _tz.gamma_sample(a_t, b_t)


__all__ = [
    "standard_normal",
    "standard_uniform",
    "standard_exponential",
    "standard_gumbel",
    "standard_cauchy",
    "gamma_sample",
]
