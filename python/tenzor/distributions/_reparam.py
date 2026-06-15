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
from tenzor.tenzor_core import Tensor  # type: ignore


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


__all__ = [
    "standard_normal",
    "standard_uniform",
    "standard_exponential",
    "standard_gumbel",
    "standard_cauchy",
]
