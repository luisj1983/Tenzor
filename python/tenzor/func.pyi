"""Type stubs for tenzor.func module (composable function transforms)."""

from __future__ import annotations
from typing import Any, Callable, Tuple
from tenzor import Tensor, Variable


def grad(f: Callable[[Variable], Variable]) -> Callable[[Variable], Variable]:
    """Return a function that computes the gradient of ``f``."""
    ...


def vmap(
    f: Callable[..., Any],
    in_dim: int = 0,
    out_dim: int = 0,
) -> Callable[..., Any]:
    """Return a vectorized version of ``f`` that maps over a batch dimension."""
    ...


def jacrev(f: Callable[[Variable], Variable]) -> Callable[[Variable], Variable]:
    """Return a function that computes the reverse-mode Jacobian of ``f``."""
    ...


def jacfwd(f: Callable[[Variable], Variable]) -> Callable[[Variable], Variable]:
    """Return a function that computes the forward-mode Jacobian of ``f``."""
    ...


def hessian(f: Callable[[Variable], Variable]) -> Callable[[Variable], Variable]:
    """Return a function that computes the Hessian of scalar-valued ``f``."""
    ...


def jvp(
    f: Callable[[Variable], Variable],
    x: Variable,
    tangent: Tensor,
    mode: str = "walker",
) -> Tuple[Variable, Tensor]:
    """Forward-mode Jacobian-vector product. Returns ``(output, J_f(x) @ tangent)``."""
    ...


def hvp(
    f: Callable[[Variable], Variable],
    x: Variable,
    v: Tensor,
) -> Tuple[Variable, Tensor]:
    """Hessian-vector product. Returns ``(output, H_f(x) @ v)``."""
    ...


def vhp(
    f: Callable[[Variable], Variable],
    x: Variable,
    v: Tensor,
) -> Tuple[Variable, Tensor]:
    """Vector-Hessian product. Returns ``(output, v^T @ H_f(x))``."""
    ...


def vjp(
    f: Callable[[Variable], Variable],
    x: Variable,
    cotangent: Tensor,
) -> Tuple[Variable, Tensor]:
    """Vector-Jacobian product. Returns ``(output, cotangent @ J_f(x))``."""
    ...
