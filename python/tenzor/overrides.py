"""Tenzor subclass-override protocol (Phase 2.6).

Tenzor exposes a subclass-override hook modeled on PyTorch's
``__torch_function__`` and NumPy's ``__array_function__``. It lets
Tensor subclasses intercept tenzor API calls that receive them as
arguments and provide their own implementation — for example, a
unit-aware tensor that wants to reject mixed-unit additions, or a
tracing wrapper that records every op call.

The hook is named ``__tensor_function__`` rather than
``__torch_function__`` so a single Python process can carry both
Tenzor subclasses and PyTorch subclasses without collisions.

Usage
-----

Define a subclass with a class-method ``__tensor_function__``::

    class TracingTensor(tz.Tensor):
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            print(f"call {func.__name__} with {len(args)} args")
            # Delegate to the default implementation by dropping the
            # subclass type from `types` and re-calling `func`.
            kwargs = kwargs or {}
            return func(*args, **kwargs)

Any Tenzor public API that has been wrapped via :func:`implements` (or
invoked through :func:`handle_tensor_function`) will route through the
subclass's ``__tensor_function__`` when it receives an instance of the
subclass as a positional argument.

Semantics match PyTorch: the highest-priority subclass (the first one
encountered, with `issubclass` breaking ties in favor of the deepest
class) receives the call. If the hook returns ``NotImplemented``, the
next candidate is tried; if every hook returns ``NotImplemented``, the
default implementation runs.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Iterable, Tuple


# The marker the protocol looks for. Exposed so users can
# `hasattr(t, tz.overrides.TENSOR_FUNCTION)` without hardcoding the name.
TENSOR_FUNCTION = "__tensor_function__"


def _overloaded_types(args: Iterable[Any]) -> Tuple[type, ...]:
    """Return the (ordered, deduplicated) set of types in ``args`` that
    define ``__tensor_function__``. Preserves the first-seen order,
    matching NumPy / PyTorch convention."""
    seen = []
    for a in args:
        t = type(a)
        if t in seen:
            continue
        if hasattr(t, TENSOR_FUNCTION):
            seen.append(t)
    return tuple(seen)


def has_tensor_function(args: Iterable[Any]) -> bool:
    """Return True if any positional argument has a ``__tensor_function__``
    hook — i.e. is an instance of a subclass that wants to intercept.

    This is the cheap fast-path check callers can use before building
    the slower ``handle_tensor_function`` dispatch:

        if tz.overrides.has_tensor_function(args):
            return tz.overrides.handle_tensor_function(api, args, kwargs)
    """
    for a in args:
        if hasattr(type(a), TENSOR_FUNCTION):
            return True
    return False


def handle_tensor_function(
    public_api: Callable,
    args: Tuple[Any, ...],
    kwargs: dict | None = None,
) -> Any:
    """Dispatch a public API call through the ``__tensor_function__``
    protocol. At least one of ``args`` must define the hook — otherwise
    this raises ``TypeError`` to signal a misuse.

    Iterates the overloaded types in "most-derived first" order (the
    order NumPy/PyTorch use: MRO-deepest wins for ties) and calls each
    class's ``__tensor_function__`` with ``(public_api, types, args, kwargs)``.
    The first return value that is not ``NotImplemented`` wins. If every
    hook defers, raises ``TypeError`` — there's no default to fall
    back to at this layer (the caller is expected to handle that).

    Parameters
    ----------
    public_api: Callable
        The tenzor API function the user originally called. Its name
        identifies which op the subclass is intercepting.
    args: tuple
        Positional arguments from the original call.
    kwargs: dict, optional
        Keyword arguments from the original call.
    """
    kwargs = kwargs or {}
    overloaded = _overloaded_types(args)
    if not overloaded:
        raise TypeError(
            f"handle_tensor_function({public_api!r}): no argument has "
            f"__tensor_function__; call the default implementation directly")

    # Sort so that the most-derived class runs first. `issubclass(a, b)`
    # means a is a subclass of b, so a is "more derived".
    def precedence_key(t: type) -> int:
        # More-derived types get smaller keys (higher priority).
        return -len(t.__mro__)

    ordered = sorted(overloaded, key=precedence_key)

    for t in ordered:
        hook = getattr(t, TENSOR_FUNCTION)
        result = hook(public_api, ordered, args, kwargs)
        if result is not NotImplemented:
            return result

    raise TypeError(
        f"no __tensor_function__ implementation handled {public_api.__name__}; "
        f"overloaded types: {[t.__name__ for t in ordered]}")


def implements(public_api: Callable) -> Callable:
    """Decorator that wraps a tenzor public API so calls through it run
    the ``__tensor_function__`` protocol before dispatching to the real
    implementation.

    Use it in ``tenzor/__init__.py`` to opt a curated set of user-facing
    ops into the protocol without touching the 500+ pybind11 bindings:

        import tenzor.overrides as _ov
        add = _ov.implements(add)
        mul = _ov.implements(mul)

    The returned wrapper has the same name and signature as ``public_api``
    and delegates to it when no subclass is involved.
    """
    @wraps(public_api)
    def wrapper(*args, **kwargs):
        if has_tensor_function(args):
            return handle_tensor_function(wrapper, args, kwargs)
        return public_api(*args, **kwargs)

    # Expose the original for callers that want to bypass the protocol.
    wrapper.__wrapped_api__ = public_api
    return wrapper


def get_default_nowrap_functions() -> set:
    """Return the set of public tenzor functions that are deliberately NOT
    wrapped by the override protocol.

    Currently empty — every wrapped op goes through the hook. Matches
    torch.overrides.get_default_nowrap_functions() for API symmetry.
    """
    return set()
