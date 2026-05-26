"""Tenzor exception hierarchy.

Audit-8 II.12: re-export the typed exceptions from the C++ extension under
the ``tenzor.exceptions`` namespace so user code can catch them explicitly
without shadowing Python builtins on ``from tenzor import *``.

At runtime the exception classes are registered in
``python/bindings.cpp`` (``py::register_exception<...>``) and live as
attributes of the ``tenzor_core`` extension module. This module simply
re-exports them under stable, namespaced names.
"""

from __future__ import annotations

from .tenzor_core import (
    TenzorError,
    ShapeError,
    DTypeError,
    DeviceError,
    AutogradError,
    BackendError,
    TensorBoardError,
    # Python-parity typed exceptions. These shadow Python builtins on
    # ``from tenzor.exceptions import *`` — code that wants both should
    # qualify access (e.g. ``tenzor.exceptions.ValueError``) or rebind:
    #   from tenzor.exceptions import ValueError as TenzorValueError
    IndexError,
    MemoryError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
)

__all__ = [
    "TenzorError",
    "ShapeError",
    "DTypeError",
    "DeviceError",
    "AutogradError",
    "BackendError",
    "TensorBoardError",
    "IndexError",
    "MemoryError",
    "NotImplementedError",
    "RuntimeError",
    "TypeError",
    "ValueError",
]
