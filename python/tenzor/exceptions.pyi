"""Type stubs for tenzor exception hierarchy.

Audit-8 II.12: the typed exceptions ``IndexError`` / ``MemoryError`` /
``NotImplementedError`` / ``RuntimeError`` / ``TypeError`` / ``ValueError``
collide with the Python builtins when imported at the root of ``tenzor``.
They are surfaced here under the ``tenzor.exceptions`` namespace so user
code can write ``except tenzor.exceptions.ValueError`` without shadowing
the builtin in their own module.

At runtime these names are still attributes of the ``tenzor_core``
extension module (registered via ``py::register_exception`` in
``python/bindings.cpp``); this stub simply gives the public-facing
namespace a typed surface.
"""

from __future__ import annotations

# Base hierarchy — all derive from RuntimeError
class TenzorError(RuntimeError):
    """Root of the Tenzor exception hierarchy."""

class ShapeError(TenzorError):
    """Raised when tensor shapes are incompatible with the requested op."""

class DTypeError(TenzorError):
    """Raised when a tensor's dtype is incompatible with the requested op."""

class DeviceError(TenzorError):
    """Raised on device mismatches or missing backend availability."""

class AutogradError(TenzorError):
    """Raised by the autograd engine for graph / gradient errors."""

class BackendError(TenzorError):
    """Raised by a backend kernel for an unrecoverable internal failure."""

class TensorBoardError(TenzorError):
    """Raised by the TensorBoard exporter."""

# Python-parity exceptions — registered under
# (PyExc_IndexError / PyExc_ValueError / ...) so they catch like the builtins
# but the named classes are exposed for explicit ``except tenzor.exceptions.X``.
class IndexError(IndexError):  # type: ignore[misc]
    """Raised on out-of-range indexing of a tensor or sequence."""

class MemoryError(MemoryError):  # type: ignore[misc]
    """Raised when a tensor allocation cannot be satisfied."""

class NotImplementedError(NotImplementedError):  # type: ignore[misc]
    """Raised for ops not yet implemented on a given backend."""

class RuntimeError(RuntimeError):  # type: ignore[misc]
    """Generic runtime error that doesn't fit another typed exception."""

class TypeError(TypeError):  # type: ignore[misc]
    """Raised when an argument type is wrong (mirrors Python ``TypeError``)."""

class ValueError(ValueError):  # type: ignore[misc]
    """Raised when an argument value is invalid (mirrors Python ``ValueError``)."""
