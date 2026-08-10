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

# S23: explicit ``builtins.X`` references avoid the recursive-base shadow
# that occurs when a stub names its class the same as the builtin it
# inherits from (``class NotImplementedError(NotImplementedError)`` resolved
# the base name to the just-being-declared class). Importing ``builtins``
# under aliases keeps the public stub names (``IndexError`` /
# ``NotImplementedError`` / …) while disambiguating the bases.
import builtins as _builtins

# Base hierarchy — all derive from RuntimeError
class TenzorError(_builtins.RuntimeError):
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
# S23: bases are spelled ``_builtins.X`` to avoid the recursive-base name
# resolution where ``class IndexError(IndexError)`` resolved the base to
# the same not-yet-finished class.
class IndexError(_builtins.IndexError):
    """Raised on out-of-range indexing of a tensor or sequence."""

class MemoryError(TenzorError, _builtins.MemoryError):
    """Raised when a tensor allocation cannot be satisfied."""

class NotImplementedError(_builtins.NotImplementedError):
    """Raised for ops not yet implemented on a given backend."""

class RuntimeError(_builtins.RuntimeError):
    """Generic runtime error that doesn't fit another typed exception."""

class TypeError(_builtins.TypeError):
    """Raised when an argument type is wrong (mirrors Python ``TypeError``)."""

class ValueError(_builtins.ValueError):
    """Raised when an argument value is invalid (mirrors Python ``ValueError``)."""
