"""tz.jit — MLIR/StableHLO/IREE pipeline (Phase 13).

@tz.jit is a decorator that traces a Python forward pass and compiles it
through the MLIR/StableHLO/IREE pipeline. Works on all 4 IREE targets in
MVP-1: llvm-cpu, cuda, vulkan, rocm.

When TENZOR_USE_MLIR_JIT is OFF at build time, calling a jit-decorated
function raises JitNotEnabledError with a clear message.
"""
from __future__ import annotations
from typing import Callable, Any
import functools


class JitNotEnabledError(RuntimeError):
    """Raised when @tz.jit is called but the C++ build was configured without
    TENZOR_USE_MLIR_JIT=ON."""


def _compile_function(fn: Callable, *, backend: str, target: str,
                      fallback_to_eager: bool) -> Any:
    """Delegate to the C++ binding if available; raise JitNotEnabledError otherwise."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        return _core.jit.compile_function(
            fn, backend=backend, target=target,
            fallback_to_eager=fallback_to_eager)
    except AttributeError:
        # tenzor_core exists but jit submodule not compiled in (MLIR JIT OFF).
        class _NotEnabled:
            """Placeholder returned when MLIR JIT is not enabled at build time."""
            def __init__(self, fn_: Callable):
                self._fn = fn_
                functools.update_wrapper(self, fn_)
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                raise JitNotEnabledError(
                    f"@tz.jit: MLIR JIT is not enabled in this build. "
                    f"Rebuild Tenzor with -DTENZOR_USE_MLIR_JIT=ON to enable it. "
                    f"See docs/jit-mlir-setup.md for installation instructions.")
        return _NotEnabled(fn)
    except ImportError:
        # tenzor_core not found at all — still return a helpful placeholder.
        class _NoCoreNotEnabled:  # type: ignore[no-redef]
            def __init__(self, fn_: Callable):
                self._fn = fn_
                functools.update_wrapper(self, fn_)
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                raise JitNotEnabledError(
                    "@tz.jit: tenzor_core C++ module not found. "
                    "Build the project and set PYTHONPATH to include build/python.")
        return _NoCoreNotEnabled(fn)


def jit(fn: Callable[..., Any] | None = None, *,
        target: str = "auto",
        fallback_to_eager: bool = False) -> Callable[..., Any]:
    """Decorate `fn` to compile and execute via the MLIR pipeline.

    Args:
        fn: function to JIT. If None, returns a parametrised decorator.
        target: "auto" (default), "llvm-cpu", "cuda", "vulkan", "rocm".
        fallback_to_eager: if True, ops not yet supported by the MLIR
            lowering fall back to eager execution silently. Default False
            — coverage gaps throw JitLoweringError loudly.

    Returns:
        A wrapped function that traces on first call (per shape+dtype),
        compiles via iree-compile, caches the compiled artifact, and
        executes via the IREE runtime on subsequent calls.
    """
    def _wrap(f: Callable) -> Callable:
        compiled = _compile_function(
            f, backend="mlir", target=target,
            fallback_to_eager=fallback_to_eager)

        @functools.wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            return compiled(*args, **kwargs)

        _call._tz_compiled = compiled  # type: ignore[attr-defined]
        return _call

    if fn is not None:
        return _wrap(fn)
    return _wrap  # type: ignore[return-value]


def show_graph(fn: Any) -> str:
    """Dump the tenzor::jit::Graph IR after tracing."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        return _core.jit.show_graph(getattr(fn, "_tz_compiled", fn))
    except AttributeError:
        raise JitNotEnabledError("show_graph requires TENZOR_USE_MLIR_JIT=ON")


def show_mlir(fn: Any) -> str:
    """Dump the Tenzor MLIR dialect module before lowering to StableHLO."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        return _core.jit.show_mlir(getattr(fn, "_tz_compiled", fn))
    except AttributeError:
        raise JitNotEnabledError("show_mlir requires TENZOR_USE_MLIR_JIT=ON")


def show_stablehlo(fn: Any) -> str:
    """Dump the final StableHLO module (textual)."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        return _core.jit.show_stablehlo(getattr(fn, "_tz_compiled", fn))
    except AttributeError:
        raise JitNotEnabledError("show_stablehlo requires TENZOR_USE_MLIR_JIT=ON")


def show_iree(fn: Any) -> str:
    """Run iree-compile --dump-ir-after-all and return the trace."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        return _core.jit.show_iree(getattr(fn, "_tz_compiled", fn))
    except AttributeError:
        raise JitNotEnabledError("show_iree requires TENZOR_USE_MLIR_JIT=ON")


def cache_stats() -> dict:
    """Return a dict of hits, misses, retraces, evictions, total compile time."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        return _core.jit.cache_stats()
    except AttributeError:
        raise JitNotEnabledError("cache_stats requires TENZOR_USE_MLIR_JIT=ON")
