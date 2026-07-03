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
    # Deferred-error wrapper: keeps the decorator usable as a no-op for builds
    # without the JIT, but loudly tells the user how to enable it the moment they
    # actually try to compile.
    def _disabled_stub(fn_: Callable, msg: str) -> Any:
        class _JitDisabledStub:
            def __init__(self, f: Callable):
                self._fn = f
                functools.update_wrapper(self, f)
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                raise JitNotEnabledError(msg)
        return _JitDisabledStub(fn_)

    _MLIR_OFF_MSG = (
        "@tz.jit: MLIR JIT is not enabled in this build. "
        "Rebuild Tenzor with -DTENZOR_USE_MLIR_JIT=ON to enable it. "
        "See docs/jit-mlir-setup.md for installation instructions.")

    try:
        from . import tenzor_core as _core  # type: ignore[import]
    except ImportError:
        return _disabled_stub(
            fn,
            "@tz.jit: tenzor_core C++ module not found. "
            "Build the project and set PYTHONPATH to include build/python.")

    try:
        binding = _core.jit.compile_function
    except AttributeError:
        # tenzor_core exists but the jit submodule was not compiled in.
        return _disabled_stub(fn, _MLIR_OFF_MSG)

    try:
        return binding(fn, backend=backend, target=target,
                       fallback_to_eager=fallback_to_eager)
    except JitNotEnabledError:
        raise
    except RuntimeError as e:
        # The jit submodule IS compiled but TENZOR_USE_MLIR_JIT is OFF, so the
        # C++ binding raises a generic RuntimeError (rather than the submodule
        # being absent). Treat that exactly like a disabled build so callers get
        # the consistent JitNotEnabledError contract.
        if "TENZOR_USE_MLIR_JIT" in str(e):
            return _disabled_stub(fn, _MLIR_OFF_MSG)
        raise


def jit(fn: Callable[..., Any] | None = None, *,
        target: str = "auto",
        fallback_to_eager: bool = False) -> Callable[..., Any]:
    """Decorate `fn` to compile and execute via the MLIR pipeline.

    Args:
        fn: function to JIT. If None, returns a parametrised decorator.
        target: "auto" (default), "llvm-cpu", "cuda", "rocm", or
            "vulkan-spirv" (the alias "vulkan" is accepted and normalized
            to "vulkan-spirv", the name IREE requires). "auto" picks the
            target from the input tensor's device.
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
            # Stash the most-recent first positional input so the show_*
            # helpers (which require an example tensor to re-trace the
            # graph) can default to it when the user calls them as
            # `tz.jit.show_graph(f)` after a compile.
            if args:
                _call._tz_last_example = args[0]  # type: ignore[attr-defined]
            return compiled(*args, **kwargs)

        _call._tz_compiled = compiled  # type: ignore[attr-defined]
        _call._tz_last_example = None  # type: ignore[attr-defined]
        return _call

    if fn is not None:
        return _wrap(fn)
    return _wrap  # type: ignore[return-value]


def _resolve_example(fn: Any, example: Any) -> Any:
    """Return the user-supplied example, or fall back to the last-invocation
    example stashed by the @tz.jit decorator."""
    if example is not None:
        return example
    last = getattr(fn, "_tz_last_example", None)
    if last is None:
        raise ValueError(
            "show_* helpers require an example tensor — either call the "
            "decorated function once first, or pass an explicit example: "
            "e.g. tz.jit.show_graph(f, example_tensor).")
    return last


def show_graph(fn: Any, example: Any = None) -> str:
    """Dump the tenzor::jit::Graph IR after tracing."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        compiled = getattr(fn, "_tz_compiled", fn)
        return _core.jit.show_graph(compiled, _resolve_example(fn, example))
    except AttributeError:
        raise JitNotEnabledError("show_graph requires TENZOR_USE_MLIR_JIT=ON")


def show_mlir(fn: Any, example: Any = None) -> str:
    """Dump the Tenzor MLIR dialect module before lowering to StableHLO."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        compiled = getattr(fn, "_tz_compiled", fn)
        return _core.jit.show_mlir(compiled, _resolve_example(fn, example))
    except AttributeError:
        raise JitNotEnabledError("show_mlir requires TENZOR_USE_MLIR_JIT=ON")


def show_stablehlo(fn: Any, example: Any = None) -> str:
    """Dump the final StableHLO module (textual)."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        compiled = getattr(fn, "_tz_compiled", fn)
        return _core.jit.show_stablehlo(compiled, _resolve_example(fn, example))
    except AttributeError:
        raise JitNotEnabledError("show_stablehlo requires TENZOR_USE_MLIR_JIT=ON")


def show_iree(fn: Any, example: Any = None) -> str:
    """Run iree-compile --dump-ir-after-all and return the trace."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        compiled = getattr(fn, "_tz_compiled", fn)
        return _core.jit.show_iree(compiled, _resolve_example(fn, example))
    except AttributeError:
        raise JitNotEnabledError("show_iree requires TENZOR_USE_MLIR_JIT=ON")


def cache_stats() -> dict:
    """Return a dict of hits, misses, retraces, evictions, total compile time."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        return _core.jit.cache_stats()
    except AttributeError:
        raise JitNotEnabledError("cache_stats requires TENZOR_USE_MLIR_JIT=ON")


def reset_cache_stats() -> None:
    """Reset the JIT compile-cache statistics counters to zero.

    Audit item H.6: the C++ binding `reset_cache_stats` was added to
    `tenzor_core.jit` but never re-exported through the Python
    `tenzor.jit` shim, so user code calling ``tz.jit.reset_cache_stats()``
    silently fell back to AttributeError instead of resetting the
    counters.
    """
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        _core.jit.reset_cache_stats()
    except AttributeError:
        raise JitNotEnabledError("reset_cache_stats requires TENZOR_USE_MLIR_JIT=ON")
