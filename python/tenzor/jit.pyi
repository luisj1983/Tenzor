"""Type stubs for tenzor.jit (CC.11).

`tenzor.jit` is the @tz.jit decorator function provided by
python/tenzor/jit.py. The drift checker (tools/check_pyi_drift.py) lifts
the public names defined in jit.py and diffs them against this stub.

Note: at runtime the @jit callable also carries pybind11-attached
attributes (compile, trace, save_graph, etc.) populated by
python/tenzor/__init__.py from the C++ `tenzor_core.jit` submodule. Those
are exposed dynamically via `setattr(jit, ...)` and so don't appear as
top-level names in jit.py — they're not part of the drift surface.
"""

from __future__ import annotations
from typing import Any, Callable, Dict


class JitNotEnabledError(RuntimeError):
    """Raised when @tz.jit is invoked but the C++ build was configured
    without -DTENZOR_USE_MLIR_JIT=ON."""


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    target: str = "auto",
    fallback_to_eager: bool = False,
) -> Callable[..., Any]:
    """Decorate `fn` to compile through the MLIR pipeline. Can be used as
    `@tz.jit` or `@tz.jit(target="cuda")`.

    Args:
        fn: function to JIT. If None, returns a parametrised decorator.
        target: "auto" (default), "llvm-cpu", "cuda", "vulkan", or "rocm".
        fallback_to_eager: if True, ops not yet lowered fall back to eager
            execution silently; default False raises on coverage gaps.
    """
    ...


def show_graph(fn: Any, example: Any = None) -> str:
    """Return the captured Tenzor graph for `fn` as a human-readable string."""
    ...


def show_mlir(fn: Any, example: Any = None) -> str:
    """Return the MLIR module produced by lowering `fn`."""
    ...


def show_stablehlo(fn: Any, example: Any = None) -> str:
    """Return the StableHLO module produced by lowering `fn`."""
    ...


def show_iree(fn: Any, example: Any = None) -> str:
    """Return the IREE compiled artifact dump for `fn`."""
    ...


def cache_stats() -> Dict[str, Any]:
    """Return @tz.jit compile-cache statistics (hits, misses, sizes)."""
    ...


def reset_cache_stats() -> None:
    """Reset @tz.jit compile-cache counters back to zero."""
    ...
