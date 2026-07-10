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
import inspect
import threading


class JitNotEnabledError(RuntimeError):
    """Raised when @tz.jit is called but the C++ build was configured without
    TENZOR_USE_MLIR_JIT=ON."""


class _JitDisabledStub:
    """Wraps `fn` in place of a real compiled variant when the JIT is
    unavailable (tenzor_core not importable, the jit submodule absent, or
    TENZOR_USE_MLIR_JIT=OFF at build time). Calling the wrapped function
    raises JitNotEnabledError with `msg`.

    Defined at module level (not nested inside `_compile_function`) so the
    show_* introspection helpers can `isinstance()`-check for it (JIT-R013)
    and raise the documented JitNotEnabledError themselves, instead of
    forwarding the stub straight into the C++ binding — which requires a
    std::shared_ptr<CompiledFunction> and rejects the stub with a confusing
    pybind11 "incompatible function arguments" TypeError.
    """
    def __init__(self, f: Callable, msg: str):
        self._fn = f
        self._msg = msg
        functools.update_wrapper(self, f)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise JitNotEnabledError(self._msg)


def _is_tensor_arg(a: Any) -> bool:
    """True if ``a`` should be forwarded to the compiled core as a tensor operand;
    False if it is a static (non-tensor) argument to be baked into the traced
    graph (JIT-061). Detects the tenzor Variable/Tensor types when the C++ module
    is importable; otherwise treats plain Python scalars/containers as static."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
        types = tuple(
            t for t in (getattr(_core, "Variable", None),
                        getattr(_core, "Tensor", None))
            if isinstance(t, type))
        if types:
            return isinstance(a, types)
    except Exception:
        pass
    return not isinstance(a, (int, float, bool, str, bytes, type(None),
                              list, tuple, dict, set))


def _compile_function(fn: Callable, *, backend: str, target: str,
                      fallback_to_eager: bool) -> Any:
    """Delegate to the C++ binding if available; raise JitNotEnabledError otherwise."""
    # Deferred-error wrapper: keeps the decorator usable as a no-op for builds
    # without the JIT, but loudly tells the user how to enable it the moment they
    # actually try to compile.
    def _disabled_stub(fn_: Callable, msg: str) -> Any:
        return _JitDisabledStub(fn_, msg)

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
        # Lazily-built compiled variants (JIT-061). With no static (non-tensor)
        # arguments there is a single variant. When a call carries static args
        # (e.g. a `dim` int, a shape tuple), we compile one variant per distinct
        # static-value combination — baking those values into the traced graph
        # and forwarding only the tensor operands — instead of forwarding a
        # non-tensor to the positional-Variable-only core (which errored or
        # mis-traced the scalar as a graph input).
        _variants: dict = {}

        # Guards the (._tz_compiled, ._tz_last_examples) pair below so a
        # concurrent reader (show_graph/show_mlir/show_stablehlo/show_iree)
        # never observes a torn write — e.g. _tz_compiled from one thread's
        # call paired with _tz_last_examples from a different thread's call
        # with a different static-argument variant (JIT-R018). The GIL makes
        # each individual attribute assignment atomic, but not the pair.
        _lock = threading.Lock()

        # Capture the wrapped function's signature so keyword arguments can be
        # normalized to positional order before dispatch.
        try:
            _sig: inspect.Signature | None = inspect.signature(f)
        except (ValueError, TypeError):
            _sig = None

        def _variant_for(call_args: tuple):
            static_items = [(i, a) for i, a in enumerate(call_args)
                            if not _is_tensor_arg(a)]
            tensor_args = [a for a in call_args if _is_tensor_arg(a)]
            # Key on the static VALUES themselves (not repr(): two distinct objects
            # can share a repr and would collapse to one baked variant, returning
            # silently wrong numerics). Only value-faithful, hashable primitives
            # are cacheable; reject anything else loudly rather than risk a
            # mis-keyed bake.
            def _key_part(i: int, v: Any):
                if isinstance(v, (int, float, bool, str, bytes, type(None))):
                    return (i, type(v).__name__, v)
                if isinstance(v, tuple) and all(
                        isinstance(e, (int, float, bool, str, bytes, type(None)))
                        for e in v):
                    return (i, "tuple", v)
                raise TypeError(
                    f"@tz.jit: static (non-tensor) argument at position {i} of "
                    f"type '{type(v).__name__}' is not supported. Static args must "
                    "be scalars (int/float/bool/str/bytes/None) or tuples of those; "
                    "pass tensors for all other operands.")
            key = (len(call_args),) + tuple(_key_part(i, v) for i, v in static_items)
            variant = _variants.get(key)
            if variant is None:
                if not static_items:
                    target_fn: Callable = f
                else:
                    _sm = dict(static_items)
                    _n = len(call_args)

                    def target_fn(*tensors: Any) -> Any:
                        merged = []
                        ti = 0
                        for i in range(_n):
                            if i in _sm:
                                merged.append(_sm[i])
                            else:
                                merged.append(tensors[ti])
                                ti += 1
                        return f(*merged)

                variant = _compile_function(
                    target_fn, backend="mlir", target=target,
                    fallback_to_eager=fallback_to_eager)
                _variants[key] = variant
            return variant, tuple(tensor_args)

        @functools.wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            call_args = args
            if kwargs:
                if _sig is None:
                    raise TypeError(
                        "@tz.jit: cannot bind keyword arguments for a callable "
                        "with no inspectable signature — pass all inputs "
                        "positionally.")
                bound = _sig.bind(*args, **kwargs)
                # The compiled core takes positional Variables only. Anything
                # left in bound.kwargs cannot be forwarded positionally — either
                # a keyword-only parameter, or a positional-or-keyword parameter
                # passed by keyword while an earlier one was omitted (so it has
                # no positional slot). Defaults are deliberately NOT applied:
                # doing so would turn non-tensor defaults into positional
                # arguments and break the cast to Variable.
                if bound.kwargs:
                    raise TypeError(
                        "@tz.jit: these arguments cannot be forwarded to the "
                        "positional-only compiled function: " +
                        ", ".join(bound.kwargs) +
                        ". Pass all inputs positionally (and don't skip an "
                        "earlier parameter while passing a later one by "
                        "keyword).")
                call_args = bound.args
            # Select (or lazily compile) the variant for this call's static args
            # and forward only the tensor operands (JIT-061).
            variant, tensor_args = _variant_for(call_args)
            # Stash the tensor operands (positional order) so the show_* helpers
            # can re-trace without an explicit example. Store STRONG references
            # (the natural `f(randn()); show_graph(f)` idiom needs the inputs to
            # survive the call) — this is bounded: only the LAST call's operands
            # are held and they are overwritten every call (incl. an empty tuple
            # for an all-static call, so the stash never goes stale). Write both
            # attributes together under `_lock` (JIT-R018) so a concurrent
            # reader always sees a matched (compiled, examples) pair.
            with _lock:
                _call._tz_compiled = variant  # type: ignore[attr-defined]
                _call._tz_last_examples = tuple(tensor_args)  # type: ignore[attr-defined]
            try:
                return variant(*tensor_args)
            except Exception as e:
                # The compiled core casts the return value to a SINGLE Variable; a
                # function returning multiple tensors fails that cast (JIT-060).
                # Add a clear hint ONLY when the error looks like that cast — do
                # NOT re-run f() (that would duplicate side effects), and do not
                # mask unrelated errors.
                msg = str(e)
                if (("tuple" in msg or "list" in msg) and
                        ("Variable" in msg or "cast" in msg.lower())):
                    raise TypeError(
                        "@tz.jit: the function appears to return multiple tensors, "
                        "which the JIT core (single output) does not support — "
                        "split it into separate single-output @tz.jit functions. "
                        f"(underlying: {e})") from e
                raise

        # _tz_compiled is (re)assigned per call to the active variant; seed it
        # with the no-static variant so show_* works before the first call.
        # Seeding happens before `_call` is returned to any caller, so no
        # concurrent reader can observe it yet — the lock isn't needed here.
        _call._tz_compiled = _variant_for(())[0]  # type: ignore[attr-defined]
        _call._tz_last_examples = ()  # type: ignore[attr-defined]
        # Exposed so the show_* helpers can read the pair above atomically
        # too (JIT-R018) — a torn read is just as bad as a torn write.
        _call._tz_lock = _lock  # type: ignore[attr-defined]
        return _call

    if fn is not None:
        return _wrap(fn)
    return _wrap  # type: ignore[return-value]


def _snapshot_compiled_and_examples(fn: Any) -> tuple:
    """Atomically read the (compiled variant, last example tensors) pair off
    `fn` under `fn._tz_lock`, if present.

    `_tz_compiled` and `_tz_last_examples` are updated together (under the
    same lock) by `_call` on every invocation. Reading them with two separate
    `getattr` calls — as this used to do — is not atomic as a pair: a
    concurrent call from another thread (compiling a different
    static-argument variant) can interleave between the two reads, handing
    show_graph/show_mlir/etc. a `_tz_compiled` from one call paired with
    `_tz_last_examples` from a different one (JIT-R018). Taking the same lock
    the writer uses closes that window. `fn` values that were never wrapped
    by `@tz.jit` (no `_tz_lock` attribute) fall back to a plain best-effort
    read.
    """
    lock = getattr(fn, "_tz_lock", None)
    if lock is not None:
        with lock:
            return getattr(fn, "_tz_compiled", fn), getattr(fn, "_tz_last_examples", ())
    return getattr(fn, "_tz_compiled", fn), getattr(fn, "_tz_last_examples", ())


def _resolve_examples(last: tuple, examples: tuple) -> tuple:
    """Return the user-supplied example(s), or fall back to `last` — the
    inputs stashed by the @tz.jit decorator on the last invocation, already
    snapshotted atomically together with the matching compiled variant by
    `_snapshot_compiled_and_examples` (JIT-R018). Returns a tuple so a
    multi-argument function is re-traced with ALL of its inputs (the old
    single-example path silently dropped every argument after the first)."""
    if examples:
        return tuple(examples)
    if not last:
        raise ValueError(
            "show_* helpers require example tensor(s) — either call the decorated "
            "function once first, or pass explicit examples: e.g. "
            "tz.jit.show_graph(f, x, y).")
    return tuple(last)


def _jit_show_fn(name: str):
    """Return the C++ ``_core.jit.<name>`` dump helper, or raise
    JitNotEnabledError if the JIT module — or tenzor_core itself — is absent.
    This separates the "JIT not enabled" check from the actual call so a
    genuine error inside the call (a bad example cast, or an exception raised
    while re-tracing the user's function) propagates instead of being
    misreported as "rebuild with TENZOR_USE_MLIR_JIT=ON" (JIT-058).

    tenzor_core is guaranteed importable in the normal import flow (see
    module docstring), but this mirrors `_compile_function`'s explicit
    ImportError -> JitNotEnabledError handling for consistency (JIT-R024)
    rather than letting an ImportError escape uncaught here."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
    except ImportError:
        raise JitNotEnabledError(
            f"{name} requires the tenzor_core C++ module. Build the project "
            "and set PYTHONPATH to include build/python.") from None
    jit_mod = getattr(_core, "jit", None)
    fn = getattr(jit_mod, name, None) if jit_mod is not None else None
    if fn is None:
        raise JitNotEnabledError(
            f"{name} requires a @tz.jit-compiled function and "
            "TENZOR_USE_MLIR_JIT=ON")
    return fn


def show_graph(fn: Any, *examples: Any) -> str:
    """Dump the tenzor::jit::Graph IR after tracing. Pass one example per
    argument of the decorated function (or none to reuse the last call)."""
    show = _jit_show_fn("show_graph")
    compiled, last = _snapshot_compiled_and_examples(fn)
    if isinstance(compiled, _JitDisabledStub):
        raise JitNotEnabledError(compiled._msg)
    return show(compiled, *_resolve_examples(last, examples))


def show_mlir(fn: Any, *examples: Any) -> str:
    """Dump the Tenzor MLIR dialect module before lowering to StableHLO."""
    show = _jit_show_fn("show_mlir")
    compiled, last = _snapshot_compiled_and_examples(fn)
    if isinstance(compiled, _JitDisabledStub):
        raise JitNotEnabledError(compiled._msg)
    return show(compiled, *_resolve_examples(last, examples))


def show_stablehlo(fn: Any, *examples: Any) -> str:
    """Dump the final StableHLO module (textual)."""
    show = _jit_show_fn("show_stablehlo")
    compiled, last = _snapshot_compiled_and_examples(fn)
    if isinstance(compiled, _JitDisabledStub):
        raise JitNotEnabledError(compiled._msg)
    return show(compiled, *_resolve_examples(last, examples))


def show_iree(fn: Any, *examples: Any) -> str:
    """Run iree-compile --dump-ir-after-all and return the trace."""
    show = _jit_show_fn("show_iree")
    compiled, last = _snapshot_compiled_and_examples(fn)
    if isinstance(compiled, _JitDisabledStub):
        raise JitNotEnabledError(compiled._msg)
    return show(compiled, *_resolve_examples(last, examples))


def cache_stats() -> dict:
    """Return a dict of hits, misses, retraces, evictions, total compile time."""
    try:
        from . import tenzor_core as _core  # type: ignore[import]
    except ImportError:
        # Mirrors _compile_function's ImportError -> JitNotEnabledError contract
        # (JIT-R024); not reachable in the normal import flow (tenzor_core is
        # imported unconditionally by tenzor/__init__.py before tz.jit is
        # wired up) but kept consistent rather than left to raise a bare
        # ImportError.
        raise JitNotEnabledError(
            "cache_stats requires the tenzor_core C++ module. Build the "
            "project and set PYTHONPATH to include build/python.") from None
    try:
        return _core.jit.cache_stats()
    except AttributeError:
        raise JitNotEnabledError("cache_stats requires TENZOR_USE_MLIR_JIT=ON") from None


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
    except ImportError:
        # See cache_stats above (JIT-R024).
        raise JitNotEnabledError(
            "reset_cache_stats requires the tenzor_core C++ module. Build "
            "the project and set PYTHONPATH to include build/python.") from None
    try:
        _core.jit.reset_cache_stats()
    except AttributeError:
        raise JitNotEnabledError("reset_cache_stats requires TENZOR_USE_MLIR_JIT=ON") from None
