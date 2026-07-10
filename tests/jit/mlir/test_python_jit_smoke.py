"""Phase 13 / 1A.10: @tz.jit decorator imports and is callable.

The decorator binds successfully even if MLIR JIT isn't enabled in
the C++ build — the underlying compile_function will raise JitNotEnabledError
on first call. Smoke test only covers the Python side.
"""
import pytest


def test_jit_is_decorator():
    import tenzor as tz
    @tz.jit
    def f(x):
        return x + x
    assert callable(f)
    assert hasattr(f, "_tz_compiled")


def test_jit_with_target_kwarg():
    """The `target` kwarg must actually be plumbed through to compilation and
    execution, not merely accepted by the decorator (JIT-R030: this used to
    only assert callable(f) and never called the decorated function, so the
    target kwarg's plumbing was never exercised)."""
    import numpy as np
    import tenzor as tz
    from tenzor.jit import JitNotEnabledError

    @tz.jit(target="llvm-cpu")
    def f(x):
        return x * 2
    assert callable(f)

    x = tz.full([4], 3.0, tz.dtype.float32)
    try:
        out = f(x)
    except JitNotEnabledError:
        pytest.skip("MLIR JIT not enabled in this build")

    out_t = out.tensor() if hasattr(out, "tensor") and callable(out.tensor) else out
    arr = out_t.cpu().numpy() if hasattr(out_t, "cpu") else np.array(out_t)
    assert np.allclose(arr, 6.0), f"target=\"llvm-cpu\" @tz.jit f(x)=x*2 expected 6.0, got {arr}"


def test_show_helpers_exist():
    # tenzor/__init__.py copies show_graph/show_mlir/show_stablehlo/show_iree/
    # cache_stats onto the `tz.jit` decorator itself (setattr loop, see
    # __init__.py's comment "`tz.jit.show_graph(fn)` works") specifically so
    # `tz.jit.show_graph(fn)` resolves through the callable -- there never was
    # a `tenzor._jit_mod` import path; the module is registered in
    # sys.modules under the name 'tenzor.jit', which `import tenzor.jit`
    # cannot reach as a plain attribute either (that resolves to the `jit`
    # function tenzor/__init__.py rebinds tenzor.jit to, shadowing the
    # submodule) -- `tz.jit` is the one correct, documented way to reach
    # these helpers from Python.
    import tenzor as tz
    for h in (tz.jit.show_graph, tz.jit.show_mlir, tz.jit.show_stablehlo,
              tz.jit.show_iree, tz.jit.cache_stats):
        assert callable(h)


def test_jit_module_importable():
    """The underlying tenzor.jit module (python/tenzor/jit.py) is reachable
    via sys.modules, distinct from the tz.jit decorator function tenzor/
    __init__.py rebinds the `tenzor.jit` attribute to (see
    test_show_helpers_exist's comment)."""
    import sys
    import tenzor as tz  # noqa: F401 — ensures tenzor.jit is registered first
    _jit_mod = sys.modules["tenzor.jit"]
    assert hasattr(_jit_mod, "jit")
    assert hasattr(_jit_mod, "JitNotEnabledError")
