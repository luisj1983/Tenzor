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
    import tenzor as tz
    @tz.jit(target="llvm-cpu")
    def f(x):
        return x * 2
    assert callable(f)


def test_show_helpers_exist():
    import tenzor._jit_mod as _jit_mod
    for h in (_jit_mod.show_graph, _jit_mod.show_mlir, _jit_mod.show_stablehlo,
              _jit_mod.show_iree, _jit_mod.cache_stats):
        assert callable(h)


def test_jit_module_importable():
    """tz.jit is importable as a module (not just a function)."""
    import tenzor._jit_mod as _jit_mod
    assert hasattr(_jit_mod, "jit")
    assert hasattr(_jit_mod, "JitNotEnabledError")
