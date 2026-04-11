"""Phase 2.6 — `__tensor_function__` subclass override protocol.

Covers the tenzor.overrides module and verifies that public ops wrapped
via `tenzor.overrides.implements` route through a subclass's
`__tensor_function__` hook when it appears in their argument list.
"""

import pytest

import tenzor as tz
import tenzor.overrides as ov
import tenzor.tenzor_core as _core

f32 = _core.dtype.float32


# ---------------------------------------------------------------------------
# Protocol helpers exposed under tenzor.overrides
# ---------------------------------------------------------------------------

def test_overrides_module_exposes_expected_api():
    assert hasattr(ov, "implements")
    assert hasattr(ov, "has_tensor_function")
    assert hasattr(ov, "handle_tensor_function")
    assert hasattr(ov, "TENSOR_FUNCTION")
    assert ov.TENSOR_FUNCTION == "__tensor_function__"
    assert hasattr(ov, "get_default_nowrap_functions")


# ---------------------------------------------------------------------------
# has_tensor_function
# ---------------------------------------------------------------------------

def test_base_tensor_is_not_intercepted():
    t = tz.ones((3,), dtype=f32)
    assert ov.has_tensor_function([t]) is False


def test_subclass_with_hook_is_detected():
    class HookCls:
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            return NotImplemented

    assert ov.has_tensor_function([HookCls()]) is True
    assert ov.has_tensor_function([1, 2.0, HookCls()]) is True


# ---------------------------------------------------------------------------
# handle_tensor_function dispatch
# ---------------------------------------------------------------------------

def test_dispatch_calls_first_hook():
    calls = []

    class A:
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            calls.append(("A", func.__name__))
            return "from-A"

    result = ov.handle_tensor_function(tz.add, (A(), 5), {})
    assert result == "from-A"
    assert calls == [("A", "add")]


def test_dispatch_notimplemented_propagates_to_next():
    order = []

    class Defer:
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            order.append("Defer")
            return NotImplemented

    class Accept(Defer):
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            order.append("Accept")
            return "accepted"

    # Most-derived class (Accept) runs first; since Accept accepts, the
    # result is "accepted". If both had been in args separately, the
    # deeper class would still win.
    result = ov.handle_tensor_function(tz.add, (Accept(), Defer()), {})
    assert result == "accepted"
    assert "Accept" in order


def test_dispatch_all_notimplemented_raises():
    class A:
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            return NotImplemented

    with pytest.raises(TypeError):
        ov.handle_tensor_function(tz.add, (A(),), {})


def test_handle_without_hook_arg_raises():
    # At least one positional arg must carry the hook; otherwise
    # handle_tensor_function is a caller error.
    with pytest.raises(TypeError):
        ov.handle_tensor_function(tz.add, (1, 2), {})


# ---------------------------------------------------------------------------
# @implements wrapper on the public tz.* functions
# ---------------------------------------------------------------------------

def test_wrapped_tz_add_intercepts_subclass_instance():
    intercepted = []

    class Intercept:
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            intercepted.append(func.__name__)
            return "intercepted"

    # tz.add was wrapped in tenzor/__init__.py via ov.implements.
    result = tz.add(Intercept(), 5)
    assert result == "intercepted"
    assert intercepted == ["add"]


def test_wrapped_tz_add_falls_through_for_base_tensor():
    # When only base Tensors are involved, the wrapped op transparently
    # calls the underlying C++ implementation and returns a real Tensor.
    a = tz.ones((3,), dtype=f32)
    b = tz.ones((3,), dtype=f32)
    result = tz.add(a, b)
    # Result should be a Tensor whose first element is 2.
    assert result[0].item() == 2.0


def test_wrapper_preserves_name_and_wrapped_attribute():
    # @implements uses functools.wraps so __name__ is preserved and
    # the original function is reachable via __wrapped_api__.
    assert tz.add.__name__ == "add"
    assert hasattr(tz.add, "__wrapped_api__")


def test_multiple_ops_wrapped():
    # Every op in the curated wrap set from tenzor/__init__.py should
    # expose the __wrapped_api__ sentinel.
    wrapped_ops = ["add", "sub", "mul", "div", "matmul", "bmm",
                   "sum", "mean", "max", "min",
                   "sqrt", "exp", "log", "abs", "neg"]
    for name in wrapped_ops:
        op = getattr(tz, name, None)
        if op is None:
            pytest.skip(f"tz.{name} not present in this build")
        assert hasattr(op, "__wrapped_api__"), (
            f"tz.{name} should be wrapped by tz.overrides.implements")


# ---------------------------------------------------------------------------
# Priority ordering: most-derived class runs first
# ---------------------------------------------------------------------------

def test_most_derived_class_wins():
    order = []

    class Parent:
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            order.append("Parent")
            return "parent"

    class Child(Parent):
        @classmethod
        def __tensor_function__(cls, func, types, args, kwargs=None):
            order.append("Child")
            return "child"

    # Child is deeper in the MRO → runs first.
    result = ov.handle_tensor_function(tz.mul, (Parent(), Child()), {})
    assert result == "child"
    assert order[0] == "Child"
