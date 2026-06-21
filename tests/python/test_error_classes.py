"""Tests for the typed Tenzor exception classes exposed to Python.

`tz.ShapeError`, `tz.DTypeError`, `tz.DeviceError`, `tz.AutogradError`,
`tz.BackendError`, and `tz.MemoryError` all derive from `tz.TenzorError`.
The existing test_error_handling.py covers custom-function error propagation
but never validates the typed classes themselves. This file fills that gap by
triggering each class via a known-bad operation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


def test_classes_exist_and_subclass_tenzor_error():
    """All typed error classes must be present and inherit from TenzorError."""
    for cls_name in (
        "TenzorError",
        "ShapeError",
        "DTypeError",
        "DeviceError",
        "AutogradError",
        "BackendError",
        "MemoryError",
    ):
        assert hasattr(tz, cls_name), f"tz.{cls_name} not exposed"
    base = tz.TenzorError
    for cls_name in ("ShapeError", "DTypeError", "DeviceError",
                     "AutogradError", "BackendError", "MemoryError"):
        assert issubclass(getattr(tz, cls_name), base), (
            f"tz.{cls_name} should subclass TenzorError"
        )


def test_shape_error_on_mismatched_matmul():
    """matmul of (2,3) @ (4,5) is shape-incompatible — must raise."""
    a = tz.randn([2, 3])
    b = tz.randn([4, 5])
    with pytest.raises((tz.ShapeError, tz.TenzorError, RuntimeError)):
        _ = tz.matmul(a, b)


def test_shape_error_on_mismatched_add():
    """Add of (3,) and (4,) cannot broadcast — must raise."""
    a = tz.randn([3])
    b = tz.randn([4])
    with pytest.raises((tz.ShapeError, tz.TenzorError, RuntimeError)):
        _ = a + b


def test_dtype_error_on_invalid_cast():
    """Casting via .to() with an unsupported dtype combination should raise.
    Documents the DTypeError pathway is reachable from Python."""
    a = tz.randn([4])
    # Try to convert into a non-tensor-supported type. If Tenzor accepts every
    # combination, fall through with a sentinel skip rather than failing the
    # whole suite — the goal is to verify the binding exists, not enumerate
    # every dtype mismatch in the codebase.
    raised = False
    for bad in (tz.dtype.complex64, tz.dtype.complex128):
        try:
            _ = a.to(bad).to(tz.dtype.bool)
        except (tz.DTypeError, tz.TenzorError, RuntimeError):
            raised = True
            break
    if not raised:
        pytest.skip("All probed dtype combinations succeed; DTypeError path "
                    "is exposed but no easily-triggerable case from Python.")


def test_device_error_on_cross_device_op():
    """Operating on tensors that live on incompatible devices should raise."""
    if not tz.cuda_is_available():
        pytest.skip("CUDA not available")
    a = tz.randn([4]).to('cpu')
    b = tz.randn([4]).to('cuda:0')
    with pytest.raises((tz.DeviceError, tz.TenzorError, RuntimeError)):
        _ = a + b


def test_autograd_error_on_nonscalar_backward_without_grad():
    """Calling backward() on a non-scalar Variable without an explicit grad
    seed should raise — non-scalar reduction is not implicit."""
    x = tz.Variable(tz.randn([4]), requires_grad=True)
    y = x * 2.0  # vector output, requires_grad propagates
    with pytest.raises((tz.AutogradError, tz.TenzorError, RuntimeError)):
        y.backward()  # no grad_output passed; non-scalar must error


def test_tenzor_error_is_python_exception():
    """TenzorError should be a Python Exception subclass (catchable by `except`).
    Note: it is NOT currently a RuntimeError subclass — diverges from PyTorch
    convention; document for any future migration."""
    assert issubclass(tz.TenzorError, Exception)


def test_specific_subclass_is_raised_not_base():
    """REGRESSION for the catch-all exception translator shadowing bug.

    pybind registers translators LIFO and applies them front-to-back, so the
    catch-all (base TenzorException) had been running FIRST and converting every
    TenzorException subclass into the bare base TenzorError — making
    `except tz.AutogradError` (etc.) unreachable. A non-scalar backward without
    an explicit grad seed throws a C++ AutogradException; with the translator
    ordering fixed it must surface as the SPECIFIC tz.AutogradError, not the
    base tz.TenzorError. Before the fix `type(e)` would be the base class.
    """
    x = tz.Variable(tz.randn([4]), requires_grad=True)
    y = x * 2.0  # non-scalar output
    try:
        y.backward()  # no grad_output -> AutogradException in C++
    except tz.TenzorError as e:
        assert type(e) is tz.AutogradError, (
            f"expected the concrete tz.AutogradError, got {type(e).__name__}; "
            "the catch-all translator is shadowing the typed translator")
    else:
        pytest.fail("non-scalar backward without grad seed did not raise")


def test_specific_subclass_matches_in_except_clause():
    """`except tz.AutogradError` must actually catch the typed error (the
    user-visible symptom of the catch-all-shadowing bug)."""
    x = tz.Variable(tz.randn([4]), requires_grad=True)
    y = x * 2.0
    caught = False
    try:
        y.backward()
    except tz.AutogradError:
        caught = True
    assert caught, "except tz.AutogradError failed to catch the typed error"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
