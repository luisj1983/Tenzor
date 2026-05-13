"""
Fifth-pass audit regression coverage for python/numpy_interop.cpp and
python/bindings/bindings_core.cpp.

B'3: Python ints that exceed int64 must throw on scalar assignment, not
     silently wrap to a negative value.
B'4: numpy object-dtype arrays must be rejected with a clear message at
     entry to numpy_to_tensor, not produce a deep "Unsupported NumPy dtype"
     deeper in the call.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "python")
import tenzor as tz  # noqa: E402

tz.initialize()


# ---------------------------------------------------------------------------
# B'4: object-dtype numpy arrays must be rejected up front
# ---------------------------------------------------------------------------
def test_object_dtype_numpy_array_throws_with_clear_message():
    obj_arr = np.array([{"a": 1}, {"b": 2}], dtype=object)
    with pytest.raises((TypeError, ValueError, RuntimeError)) as ei:
        tz.Tensor.from_numpy(obj_arr)
    msg = str(ei.value).lower()
    # The new pre-check message must mention "object" so users know what
    # to fix. Pre-fix, the error was a generic "Unsupported NumPy dtype"
    # from deeper in the call.
    assert "object" in msg


def test_numeric_dtype_numpy_arrays_still_round_trip():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = tz.Tensor.from_numpy(arr)
    assert list(t.shape) == [3, 4]
    assert t.dtype == tz.dtype.float32
    arr2 = t.numpy()
    assert np.array_equal(arr, arr2)


# ---------------------------------------------------------------------------
# B'3: int64 overflow on scalar assignment must throw a clear error
# ---------------------------------------------------------------------------
def test_python_int_overflow_in_scalar_assignment_throws():
    t = tz.zeros((4,), dtype=tz.dtype.int64)
    # 1 << 63 is one past INT64_MAX
    with pytest.raises((OverflowError, RuntimeError, ValueError)) as ei:
        t[0] = 1 << 63
    msg = str(ei.value).lower()
    assert "overflow" in msg or "fit" in msg or "too large" in msg


def test_python_int_in_range_assignment_still_works():
    # Note: __setitem__ converts to double internally on the broadcast path,
    # so values above 2^53 may lose precision (a separate bug not in scope
    # for B'3). Use a value that survives the double round-trip.
    t = tz.zeros((4,), dtype=tz.dtype.int64)
    safe_max = (1 << 52) + 7
    t[0] = safe_max
    assert int(t[0].item()) == safe_max
