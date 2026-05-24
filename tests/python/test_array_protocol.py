"""Phase 2.4 — NumPy `__array__` protocol binding.

Verifies that Tenzor tensors can be consumed transparently by NumPy via
`np.asarray(tensor)` / `np.array(tensor)`, and that the dtype argument
supported by the NumPy 2.0+ protocol is honored.

Skipped if numpy is not installed.
"""

import pytest

np = pytest.importorskip("numpy")

import tenzor as tz
import tenzor.tenzor_core as _core

f32 = _core.dtype.float32
f64 = _core.dtype.float64
i32 = _core.dtype.int32


# ---------------------------------------------------------------------------
# Presence and basic shape round-trip
# ---------------------------------------------------------------------------

def test_array_protocol_is_bound():
    t = tz.ones((3,), dtype=f32)
    assert hasattr(t, "__array__"), "Tensor should expose __array__"


def test_np_asarray_roundtrip_shape():
    t = tz.ones((2, 3), dtype=f32)
    arr = np.asarray(t)
    assert arr.shape == (2, 3)


def test_np_array_roundtrip_values():
    t = tz.full((2, 3), 7.5, dtype=f32)
    arr = np.asarray(t)
    assert arr.dtype == np.float32
    assert np.all(arr == 7.5)


def test_np_asarray_on_1d():
    t = tz.arange(0, 10)
    arr = np.asarray(t)
    assert list(arr) == list(range(10))


# ---------------------------------------------------------------------------
# dtype casting via the NumPy 2.0+ protocol
# ---------------------------------------------------------------------------

def test_array_protocol_dtype_promotion_to_float64():
    t = tz.ones((3,), dtype=f32)
    arr = np.asarray(t, dtype=np.float64)
    assert arr.dtype == np.float64
    assert arr.tolist() == [1.0, 1.0, 1.0]


def test_array_protocol_dtype_demotion_to_int32():
    t = tz.full((3,), 3.7, dtype=f32)
    arr = np.asarray(t, dtype=np.int32)
    assert arr.dtype == np.int32
    # float->int cast truncates (numpy default)
    assert arr.tolist() == [3, 3, 3]


# ---------------------------------------------------------------------------
# Ufunc dispatch — NumPy ufuncs call __array__ internally
# ---------------------------------------------------------------------------

def test_np_ufunc_dispatch_add():
    t = tz.full((4,), 2.0, dtype=f32)
    result = np.add(t, 3.0)
    # Result is a numpy ndarray, not a Tenzor tensor — __array__ is a
    # one-way coercion, not a wrapping protocol.
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [5.0, 5.0, 5.0, 5.0]


def test_np_sum_over_tenzor_tensor():
    t = tz.arange(0, 10)
    total = np.sum(t)
    assert int(total) == sum(range(10))


# ---------------------------------------------------------------------------
# GPU tensors get copied to CPU automatically (numpy can't read device mem)
# ---------------------------------------------------------------------------

def test_np_asarray_on_cuda_tensor():
    try:
        cuda = _core.Device(_core.DeviceType.CUDA, 0)
        t = tz.full((3,), 4.0, dtype=f32, device=cuda)
    except Exception:
        pytest.skip("no CUDA device")
    arr = np.asarray(t)
    assert arr.shape == (3,)
    assert arr.tolist() == [4.0, 4.0, 4.0]


# ---------------------------------------------------------------------------
# Audit-5 Z.26 — __array__(copy=True/False/None) semantics.
#
# W.16 + Y.25 implement the NumPy 2.0 strict copy= contract:
#   copy=None  → may share storage (zero-copy view permitted)
#   copy=True  → MUST return a fresh allocation independent of Tenzor storage
#   copy=False → MUST raise ValueError if a copy would be required
# ---------------------------------------------------------------------------

def test_array_protocol_copy_true_forces_fresh_allocation():
    """copy=True must guarantee the returned array does not alias the tensor.

    The CPU+contiguous+no-dtype-cast path is the only one where the zero-copy
    view path is reachable; the implementation must force an explicit
    ``.copy()`` on top of it so writes through the returned array do not
    propagate back into Tenzor storage.
    """
    t = tz.full((4,), 1.5, dtype=f32)
    arr_view = t.__array__(copy=None)       # zero-copy view permitted
    arr_copy = t.__array__(copy=True)       # must force fresh allocation

    # Mutate the (potentially) aliased view to seed both arrays.
    # The copy=True result must NOT change.
    pre_copy_value = arr_copy[0]
    arr_view[0] = 99.0
    assert arr_copy[0] == pre_copy_value, (
        "copy=True returned a view that aliases Tenzor storage — "
        "W.16 contract broken (mutating the no-copy view also changed the "
        "copy=True array)."
    )


def test_array_protocol_copy_true_with_dtype_cast_is_independent():
    """copy=True + dtype= must also yield an independent allocation."""
    t = tz.full((3,), 2.0, dtype=f32)
    arr = t.__array__(dtype=np.float64, copy=True)
    assert arr.dtype == np.float64
    # Mutating the result must not affect a subsequent re-read.
    arr[0] = 999.0
    arr2 = np.asarray(t)
    assert arr2[0] == 2.0


def test_array_protocol_copy_false_raises_on_dtype_mismatch():
    """copy=False must raise when a dtype cast would force a copy."""
    t = tz.full((3,), 1.0, dtype=f32)
    with pytest.raises(ValueError, match="copy"):
        t.__array__(dtype=np.float64, copy=False)


def test_array_protocol_copy_false_raises_on_device_copy():
    """copy=False must raise for non-CPU tensors (host transfer is a copy)."""
    try:
        cuda = _core.Device(_core.DeviceType.CUDA, 0)
        t = tz.full((3,), 4.0, dtype=f32, device=cuda)
    except Exception:
        pytest.skip("no CUDA device")
    with pytest.raises(ValueError, match="copy"):
        t.__array__(copy=False)
