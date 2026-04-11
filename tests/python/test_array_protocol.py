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
