"""Phase 2.2 — Python tensor accessors matching torch.Tensor shape/layout API.

Covers:
    .T, .mT, .H (shape accessors)
    .stride() / .stride(dim) / .storage_offset() / .data_ptr()
    .is_pinned / .pin_memory()

Each test builds a small tensor and verifies the accessor matches the
expected PyTorch semantics.
"""

import pytest

import tenzor as tz
import tenzor.tenzor_core as _core

f32 = _core.dtype.float32


# ---------------------------------------------------------------------------
# Shape transposes (.T / .mT / .H)
# ---------------------------------------------------------------------------

def test_T_transposes_2d():
    t = tz.ones((3, 4), dtype=f32)
    u = t.T
    assert list(u.shape) == [4, 3]


def test_T_errors_on_non_2d():
    t = tz.ones((2, 3, 4), dtype=f32)
    with pytest.raises(ValueError):
        _ = t.T


def test_mT_swaps_last_two_dims_batched():
    t = tz.ones((2, 3, 4), dtype=f32)
    u = t.mT
    assert list(u.shape) == [2, 4, 3]


def test_mT_errors_on_1d():
    t = tz.ones((5,), dtype=f32)
    with pytest.raises(ValueError):
        _ = t.mT


def test_H_matches_mT_for_real_dtype():
    # For real-valued tensors, conj is a no-op so .H == .mT.
    t = tz.ones((2, 3, 4), dtype=f32)
    h = t.H
    mt = t.mT
    assert list(h.shape) == list(mt.shape)


# ---------------------------------------------------------------------------
# stride() / storage_offset() / data_ptr()
# ---------------------------------------------------------------------------

def test_stride_returns_tuple():
    # Row-major 3x4 float32: strides should be (4, 1).
    t = tz.ones((3, 4), dtype=f32)
    s = t.stride()
    assert isinstance(s, tuple)
    assert s == (4, 1)


def test_stride_single_dim():
    t = tz.ones((3, 4), dtype=f32)
    assert t.stride(0) == 4
    assert t.stride(1) == 1


def test_stride_negative_index():
    t = tz.ones((3, 4), dtype=f32)
    assert t.stride(-1) == t.stride(1)
    assert t.stride(-2) == t.stride(0)


def test_stride_out_of_range_raises():
    t = tz.ones((3, 4), dtype=f32)
    with pytest.raises(IndexError):
        _ = t.stride(5)


def test_storage_offset_fresh_tensor_is_zero():
    t = tz.ones((3, 4), dtype=f32)
    assert t.storage_offset() == 0


def test_data_ptr_returns_int_address():
    t = tz.ones((3, 4), dtype=f32)
    p = t.data_ptr()
    assert isinstance(p, int)
    assert p != 0, "data_ptr of a populated tensor should be non-null"


def test_data_ptr_same_for_views_of_same_storage():
    # A simple reshape is a view, so data_ptr should be equal to the
    # original (offset 0).
    t = tz.ones((6,), dtype=f32)
    v = t.reshape((2, 3))
    assert t.data_ptr() == v.data_ptr()


# ---------------------------------------------------------------------------
# is_pinned / pin_memory
# ---------------------------------------------------------------------------

def test_fresh_cpu_tensor_is_not_pinned():
    t = tz.ones((4,), dtype=f32)
    assert t.is_pinned is False


def test_pin_memory_returns_self_and_updates_is_pinned_when_cuda_available():
    # pin_memory() is documented as a no-op on non-CUDA builds.
    # Detect whether CUDA is present via tenzor_core.DeviceType, and only
    # assert the post-pin state in that case; otherwise just verify the
    # call is idempotent and returns the same object.
    t = tz.ones((16,), dtype=f32)
    returned = t.pin_memory()
    # Match torch semantics: returns a Tensor referring to the same
    # underlying storage. pybind11 returns the same Python object for a
    # `Tensor&` return.
    assert returned is t or returned.data_ptr() == t.data_ptr()

    have_cuda = False
    try:
        # Cheap probe — constructing a CUDA tensor throws if CUDA is
        # unavailable at this build.
        _ = tz.zeros((1,), dtype=f32, device=_core.DeviceType.CUDA)
        have_cuda = True
    except Exception:
        have_cuda = False

    if have_cuda:
        assert t.is_pinned is True, (
            "With CUDA available, pin_memory() should successfully "
            "page-lock a contiguous CPU allocation")
        # Pinning is idempotent.
        t.pin_memory()
        assert t.is_pinned is True
