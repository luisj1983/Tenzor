"""Phase 2.3 — DLPack protocol bindings.

Covers:
    Tensor.__dlpack__(stream=None)     -> PyCapsule "dltensor"
    Tensor.__dlpack_device__()         -> (device_type_code, device_id)
    tz.from_dlpack(obj)                -> Tensor (accepts __dlpack__ or capsule)

The test exercises CPU round-trips always, and adds CUDA / third-party
round-trips only when the required infrastructure is present.
"""

import pytest

import tenzor as tz
import tenzor.tenzor_core as _core

f32 = _core.dtype.float32


def _cuda_device():
    """Return a CUDA Device if CUDA is available, else None."""
    try:
        return _core.Device(_core.DeviceType.CUDA, 0)
    except Exception:
        return None


def _tensor_values(t):
    """Flatten a 1- or 2-D tensor to a Python list of floats."""
    if len(t.shape) == 1:
        return [t[i].item() for i in range(t.shape[0])]
    assert len(t.shape) == 2
    return [t[i, j].item() for i in range(t.shape[0]) for j in range(t.shape[1])]


# ---------------------------------------------------------------------------
# __dlpack_device__
# ---------------------------------------------------------------------------

def test_dlpack_device_cpu():
    t = tz.ones((4,), dtype=f32)
    dev = t.__dlpack_device__()
    # kDLCPU == 1 in DLPack (see dlpack.h); device_id = 0 for CPU.
    assert dev == (1, 0)


def test_dlpack_device_cuda():
    d = _cuda_device()
    if d is None:
        pytest.skip("no CUDA device")
    t = tz.ones((4,), dtype=f32, device=d)
    dev = t.__dlpack_device__()
    # kDLCUDA == 2
    assert dev[0] == 2
    assert dev[1] == 0


# ---------------------------------------------------------------------------
# __dlpack__ -> capsule, tz.from_dlpack round-trip
# ---------------------------------------------------------------------------

def test_dlpack_roundtrip_cpu_shape():
    t = tz.ones((3, 4), dtype=f32)
    t2 = tz.from_dlpack(t)
    assert list(t2.shape) == [3, 4]
    assert t2.dtype == f32


def test_dlpack_roundtrip_cpu_values():
    t = tz.full((2, 3), 7.5, dtype=f32)
    t2 = tz.from_dlpack(t)
    # Values share memory; validate semantics via elementwise compare.
    assert _tensor_values(t2) == _tensor_values(t)


def test_dlpack_direct_capsule_path():
    # tz.from_dlpack also accepts a raw capsule directly (not just an
    # object with __dlpack__). The capsule path is the legacy API used
    # by older producers.
    t = tz.ones((5,), dtype=f32)
    capsule = t.__dlpack__()
    t2 = tz.from_dlpack(capsule)
    assert list(t2.shape) == [5]


def test_dlpack_stream_kwarg_accepted():
    # Protocol compat: __dlpack__(stream=None) must accept the argument
    # without erroring, even though the producer doesn't use it.
    t = tz.ones((2,), dtype=f32)
    capsule = t.__dlpack__(stream=None)
    # Consumer can still use the capsule.
    t2 = tz.from_dlpack(capsule)
    assert list(t2.shape) == [2]


# ---------------------------------------------------------------------------
# CUDA device preservation across the DLPack boundary
# ---------------------------------------------------------------------------

def test_dlpack_roundtrip_cuda_preserves_device():
    d = _cuda_device()
    if d is None:
        pytest.skip("no CUDA device")
    t = tz.ones((2, 3), dtype=f32, device=d)
    t2 = tz.from_dlpack(t)
    assert list(t2.shape) == [2, 3]
    # The imported tensor should sit on the same device — no implicit
    # device move across DLPack boundaries.
    assert str(t2.device) == str(t.device)
    # Round-trip values match (compare on CPU).
    cpu = t2.cpu()
    assert _tensor_values(cpu) == [1.0] * 6


# ---------------------------------------------------------------------------
# Capsule lifecycle: after from_dlpack consumes it, the capsule's name
# is "used_dltensor" so the original tensor's destructor doesn't double
# free. We can't inspect the rename directly (pybind11 capsule type
# doesn't expose name in Python), but we verify no crash on gc.
# ---------------------------------------------------------------------------

def test_dlpack_no_double_free_on_gc():
    t = tz.ones((4,), dtype=f32)
    capsule = t.__dlpack__()
    t2 = tz.from_dlpack(capsule)
    # Drop all references; GC will destroy the capsule. If the destructor
    # called the DLPack deleter after from_dlpack already transferred
    # ownership, we'd segfault here or in the next gc pass.
    del capsule
    import gc
    gc.collect()
    # t2 is still valid and usable
    assert t2.shape == [4]
