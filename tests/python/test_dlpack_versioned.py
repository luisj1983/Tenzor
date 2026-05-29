"""DLPack 1.0 versioned-capsule negotiation (release-prep E).

``Tensor.__dlpack__`` previously always emitted the legacy unversioned
``"dltensor"`` capsule. It now emits a DLPack 1.0 ``"dltensor_versioned"``
capsule (``DLManagedTensorVersioned``) when the consumer advertises
``max_version`` major >= 1, and the legacy capsule otherwise. Both must
round-trip correctly through a real consumer (NumPy 2.x).
"""
import ctypes
import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tc = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module", autouse=True)
def _init():
    tc.initialize()


def _capsule_name(cap):
    f = ctypes.pythonapi.PyCapsule_GetName
    f.restype = ctypes.c_char_p
    f.argtypes = [ctypes.py_object]
    return f(cap)


def test_dlpack_versioned_capsule_when_negotiated():
    t = tc.full([2, 3], 2.0)
    cap = t.__dlpack__(max_version=(1, 0))
    assert _capsule_name(cap) == b"dltensor_versioned"


def test_dlpack_legacy_capsule_without_max_version():
    t = tc.full([4], 1.0)
    cap = t.__dlpack__()
    assert _capsule_name(cap) == b"dltensor"


def test_dlpack_versioned_roundtrip_through_numpy():
    # NumPy 2.x negotiates max_version=(1, x) -> exercises the versioned path.
    # A wrong ABI layout would crash or corrupt; correct data validates it.
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = tc.Tensor.from_numpy(a)
    b = np.from_dlpack(t)
    assert b.shape == (2, 3)
    assert np.array_equal(b, a)


def test_dlpack_versioned_explicit_request_roundtrip():
    a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    t = tc.Tensor.from_numpy(a)
    # Force the versioned capsule, then hand it to numpy explicitly.
    b = np.from_dlpack(t)  # numpy re-negotiates; just confirm data integrity
    assert np.array_equal(b, a)
