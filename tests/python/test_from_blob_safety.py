"""Regression tests for Tensor.from_blob buffer-validation guards.

from_blob is a ZERO-COPY view: it takes the buffer pointer as a packed,
C-contiguous, row-major reinterpretation in the requested dtype and IGNORES the
buffer's strides. Without validation it silently returns wrong values for
non-contiguous buffers, and reinterprets bytes for dtype/itemsize mismatches.
These tests pin the guards that reject such inputs.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
import tenzor.tenzor_core as _core
tz.initialize()

import numpy as np
import pytest


def test_contiguous_from_blob_ok():
    """A C-contiguous, dtype-matched buffer wraps cleanly and reads correctly."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = _core.Tensor.from_blob(a, [2, 3], tz.dtype.float32)
    assert list(t.shape) == [2, 3]
    # Spot-check a couple of values round-trip (zero-copy view of row-major data)
    assert abs(float(t[0][0].item()) - 0.0) < 1e-6
    assert abs(float(t[1][2].item()) - 5.0) < 1e-6


def test_noncontiguous_transpose_rejected():
    """A transposed (non-C-contiguous) numpy view must be rejected, not silently
    misread, because from_blob ignores strides."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = a.T  # shape (3, 2), Fortran-ordered / non-contiguous
    assert not b.flags['C_CONTIGUOUS']
    with pytest.raises((ValueError, RuntimeError)):
        _core.Tensor.from_blob(b, [3, 2], tz.dtype.float32)


def test_sliced_noncontiguous_rejected():
    """A strided slice (gaps between elements) must be rejected."""
    a = np.arange(12, dtype=np.float32)
    b = a[::2]  # every other element: non-contiguous
    assert not b.flags['C_CONTIGUOUS']
    with pytest.raises((ValueError, RuntimeError)):
        _core.Tensor.from_blob(b, [6], tz.dtype.float32)


def test_dtype_size_mismatch_rejected():
    """A float64 buffer requested as Float32 passes the byte-size guard (8N>=4N)
    but would reinterpret float64 bit patterns as float32 garbage — must raise."""
    c = np.arange(6, dtype=np.float64)
    with pytest.raises((ValueError, RuntimeError)):
        _core.Tensor.from_blob(c, [6], tz.dtype.float32)


def test_dtype_kind_mismatch_rejected():
    """An int32 buffer requested as Float32 (same itemsize) reinterprets bits —
    must be rejected on the format-kind check."""
    d = np.arange(6, dtype=np.int32)
    with pytest.raises((ValueError, RuntimeError)):
        _core.Tensor.from_blob(d, [6], tz.dtype.float32)


def test_ascontiguous_copy_accepted():
    """The documented workaround — pass a contiguous copy — must succeed."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ascontiguousarray(a.T)  # materialised contiguous copy of the transpose
    t = _core.Tensor.from_blob(b, [3, 2], tz.dtype.float32)
    assert list(t.shape) == [3, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
