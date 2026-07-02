"""
B7: differentiable Variable.__setitem__.

``var[key] = value`` participates in autograd: gradient flows to ``value`` at
the written positions and to ``var``'s pre-write upstream graph at the
non-written positions. The write is routed through a functional
masked_scatter / masked_fill against a pre-write *snapshot* of ``var`` (so the
graph contains no cycle), and ``var`` is rebound to the result.

Routing contract:
  * leaf  + requires_grad=True   -> ValueError  (PyTorch in-place parity)
  * non-leaf + requires_grad=True -> differentiable scatter (B7)
  * requires_grad=False value into a no-grad var -> plain in-place mutation
  * requires_grad=True value into a no-grad var  -> var becomes grad-tracked
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module", autouse=True)
def _init_tenzor():
    tz.initialize()
    tz.manual_seed(0)


def _np(v):
    t = v.tensor() if hasattr(v, "tensor") else v
    return np.asarray(t)


def _zeros4_var(requires_grad: bool) -> "tz.Variable":
    return tz.Variable(tz.zeros([4], tz.dtype.float32), requires_grad)


def _bool_mask(values):
    return tz.Tensor.from_numpy(np.array(values, dtype=bool))


# ---------------------------------------------------------------------------
# Routing guards (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_setitem_leaf_no_grad_succeeds():
    x = _zeros4_var(requires_grad=False)
    assert x.is_leaf
    x[0] = 1.0  # must NOT raise
    assert float(x.tensor().numpy()[0]) == 1.0


def test_setitem_leaf_with_grad_raises():
    x = _zeros4_var(requires_grad=True)
    assert x.is_leaf and x.requires_grad
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        x[0] = 1.0
    msg = str(excinfo.value).lower()
    assert "leaf" in msg
    assert "in-place" in msg or "in place" in msg


def test_setitem_nonleaf_no_grad_succeeds():
    x = _zeros4_var(requires_grad=False)
    y = x + 1.0
    assert not y.requires_grad
    y[0] = 7.0  # must NOT raise
    assert float(y.tensor().numpy()[0]) == 7.0


def test_setitem_rebuild_as_leaf_workaround_succeeds():
    x = _zeros4_var(requires_grad=True)
    y = x + 1.0
    z = tz.Variable(y.tensor(), requires_grad=False)
    assert not z.requires_grad and z.is_leaf
    z[0] = 1.0  # must NOT raise
    assert float(z.tensor().numpy()[0]) == 1.0


# ---------------------------------------------------------------------------
# B7: differentiable scatter (the case that previously raised)
# ---------------------------------------------------------------------------

def test_setitem_nonleaf_scalar_is_differentiable():
    # Previously raised "CopySlices not supported"; now does masked_fill.
    x = _zeros4_var(requires_grad=True)
    y = x + 1.0                                    # non-leaf, [1,1,1,1]
    y[0] = 5.0                                     # masked_fill at index 0
    assert float(y.tensor().numpy()[0]) == 5.0
    tz.sum(y).backward()
    # index 0 overwritten -> no grad to x there; others -> grad 1
    assert np.array_equal(_np(x.grad), np.array([0, 1, 1, 1], dtype=np.float32))


def test_setitem_boolmask_grad_to_both_sides():
    x = tz.Variable(tz.full([4], 2.0), True)       # leaf param
    y = x * 1.0                                    # non-leaf (writable)
    w = tz.Variable(tz.full([2], 7.0), True)       # value for the 2 masked slots
    mask = _bool_mask([True, False, True, False])

    y[mask] = w                                    # y -> [7, 2, 7, 2]
    assert np.array_equal(_np(y), np.array([7, 2, 7, 2], dtype=np.float32))

    tz.sum(y).backward()
    assert np.array_equal(_np(x.grad), np.array([0, 1, 0, 1], dtype=np.float32))
    assert np.array_equal(_np(w.grad), np.array([1, 1], dtype=np.float32))


def test_setitem_slice_grad_to_both_sides():
    x = tz.Variable(tz.full([4], 2.0), True)
    y = x * 1.0
    w = tz.Variable(tz.full([2], 9.0), True)
    y[1:3] = w                                     # y -> [2, 9, 9, 2]
    assert np.array_equal(_np(y), np.array([2, 9, 9, 2], dtype=np.float32))

    tz.sum(y).backward()
    assert np.array_equal(_np(x.grad), np.array([1, 0, 0, 1], dtype=np.float32))
    assert np.array_equal(_np(w.grad), np.array([1, 1], dtype=np.float32))


def test_setitem_into_nongrad_buffer_flows_to_value():
    buf = tz.Variable(tz.zeros([3], tz.dtype.float32), False)
    w = tz.Variable(tz.full([3], 4.0), True)
    buf[:] = w
    assert buf.requires_grad
    tz.sum(buf).backward()
    assert np.array_equal(_np(w.grad), np.array([1, 1, 1], dtype=np.float32))


def test_setitem_no_grad_context_is_plain_inplace():
    x = tz.Variable(tz.full([4], 2.0), False)
    y = x * 1.0
    w = tz.Variable(tz.full([2], 3.0), False)
    mask = _bool_mask([True, False, True, False])
    y[mask] = w
    assert np.array_equal(_np(y), np.array([3, 2, 3, 2], dtype=np.float32))


# ---------------------------------------------------------------------------
# Regression: Tensor.__setitem__ must cast a dtype-mismatched RHS to the
# destination dtype rather than byte-reinterpreting it. Previously only the
# scalar branch cast; the exact-shape and broadcast branches memcpy'd raw
# bytes, so an int64 RHS was reinterpreted as float32 (garbage) or copied with
# the wrong byte count.
# ---------------------------------------------------------------------------

def test_setitem_tensor_dtype_mismatch_exact_shape_casts():
    x = tz.zeros([4], tz.dtype.float32)
    x[0:2] = tz.Tensor.from_numpy(np.array([7, 9], dtype=np.int64))
    out = np.asarray(x)
    assert float(out[0]) == 7.0 and float(out[1]) == 9.0, f"exact-shape cast failed: {out.tolist()}"


def test_setitem_tensor_dtype_mismatch_same_size_casts():
    # int32 -> float32 are both 4 bytes: a raw copy would reinterpret the bits.
    x = tz.zeros([3], tz.dtype.float32)
    x[0:3] = tz.Tensor.from_numpy(np.array([5, 6, 7], dtype=np.int32))
    out = np.asarray(x)
    assert out.tolist() == [5.0, 6.0, 7.0], f"same-size cast failed: {out.tolist()}"


def test_setitem_tensor_dtype_mismatch_broadcast_casts():
    x = tz.zeros([2, 3], tz.dtype.float32)
    x[:] = tz.Tensor.from_numpy(np.array([1, 2, 3], dtype=np.int64))
    out = np.asarray(x)
    assert out.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], f"broadcast cast failed: {out.tolist()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
