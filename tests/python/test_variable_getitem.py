#!/usr/bin/env python3
"""
Audit-5 Z.28 (U.11 / V.35 follow-up): Variable.__getitem__ across the
indexing modes that V.35 added to Tensor.__getitem__.

V.35 wired int / slice / tuple / fancy / mask indexing through the Tensor
binding; this test pins the parallel behaviour for Variable, including the
autograd contract: every indexing op must keep grad_fn alive so a backward
pass through the indexed view flows back to the original Variable.

Skipped if numpy is not installed (used only for sanity-checking values).
"""

import os
import sys

build_python_dir = os.path.join(os.path.dirname(__file__), "../../build/python")
sys.path.insert(0, build_python_dir)

import pytest

import tenzor.tenzor_core as tz


def _grad_max_abs(var):
    """Return max(|var.grad|) as a Python float, or None if grad is missing."""
    g = var.grad
    if g is None:
        return None
    return float(tz.max(tz.abs(g.to(tz.dtype.float64))).item())


def _assert_grad_flows(var, msg=""):
    g_max = _grad_max_abs(var)
    assert g_max is not None, f"{msg}: var.grad is None — backward never ran"
    assert g_max > 0.0, (
        f"{msg}: var.grad is identically zero after backward — grad_fn likely "
        f"severed by the indexing op."
    )


def setup_module(_module):
    tz.initialize()


# ---------------------------------------------------------------------------
# Integer indexing
# ---------------------------------------------------------------------------

def test_variable_getitem_int_keeps_grad_flow():
    x = tz.Variable(tz.randn([4, 3], tz.dtype.float32), True)
    y = x[2]                       # picks row 2 → shape [3]
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "int indexing")


def test_variable_getitem_negative_int_keeps_grad_flow():
    x = tz.Variable(tz.randn([4, 3], tz.dtype.float32), True)
    y = x[-1]                      # last row
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "negative int indexing")


# ---------------------------------------------------------------------------
# Slice indexing
# ---------------------------------------------------------------------------

def test_variable_getitem_slice_keeps_grad_flow():
    x = tz.Variable(tz.randn([6, 4], tz.dtype.float32), True)
    y = x[1:4]                     # shape [3, 4]
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "slice indexing")


def test_variable_getitem_slice_with_step_keeps_grad_flow():
    x = tz.Variable(tz.randn([8, 2], tz.dtype.float32), True)
    y = x[::2]                     # every other row → shape [4, 2]
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "strided slice indexing")


# ---------------------------------------------------------------------------
# Tuple indexing (mixing ints + slices)
# ---------------------------------------------------------------------------

def test_variable_getitem_tuple_int_slice_keeps_grad_flow():
    x = tz.Variable(tz.randn([4, 5, 3], tz.dtype.float32), True)
    y = x[1, 0:3, :]               # mixed int + slice + : → shape [3, 3]
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "tuple (int, slice, :) indexing")


def test_variable_getitem_tuple_all_slices_keeps_grad_flow():
    x = tz.Variable(tz.randn([5, 5, 5], tz.dtype.float32), True)
    y = x[1:4, 1:4, 1:4]           # 3D inner cube
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "tuple (slice, slice, slice) indexing")


# ---------------------------------------------------------------------------
# Fancy / advanced indexing — integer-tensor index
# ---------------------------------------------------------------------------

def test_variable_getitem_fancy_keeps_grad_flow():
    x = tz.Variable(tz.randn([6, 4], tz.dtype.float32), True)
    idx = tz.tensor([0, 2, 5], tz.dtype.int64)  # from-data (lowercase); Tensor([...]) would be a shape
    y = x[idx]                     # shape [3, 4]
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "fancy (int64 tensor) indexing")


# ---------------------------------------------------------------------------
# Boolean mask indexing
# ---------------------------------------------------------------------------

def test_variable_getitem_bool_mask_keeps_grad_flow():
    x = tz.Variable(tz.randn([5, 3], tz.dtype.float32), True)
    # Mask along dim-0: pick rows 0, 2, 4.
    mask = tz.tensor([True, False, True, False, True], tz.dtype.bool)  # from-data (lowercase)
    y = x[mask]                    # shape [3, 3]
    loss = tz.sum(y)
    loss.backward()
    _assert_grad_flows(x, "boolean mask indexing")


# ---------------------------------------------------------------------------
# Shape sanity — at least one indexing form should narrow as expected.
# ---------------------------------------------------------------------------

def test_variable_getitem_shape_is_correct():
    x = tz.Variable(tz.randn([4, 6, 8], tz.dtype.float32), False)
    assert tuple(x[0].data.shape) == (6, 8)
    assert tuple(x[:, 1:4].data.shape) == (4, 3, 8)
    assert tuple(x[0, :, 2].data.shape) == (6,)


# ---------------------------------------------------------------------------
# NN.22: Ellipsis-in-middle of tuple index must not miscount consumed dims.
# Previously ``x[0, ..., 2]`` on a 4-D tensor mis-subtracted the leading Int
# from the ellipsis fill, producing the wrong shape (or out-of-range error).
# ---------------------------------------------------------------------------

def test_tensor_getitem_ellipsis_in_middle_shape():
    x = tz.randn([4, 5, 6, 7], tz.dtype.float32)
    # Int + ellipsis + Int: should consume first & last dim, keep middle two.
    assert tuple(x[0, ..., 2].shape) == (5, 6)
    # Ellipsis at front, trailing Int: drops last dim only.
    assert tuple(x[..., 0].shape) == (4, 5, 6)
    # Ellipsis at back, leading Int: drops first dim only.
    assert tuple(x[0, ...].shape) == (5, 6, 7)
    # Ellipsis between two slices: middle dims survive untouched.
    assert tuple(x[1:3, ..., 0:2].shape) == (2, 5, 6, 2)
    # 5-D with Int + ellipsis + Int.
    y = tz.randn([2, 3, 4, 5, 6], tz.dtype.float32)
    assert tuple(y[0, ..., 1].shape) == (3, 4, 5)


def test_variable_getitem_ellipsis_in_middle_shape():
    x = tz.Variable(tz.randn([4, 5, 6, 7], tz.dtype.float32), False)
    assert tuple(x[0, ..., 2].data.shape) == (5, 6)
    assert tuple(x[..., 0].data.shape) == (4, 5, 6)
    assert tuple(x[0, ...].data.shape) == (5, 6, 7)


def test_variable_getitem_ellipsis_in_middle_grad_flows():
    x = tz.Variable(tz.randn([4, 5, 6, 7], tz.dtype.float32), True)
    y = x[0, ..., 2]
    assert tuple(y.data.shape) == (5, 6)
    tz.sum(y).backward()
    _assert_grad_flows(x, "ellipsis-in-middle indexing")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
