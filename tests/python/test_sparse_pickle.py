"""Audit-11 QQ.15: SparseTensor must round-trip through pickle.

KK.22 claimed pickle support but the actual ``py::pickle`` binding was
never added to ``py::class_<SparseTensor>``, so ``pickle.dumps`` raised
``TypeError``. This regression test exercises every supported layout
(COO/CSR/CSC/BSR) end-to-end.
"""
from __future__ import annotations

import pickle

import pytest

import tenzor as tz
from tenzor import sparse as tzs


def _assert_tensor_equal(a, b):
    """Compare two Tensors element-wise via the to_list bridge."""
    assert list(a.shape) == list(b.shape)
    assert str(a.dtype) == str(b.dtype)
    assert a.to_list() == b.to_list()


def test_sparse_coo_pickle_roundtrip():
    indices = tz.tensor([[0, 1, 2], [1, 2, 0]], dtype=tz.int64)
    values = tz.tensor([1.0, 2.0, 3.0], dtype=tz.float32)
    st = tzs.sparse_coo(indices, values, [3, 3])

    restored = pickle.loads(pickle.dumps(st))

    assert restored.layout == st.layout
    assert list(restored.shape) == list(st.shape)
    assert str(restored.dtype) == str(st.dtype)
    assert str(restored.device) == str(st.device)
    _assert_tensor_equal(restored.indices(), st.indices())
    _assert_tensor_equal(restored.values(), st.values())


def test_sparse_csr_pickle_roundtrip():
    crow = tz.tensor([0, 1, 2, 3], dtype=tz.int64)
    cols = tz.tensor([1, 2, 0], dtype=tz.int64)
    vals = tz.tensor([1.0, 2.0, 3.0], dtype=tz.float32)
    st = tzs.sparse_csr(crow, cols, vals, [3, 3])

    restored = pickle.loads(pickle.dumps(st))

    assert restored.layout == st.layout
    assert list(restored.shape) == list(st.shape)
    _assert_tensor_equal(restored.crow_indices(), st.crow_indices())
    _assert_tensor_equal(restored.col_indices(), st.col_indices())
    _assert_tensor_equal(restored.values(), st.values())


def test_sparse_csc_pickle_roundtrip():
    ccol = tz.tensor([0, 1, 2, 3], dtype=tz.int64)
    rows = tz.tensor([2, 0, 1], dtype=tz.int64)
    vals = tz.tensor([1.0, 2.0, 3.0], dtype=tz.float32)
    st = tzs.sparse_csc(ccol, rows, vals, [3, 3])

    restored = pickle.loads(pickle.dumps(st))

    assert restored.layout == st.layout
    assert list(restored.shape) == list(st.shape)
    _assert_tensor_equal(restored.ccol_indices(), st.ccol_indices())
    _assert_tensor_equal(restored.row_indices(), st.row_indices())
    _assert_tensor_equal(restored.values(), st.values())


def test_sparse_bsr_pickle_roundtrip():
    row_ptr = tz.tensor([0, 1, 2], dtype=tz.int64)
    col_ind = tz.tensor([0, 1], dtype=tz.int64)
    # Two 2x2 blocks
    vals = tz.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=tz.float32,
    )
    st = tzs.sparse_bsr(row_ptr, col_ind, vals, [4, 4], (2, 2))

    restored = pickle.loads(pickle.dumps(st))

    assert restored.layout == st.layout
    assert list(restored.shape) == list(st.shape)
    assert restored.block_size() == st.block_size()
    _assert_tensor_equal(restored.bsr_row_ptr(), st.bsr_row_ptr())
    _assert_tensor_equal(restored.bsr_col_ind(), st.bsr_col_ind())
    _assert_tensor_equal(restored.values(), st.values())
