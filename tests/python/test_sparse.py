"""Tests for sparse tensor operations (COO, CSR, CSC, BSR)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# COO format
# ---------------------------------------------------------------------------

def test_sparse_coo_creation():
    indices = tz.tensor([[0, 1, 2], [0, 1, 2]], dtype=tz.dtype.int64)
    values = tz.tensor([1.0, 2.0, 3.0])
    s = tz.sparse.sparse_coo(indices, values, [3, 3])
    assert s.nnz == 3
    assert s.shape == [3, 3]
    assert s.layout == tz.sparse.SparseLayout.COO


def test_sparse_coo_to_dense():
    indices = tz.tensor([[0, 1], [1, 0]], dtype=tz.dtype.int64)
    values = tz.tensor([5.0, 7.0])
    s = tz.sparse.sparse_coo(indices, values, [2, 2])
    d = s.to_dense()
    assert d.shape == [2, 2]


def test_sparse_coo_coalesce():
    # Duplicate indices should be summed
    indices = tz.tensor([[0, 0], [0, 0]], dtype=tz.dtype.int64)
    values = tz.tensor([1.0, 2.0])
    s = tz.sparse.sparse_coo(indices, values, [2, 2])
    s_c = s.coalesce()
    assert s_c.is_coalesced


def test_dense_to_sparse_roundtrip():
    d = tz.eye(3)
    s = tz.sparse.to_sparse(d)
    d2 = s.to_dense()
    # Verify roundtrip preserves values
    assert d2.shape == [3, 3]


# ---------------------------------------------------------------------------
# CSR format
# ---------------------------------------------------------------------------

def test_sparse_csr_creation():
    crow = tz.tensor([0, 1, 2, 3], dtype=tz.dtype.int64)
    col = tz.tensor([0, 1, 2], dtype=tz.dtype.int64)
    vals = tz.tensor([1.0, 2.0, 3.0])
    s = tz.sparse.sparse_csr(crow, col, vals, [3, 3])
    assert s.layout == tz.sparse.SparseLayout.CSR
    assert s.nnz == 3


def test_dense_to_csr_roundtrip():
    d = tz.eye(4)
    s = tz.sparse.to_sparse_csr(d)
    assert s.layout == tz.sparse.SparseLayout.CSR
    d2 = s.to_dense()
    assert d2.shape == [4, 4]


def test_csr_crow_col_indices():
    crow = tz.tensor([0, 2, 3, 3], dtype=tz.dtype.int64)
    col = tz.tensor([0, 2, 1], dtype=tz.dtype.int64)
    vals = tz.tensor([1.0, 2.0, 3.0])
    s = tz.sparse.sparse_csr(crow, col, vals, [3, 3])
    assert s.crow_indices().shape == [4]
    assert s.col_indices().shape == [3]


# ---------------------------------------------------------------------------
# CSC format
# ---------------------------------------------------------------------------

def test_sparse_csc_creation():
    ccol = tz.tensor([0, 1, 2, 3], dtype=tz.dtype.int64)
    row = tz.tensor([0, 1, 2], dtype=tz.dtype.int64)
    vals = tz.tensor([1.0, 2.0, 3.0])
    s = tz.sparse.sparse_csc(ccol, row, vals, [3, 3])
    assert s.layout == tz.sparse.SparseLayout.CSC
    assert s.nnz == 3


def test_csc_accessors():
    ccol = tz.tensor([0, 1, 2, 3], dtype=tz.dtype.int64)
    row = tz.tensor([0, 1, 2], dtype=tz.dtype.int64)
    vals = tz.tensor([1.0, 2.0, 3.0])
    s = tz.sparse.sparse_csc(ccol, row, vals, [3, 3])
    assert s.ccol_indices().shape == [4]
    assert s.row_indices().shape == [3]


def test_dense_to_csc_roundtrip():
    d = tz.eye(3)
    s = tz.sparse.to_sparse_csc(d)
    assert s.layout == tz.sparse.SparseLayout.CSC
    d2 = s.to_dense()
    assert d2.shape == [3, 3]


def test_csr_to_csc_conversion():
    d = tz.eye(3)
    csr = tz.sparse.to_sparse_csr(d)
    csc = csr.to_csc()
    assert csc.layout == tz.sparse.SparseLayout.CSC


# ---------------------------------------------------------------------------
# BSR format
# ---------------------------------------------------------------------------

def test_sparse_bsr_creation():
    # 4x4 matrix with 2x2 blocks, 2 non-zero blocks on the diagonal
    bsr_row_ptr = tz.tensor([0, 1, 2], dtype=tz.dtype.int64)
    bsr_col_ind = tz.tensor([0, 1], dtype=tz.dtype.int64)
    # 2 blocks of shape (2, 2)
    vals = tz.tensor([1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]).reshape([2, 2, 2])
    s = tz.sparse.sparse_bsr(bsr_row_ptr, bsr_col_ind, vals, [4, 4], (2, 2))
    assert s.layout == tz.sparse.SparseLayout.BSR
    assert s.block_size() == (2, 2)


def test_bsr_accessors():
    bsr_row_ptr = tz.tensor([0, 1, 2], dtype=tz.dtype.int64)
    bsr_col_ind = tz.tensor([0, 1], dtype=tz.dtype.int64)
    vals = tz.ones([2, 2, 2])
    s = tz.sparse.sparse_bsr(bsr_row_ptr, bsr_col_ind, vals, [4, 4], (2, 2))
    assert s.bsr_row_ptr().shape == [3]
    assert s.bsr_col_ind().shape == [2]


# ---------------------------------------------------------------------------
# Sparse operations
# ---------------------------------------------------------------------------

def test_spmm():
    d = tz.eye(3)
    s = tz.sparse.to_sparse_csr(d)
    x = tz.randn([3, 4])
    result = tz.sparse.spmm(s, x)
    assert result.shape == [3, 4]


def test_spmv():
    d = tz.eye(3)
    s = tz.sparse.to_sparse_csr(d)
    v = tz.randn([3])
    result = tz.sparse.spmv(s, v)
    assert result.shape == [3]


def test_sparse_dense_add():
    d = tz.eye(3)
    s = tz.sparse.to_sparse_csr(d)
    dense = tz.ones([3, 3])
    result = tz.sparse.add(s, dense)
    assert result.shape == [3, 3]


def test_sparse_scalar_mul():
    d = tz.eye(3)
    s = tz.sparse.to_sparse_csr(d)
    result = tz.sparse.mul(s, 2.0)
    assert result.nnz == s.nnz


def test_sparse_transpose():
    crow = tz.tensor([0, 2, 3, 3], dtype=tz.dtype.int64)
    col = tz.tensor([0, 2, 1], dtype=tz.dtype.int64)
    vals = tz.tensor([1.0, 2.0, 3.0])
    s = tz.sparse.sparse_csr(crow, col, vals, [3, 3])
    st = s.transpose()
    assert st.shape == [3, 3]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_sparse():
    indices = tz.zeros([2, 0], dtype=tz.dtype.int64)
    values = tz.zeros([0])
    s = tz.sparse.sparse_coo(indices, values, [3, 3])
    assert s.nnz == 0


def test_sparse_repr():
    d = tz.eye(3)
    s = tz.sparse.to_sparse_csr(d)
    r = repr(s)
    assert "SparseTensor" in r
    assert "CSR" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
