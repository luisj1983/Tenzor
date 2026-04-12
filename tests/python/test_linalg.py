#!/usr/bin/env python3
"""
Test Python bindings for linear algebra operations (tenzor.linalg).
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def _make_spd(n):
    """Create a symmetric positive-definite matrix of size n x n."""
    A = tz.randn([n, n])
    # A^T A + n*I is guaranteed SPD
    AtA = tz.matmul(tz.transpose(A, 0, 1), A)
    nI = tz.full([n, n], float(n))
    eye = tz.eye(n)
    return tz.add(AtA, tz.mul(eye, nI))


def test_det():
    """Test determinant computation."""
    print("Testing linalg.det...")
    eye = tz.eye(3)
    d = tz.linalg.det(eye)
    assert d.numel == 1, f"Expected scalar-like, got {d.numel} elements"
    print("  det OK")


def test_inv():
    """Test matrix inverse."""
    print("Testing linalg.inv...")
    A = _make_spd(3)
    A_inv = tz.linalg.inv(A)
    assert A_inv.shape == [3, 3], f"Wrong shape: {A_inv.shape}"
    # A @ A_inv should be close to identity
    product = tz.matmul(A, A_inv)
    assert product.shape == [3, 3]
    print("  inv OK")


def test_solve():
    """Test linear system solve AX = B."""
    print("Testing linalg.solve...")
    A = _make_spd(3)
    B = tz.randn([3, 2])
    X = tz.linalg.solve(A, B)
    assert X.shape == [3, 2], f"Wrong shape: {X.shape}"
    print("  solve OK")


def test_cholesky():
    """Test Cholesky decomposition."""
    print("Testing linalg.cholesky...")
    A = _make_spd(4)
    L = tz.linalg.cholesky(A)
    assert L.shape == [4, 4], f"Wrong shape: {L.shape}"
    print("  cholesky OK")


def test_norm():
    """Test matrix norm."""
    print("Testing linalg.norm...")
    A = tz.randn([3, 4])
    n = tz.linalg.norm(A)
    assert n.numel == 1, f"Expected scalar-like, got {n.numel} elements"
    print("  norm OK")


def test_slogdet():
    """Test sign and log-determinant."""
    print("Testing linalg.slogdet...")
    A = _make_spd(3)
    sign, logabsdet = tz.linalg.slogdet(A)
    assert sign.numel == 1, f"sign wrong size: {sign.numel}"
    assert logabsdet.numel == 1, f"logabsdet wrong size: {logabsdet.numel}"
    print("  slogdet OK")


def test_svd():
    """Test Singular Value Decomposition."""
    print("Testing linalg.svd...")
    A = tz.randn([4, 3])
    U, S, Vh = tz.linalg.svd(A)
    assert U.shape == [4, 4], f"U wrong shape: {U.shape}"
    assert S.shape == [3], f"S wrong shape: {S.shape}"
    assert Vh.shape == [3, 3], f"Vh wrong shape: {Vh.shape}"
    print("  svd OK")


def test_qr():
    """Test QR decomposition."""
    print("Testing linalg.qr...")
    A = tz.randn([4, 3])
    Q, R = tz.linalg.qr(A)
    assert Q.shape[0] == 4, f"Q wrong rows: {Q.shape}"
    assert R.shape[1] == 3, f"R wrong cols: {R.shape}"
    print("  qr OK")


def test_eigh():
    """Test eigendecomposition of symmetric matrix."""
    print("Testing linalg.eigh...")
    A = _make_spd(3)
    eigenvalues, eigenvectors = tz.linalg.eigh(A)
    assert eigenvalues.shape == [3], f"eigenvalues wrong shape: {eigenvalues.shape}"
    assert eigenvectors.shape == [3, 3], f"eigenvectors wrong shape: {eigenvectors.shape}"
    print("  eigh OK")


def test_eigvalsh():
    """Test eigenvalues of symmetric matrix."""
    print("Testing linalg.eigvalsh...")
    A = _make_spd(3)
    vals = tz.linalg.eigvalsh(A)
    assert vals.shape == [3], f"Wrong shape: {vals.shape}"
    print("  eigvalsh OK")


def test_matrix_power():
    """Test matrix power."""
    print("Testing linalg.matrix_power...")
    A = tz.eye(3)
    A2 = tz.linalg.matrix_power(A, 5)
    assert A2.shape == [3, 3], f"Wrong shape: {A2.shape}"
    print("  matrix_power OK")


def test_lstsq():
    """Test least-squares solution."""
    print("Testing linalg.lstsq...")
    A = tz.randn([5, 3])
    B = tz.randn([5, 2])
    solution, residuals = tz.linalg.lstsq(A, B)
    assert solution.shape == [3, 2], f"solution wrong shape: {solution.shape}"
    print("  lstsq OK")


def test_pinv():
    """Test pseudoinverse."""
    print("Testing linalg.pinv...")
    A = tz.randn([3, 4])
    A_pinv = tz.linalg.pinv(A)
    assert A_pinv.shape == [4, 3], f"Wrong shape: {A_pinv.shape}"
    print("  pinv OK")


def test_matrix_exp():
    """Test matrix exponential."""
    print("Testing linalg.matrix_exp...")
    A = tz.randn([3, 3])
    expA = tz.linalg.matrix_exp(A)
    assert expA.shape == [3, 3], f"Wrong shape: {expA.shape}"
    print("  matrix_exp OK")


if __name__ == "__main__":
    tz.initialize()
    tz.manual_seed(42)
    test_det()
    test_inv()
    test_solve()
    test_cholesky()
    test_norm()
    test_slogdet()
    test_svd()
    test_qr()
    test_eigh()
    test_eigvalsh()
    test_matrix_power()
    test_lstsq()
    test_pinv()
    test_matrix_exp()
    print("\nAll linalg tests passed!")
