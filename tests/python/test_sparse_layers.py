#!/usr/bin/env python3
"""
Test Python bindings for sparse neural network layers (SparseLinear, SparseEmbedding).
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_sparse_linear_creation():
    """Test SparseLinear layer construction."""
    print("Testing SparseLinear creation...")
    layer = tz.nn.SparseLinear(10, 5, 0.5)  # in=10, out=5, sparsity=0.5
    params = layer.parameters()
    assert len(params) > 0, "Expected at least 1 parameter"
    print("  SparseLinear creation OK")


def test_sparse_linear_forward():
    """Test SparseLinear forward pass."""
    print("Testing SparseLinear forward...")
    layer = tz.nn.SparseLinear(10, 5, 0.5)
    x = tz.Variable(tz.randn([2, 10]), False)
    y = layer.forward(x)
    assert y.tensor().shape == [2, 5], f"Wrong shape: {y.tensor().shape}"
    print("  SparseLinear forward OK")


def test_sparse_embedding_creation():
    """Test SparseEmbedding layer construction."""
    print("Testing SparseEmbedding creation...")
    layer = tz.nn.SparseEmbedding(100, 32)  # num_embeddings=100, dim=32
    params = layer.parameters()
    assert len(params) > 0, "Expected at least 1 parameter"
    print("  SparseEmbedding creation OK")


def test_sparse_embedding_forward():
    """Test SparseEmbedding forward pass."""
    print("Testing SparseEmbedding forward...")
    layer = tz.nn.SparseEmbedding(100, 32)
    indices = tz.Variable(tz.zeros([4], tz.dtype.int64), False)
    y = layer.forward(indices)
    assert y.tensor().shape == [4, 32], f"Wrong shape: {y.tensor().shape}"
    print("  SparseEmbedding forward OK")


if __name__ == "__main__":
    tz.initialize()
    tz.manual_seed(42)
    test_sparse_linear_creation()
    test_sparse_linear_forward()
    test_sparse_embedding_creation()
    test_sparse_embedding_forward()
    print("\nAll sparse layer tests passed!")
