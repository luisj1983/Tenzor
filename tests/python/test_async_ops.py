#!/usr/bin/env python3
"""
Test Python bindings for async operations.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_async_matmul():
    """Test async matrix multiplication."""
    print("Testing async_matmul...")
    a = tz.randn([4, 3])
    b = tz.randn([3, 5])
    result = tz.async_ops.async_matmul(a, b)
    assert result.shape == [4, 5], f"Wrong shape: {result.shape}"
    print("  async_matmul OK")


def test_async_add():
    """Test async element-wise addition."""
    print("Testing async_add...")
    a = tz.randn([3, 4])
    b = tz.randn([3, 4])
    result = tz.async_ops.async_add(a, b)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_add OK")


def test_async_mul():
    """Test async element-wise multiplication."""
    print("Testing async_mul...")
    a = tz.randn([3, 4])
    b = tz.randn([3, 4])
    result = tz.async_ops.async_mul(a, b)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_mul OK")


def test_async_sub():
    """Test async element-wise subtraction."""
    print("Testing async_sub...")
    a = tz.randn([3, 4])
    b = tz.randn([3, 4])
    result = tz.async_ops.async_sub(a, b)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_sub OK")


def test_async_div():
    """Test async element-wise division."""
    print("Testing async_div...")
    a = tz.randn([3, 4])
    b = tz.full([3, 4], 2.0)
    result = tz.async_ops.async_div(a, b)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_div OK")


def test_async_relu():
    """Test async ReLU activation."""
    print("Testing async_relu...")
    a = tz.randn([3, 4])
    result = tz.async_ops.async_relu(a)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_relu OK")


def test_async_sigmoid():
    """Test async sigmoid activation."""
    print("Testing async_sigmoid...")
    a = tz.randn([3, 4])
    result = tz.async_ops.async_sigmoid(a)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_sigmoid OK")


def test_async_tanh():
    """Test async tanh activation."""
    print("Testing async_tanh...")
    a = tz.randn([3, 4])
    result = tz.async_ops.async_tanh(a)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_tanh OK")


def test_async_softmax():
    """Test async softmax."""
    print("Testing async_softmax...")
    a = tz.randn([3, 4])
    result = tz.async_ops.async_softmax(a)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  async_softmax OK")


if __name__ == "__main__":
    tz.initialize()
    tz.manual_seed(42)

    # NOTE: Async ops return Future<Tensor> which is not yet bound to Python.
    # These tests are skipped until the Future type is exposed via pybind11.
    # When Future bindings are added, uncomment the calls below and use
    # result.wait() or result.get() to obtain the Tensor.
    print("SKIPPED: async ops return Future<Tensor> (not yet bound to Python)")
    print("  Once Future<T> is exposed, enable these tests.")
    print("\nAsync ops tests skipped (pre-existing binding gap).")
