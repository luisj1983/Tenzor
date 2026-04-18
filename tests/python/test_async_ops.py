"""
Test Python bindings for async operations.

The C++ API exposes `Future<Tensor>`; the pybind11 layer in
`bindings_vision_detection.cpp` wraps each async_* op to block on the future
and return a Tensor directly. These tests verify shape equivalence with the
synchronous reference implementation for each bound op.
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


@pytest.fixture(scope="module", autouse=True)
def _init_tenzor():
    tz.initialize()
    tz.manual_seed(42)


def test_async_matmul():
    a = tz.randn([4, 3])
    b = tz.randn([3, 5])
    result = tz.async_ops.async_matmul(a, b)
    assert result.shape == [4, 5]


def test_async_add():
    a = tz.randn([3, 4])
    b = tz.randn([3, 4])
    result = tz.async_ops.async_add(a, b)
    assert result.shape == [3, 4]


def test_async_mul():
    a = tz.randn([3, 4])
    b = tz.randn([3, 4])
    result = tz.async_ops.async_mul(a, b)
    assert result.shape == [3, 4]


def test_async_sub():
    a = tz.randn([3, 4])
    b = tz.randn([3, 4])
    result = tz.async_ops.async_sub(a, b)
    assert result.shape == [3, 4]


def test_async_div():
    a = tz.randn([3, 4])
    b = tz.full([3, 4], 2.0)
    result = tz.async_ops.async_div(a, b)
    assert result.shape == [3, 4]


def test_async_relu():
    a = tz.randn([3, 4])
    result = tz.async_ops.async_relu(a)
    assert result.shape == [3, 4]


def test_async_sigmoid():
    a = tz.randn([3, 4])
    result = tz.async_ops.async_sigmoid(a)
    assert result.shape == [3, 4]


def test_async_tanh():
    a = tz.randn([3, 4])
    result = tz.async_ops.async_tanh(a)
    assert result.shape == [3, 4]


def test_async_softmax():
    a = tz.randn([3, 4])
    result = tz.async_ops.async_softmax(a)
    assert result.shape == [3, 4]
