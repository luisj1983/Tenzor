#!/usr/bin/env python3
"""
Test Python bindings for fused operations.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_fused_linear_relu():
    """Test fused linear + ReLU."""
    print("Testing fused_linear_relu...")
    x = tz.randn([2, 10])
    weight = tz.randn([5, 10])
    bias = tz.randn([5])
    result = tz.fused.fused_linear_relu(x, weight, bias)
    assert result.shape == [2, 5], f"Wrong shape: {result.shape}"
    print("  fused_linear_relu OK")


def test_fused_add_relu():
    """Test fused element-wise add + ReLU."""
    print("Testing fused_add_relu...")
    a = tz.randn([3, 4])
    b = tz.randn([3, 4])
    result = tz.fused.fused_add_relu(a, b)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  fused_add_relu OK")


def test_fused_gelu():
    """Test fused GELU activation."""
    print("Testing fused_gelu...")
    x = tz.randn([3, 4])
    result = tz.fused.fused_gelu(x)
    assert result.shape == [3, 4], f"Wrong shape: {result.shape}"
    print("  fused_gelu OK")


def test_fused_layer_norm():
    """Test fused layer normalization."""
    print("Testing fused_layer_norm...")
    x = tz.randn([2, 3, 4])
    weight = tz.ones([4])
    bias = tz.zeros([4])
    result = tz.fused.fused_layer_norm(x, [4], weight, bias, 1e-5)
    assert result.shape == [2, 3, 4], f"Wrong shape: {result.shape}"
    print("  fused_layer_norm OK")


def test_fused_softmax_cross_entropy():
    """Test fused softmax + cross entropy."""
    print("Testing fused_softmax_cross_entropy...")
    logits = tz.randn([4, 10])
    # Create target indices as int64
    targets = tz.zeros([4], tz.dtype.int64)
    result = tz.fused.fused_softmax_cross_entropy(logits, targets)
    assert result.shape == [] or result.numel == 1, f"Expected scalar loss, got shape: {result.shape}"
    print("  fused_softmax_cross_entropy OK")


def test_fused_conv2d_relu():
    """Test fused conv2d + ReLU."""
    print("Testing fused_conv2d_relu...")
    x = tz.randn([1, 3, 8, 8])
    weight = tz.randn([16, 3, 3, 3])
    bias = tz.randn([16])
    result = tz.fused.fused_conv2d_relu(x, weight, bias)
    assert result.shape[0] == 1, f"Batch wrong: {result.shape}"
    assert result.shape[1] == 16, f"Channels wrong: {result.shape}"
    print("  fused_conv2d_relu OK")


def test_fused_batchnorm_relu():
    """Test fused batch normalization + ReLU."""
    print("Testing fused_batchnorm_relu...")
    x = tz.randn([2, 8, 4, 4])
    weight = tz.ones([8])
    bias = tz.zeros([8])
    mean = tz.zeros([8])
    var = tz.ones([8])
    result = tz.fused.fused_batchnorm_relu(x, mean, var, weight, bias, 1e-5)
    assert result.shape == [2, 8, 4, 4], f"Wrong shape: {result.shape}"
    print("  fused_batchnorm_relu OK")


if __name__ == "__main__":
    tz.initialize()
    tz.manual_seed(42)
    test_fused_linear_relu()
    test_fused_add_relu()
    test_fused_gelu()
    test_fused_layer_norm()
    test_fused_softmax_cross_entropy()
    test_fused_conv2d_relu()
    test_fused_batchnorm_relu()
    print("\nAll fused ops tests passed!")
