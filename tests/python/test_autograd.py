#!/usr/bin/env python3
"""
Test Python bindings for autograd: backward, no_grad, retain_grad,
scalar ops grad, and anomaly mode.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def make_scalar_loss(var):
    """Reduce a Variable to scalar via MSELoss against zeros for backward()."""
    target = tz.Variable(tz.zeros(var.data.shape, tz.dtype.float32), False)
    return tz.nn.MSELoss()(var, target)


def test_variable_creation():
    """Test Variable creation with requires_grad."""
    print("Testing Variable creation...")
    t = tz.Tensor([2, 3], tz.dtype.float32)
    v = tz.Variable(t, True)
    assert v.requires_grad(), "Should require grad"

    v2 = tz.Variable(t, False)
    assert not v2.requires_grad(), "Should not require grad"
    print("  Variable creation OK")


def test_basic_backward():
    """Test basic backward pass on scalar Variable."""
    print("Testing basic backward...")
    x = tz.Variable(tz.ones([1], tz.dtype.float32), True)
    y = x * 3.0
    y.backward()

    grad = x.grad
    assert grad is not None, "Gradient should not be None"
    print("  basic backward OK")


def test_matmul_backward():
    """Test backward through matmul."""
    print("Testing matmul backward...")
    x = tz.Variable(tz.randn([2, 3], tz.dtype.float32), True)
    w = tz.Variable(tz.randn([3, 4], tz.dtype.float32), True)

    y = x @ w
    loss = make_scalar_loss(y)
    loss.backward()

    assert x.grad is not None, "x gradient should exist"
    assert w.grad is not None, "w gradient should exist"
    assert x.grad.shape == [2, 3], f"x grad wrong shape: {x.grad.shape}"
    assert w.grad.shape == [3, 4], f"w grad wrong shape: {w.grad.shape}"
    print("  matmul backward OK")


def test_no_grad():
    """Test grad enabled/disabled state."""
    print("Testing no_grad...")
    assert tz.is_grad_enabled(), "Grad should be enabled by default"

    tz.set_grad_enabled(False)
    assert not tz.is_grad_enabled(), "Grad should be disabled"

    tz.set_grad_enabled(True)
    assert tz.is_grad_enabled(), "Grad should be re-enabled"

    # Test context manager if available
    if hasattr(tz, 'no_grad'):
        with tz.no_grad():
            assert not tz.is_grad_enabled(), "Grad should be off inside no_grad"
        assert tz.is_grad_enabled(), "Grad should be restored after no_grad"
    print("  no_grad OK")


def test_scalar_ops_grad():
    """Test gradients through scalar operators."""
    print("Testing scalar ops grad...")

    # Scalar multiply
    x = tz.Variable(tz.ones([1], tz.dtype.float32), True)
    y = x * 2.0
    y.backward()
    assert x.grad is not None, "Gradient from scalar mul should exist"

    # Scalar add
    x2 = tz.Variable(tz.ones([1], tz.dtype.float32), True)
    y2 = x2 + 1.0
    y2.backward()
    assert x2.grad is not None, "Gradient from scalar add should exist"
    print("  scalar ops grad OK")


def test_anomaly_detection():
    """Test anomaly detection mode."""
    print("Testing anomaly detection...")
    assert not tz.is_anomaly_detection_enabled(), "Should be off by default"

    tz.set_anomaly_detection(True)
    assert tz.is_anomaly_detection_enabled(), "Should be on"

    tz.set_anomaly_detection(False)
    assert not tz.is_anomaly_detection_enabled(), "Should be off again"

    # Test context manager
    with tz.detect_anomaly():
        assert tz.is_anomaly_detection_enabled(), "Should be on inside context"
    assert not tz.is_anomaly_detection_enabled(), "Should be off outside context"
    print("  anomaly detection OK")


def test_add_backward():
    """Test backward through addition."""
    print("Testing add backward...")
    a = tz.Variable(tz.randn([3, 3], tz.dtype.float32), True)
    b = tz.Variable(tz.randn([3, 3], tz.dtype.float32), True)

    c = a + b
    loss = make_scalar_loss(c)
    loss.backward()

    assert a.grad is not None, "a gradient should exist"
    assert b.grad is not None, "b gradient should exist"
    print("  add backward OK")


def test_mul_backward():
    """Test backward through element-wise multiply."""
    print("Testing mul backward...")
    a = tz.Variable(tz.randn([3, 3], tz.dtype.float32), True)
    b = tz.Variable(tz.randn([3, 3], tz.dtype.float32), True)

    c = a * b
    loss = make_scalar_loss(c)
    loss.backward()

    assert a.grad is not None, "a gradient should exist"
    assert b.grad is not None, "b gradient should exist"
    print("  mul backward OK")


def test_chain_backward():
    """Test backward through a chain of operations."""
    print("Testing chain backward...")
    x = tz.Variable(tz.randn([2, 3], tz.dtype.float32), True)
    w = tz.Variable(tz.randn([3, 4], tz.dtype.float32), True)

    # x @ w + bias, then relu, then loss
    y = x @ w
    y = tz.nn.relu(y)
    loss = make_scalar_loss(y)
    loss.backward()

    assert x.grad is not None, "x gradient through chain should exist"
    assert w.grad is not None, "w gradient through chain should exist"
    print("  chain backward OK")


def main():
    print("=" * 60)
    print("Testing Autograd Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        test_variable_creation()
        test_basic_backward()
        test_matmul_backward()
        test_no_grad()
        test_scalar_ops_grad()
        test_anomaly_detection()
        test_add_backward()
        test_mul_backward()
        test_chain_backward()

        print("\n" + "=" * 60)
        print("ALL AUTOGRAD TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
