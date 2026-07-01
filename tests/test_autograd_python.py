#!/usr/bin/env python3
"""
Test backward pass / autograd functionality from Python bindings.
Tests gradient computation, chain rule, and no_grad context.
"""

import sys
import os
import math

# tenzor_core.so lives under build/python/tenzor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python', 'tenzor'))

# Import tenzor
try:
    import tenzor_core as tz
except ImportError:
    print("Error: Could not import tenzor_core module")
    print("Please build the Python bindings first")
    sys.exit(1)


def approx_equal(a, b, tol=1e-4):
    """Check if two floats are approximately equal."""
    return abs(a - b) < tol


def test_simple_scalar_backward():
    """Test simple backward: y = x * 2, dy/dx = 2."""
    print("\n=== Test: Simple Scalar Backward ===")

    x = tz.Variable(tz.full([1], 3.0), requires_grad=True)
    # y = x * 2
    two = tz.Variable(tz.full([1], 2.0), requires_grad=False)
    y = x * two
    y.backward()

    grad_val = x.grad.item()
    print(f"  x = 3.0, y = x * 2, dy/dx = {grad_val}")
    assert approx_equal(grad_val, 2.0), f"Expected grad 2.0, got {grad_val}"
    print("  PASSED")


def test_chain_rule():
    """Test chain rule: y = x^2, dy/dx = 2*x."""
    print("\n=== Test: Chain Rule (x^2) ===")

    x = tz.Variable(tz.full([1], 3.0), requires_grad=True)
    # y = x * x
    y = x * x
    y.backward()

    grad_val = x.grad.item()
    print(f"  x = 3.0, y = x^2, dy/dx = {grad_val}")
    assert approx_equal(grad_val, 6.0), f"Expected grad 6.0, got {grad_val}"
    print("  PASSED")


def test_addition_backward():
    """Test addition: z = x + y, dz/dx = 1, dz/dy = 1."""
    print("\n=== Test: Addition Backward ===")

    x = tz.Variable(tz.full([1], 2.0), requires_grad=True)
    y = tz.Variable(tz.full([1], 3.0), requires_grad=True)
    z = x + y
    z.backward()

    grad_x = x.grad.item()
    grad_y = y.grad.item()
    print(f"  z = x + y, dz/dx = {grad_x}, dz/dy = {grad_y}")
    assert approx_equal(grad_x, 1.0), f"Expected dz/dx = 1.0, got {grad_x}"
    assert approx_equal(grad_y, 1.0), f"Expected dz/dy = 1.0, got {grad_y}"
    print("  PASSED")


def test_multi_variable_backward():
    """Test multi-variable: z = x * y, dz/dx = y, dz/dy = x."""
    print("\n=== Test: Multi-Variable Backward ===")

    x = tz.Variable(tz.full([1], 4.0), requires_grad=True)
    y = tz.Variable(tz.full([1], 5.0), requires_grad=True)
    z = x * y
    z.backward()

    grad_x = x.grad.item()
    grad_y = y.grad.item()
    print(f"  x = 4.0, y = 5.0, z = x * y")
    print(f"  dz/dx = {grad_x} (expected {5.0})")
    print(f"  dz/dy = {grad_y} (expected {4.0})")
    assert approx_equal(grad_x, 5.0), f"Expected dz/dx = 5.0, got {grad_x}"
    assert approx_equal(grad_y, 4.0), f"Expected dz/dy = 4.0, got {grad_y}"
    print("  PASSED")


def test_mse_loss_backward():
    """Test MSE loss backward pass."""
    print("\n=== Test: MSE Loss Backward ===")

    # pred = [2.0], target = [1.0], loss = (2-1)^2 = 1.0
    # dloss/dpred = 2*(pred - target) / n = 2*(2-1)/1 = 2.0
    pred = tz.Variable(tz.full([1], 2.0), requires_grad=True)
    target = tz.Variable(tz.full([1], 1.0), requires_grad=False)

    mse = tz.nn.MSELoss(reduction=tz.nn.Reduction.mean)
    loss = mse(pred, target)

    loss_val = loss.data.item()
    print(f"  pred = 2.0, target = 1.0, MSE loss = {loss_val}")
    assert approx_equal(loss_val, 1.0), f"Expected loss 1.0, got {loss_val}"

    loss.backward()

    grad_val = pred.grad.item()
    print(f"  dloss/dpred = {grad_val}")
    assert approx_equal(grad_val, 2.0), f"Expected grad 2.0, got {grad_val}"
    print("  PASSED")


def test_retain_graph():
    """Test retain_graph=True allows multiple backward passes."""
    print("\n=== Test: retain_graph ===")

    x = tz.Variable(tz.full([1], 3.0), requires_grad=True)
    y = x * x  # y = x^2

    # First backward with retain_graph
    y.backward(retain_graph=True)
    grad1 = x.grad.item()
    print(f"  First backward: grad = {grad1}")
    assert approx_equal(grad1, 6.0), f"Expected 6.0, got {grad1}"

    # Second backward should work because graph is retained
    try:
        y.backward(retain_graph=True)
        print("  Second backward succeeded (graph retained)")
    except RuntimeError as e:
        print(f"  Second backward failed (unexpected): {e}")
        raise

    print("  PASSED")


def test_no_grad():
    """Test no_grad context prevents gradient tracking."""
    print("\n=== Test: no_grad ===")

    x = tz.Variable(tz.randn([3, 3]), requires_grad=True)

    # Operations inside no_grad should not track gradients
    with tz.no_grad():
        y = x * x
        print(f"  Inside no_grad: y.requires_grad = {y.requires_grad}")
        assert not y.requires_grad, "Variable should not require grad inside no_grad"

    # Operations outside should still track
    z = x * x
    print(f"  Outside no_grad: z.requires_grad = {z.requires_grad}")
    assert z.requires_grad, "Variable should require grad outside no_grad"

    print("  PASSED")


def test_linear_backward():
    """Test backward through a Linear layer."""
    print("\n=== Test: Linear Layer Backward ===")

    linear = tz.nn.Linear(4, 2)
    x = tz.Variable(tz.randn([1, 4]), requires_grad=True)

    out = linear(x)
    print(f"  Linear(4, 2) output shape: {list(out.data.shape)}")

    # Sum to get scalar for backward
    loss = tz.sum(out.data)
    loss_var = tz.Variable(loss, requires_grad=True)

    # Use MSE loss to drive backward
    target = tz.Variable(tz.zeros([1, 2]), requires_grad=False)
    mse = tz.nn.MSELoss(reduction=tz.nn.Reduction.mean)
    loss = mse(out, target)
    loss.backward()

    has_grad = x.grad is not None
    print(f"  Input has gradient: {has_grad}")
    if has_grad:
        print(f"  Input gradient shape: {list(x.grad.shape)}")
        assert list(x.grad.shape) == [1, 4], "Input grad shape mismatch"

    print("  PASSED")


def main():
    """Run all autograd tests."""
    print("=" * 60)
    print("Testing Tenzor Autograd / Backward Pass (Python)")
    print("=" * 60)

    # Initialize library
    tz.initialize()

    try:
        test_simple_scalar_backward()
        test_chain_rule()
        test_addition_backward()
        test_multi_variable_backward()
        test_mse_loss_backward()
        test_retain_graph()
        test_no_grad()
        test_linear_backward()

        print("\n" + "=" * 60)
        print("ALL AUTOGRAD TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
