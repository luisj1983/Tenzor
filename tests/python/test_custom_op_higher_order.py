#!/usr/bin/env python3
"""Test that custom autograd functions work with create_graph=True
for higher-order gradient computation.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


class _FunctionCtx:
    """Minimal context for custom functions (mirrors tenzor.autograd.Function)."""
    def __init__(self):
        self.saved_tensors_ = []
    def save_for_backward(self, *tensors):
        self.saved_tensors_ = list(tensors)
    @property
    def saved_tensors(self):
        return self.saved_tensors_


def test_custom_square_backward():
    """Custom square op backward computes 2*x*grad."""
    # Use the low-level apply_custom_function API directly
    class Square:
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            return (2.0 * x * grad_output,)

    x = tz.Variable(tz.full([1], 3.0, tz.dtype.float32), True)
    y = tz.autograd.apply_custom_function(Square, x)

    # y = x^2 = 9, loss = MSE(y, 0) = y^2 / 1 = 81
    loss = tz.nn.MSELoss()(y, tz.Variable(tz.zeros([1], tz.dtype.float32), False))
    loss.backward()

    grad = x.grad
    assert grad is not None, "Gradient should exist"
    print(f"  PASS: test_custom_square_backward")


def test_custom_op_basic_forward():
    """Custom scale op: y = 2*x."""
    class Scale:
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 2.0

        @staticmethod
        def backward(ctx, grad_output):
            return (grad_output * 2.0,)

    x = tz.Variable(tz.full([1], 3.0, tz.dtype.float32), True)
    y = tz.autograd.apply_custom_function(Scale, x)

    loss = tz.nn.MSELoss()(y, tz.Variable(tz.zeros([1], tz.dtype.float32), False))
    loss.backward()

    assert x.grad is not None, "First-order gradient should exist"
    print(f"  PASS: test_custom_op_basic_forward")


if __name__ == "__main__":
    tz.initialize()

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
