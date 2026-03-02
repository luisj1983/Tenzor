"""Tests for error handling in autograd custom functions."""
import sys
sys.path.insert(0, "python")
import tenzor as tz

tz.initialize()

def test_custom_function_forward_exception():
    """PyCustomFunction.forward() should propagate Python exceptions cleanly."""
    class BadForward(tz.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            raise ValueError("Test error in forward")

        @staticmethod
        def backward(ctx, grad):
            return grad

    x = tz.Variable(tz.randn([3], dtype=tz.dtype.Float32), True)
    try:
        BadForward.apply(x)
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "forward" in str(e).lower() or "Test error" in str(e), f"Got: {e}"

def test_custom_function_backward_exception():
    """PyCustomFunction.backward() should propagate Python exceptions cleanly."""
    class BadBackward(tz.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, grad):
            raise ValueError("Test error in backward")

    x = tz.Variable(tz.randn([3], dtype=tz.dtype.Float32), True)
    y = BadBackward.apply(x)
    try:
        y.backward()
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "backward" in str(e).lower() or "Test error" in str(e), f"Got: {e}"

def test_gradient_shape_mismatch():
    """User-supplied gradient with wrong shape should raise error."""
    x = tz.Variable(tz.randn([3, 4], dtype=tz.dtype.Float32), True)
    y = x.sum()
    try:
        y.backward(tz.randn([3, 4], dtype=tz.dtype.Float32))
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "shape" in str(e).lower() or "mismatch" in str(e).lower(), f"Got: {e}"

if __name__ == "__main__":
    test_custom_function_forward_exception()
    print("  forward exception test passed")
    test_custom_function_backward_exception()
    print("  backward exception test passed")
    test_gradient_shape_mismatch()
    print("  gradient shape mismatch test passed")
    print("All error handling tests passed!")
