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

    x = tz.Variable(tz.randn([3], dtype=tz.dtype.float32), True)
    try:
        BadForward.apply(x)
        assert False, "Should have raised"
    except (RuntimeError, ValueError) as e:
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

    x = tz.Variable(tz.randn([3], dtype=tz.dtype.float32), True)
    y = BadBackward.apply(x)
    try:
        # Non-scalar output needs an explicit gradient for backward.
        y.backward(tz.ones([3], dtype=tz.dtype.float32))
        assert False, "Should have raised"
    except (RuntimeError, ValueError) as e:
        assert "backward" in str(e).lower() or "Test error" in str(e), f"Got: {e}"

def test_gradient_shape_mismatch():
    """User-supplied gradient with wrong shape should raise error."""
    x = tz.Variable(tz.randn([3, 4], dtype=tz.dtype.float32), True)
    # Multiply by scalar 1 to build a valid autograd graph — Variable.sum
    # is not bound, and raw Tensor.sum() would sever the graph.
    y = x * 1.0
    try:
        # Wrong-shape gradient: y has shape [3, 4] but we pass [3, 5].
        y.backward(tz.randn([3, 5], dtype=tz.dtype.float32))
        assert False, "Should have raised"
    except Exception as e:
        msg = str(e).lower()
        assert "shape" in msg or "mismatch" in msg or "size" in msg, f"Got: {e}"

if __name__ == "__main__":
    test_custom_function_forward_exception()
    print("  forward exception test passed")
    test_custom_function_backward_exception()
    print("  backward exception test passed")
    test_gradient_shape_mismatch()
    print("  gradient shape mismatch test passed")
    print("All error handling tests passed!")
