"""
Tenzor Autograd Module

Provides automatic differentiation with support for custom differentiable
operations via the Function base class.

Usage:
    import tenzor as tz

    class MyReLU(tz.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return tz.clamp_min(input, 0.0)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad = grad_output * (input > 0).to(tz.dtype.float32)
            return (grad,)

    result = MyReLU.apply(x)
"""

from . import tenzor_core as _core

_autograd_cpp = _core.autograd


class Function:
    """Base class for custom autograd functions.

    Subclass this and implement ``forward`` and ``backward`` as
    ``@staticmethod`` methods.  Call ``cls.apply(*inputs)`` to run the
    function and wire it into the autograd graph.

    Parameters
    ----------
    ctx : FunctionCtx
        Context object.  Use ``ctx.save_for_backward(...)`` in forward
        and ``ctx.saved_tensors`` in backward.

    Example
    -------
    >>> class Square(tz.autograd.Function):
    ...     @staticmethod
    ...     def forward(ctx, x):
    ...         ctx.save_for_backward(x)
    ...         return x * x
    ...     @staticmethod
    ...     def backward(ctx, grad_output):
    ...         x, = ctx.saved_tensors
    ...         return (grad_output * 2.0 * x,)
    >>> y = Square.apply(x)
    """

    @staticmethod
    def forward(ctx, *inputs):
        """Perform the forward computation.

        Must be overridden by subclass.
        """
        raise NotImplementedError(
            "Subclasses of Function must implement forward()")

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Compute gradients of the forward operation.

        Must be overridden by subclass.  Return a tuple with one
        gradient per forward input (use ``None`` for non-differentiable
        inputs).
        """
        raise NotImplementedError(
            "Subclasses of Function must implement backward()")

    @classmethod
    def apply(cls, *inputs):
        """Apply the custom function and register it in the autograd graph.

        Parameters
        ----------
        *inputs : Variable or Tensor
            Input tensors.

        Returns
        -------
        Variable or tuple of Variable
            Output(s) of the forward pass.
        """
        return _autograd_cpp.apply_custom_function(cls, *inputs)


# Re-export C++ autograd utilities
# FunctionCtx is registered on the root module, not the autograd submodule
FunctionCtx = _core.FunctionCtx
grad = _autograd_cpp.grad

# Audit E.11: gradcheck / gradgradcheck now have Python bindings. Surface
# them through `tenzor.autograd` so the autograd.pyi declarations match
# the runtime.
gradcheck = _autograd_cpp.gradcheck
gradgradcheck = _autograd_cpp.gradgradcheck


def backward(tensors, grad_tensors=None, retain_graph: bool = False,
             create_graph: bool = False):
    """Compute gradients of a set of output tensors via reverse mode.

    Mirrors ``torch.autograd.backward``: walks the autograd graph backward
    from each tensor with the corresponding incoming gradient, accumulating
    gradients on every leaf Variable that has ``requires_grad=True``.

    Parameters
    ----------
    tensors : Variable or sequence of Variable
        The output Variable(s) whose gradients to propagate. A scalar
        Variable can be passed without a grad_tensor; non-scalars require
        an explicit corresponding grad_tensor.
    grad_tensors : Tensor or sequence of Tensor, optional
        Per-output incoming gradients. ``None`` is allowed only when the
        matching `tensors` entry is scalar.
    retain_graph : bool, default False
        If True, the autograd graph is kept after the backward pass so
        the same outputs can be used in a second backward.
    create_graph : bool, default False
        If True, the backward pass is itself differentiable (higher-order
        gradients).
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    if grad_tensors is None:
        grad_tensors = [None] * len(tensors)
    elif not isinstance(grad_tensors, (list, tuple)):
        grad_tensors = [grad_tensors]
    if len(grad_tensors) != len(tensors):
        raise ValueError(
            f"backward: tensors and grad_tensors must have matching length "
            f"({len(tensors)} vs {len(grad_tensors)})"
        )
    for i, (t, g) in enumerate(zip(tensors, grad_tensors)):
        # Retain the graph if any subsequent output still needs to fire,
        # mirroring the C++ side's `retain_graph=true` for non-final
        # outputs.
        last = (i == len(tensors) - 1)
        retain = retain_graph or create_graph or not last
        t.backward(g, retain, create_graph)
