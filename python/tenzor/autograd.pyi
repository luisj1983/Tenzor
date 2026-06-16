"""Type stubs for tenzor.autograd module (automatic differentiation)."""

from __future__ import annotations
from typing import Optional, List, Tuple, Callable, Any
from tenzor import Tensor, Variable, DType, Device

def backward(
    tensors: List[Variable],
    grad_tensors: Optional[List[Tensor]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> None:
    """Compute gradients of given tensors with respect to graph leaves."""
    ...

def grad(
    outputs: List[Variable],
    inputs: List[Variable],
    grad_outputs: Optional[List[Tensor]] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> Tuple[Optional[Tensor], ...]:
    """Compute and return gradients of outputs with respect to inputs."""
    ...

class FunctionCtx:
    """Context object passed to Function.forward / Function.backward.

    Holds the ``saved_tensors`` saved via ``ctx.save_for_backward(...)``
    during forward; ``backward`` reads them back out.
    """

    saved_tensors: Tuple[Variable, ...]

    def save_for_backward(self, *tensors: Variable) -> None:
        """Save tensors for retrieval in backward."""
        ...


class Function:
    """Base class for custom autograd functions."""

    @staticmethod
    def forward(ctx: Any, *args: Any) -> Any:
        """Define the forward pass of the custom function."""
        ...

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Define the backward pass (gradient computation)."""
        ...

    @staticmethod
    def apply(*args: Any) -> Any:
        """Apply the custom function."""
        ...

class no_grad:
    """Context manager that disables gradient computation."""

    def __enter__(self) -> no_grad: ...
    def __exit__(self, *args: Any) -> None: ...

class enable_grad:
    """Context manager that enables gradient computation."""

    def __enter__(self) -> enable_grad: ...
    def __exit__(self, *args: Any) -> None: ...

class inference_mode:
    """Context manager / decorator for inference mode.

    Stronger than no_grad(): disables gradient computation AND skips version
    counter increments for in-place ops.
    """

    def __init__(self, mode: bool = True) -> None: ...
    def __enter__(self) -> inference_mode: ...
    def __exit__(self, *args: Any) -> None: ...
    def __call__(self, func: Callable) -> Callable: ...

def gradcheck(
    func: Callable,
    inputs: Tuple[Variable, ...],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Check gradients computed via backprop against numerical gradients."""
    ...

def gradgradcheck(
    func: Callable,
    inputs: Tuple[Variable, ...],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = False,
) -> bool:
    """Check second-order gradients by applying gradcheck to the gradient function."""
    ...

# audit-10 OO.12: graph utilities exposed via tenzor_core.autograd; re-exported
# from tenzor/autograd.py so they appear under the tenzor.autograd module.
def make_dot(root: Variable, params: Any = ...) -> str:
    """Generate a Graphviz DOT-format string for the computation graph rooted at ``root``."""
    ...

def optimize_graph(root: Variable) -> Any:
    """Optimize the computation graph in-place; returns a stats dict with fusion / DCE counts."""
    ...
