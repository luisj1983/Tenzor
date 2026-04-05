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
