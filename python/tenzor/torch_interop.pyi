"""Type stubs for tenzor.torch_interop module (PyTorch tensor interoperability).

S23: this stub is *conditional* — the underlying C++ binding is only
registered when the extension was built with ``TENZOR_HAS_TORCH``
defined (PyTorch headers available at configure time). When the
extension is built WITHOUT torch support, ``import tenzor.torch_interop``
raises ``ModuleNotFoundError`` at runtime. The drift checker
(``tools/check_pyi_parity.py``) skips this stub in that case via its
OPTIONAL_RUNTIME_STUBS list, so type-checking-only consumers can still
see the signatures.

If you depend on this module, gate your import:

    if TYPE_CHECKING:
        from tenzor import torch_interop  # type-checker sees stub
    else:
        torch_interop = None  # runtime fallback
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
from tenzor import Tensor, Device

def can_zero_copy_to_torch(tensor: Tensor) -> bool:
    """
    Check if zero-copy conversion from Tenzor to PyTorch is possible.

    Zero-copy is possible when:
    - Tensor is contiguous
    - Device is compatible (CPU-CPU, CUDA-CUDA)
    - Data type is supported by PyTorch

    Args:
        tensor: Tenzor tensor to check

    Returns:
        True if zero-copy conversion is possible
    """
    ...

def can_zero_copy_from_torch(torch_tensor: Any) -> bool:
    """
    Check if zero-copy conversion from PyTorch to Tenzor is possible.

    Args:
        torch_tensor: PyTorch tensor to check

    Returns:
        True if zero-copy conversion is possible
    """
    ...

def tensor_to_torch(tensor: Tensor, requires_grad: bool = False) -> Any:
    """
    Convert Tenzor tensor to PyTorch tensor.

    Performs zero-copy conversion when possible, otherwise copies data.
    For CUDA tensors, uses the same device pointer.

    Args:
        tensor: Tenzor tensor to convert
        requires_grad: Whether PyTorch tensor should track gradients

    Returns:
        PyTorch tensor

    Raises:
        RuntimeError: If conversion fails
    """
    ...

def tensor_from_torch(torch_tensor: Any, device: Optional[Device] = None) -> Tensor:
    """
    Convert PyTorch tensor to Tenzor tensor.

    Performs zero-copy conversion when possible, otherwise copies data.

    Args:
        torch_tensor: PyTorch tensor to convert
        device: Target device (if not specified, uses PyTorch tensor's device)

    Returns:
        Tenzor tensor

    Raises:
        RuntimeError: If conversion fails
    """
    ...

def variable_to_torch(variable: Tensor) -> Any:
    """
    Convert Tenzor Variable to PyTorch Variable (with autograd).

    Converts gradient-tracking Variable to PyTorch Variable.
    If both have gradients, they share gradient storage when possible.

    Args:
        variable: Tenzor Variable

    Returns:
        PyTorch Variable (autograd::Variable)
    """
    ...

def variable_from_torch(torch_variable: Any) -> Tensor:
    """
    Convert PyTorch Variable to Tenzor Variable.

    Args:
        torch_variable: PyTorch Variable

    Returns:
        Tenzor Variable
    """
    ...

def sync_gradients(
    tenzor_var: Tensor,
    torch_var: Any,
    tenzor_to_torch: bool = True
) -> None:
    """
    Synchronize gradient storage between Tenzor and PyTorch.

    After backward pass, synchronize gradients so both frameworks
    see the same gradient values.

    Args:
        tenzor_var: Tenzor Variable
        torch_var: PyTorch Variable
        tenzor_to_torch: Direction of sync (True = Tenzor -> PyTorch, False = PyTorch -> Tenzor)
    """
    ...
