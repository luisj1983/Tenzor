"""Type stubs for tenzor.nn module (neural network layers and functions)."""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Callable, Iterator, overload
from tenzor import Tensor, Variable, DType, Device

class Module:
    """Base class for all neural network modules."""

    training: bool

    def __init__(self) -> None: ...

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def parameters(self) -> Iterator[Variable]: ...
    def named_parameters(self, prefix: str = '') -> Iterator[Tuple[str, Variable]]: ...
    def children(self) -> Iterator[Module]: ...
    def named_children(self) -> Iterator[Tuple[str, Module]]: ...
    def modules(self) -> Iterator[Module]: ...
    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, Module]]: ...

    def train(self, mode: bool = True) -> Module: ...
    def eval(self) -> Module: ...

    @overload
    def to(self, device: Device) -> Module: ...
    @overload
    def to(self, dtype: DType) -> Module: ...
    @overload
    def to(self, device: Device, dtype: DType) -> Module: ...
    def cpu(self) -> Module: ...
    def cuda(self, device: int = 0) -> Module: ...
    def type(self, dtype: DType) -> Module: ...
    def float(self) -> Module: ...
    def double(self) -> Module: ...
    def half(self) -> Module: ...

    def zero_grad(self) -> None: ...
    def state_dict(self) -> Dict[str, Tensor]: ...
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True) -> None: ...

    def register_parameter(self, name: str, param: Optional[Variable]) -> None: ...
    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None: ...
    def register_module(self, name: str, module: Optional[Module]) -> None: ...

    def apply(self, fn: Callable[[Module], None]) -> Module: ...

class Sequential(Module):
    """Sequential container for neural network modules."""

    def __init__(self, *args: Module) -> None: ...
    def append(self, module: Module) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ModuleList(Module):
    """List container for neural network modules."""

    def __init__(self, modules: Optional[List[Module]] = None) -> None: ...
    def append(self, module: Module) -> None: ...
    def extend(self, modules: List[Module]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Module: ...
    def __setitem__(self, idx: int, module: Module) -> None: ...
    def __iter__(self) -> Any: ...

class ModuleDict(Module):
    """Dictionary container for neural network modules."""

    def __init__(self, modules: Optional[Dict[str, Module]] = None) -> None: ...
    def __getitem__(self, key: str) -> Module: ...
    def __setitem__(self, key: str, module: Module) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def keys(self) -> List[str]: ...
    def values(self) -> List[Module]: ...
    def items(self) -> List[Tuple[str, Module]]: ...

class ParameterList(Module):
    """List container for parameters."""

    def __init__(self, parameters: Optional[List[Any]] = None) -> None: ...
    def append(self, param: Any) -> "ParameterList": ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Any: ...
    def __iter__(self) -> Any: ...

class ParameterDict(Module):
    """Dictionary container for parameters."""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, param: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def keys(self) -> List[str]: ...
    def __contains__(self, key: str) -> bool: ...

# Linear layers
class Linear(Module):
    """Fully connected linear layer."""

    in_features: int
    out_features: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class LazyLinear(Module):
    """Linear layer with lazy initialization of in_features."""

    out_features: int
    weight: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(
        self,
        out_features: int,
        bias: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

# X.9: Bilinear is declared in .pyi but not bound from C++. Removed pending
# either a C++ binding (bindings_nn.cpp) or pure-Python implementation in nn.py.

# Convolutional layers
class Conv1d(Module):
    """1D convolution layer."""

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class Conv2d(Module):
    """2D convolution layer."""

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class Conv3d(Module):
    """3D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

# Transposed convolution layers
class ConvTranspose3d(Module):
    """3D transposed convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class ConvTranspose1d(Module):
    """1D transposed convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class ConvTranspose2d(Module):
    """2D transposed convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        output_padding: int | Tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | Tuple[int, int] = 1,
        padding_mode: str = 'zeros'
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

# Pooling layers
class MaxPool1d(Module):
    """1D max pooling layer."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class MaxPool2d(Module):
    """2D max pooling layer."""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool1d(Module):
    """1D average pooling layer."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool2d(Module):
    """2D average pooling layer."""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveAvgPool1d(Module):
    """1D adaptive average pooling layer."""

    def __init__(self, output_size: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveAvgPool2d(Module):
    """2D adaptive average pooling layer."""

    def __init__(self, output_size: int | Tuple[int, int]) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveMaxPool1d(Module):
    """1D adaptive max pooling layer."""

    def __init__(self, output_size: int, return_indices: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveMaxPool2d(Module):
    """2D adaptive max pooling layer."""

    def __init__(self, output_size: int | Tuple[int, int], return_indices: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class MaxPool3d(Module):
    """3D max pooling layer."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool3d(Module):
    """3D average pooling layer."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveAvgPool3d(Module):
    """3D adaptive average pooling layer."""

    @overload
    def __init__(self, output_size: int) -> None: ...
    @overload
    def __init__(self, output_d: int, output_h: int, output_w: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveMaxPool3d(Module):
    """3D adaptive max pooling layer."""

    @overload
    def __init__(self, output_size: int) -> None: ...
    @overload
    def __init__(self, output_d: int, output_h: int, output_w: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# Padding layers
class ConstantPad1d(Module):
    """1D constant padding layer."""

    @overload
    def __init__(self, padding_left: int, padding_right: int, value: float = 0.0) -> None: ...
    @overload
    def __init__(self, padding: int, value: float = 0.0) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ConstantPad2d(Module):
    """2D constant padding layer."""

    @overload
    def __init__(self, padding_left: int, padding_right: int, padding_top: int, padding_bottom: int, value: float = 0.0) -> None: ...
    @overload
    def __init__(self, padding: int, value: float = 0.0) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ConstantPad3d(Module):
    """3D constant padding layer."""

    @overload
    def __init__(self, padding: List[int], value: float = 0.0) -> None: ...
    @overload
    def __init__(self, padding: int, value: float = 0.0) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReflectionPad1d(Module):
    """1D reflection padding layer."""

    @overload
    def __init__(self, padding_left: int, padding_right: int) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReflectionPad2d(Module):
    """2D reflection padding layer."""

    @overload
    def __init__(self, padding_left: int, padding_right: int, padding_top: int, padding_bottom: int) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReplicationPad1d(Module):
    """1D replication padding layer."""

    @overload
    def __init__(self, padding_left: int, padding_right: int) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReplicationPad2d(Module):
    """2D replication padding layer."""

    @overload
    def __init__(self, padding_left: int, padding_right: int, padding_top: int, padding_bottom: int) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReplicationPad3d(Module):
    """3D replication padding layer."""

    @overload
    def __init__(self, padding: List[int]) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ZeroPad2d(Module):
    """2D zero padding layer."""

    @overload
    def __init__(self, padding_left: int, padding_right: int, padding_top: int, padding_bottom: int) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class CircularPad1d(Module):
    """1D circular padding layer. Wraps values from the opposite end."""

    @overload
    def __init__(self, padding_left: int, padding_right: int) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class CircularPad2d(Module):
    """2D circular padding layer. Wraps values from the opposite end."""

    @overload
    def __init__(self, padding_left: int, padding_right: int, padding_top: int, padding_bottom: int) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class CircularPad3d(Module):
    """3D circular padding layer. Wraps values from the opposite end."""

    @overload
    def __init__(self, padding: List[int]) -> None: ...
    @overload
    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# Upsample layer
class Upsample(Module):
    """Upsamples input tensor to given size or scale factor."""

    def __init__(
        self,
        size: Optional[List[int]] = None,
        scale_factor: Optional[float] = None,
        mode: str = 'nearest',
        align_corners: bool = False
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

# Normalization layers
class BatchNorm1d(Module):
    """1D batch normalization layer."""

    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    weight: Optional[Tensor]
    bias: Optional[Tensor]
    running_mean: Optional[Tensor]
    running_var: Optional[Tensor]

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class BatchNorm2d(Module):
    """2D batch normalization layer."""

    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    weight: Optional[Tensor]
    bias: Optional[Tensor]
    running_mean: Optional[Tensor]
    running_var: Optional[Tensor]

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class LayerNorm(Module):
    """Layer normalization."""

    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    weight: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(
        self,
        normalized_shape: int | Tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class GroupNorm(Module):
    """Group normalization."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class InstanceNorm1d(Module):
    """1D instance normalization."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class InstanceNorm2d(Module):
    """2D instance normalization."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class InstanceNorm3d(Module):
    """3D instance normalization."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class RMSNorm(Module):
    """Root-Mean-Square layer normalization."""

    normalized_shape: int
    eps: float

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class GLU(Module):
    """Gated Linear Unit: splits the input along ``dim`` into halves a, b
    and returns ``a * sigmoid(b)``."""

    dim: int

    def __init__(self, dim: int = -1) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# Recurrent layers
class RNN(Module):
    """Multi-layer Elman RNN."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False
    ) -> None: ...

    def forward(
        self,
        input: Tensor,
        h_0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]: ...

class LSTM(Module):
    """Multi-layer Long Short-Term Memory (LSTM) RNN."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False
    ) -> None: ...

    def forward(
        self,
        input: Tensor,
        h_0: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]: ...

class GRU(Module):
    """Multi-layer Gated Recurrent Unit (GRU) RNN."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False
    ) -> None: ...

    def forward(
        self,
        input: Tensor,
        h_0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]: ...

# RNN Cell layers
class RNNCell(Module):
    """An Elman RNN cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = "tanh") -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor: ...

class LSTMCell(Module):
    """A long short-term memory (LSTM) cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]: ...

class GRUCell(Module):
    """A gated recurrent unit (GRU) cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor: ...

# Transformer layers
class MultiheadAttention(Module):
    """Multi-head attention mechanism."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False
    ) -> None: ...

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]: ...

class TransformerEncoderLayer(Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        batch_first: bool = False,
        norm_first: bool = False
    ) -> None: ...

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor: ...

class TransformerDecoderLayer(Module):
    """Transformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        batch_first: bool = False,
        norm_first: bool = False
    ) -> None: ...

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor: ...

# Dropout layers
class Dropout(Module):
    """Dropout layer."""

    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Dropout2d(Module):
    """2D dropout layer."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# Activation functions (as modules)
class ReLU(Module):
    """ReLU activation function."""

    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LeakyReLU(Module):
    """Leaky ReLU activation function."""

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class PReLU(Module):
    """Parametric ReLU activation function."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ELU(Module):
    """ELU activation function."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class SELU(Module):
    """SELU activation function."""

    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class GELU(Module):
    """GELU activation function."""

    def __init__(self, approximate: str = 'none') -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Sigmoid(Module):
    """Sigmoid activation function."""

    def forward(self, input: Tensor) -> Tensor: ...

class Tanh(Module):
    """Tanh activation function."""

    def forward(self, input: Tensor) -> Tensor: ...

class Softmax(Module):
    """Softmax activation function."""

    def __init__(self, dim: Optional[int] = None) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LogSoftmax(Module):
    """Log-Softmax activation function."""

    def __init__(self, dim: Optional[int] = None) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Hardswish(Module):
    """Hard Swish activation function. (Hardswish to match torch naming.)"""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Hardsigmoid(Module):
    """Hard Sigmoid activation function. (Hardsigmoid to match torch naming.)"""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Mish(Module):
    """Mish activation function."""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class SiLU(Module):
    """SiLU (Swish) activation function."""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Parameter:
    """A tensor that is automatically registered as a parameter."""

    def __init__(self, data: Tensor, requires_grad: bool = True) -> None: ...

# Functional activation functions
def relu(input: Tensor, inplace: bool = False) -> Tensor: ...
def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor: ...
def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor: ...
def selu(input: Tensor, inplace: bool = False) -> Tensor: ...
def gelu(input: Tensor, approximate: str = 'none') -> Tensor: ...
def sigmoid(input: Tensor) -> Tensor: ...
def tanh(input: Tensor) -> Tensor: ...
def softmax(input: Tensor, dim: int) -> Tensor: ...
def log_softmax(input: Tensor, dim: int) -> Tensor: ...

# Loss functions
class Loss(Module):
    """Base class for loss functions."""

    reduction: str

    def __init__(self, reduction: str = 'mean') -> None: ...

class MSELoss(Loss):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = 'mean') -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class CrossEntropyLoss(Loss):
    """Cross-entropy loss."""

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class BCELoss(Loss):
    """Binary Cross-Entropy loss."""

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean'
    ) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class BCEWithLogitsLoss(Loss):
    """Binary Cross-Entropy with Logits loss."""

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        pos_weight: Optional[Tensor] = None
    ) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class NLLLoss(Loss):
    """Negative Log-Likelihood loss."""

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class L1Loss(Loss):
    """L1 loss (Mean Absolute Error)."""

    def __init__(self, reduction: str = 'mean') -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class SmoothL1Loss(Loss):
    """Smooth L1 loss."""

    def __init__(self, reduction: str = 'mean', beta: float = 1.0) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class HuberLoss(Loss):
    """Huber loss."""

    def __init__(self, reduction: str = 'mean', delta: float = 1.0) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class KLDivLoss(Loss):
    """Kullback-Leibler divergence loss."""

    def __init__(self, reduction: str = 'mean', log_target: bool = False) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

# Embedding layers
class Embedding(Module):
    """Embedding layer."""

    num_embeddings: int
    embedding_dim: int
    weight: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

class EmbeddingBag(Module):
    """Embedding bag layer (sum/mean of embeddings)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = 'mean',
        sparse: bool = False,
        include_last_offset: bool = False
    ) -> None: ...

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None
    ) -> Tensor: ...

# Audit E.10: functional wrappers (dropout/linear/max_pool2d/avg_pool2d/
# batch_norm/layer_norm/group_norm/interpolate/grid_sample/affine_grid/
# embedding/binary_cross_entropy_with_logits) live in `tenzor.nn.functional`
# at runtime, not directly on `tenzor.nn`.  The canonical declarations are
# in functional.pyi.  Removed false top-level declarations that drove
# typecheckers to undefined-attribute errors on the real `tz.nn.functional.*`
# paths.

# Gradient clipping utilities (these *do* exist directly on `tenzor.nn`).
def clip_grad_norm_(parameters: List[Variable], max_norm: float, norm_type: float = 2.0) -> float: ...
def clip_grad_value_(parameters: List[Variable], clip_value: float) -> None: ...

# Audit E.10: init_weights / count_parameters were declared here but never
# existed at runtime.  Removed.

# X.9: utility classes / helpers defined in python/tenzor/nn.py — declared
# here so the stub matches the runtime surface area.
class RemovableHandle:
    """Handle returned by Module.register_*_hook — call .remove() to unhook."""

    id: int

    def remove(self) -> None: ...


class PackedSequence:
    """Container for variable-length sequences used by RNN/LSTM/GRU."""

    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Optional[Tensor]
    unsorted_indices: Optional[Tensor]


def pack_padded_sequence(
    input: Tensor,
    lengths: Any,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence: ...


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor]: ...


def pack_sequence(sequences: List[Tensor], enforce_sorted: bool = True) -> PackedSequence: ...


# Sub-module declaration
from tenzor import functional as functional


# ---------------------------------------------------------------------------
# Stream S23: .pyi parity sweep — surface the full ``tenzor.nn`` namespace
# registered by the C++ extension. These names are bound by
# ``python/bindings/bindings_nn*.cpp`` (and small Python wrappers in
# ``python/tenzor/nn.py``); without declaring them here, static checkers
# rejected legitimate ``tz.nn.Transformer(...)`` / ``tz.nn.FocalLoss(...)``
# attribute access. Constructor / forward signatures use the minimal
# ``*args/**kwargs`` form where the exact binding surface varies across
# overloads; the load-bearing change is that the names exist.
# ---------------------------------------------------------------------------

# --- Activations / pointwise transforms (Module form) ---
class ReLU6(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Hardtanh(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Swish(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Identity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Convolutional / locality layers ---
class AtrousSeparableConv2d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class DeformableConv2d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LazyConv1d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LazyConv2d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LazyConv3d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class FusedMBConv(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class InvertedResidual(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class SqueezeExcitation(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ASPP(Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Normalization ---
class BatchNorm3d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LocalResponseNorm(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class SyncBatchNorm(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Pooling ---
class LPPool1d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LPPool2d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Shape utilities ---
class Flatten(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Unflatten(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Fold(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Unfold(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ChannelShuffle(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class PixelShuffle(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class PixelUnshuffle(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Attention / transformer ---
class ALiBi(Module):
    """Attention with Linear Biases (positional encoding)."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

class GroupedQueryAttention(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

class WindowAttention(Module):
    """Window-based multi-head self-attention (Swin Transformer)."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

class Transformer(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

class TransformerEncoder(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

class TransformerDecoder(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

class PatchEmbedding(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class PositionalEncoding(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Regularization ---
class AlphaDropout(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class DropPath(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Loss functions (Module form) ---
class CTCLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class CosineEmbeddingLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class CosineSimilarity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class DiceLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class FocalLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class GaussianNLLLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class HingeEmbeddingLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class InfoNCELoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MarginRankingLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MultiLabelMarginLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MultiLabelSoftMarginLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MultiMarginLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class NTXentLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class PairwiseDistance(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class PoissonNLLLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class SoftMarginLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class TripletLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class TripletMarginLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class TripletMarginWithDistanceLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...

class Reduction:
    """Enum-like sentinel for loss reduction modes (NONE / MEAN / SUM)."""
    NONE: "Reduction"
    MEAN: "Reduction"
    SUM: "Reduction"

NONE: Reduction
MEAN: Reduction
SUM: Reduction

# --- Sparse / embedding ---
class SparseEmbedding(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class SparseLinear(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# --- Metrics ---
class Metric:
    """Base class for metric-tracking helpers."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...
    def compute(self) -> Any: ...
    def reset(self) -> None: ...

class Accuracy(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class AUROC(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class AverageMode:
    """Averaging strategy for multi-class metrics (NONE / MICRO / MACRO / ...)."""

class ConfusionMatrix(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class F1Score(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class MeanAbsoluteError(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class MeanSquaredError(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Precision(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Recall(Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

# --- Training callbacks / checkpoints ---
class Callback:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class CallbackList(Callback):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class EarlyStoppingCallback(Callback):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LRSchedulerCallback(Callback):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ModelCheckpointCallback(Callback):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ProgressCallback(Callback):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Checkpoint:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class CheckpointConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ModelCheckpoint:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class AutoCheckpoint:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class TrainingMetadata:
    """Captures epoch/step/lr/loss bookkeeping during training."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

# --- High-level helpers ---
class NeuralNetwork(Module):
    """High-level convenience wrapper around Module + Optimizer + DataLoader."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class SimpleDataLoader:
    """Lightweight DataLoader wrapper (no multi-processing)."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

# --- Functional surface (C++ bindings exposed at tenzor.nn level) ---
def bce_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def cosine_embedding_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def create_shifted_window_mask(*args: Any, **kwargs: Any) -> Tensor: ...
def cross_entropy(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_adaptive_avg_pool1d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_adaptive_avg_pool2d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_adaptive_avg_pool3d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_adaptive_max_pool1d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_adaptive_max_pool2d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_adaptive_max_pool3d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_avg_pool1d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_avg_pool2d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_avg_pool3d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_batch_norm(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_binary_cross_entropy_with_logits(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_conv1d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_conv2d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_conv3d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_conv_transpose2d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_cosine_similarity(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_dropout(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_embedding(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_group_norm(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_instance_norm(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_interpolate(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_layer_norm(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_linear(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_max_pool1d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_max_pool2d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_max_pool3d(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_multi_head_attention_forward(*args: Any, **kwargs: Any) -> Any: ...
def functional_nll_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_normalize(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_pad(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_prelu(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_rms_norm(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_scaled_dot_product_attention(*args: Any, **kwargs: Any) -> Tensor: ...
def functional_smooth_l1_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def gaussian_nll_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def glu(*args: Any, **kwargs: Any) -> Tensor: ...
def hardsigmoid(*args: Any, **kwargs: Any) -> Tensor: ...
def hardswish(*args: Any, **kwargs: Any) -> Tensor: ...
def hardtanh(*args: Any, **kwargs: Any) -> Tensor: ...
def hinge_embedding_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def huber_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def info_nce_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def kl_div_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def l1_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def log_sigmoid(*args: Any, **kwargs: Any) -> Tensor: ...
def margin_ranking_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def mish(*args: Any, **kwargs: Any) -> Tensor: ...
def mse_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def multi_label_soft_margin_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def multi_margin_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def nll_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def nt_xent_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def pad_sequence(*args: Any, **kwargs: Any) -> Tensor: ...
def poisson_nll_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def rrelu(*args: Any, **kwargs: Any) -> Tensor: ...
def smooth_l1_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def soft_margin_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def softplus(*args: Any, **kwargs: Any) -> Tensor: ...
def swish(*args: Any, **kwargs: Any) -> Tensor: ...
def triplet_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def triplet_margin_loss(*args: Any, **kwargs: Any) -> Tensor: ...
def window_partition(*args: Any, **kwargs: Any) -> Tensor: ...
def window_reverse(*args: Any, **kwargs: Any) -> Tensor: ...

# Lowercase reduction aliases (Reduction.NONE / .MEAN / .SUM are also
# exposed as bare module attributes for the pythonic enum style).
none: Reduction
mean: Reduction
sum: Reduction

# --- Submodules surfaced under tenzor.nn ---
hrm: Any            # HRM (Hierarchical Reasoning Module) layer namespace
init: Any           # tenzor.nn.init — weight-init helpers (xavier, kaiming, ...)
quantization: Any   # nn-level quantization helpers (Conv2dQuant, etc.)
