"""Type stubs for tenzor.nn module (neural network layers and functions)."""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Callable
from tenzor import Tensor, DType, Device

class Module:
    """Base class for all neural network modules."""

    training: bool

    def __init__(self) -> None: ...

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def parameters(self) -> List[Tensor]: ...
    def named_parameters(self, prefix: str = '') -> List[Tuple[str, Tensor]]: ...
    def children(self) -> List[Module]: ...
    def named_children(self) -> List[Tuple[str, Module]]: ...
    def modules(self) -> List[Module]: ...
    def named_modules(self, prefix: str = '') -> List[Tuple[str, Module]]: ...

    def train(self, mode: bool = True) -> Module: ...
    def eval(self) -> Module: ...
    def to(self, device: Device) -> Module: ...
    def cpu(self) -> Module: ...
    def cuda(self, device: int = 0) -> Module: ...
    def type(self, dtype: DType) -> Module: ...
    def float(self) -> Module: ...
    def double(self) -> Module: ...
    def half(self) -> Module: ...

    def zero_grad(self) -> None: ...
    def state_dict(self) -> Dict[str, Tensor]: ...
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True) -> None: ...

    def register_parameter(self, name: str, param: Optional[Tensor]) -> None: ...
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

class Bilinear(Module):
    """Bilinear layer."""

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True
    ) -> None: ...

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor: ...

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
        kernel_size: int | Tuple[int, int, int],
        stride: int | Tuple[int, int, int] = 1,
        padding: int | Tuple[int, int, int] = 0,
        dilation: int | Tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

# Transposed convolution layers
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
        dilation: int = 1,
        padding_mode: str = 'zeros'
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
        batch_first: bool = False
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
        batch_first: bool = False
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

# Functional API (stateless operation wrappers)
def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor: ...
def max_pool2d(input: Tensor, kernel_size: int, stride: int = -1, padding: int = 0) -> Tensor: ...
def avg_pool2d(input: Tensor, kernel_size: int, stride: int = -1, padding: int = 0) -> Tensor: ...
def batch_norm(input: Tensor, num_features: int, training: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Tensor: ...
def layer_norm(input: Tensor, normalized_shape: List[int], eps: float = 1e-5) -> Tensor: ...
def group_norm(input: Tensor, num_groups: int, num_channels: int, eps: float = 1e-5) -> Tensor: ...
def interpolate(input: Tensor, size: List[int], mode: str = 'bilinear', align_corners: bool = False) -> Tensor: ...
def embedding(input: Tensor, weight: Tensor, padding_idx: int = -1) -> Tensor: ...
def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor: ...

# Gradient clipping utilities
def clip_grad_norm_(parameters: List[Tensor], max_norm: float, norm_type: float = 2.0) -> float: ...
def clip_grad_value_(parameters: List[Tensor], clip_value: float) -> None: ...

# Utility functions
def init_weights(module: Module, method: str = 'xavier_uniform') -> None:
    """Initialize module weights."""
    ...

def count_parameters(module: Module, trainable_only: bool = True) -> int:
    """Count total number of parameters in a module."""
    ...
