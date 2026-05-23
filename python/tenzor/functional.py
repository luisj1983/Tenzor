"""
Tenzor Neural Network Functional API

Provides functional versions of neural network operations, mirroring
torch.nn.functional (commonly imported as F).

Usage:
    import tenzor as tz
    import tenzor.nn.functional as F

    x = model.fc1(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=model.is_training())
    loss = F.cross_entropy(x, target)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

if TYPE_CHECKING:
    from ..tenzor_core import Variable

from .. import tenzor_core as _core

_nn = _core.nn


def _reduction(r):
    """Accept a string reduction ('mean'|'sum'|'none') OR a Reduction enum.

    The C++ bindings expose ``tenzor_core.nn.Reduction`` as a pybind11
    enum, but all functional loss wrappers in this module accept a plain
    Python string to match PyTorch's API. Pass-through enum values are
    returned untouched so callers that already use the enum still work.
    """
    if isinstance(r, str):
        key = r.lower()
        if key == "mean":
            return _nn.Reduction.MEAN
        if key == "sum":
            return _nn.Reduction.SUM
        if key == "none":
            return _nn.Reduction.NONE
        raise ValueError(
            f"Invalid reduction {r!r}. Expected 'mean', 'sum', or 'none'."
        )
    return r


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(input: Variable) -> Variable:
    """Apply the Rectified Linear Unit function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        ``max(0, x)`` applied element-wise.

    Example
    -------
    >>> x = tz.Variable(tz.randn([2, 3]))
    >>> y = F.relu(x)
    """
    return _nn.relu(input)


def leaky_relu(input: Variable, negative_slope: float = 0.01) -> Variable:
    """Apply the Leaky ReLU function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.
    negative_slope : float, optional
        Slope for negative values.  Default: ``0.01``.

    Returns
    -------
    Variable
        ``max(0, x) + negative_slope * min(0, x)`` applied element-wise.

    Example
    -------
    >>> y = F.leaky_relu(x, negative_slope=0.2)
    """
    return _nn.leaky_relu(input, negative_slope)


def elu(input: Variable, alpha: float = 1.0) -> Variable:
    """Apply the Exponential Linear Unit function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.
    alpha : float, optional
        Scale for the negative portion.  Default: ``1.0``.

    Returns
    -------
    Variable
        ELU activation applied element-wise.

    Example
    -------
    >>> y = F.elu(x, alpha=1.0)
    """
    return _nn.elu(input, alpha)


def gelu(input: Variable, approximate: str = "none") -> Variable:
    """Apply the Gaussian Error Linear Unit function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.
    approximate : str, optional
        ``"none"`` for the exact GELU using the error function (default), or
        ``"tanh"`` for the tanh-based approximation used in many transformer
        implementations.

    Returns
    -------
    Variable
        GELU activation applied element-wise.

    Example
    -------
    >>> y = F.gelu(x)
    >>> y = F.gelu(x, approximate="tanh")
    """
    return _nn.gelu(input, approximate)


def sigmoid(input: Variable) -> Variable:
    """Apply the sigmoid function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        ``1 / (1 + exp(-x))`` applied element-wise.

    Example
    -------
    >>> y = F.sigmoid(x)
    """
    return _nn.sigmoid(input)


def tanh(input: Variable) -> Variable:
    """Apply the hyperbolic tangent function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        ``tanh(x)`` applied element-wise.

    Example
    -------
    >>> y = F.tanh(x)
    """
    return _nn.tanh(input)


def softmax(input: Variable, dim: int = -1) -> Variable:
    """Apply the softmax function along a dimension.

    Parameters
    ----------
    input : Variable
        Input tensor.
    dim : int, optional
        Dimension along which softmax is computed.  Default: ``-1``.

    Returns
    -------
    Variable
        Tensor of same shape with values in ``[0, 1]`` summing to 1 along *dim*.

    Example
    -------
    >>> probs = F.softmax(logits, dim=-1)
    """
    return _nn.softmax(input, dim)


def log_softmax(input: Variable, dim: int = -1) -> Variable:
    """Apply log-softmax along a dimension.

    Numerically more stable than ``log(softmax(x))``.

    Parameters
    ----------
    input : Variable
        Input tensor.
    dim : int, optional
        Dimension along which log-softmax is computed.  Default: ``-1``.

    Returns
    -------
    Variable
        Log-probabilities of same shape as *input*.

    Example
    -------
    >>> log_probs = F.log_softmax(logits, dim=-1)
    """
    return _nn.log_softmax(input, dim)


def selu(input: Variable) -> Variable:
    """Apply the Scaled Exponential Linear Unit function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        SELU activation applied element-wise.

    Example
    -------
    >>> y = F.selu(x)
    """
    return _nn.selu(input)


def swish(input: Variable) -> Variable:
    """Apply the Swish (SiLU) activation function element-wise.

    Computes ``x * sigmoid(x)``.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        Swish activation applied element-wise.

    Example
    -------
    >>> y = F.swish(x)
    """
    return _nn.swish(input)


def mish(input: Variable) -> Variable:
    """Apply the Mish activation function element-wise.

    Computes ``x * tanh(softplus(x))``.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        Mish activation applied element-wise.

    Example
    -------
    >>> y = F.mish(x)
    """
    return _nn.mish(input)


def hardswish(input: Variable) -> Variable:
    """Apply the Hard Swish activation function element-wise.

    A piecewise-linear approximation to Swish.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        Hard Swish applied element-wise.

    Example
    -------
    >>> y = F.hardswish(x)
    """
    return _nn.hardswish(input)


def hardsigmoid(input: Variable) -> Variable:
    """Apply the Hard Sigmoid activation function element-wise.

    A piecewise-linear approximation to the sigmoid function.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        Hard Sigmoid applied element-wise.

    Example
    -------
    >>> y = F.hardsigmoid(x)
    """
    return _nn.hardsigmoid(input)


def glu(input: Variable, dim: int = -1) -> Variable:
    """Apply the Gated Linear Unit function.

    Splits the input in half along *dim*, applies sigmoid to the second half,
    and multiplies element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor whose size along *dim* must be even.
    dim : int, optional
        Dimension to split on.  Default: ``-1``.

    Returns
    -------
    Variable
        Tensor with size along *dim* halved.

    Example
    -------
    >>> y = F.glu(x, dim=-1)
    """
    return _nn.glu(input, dim)


def softplus(input: Variable, beta: float = 1.0,
             threshold: float = 20.0) -> Variable:
    """Apply the Softplus function element-wise.

    Computes ``(1/beta) * log(1 + exp(beta * x))``. For numerical stability,
    when ``beta * x > threshold`` the output reverts to the linear identity
    ``x`` (which softplus already approximates to within FP precision there).

    Parameters
    ----------
    input : Variable
        Input tensor.
    beta : float, optional
        Scaling factor.  Default: ``1.0``.
    threshold : float, optional
        Switch to a linear-region approximation when ``beta * x > threshold``.
        Default: ``20.0``.

    Returns
    -------
    Variable
        Softplus activation applied element-wise.

    Example
    -------
    >>> y = F.softplus(x, beta=1.0)
    """
    return _nn.softplus(input, beta, threshold)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(input: Variable, target: Variable, reduction: str = "mean") -> Variable:
    """Compute the mean squared error loss.

    Parameters
    ----------
    input : Variable
        Predicted values.
    target : Variable
        Ground truth values (same shape as *input*).
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value (scalar if reduction is ``'mean'`` or ``'sum'``).

    Example
    -------
    >>> loss = F.mse_loss(predictions, targets)
    """
    return _nn.mse_loss(input, target, _reduction(reduction))


def l1_loss(input: Variable, target: Variable, reduction: str = "mean") -> Variable:
    """Compute the mean absolute error loss.

    Parameters
    ----------
    input : Variable
        Predicted values.
    target : Variable
        Ground truth values (same shape as *input*).
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value (scalar if reduction is ``'mean'`` or ``'sum'``).

    Example
    -------
    >>> loss = F.l1_loss(predictions, targets)
    """
    return _nn.l1_loss(input, target, _reduction(reduction))


def cross_entropy(input: Variable, target: Variable, reduction: str = "mean",
                  ignore_index: int = -100) -> Variable:
    """Compute the cross-entropy loss between logits and class indices.

    Combines ``log_softmax`` and ``nll_loss`` in a single function for
    numerical stability.

    Parameters
    ----------
    input : Variable
        Unnormalized logits of shape ``(N, C)`` where *C* is the number
        of classes.
    target : Variable
        Class indices of shape ``(N,)`` with values in ``[0, C)``.
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.
    ignore_index : int, optional
        Class index that should be skipped. Samples whose target equals
        ``ignore_index`` contribute zero to the loss and are excluded from
        the mean denominator. Default: ``-100`` (matches PyTorch). Only
        honoured for the class-index target form; one-hot / soft targets
        are unaffected.

    Returns
    -------
    Variable
        Loss value (scalar if reduction is ``'mean'`` or ``'sum'``).

    Example
    -------
    >>> loss = F.cross_entropy(logits, labels)
    """
    return _nn.cross_entropy(input, target, _reduction(reduction), ignore_index)


def nll_loss(input: Variable, target: Variable, reduction: str = "mean",
             ignore_index: int = -100) -> Variable:
    """Compute the negative log-likelihood loss.

    Parameters
    ----------
    input : Variable
        Log-probabilities of shape ``(N, C)``.
    target : Variable
        Class indices of shape ``(N,)`` with values in ``[0, C)``.
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.
    ignore_index : int, optional
        Class index that should be skipped. Samples whose target equals
        ``ignore_index`` contribute zero to the loss and are excluded from
        the mean denominator. Default: ``-100`` (matches PyTorch).

    Returns
    -------
    Variable
        Loss value (scalar if reduction is ``'mean'`` or ``'sum'``).

    Example
    -------
    >>> log_probs = F.log_softmax(logits, dim=-1)
    >>> loss = F.nll_loss(log_probs, labels)
    """
    return _nn.nll_loss(input, target, _reduction(reduction), ignore_index)


def bce_loss(input: Variable, target: Variable, reduction: str = "mean") -> Variable:
    """Compute the binary cross-entropy loss.

    Parameters
    ----------
    input : Variable
        Probabilities in ``[0, 1]`` of shape ``(*)``.
    target : Variable
        Binary ground truth of the same shape as *input*.
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value (scalar if reduction is ``'mean'`` or ``'sum'``).

    Example
    -------
    >>> loss = F.bce_loss(F.sigmoid(logits), labels)
    """
    return _nn.bce_loss(input, target, _reduction(reduction))


def kl_div(input: Variable, target: Variable, reduction: str = "mean", log_target: bool = False) -> Variable:
    """Compute the Kullback-Leibler divergence loss.

    Parameters
    ----------
    input : Variable
        Log-probabilities of the predicted distribution.
    target : Variable
        Probabilities of the target distribution (same shape as *input*).
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.
    log_target : bool, optional
        If ``True``, *target* is given in log-space.  Default: ``False``.

    Returns
    -------
    Variable
        Scalar KL divergence value.

    Example
    -------
    >>> loss = F.kl_div(F.log_softmax(logits, dim=-1), target_probs)
    """
    return _nn.kl_div_loss(input, target, reduction, log_target)


def huber_loss(input: Variable, target: Variable, delta: float = 1.0, reduction: str = "mean") -> Variable:
    """Compute the Huber loss (smooth L1 with configurable threshold).

    Uses quadratic loss for small errors and linear loss for large errors.

    Parameters
    ----------
    input : Variable
        Predicted values.
    target : Variable
        Ground truth values (same shape as *input*).
    delta : float, optional
        Threshold at which to switch between quadratic and linear.
        Default: ``1.0``.
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.

    Example
    -------
    >>> loss = F.huber_loss(predictions, targets, delta=1.0)
    """
    return _nn.huber_loss(input, target, delta, reduction)


def smooth_l1_loss(input: Variable, target: Variable, reduction: str = "mean", beta: float = 1.0) -> Variable:
    """Compute the smooth L1 loss.

    Similar to Huber loss but parameterized by *beta*.

    Parameters
    ----------
    input : Variable
        Predicted values.
    target : Variable
        Ground truth values (same shape as *input*).
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.
    beta : float, optional
        Specifies the threshold.  Default: ``1.0``.

    Returns
    -------
    Variable
        Loss value.

    Example
    -------
    >>> loss = F.smooth_l1_loss(predictions, targets)
    """
    return _nn.smooth_l1_loss(input, target, _reduction(reduction), beta)


def soft_margin_loss(input: Variable, target: Variable, reduction: str = "mean") -> Variable:
    """Compute the two-class soft margin loss.

    Parameters
    ----------
    input : Variable
        Predicted values.
    target : Variable
        Target values (+1 or -1).
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.soft_margin_loss(input, target, _reduction(reduction))


def hinge_embedding_loss(input: Variable, target: Variable, margin: float = 1.0, reduction: str = "mean") -> Variable:
    """Compute the hinge embedding loss.

    Parameters
    ----------
    input : Variable
        Input distances.
    target : Variable
        Labels (+1 for similar, -1 for dissimilar).
    margin : float, optional
        Margin threshold.  Default: ``1.0``.
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.hinge_embedding_loss(input, target, margin, _reduction(reduction))


def poisson_nll_loss(input: Variable, target: Variable, log_input: bool = True, full: bool = False, eps: float = 1e-8,
                     reduction: str = "mean") -> Variable:
    """Compute the Poisson negative log-likelihood loss.

    Parameters
    ----------
    input : Variable
        Expected rate (or log-rate if *log_input* is ``True``).
    target : Variable
        Observed counts.
    log_input : bool, optional
        If ``True``, input is in log-space.  Default: ``True``.
    full : bool, optional
        Include Stirling approximation term.  Default: ``False``.
    eps : float, optional
        Small value for numerical stability.  Default: ``1e-8``.
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.poisson_nll_loss(input, target, log_input, full, eps, _reduction(reduction))


def cosine_embedding_loss(input1: Variable, input2: Variable, target: Variable, margin: float = 0.0, reduction: str = "mean") -> Variable:
    """Compute the cosine embedding loss.

    Parameters
    ----------
    input1, input2 : Variable
        Feature tensors of shape ``(N, D)``.
    target : Variable
        Labels (+1 for similar, -1 for dissimilar), shape ``(N,)``.
    margin : float, optional
        Margin for dissimilar pairs.  Default: ``0.0``.
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.cosine_embedding_loss(input1, input2, target, margin, _reduction(reduction))


def triplet_margin_loss(anchor: Variable, positive: Variable, negative: Variable, margin: float = 1.0, p: float = 2.0,
                        swap: bool = False, reduction: str = "mean") -> Variable:
    """Compute the triplet margin loss.

    Parameters
    ----------
    anchor, positive, negative : Variable
        Embedding tensors of shape ``(N, D)``.
    margin : float, optional
        Margin between positive/negative distances.  Default: ``1.0``.
    p : float, optional
        Norm degree for distance.  Default: ``2.0``.
    swap : bool, optional
        Use distance swap heuristic.  Default: ``False``.
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.triplet_margin_loss(anchor, positive, negative, margin, p,
                                   swap, _reduction(reduction))


def multi_label_soft_margin_loss(input: Variable, target: Variable, reduction: str = "mean") -> Variable:
    """Compute the multi-label soft margin loss.

    Parameters
    ----------
    input : Variable
        Raw logits of shape ``(N, C)``.
    target : Variable
        Multi-hot target of shape ``(N, C)``.
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.multi_label_soft_margin_loss(input, target, _reduction(reduction))


def multi_margin_loss(input: Variable, target: Variable, p: int = 1, margin: float = 1.0, reduction: str = "mean") -> Variable:
    """Compute the multi-class margin (hinge) loss.

    Parameters
    ----------
    input : Variable
        Class scores of shape ``(N, C)``.
    target : Tensor
        Class indices of shape ``(N,)``.
    p : int, optional
        Exponent (1 or 2).  Default: ``1``.
    margin : float, optional
        Margin threshold.  Default: ``1.0``.
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.multi_margin_loss(input, target, p, margin, _reduction(reduction))


def gaussian_nll_loss(input: Variable, target: Variable, var: Variable, full: bool = False, eps: float = 1e-6,
                      reduction: str = "mean") -> Variable:
    """Compute the Gaussian negative log-likelihood loss.

    Parameters
    ----------
    input : Variable
        Predicted means.
    target : Variable
        Observed values.
    var : Variable
        Predicted variances (must be positive).
    full : bool, optional
        Include constant log(2*pi) term.  Default: ``False``.
    eps : float, optional
        Minimum variance for stability.  Default: ``1e-6``.
    reduction : str, optional
        ``'mean'``, ``'sum'``, or ``'none'``.  Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.
    """
    return _nn.gaussian_nll_loss(input, target, var, full, eps, _reduction(reduction))


# ---------------------------------------------------------------------------
# Functional operations (stateless wrappers)
# ---------------------------------------------------------------------------

def dropout(input: Variable, p: float = 0.5, training: bool = True) -> Variable:
    """Apply dropout to the input during training.

    Randomly zeroes elements with probability *p* and scales remaining
    elements by ``1 / (1 - p)`` during training.  During evaluation the
    input is returned unchanged.

    Parameters
    ----------
    input : Variable
        Input tensor.
    p : float, optional
        Probability of an element being zeroed.  Default: ``0.5``.
    training : bool, optional
        Apply dropout if ``True``.  Default: ``True``.

    Returns
    -------
    Variable
        Tensor with dropout applied (if training).

    Example
    -------
    >>> x = F.dropout(x, p=0.5, training=model.is_training())
    """
    return _nn.functional_dropout(input, p, training)


def linear(input: Variable, weight: Variable, bias: Optional[Variable] = None) -> Variable:
    """Apply a linear transformation: ``y = x @ W^T + b``.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(*, in_features)``.
    weight : Variable
        Weight matrix of shape ``(out_features, in_features)``.
    bias : Variable or None, optional
        Bias vector of shape ``(out_features,)``.  Default: ``None``.

    Returns
    -------
    Variable
        Output of shape ``(*, out_features)``.

    Example
    -------
    >>> y = F.linear(x, weight, bias)
    """
    return _nn.functional_linear(input, weight, bias)


def max_pool2d(input: Variable, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> Variable:
    """Apply 2D max pooling over an input signal.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, C, H, W)``.
    kernel_size : int
        Size of the pooling window.
    stride : int or None, optional
        Stride of the pooling window.  Defaults to *kernel_size*.
    padding : int, optional
        Implicit zero padding on both sides.  Default: ``0``.

    Returns
    -------
    Variable
        Pooled tensor.

    Example
    -------
    >>> y = F.max_pool2d(x, kernel_size=2, stride=2)
    """
    if stride is None:
        stride = kernel_size
    return _nn.functional_max_pool2d(input, kernel_size, stride, padding)


def avg_pool2d(input: Variable, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> Variable:
    """Apply 2D average pooling over an input signal.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, C, H, W)``.
    kernel_size : int
        Size of the pooling window.
    stride : int or None, optional
        Stride of the pooling window.  Defaults to *kernel_size*.
    padding : int, optional
        Implicit zero padding on both sides.  Default: ``0``.

    Returns
    -------
    Variable
        Pooled tensor.

    Example
    -------
    >>> y = F.avg_pool2d(x, kernel_size=2, stride=2)
    """
    if stride is None:
        stride = kernel_size
    return _nn.functional_avg_pool2d(input, kernel_size, stride, padding)


def adaptive_avg_pool2d(input: Variable, output_size: Union[int, tuple[int, int]]) -> Variable:
    """Apply 2D adaptive average pooling.

    The output spatial dimensions match *output_size* regardless of input size.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, C, H, W)``.
    output_size : int or tuple[int, int]
        Target spatial dimensions ``(H_out, W_out)``.

    Returns
    -------
    Variable
        Tensor of shape ``(N, C, *output_size)``.

    Example
    -------
    >>> y = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pool
    """
    return _nn.functional_adaptive_avg_pool2d(input, output_size)


def adaptive_max_pool2d(input: Variable, output_size: Union[int, tuple[int, int]]) -> Variable:
    """Apply 2D adaptive max pooling.

    The output spatial dimensions match *output_size* regardless of input size.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, C, H, W)``.
    output_size : int or tuple[int, int]
        Target spatial dimensions ``(H_out, W_out)``.

    Returns
    -------
    Variable
        Tensor of shape ``(N, C, *output_size)``.

    Example
    -------
    >>> y = F.adaptive_max_pool2d(x, (7, 7))
    """
    return _nn.functional_adaptive_max_pool2d(input, output_size)


def batch_norm(input: Variable, num_features: int, training: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Variable:
    """Apply batch normalization over a mini-batch of inputs.

    Creates a transient BatchNorm layer with fresh running statistics.
    For stateful batch normalization with persistent running stats, use
    ``tz.nn.BatchNorm2d`` as a module instead.

    Parameters
    ----------
    input : Variable
        Input of shape ``(N, C, ...)`` where *C* is the channel dimension.
    num_features : int
        Number of channels *C*.
    training : bool, optional
        Use mini-batch statistics when ``True``, running stats when
        ``False``.  Default: ``True``.
    momentum : float, optional
        Value used for running mean/var update.  Default: ``0.1``.
    eps : float, optional
        Added to denominator for numerical stability.  Default: ``1e-5``.

    Returns
    -------
    Variable
        Normalized output of the same shape as *input*.

    Example
    -------
    >>> y = F.batch_norm(x, num_features=64, training=True)
    """
    return _nn.functional_batch_norm(input, num_features, training, momentum, eps)


def layer_norm(input: Variable, normalized_shape: Sequence[int], eps: float = 1e-5) -> Variable:
    """Apply layer normalization over the last *D* dimensions.

    Creates a transient LayerNorm layer with learnable affine parameters.
    For persistent parameters, use ``tz.nn.LayerNorm`` as a module.

    Parameters
    ----------
    input : Variable
        Input tensor.
    normalized_shape : list[int]
        Shape of the dimensions to normalize over (trailing dims of *input*).
    eps : float, optional
        Added to denominator for numerical stability.  Default: ``1e-5``.

    Returns
    -------
    Variable
        Normalized output of the same shape as *input*.

    Example
    -------
    >>> y = F.layer_norm(x, [hidden_size])
    """
    return _nn.functional_layer_norm(input, normalized_shape, eps)


def group_norm(input: Variable, num_groups: int, num_channels: int, eps: float = 1e-5) -> Variable:
    """Apply group normalization.

    Parameters
    ----------
    input : Variable
        Input of shape ``(N, C, ...)`` where *C* must be divisible by
        *num_groups*.
    num_groups : int
        Number of groups to divide channels into.
    num_channels : int
        Number of channels *C* in the input.
    eps : float, optional
        Added to denominator for numerical stability.  Default: ``1e-5``.

    Returns
    -------
    Variable
        Normalized output of the same shape as *input*.

    Example
    -------
    >>> y = F.group_norm(x, num_groups=32, num_channels=256)
    """
    return _nn.functional_group_norm(input, num_groups, num_channels, eps)


def instance_norm(input: Variable, num_features: int, eps: float = 1e-5, affine: bool = False) -> Variable:
    """Apply instance normalization.

    Normalizes each sample independently across spatial dimensions.

    Parameters
    ----------
    input : Variable
        Input of shape ``(N, C, ...)``.
    num_features : int
        Number of channels *C* in the input.
    eps : float, optional
        Added to denominator for numerical stability.  Default: ``1e-5``.
    affine : bool, optional
        If ``True``, apply learnable affine parameters.  Default: ``False``.

    Returns
    -------
    Variable
        Normalized output of the same shape as *input*.

    Example
    -------
    >>> y = F.instance_norm(x, num_features=64)
    """
    return _nn.functional_instance_norm(input, num_features, eps, affine)


def rms_norm(input: Variable, normalized_shape: int, eps: float = 1e-6) -> Variable:
    """Apply Root Mean Square layer normalization.

    Parameters
    ----------
    input : Variable
        Input tensor.
    normalized_shape : int
        Size of the last dimension to normalize over.
    eps : float, optional
        Added to denominator for numerical stability.  Default: ``1e-6``.

    Returns
    -------
    Variable
        Normalized output of the same shape as *input*.

    Example
    -------
    >>> y = F.rms_norm(x, hidden_size)
    """
    return _nn.functional_rms_norm(input, normalized_shape, eps)


def interpolate(input: Variable, size: Sequence[int], mode: str = 'bilinear', align_corners: bool = False) -> Variable:
    """Resize the input using interpolation.

    Parameters
    ----------
    input : Variable
        Input tensor (typically 4-D for images).
    size : list[int]
        Target spatial size.
    mode : str, optional
        Interpolation mode: ``'nearest'``, ``'bilinear'``, ``'bicubic'``,
        ``'trilinear'``.  Default: ``'bilinear'``.
    align_corners : bool, optional
        If ``True``, align corner pixels of input and output.
        Default: ``False``.

    Returns
    -------
    Variable
        Resized tensor.

    Example
    -------
    >>> y = F.interpolate(x, size=[224, 224], mode='bilinear')
    """
    return _nn.functional_interpolate(input, size, mode, align_corners)


def grid_sample(
    input: Variable,
    grid: Variable,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Variable:
    """Sample from *input* using a spatial grid of coordinates.

    Implements the spatial transformer sampling operation.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, C, H_in, W_in)``.
    grid : Variable
        Sampling grid of shape ``(N, H_out, W_out, 2)`` with values
        in ``[-1, 1]`` (normalised coordinates).
    mode : str, optional
        Interpolation mode: ``'bilinear'`` or ``'nearest'``.
        Default: ``'bilinear'``.
    padding_mode : str, optional
        Padding mode for out-of-bound coordinates: ``'zeros'``,
        ``'border'``, or ``'reflection'``.  Default: ``'zeros'``.
    align_corners : bool, optional
        If ``True``, corner pixels are aligned.  Default: ``False``.

    Returns
    -------
    Variable
        Sampled output of shape ``(N, C, H_out, W_out)``.

    Example
    -------
    >>> theta = tz.eye(2, 3).unsqueeze(0)
    >>> grid = F.affine_grid(theta, x.shape)
    >>> y = F.grid_sample(x, grid)
    """
    return _nn.functional_grid_sample(input, grid, mode, padding_mode, align_corners)


def affine_grid(theta: Variable, size: Sequence[int], align_corners: bool = False) -> Variable:
    """Generate a 2-D affine sampling grid from a batch of 2x3 matrices.

    Parameters
    ----------
    theta : Variable
        Affine matrices of shape ``(N, 2, 3)``.
    size : Sequence[int]
        Output spatial size ``(N, C, H, W)``.
    align_corners : bool, optional
        Default: ``False``.

    Returns
    -------
    Variable
        Grid tensor of shape ``(N, H, W, 2)``.
    """
    return _nn.functional_affine_grid(theta, size, align_corners)


def checkpoint(fn, *args, **kwargs):
    """Gradient checkpointing: trade compute for memory.

    Runs *fn* without saving intermediate activations.  During the
    backward pass the forward is re-executed to recompute them.

    Parameters
    ----------
    fn : callable
        Function to checkpoint (typically a module's ``forward``).
    *args
        Positional arguments forwarded to *fn*.
    **kwargs
        Keyword arguments forwarded to *fn*.

    Returns
    -------
    Variable
        Output of ``fn(*args, **kwargs)``.

    Example
    -------
    >>> y = F.checkpoint(block, x)
    """
    return _nn.functional_checkpoint(fn, *args, **kwargs)



def embedding(input: Variable, weight: Variable, padding_idx: int = -1) -> Variable:
    """Look up embeddings in a fixed dictionary and size.

    Parameters
    ----------
    input : Variable
        Tensor of indices (int64) of shape ``(*)``.
    weight : Variable
        Embedding matrix of shape ``(num_embeddings, embedding_dim)``.
    padding_idx : int, optional
        If non-negative, pads the output with zeros at this index.
        Default: ``-1`` (no padding).

    Returns
    -------
    Variable
        Tensor of shape ``(*, embedding_dim)``.

    Example
    -------
    >>> emb = F.embedding(token_ids, embed_weight)
    """
    return _nn.functional_embedding(input, weight, padding_idx)


def one_hot(input, num_classes: int = -1):
    """Create a one-hot encoded tensor from class indices.

    Parameters
    ----------
    input : Tensor
        Tensor of class indices (integer dtype).
    num_classes : int, optional
        Total number of classes. If ``-1``, inferred as ``max(input) + 1``.
        Default: ``-1``.

    Returns
    -------
    Tensor
        One-hot tensor of shape ``(*input.shape, num_classes)``.

    Example
    -------
    >>> labels = tz.tensor([0, 2, 1, 3], dtype=tz.dtype.int64)
    >>> oh = F.one_hot(labels, num_classes=4)
    """
    return _core.one_hot(input, num_classes)


def unfold(input, kernel_size: int, dilation: int = 1, padding: int = 0, stride: int = 1):
    """Extract sliding local blocks from a batched input tensor (im2col).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(N, C, H, W)``.
    kernel_size : int
        Size of the sliding blocks.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    padding : int, optional
        Zero-padding added to both sides. Default: ``0``.
    stride : int, optional
        Stride of the sliding blocks. Default: ``1``.

    Returns
    -------
    Tensor
        Unfolded tensor of shape ``(N, C * kernel_size * kernel_size, L)``.

    Example
    -------
    >>> blocks = F.unfold(x, kernel_size=3, padding=1)
    """
    return _core.vision.unfold(input, kernel_size, stride, padding, dilation)


def fold(input, output_size, kernel_size: int, dilation: int = 1, padding: int = 0, stride: int = 1):
    """Combine an array of sliding local blocks into a large tensor (col2im).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(N, C * kernel_size * kernel_size, L)``.
    output_size : tuple of int
        Spatial dimensions ``(H, W)`` of the output.
    kernel_size : int
        Size of the sliding blocks.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    padding : int, optional
        Zero-padding that was used in :func:`unfold`. Default: ``0``.
    stride : int, optional
        Stride of the sliding blocks. Default: ``1``.

    Returns
    -------
    Tensor
        Folded tensor of shape ``(N, C, H, W)``.

    Example
    -------
    >>> img = F.fold(blocks, output_size=(32, 32), kernel_size=3, padding=1)
    """
    return _core.vision.fold(input, list(output_size), kernel_size, stride, padding, dilation)


def binary_cross_entropy_with_logits(input: Variable, target: Variable, reduction: str = "mean") -> Variable:
    """Compute binary cross-entropy loss from logits.

    Combines a sigmoid layer and binary cross-entropy in a single function
    for numerical stability.

    Parameters
    ----------
    input : Variable
        Raw logits (unnormalized scores).
    target : Variable
        Binary ground truth of the same shape as *input*.
    reduction : str, optional
        Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        Default: ``'mean'``.

    Returns
    -------
    Variable
        Loss value.

    Example
    -------
    >>> loss = F.binary_cross_entropy_with_logits(logits, labels)
    """
    return _nn.functional_binary_cross_entropy_with_logits(input, target, _reduction(reduction))


# ---------------------------------------------------------------------------
# Gradient clipping utilities
# ---------------------------------------------------------------------------

def clip_grad_norm_(parameters: Iterable[Variable], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip the gradient norm of a set of parameters.

    Gradients are modified in-place.  The total norm is computed over all
    gradients together, as if they were concatenated into a single vector.

    Parameters
    ----------
    parameters : iterable of Variable
        Parameters whose gradients will be clipped.
    max_norm : float
        Maximum allowed norm value.
    norm_type : float, optional
        Type of norm (e.g., 2.0 for L2).  Default: ``2.0``.

    Returns
    -------
    float
        Total norm of the gradients (before clipping).

    Example
    -------
    >>> total_norm = F.clip_grad_norm_(model.parameters(), max_norm=1.0)
    """
    return _nn.clip_grad_norm_(parameters, max_norm, norm_type)


def clip_grad_value_(parameters: Iterable[Variable], clip_value: float) -> None:
    """Clip the gradient values of a set of parameters to a specified range.

    Gradients are modified in-place.

    Parameters
    ----------
    parameters : iterable of Variable
        Parameters whose gradients will be clipped.
    clip_value : float
        Maximum allowed absolute value for each gradient element.

    Returns
    -------
    None

    Example
    -------
    >>> F.clip_grad_value_(model.parameters(), clip_value=0.5)
    """
    return _nn.clip_grad_value_(parameters, clip_value)


def cosine_similarity(x1: Variable, x2: Variable, dim: int = 1, eps: float = 1e-8) -> Variable:
    """Compute cosine similarity between two tensors along a dimension.

    Parameters
    ----------
    x1, x2 : Variable
        Input tensors.
    dim : int, optional
        Dimension along which to compute similarity. Default: ``1``.
    eps : float, optional
        Small value to avoid division by zero. Default: ``1e-8``.

    Returns
    -------
    Variable
        Cosine similarity values.
    """
    return _nn.functional_cosine_similarity(x1, x2, dim, eps)


def pairwise_distance(x1: Variable, x2: Variable, p: float = 2.0) -> Variable:
    """Compute pairwise p-norm distance between corresponding rows.

    Parameters
    ----------
    x1, x2 : Variable
        Input tensors of shape ``(N, D)``.
    p : float, optional
        Norm degree. Default: ``2.0`` (Euclidean).

    Returns
    -------
    Variable
        Distance tensor of shape ``(N,)``.
    """
    return _core.pairwise_distance(x1, x2, p)


def pdist(input: Variable, p: float = 2.0) -> Variable:
    """Compute all-pairs p-norm distances between rows of a matrix.

    Returns the condensed distance matrix (upper triangle, row-major).
    Output size is ``N*(N-1)/2`` for input with ``N`` rows.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, D)``.
    p : float, optional
        Norm degree. Default: ``2.0`` (Euclidean).

    Returns
    -------
    Variable
        Condensed distance tensor of shape ``(N*(N-1)/2,)``.
    """
    return _core.pdist(input, p)


def channel_shuffle(input: Variable, groups: int) -> Variable:
    """Rearrange channels by dividing into groups and transposing.

    Used in ShuffleNet architectures.

    Parameters
    ----------
    input : Variable
        4D input tensor of shape ``(N, C, H, W)``.
    groups : int
        Number of groups (must divide ``C`` evenly).

    Returns
    -------
    Variable
        Tensor with shuffled channels, same shape as input.
    """
    return _core.channel_shuffle(input, groups)


def conv2d(input: Variable, weight: Variable, bias: Optional[Variable] = None,
           stride: Union[int, tuple[int, int]] = 1,
           padding: Union[int, tuple[int, int]] = 0,
           dilation: Union[int, tuple[int, int]] = 1,
           groups: int = 1) -> Variable:
    """Apply a 2D convolution over an input signal.

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, C_in, H, W)``.
    weight : Variable
        Filters of shape ``(C_out, C_in/groups, kH, kW)``.
    bias : Variable or None, optional
        Bias of shape ``(C_out,)``. Default: ``None``.
    stride : int or tuple, optional
        Stride of the convolution. Default: ``1``.
    padding : int or tuple, optional
        Zero-padding added to both sides. Default: ``0``.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.

    Returns
    -------
    Variable
        Output tensor.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return _nn.functional_conv2d(input, weight, bias, stride, padding, dilation, groups)


def deformable_conv2d(
    input: Variable, offset: Variable, weight: Variable,
    bias: Optional[Variable] = None,
    mask: Optional[Variable] = None,
    stride: Union[int, tuple[int, int]] = 1,
    padding: Union[int, tuple[int, int]] = 0,
    dilation: Union[int, tuple[int, int]] = 1,
    groups: int = 1,
    offset_groups: int = 1,
) -> Variable:
    """Apply deformable 2D convolution (DCNv2).

    Parameters
    ----------
    input : Variable
        Input tensor of shape ``(N, C_in, H, W)``.
    offset : Variable
        Offset tensor of shape ``(N, offset_groups * 2 * kH * kW, H_out, W_out)``.
    weight : Variable
        Filters of shape ``(C_out, C_in/groups, kH, kW)``.
    bias : Variable or None, optional
        Bias of shape ``(C_out,)``. Default: ``None``.
    mask : Variable or None, optional
        Modulation mask of shape ``(N, offset_groups * kH * kW, H_out, W_out)``.
        Default: ``None`` (DCNv1 behavior).
    stride : int or tuple, optional
        Stride of the convolution. Default: ``1``.
    padding : int or tuple, optional
        Zero-padding. Default: ``0``.
    dilation : int or tuple, optional
        Dilation. Default: ``1``.
    groups : int, optional
        Number of channel groups. Default: ``1``.
    offset_groups : int, optional
        Number of offset groups. Default: ``1``.

    Returns
    -------
    Variable
        Output tensor.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return _nn.functional_deformable_conv2d(
        input, offset, weight, bias, mask,
        stride, padding, dilation, groups, offset_groups)


def scaled_dot_product_attention(
    query: Variable, key: Variable, value: Variable,
    attn_mask: Optional[Variable] = None,
    dropout_p: float = 0.0, is_causal: bool = False
) -> Variable:
    """Compute scaled dot-product attention.

    Computes ``softmax(Q @ K^T / sqrt(d_k) + mask) @ V``.

    Parameters
    ----------
    query : Variable
        Query tensor ``[B, H, L, E]``.
    key : Variable
        Key tensor ``[B, H, S, E]``.
    value : Variable
        Value tensor ``[B, H, S, Ev]``.
    attn_mask : Variable or None, optional
        Additive attention mask. Default: ``None``.
    dropout_p : float, optional
        Dropout probability on attention weights. Default: ``0.0``.
    is_causal : bool, optional
        Apply causal (lower-triangular) mask. Default: ``False``.

    Returns
    -------
    Variable
        Output tensor ``[B, H, L, Ev]``.
    """
    return _nn.functional_scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal)


def multi_head_attention_forward(
    query: 'Tensor', key: 'Tensor', value: 'Tensor',
    num_heads: int,
    in_proj_weight: 'Tensor', in_proj_bias: 'Tensor',
    out_proj_weight: 'Tensor', out_proj_bias: 'Tensor',
    attn_mask: 'Optional[Tensor]' = None,
    dropout_p: float = 0.0, training: bool = False,
    need_weights: bool = False
) -> 'tuple[Tensor, Tensor]':
    """Functional multi-head attention forward pass.

    Projects query/key/value through ``in_proj_weight``/``in_proj_bias``,
    splits into ``num_heads`` attention heads, applies scaled dot-product
    attention, merges heads, and projects the output.

    Parameters
    ----------
    query : Tensor
        Query tensor ``[B, Sq, E]``.
    key : Tensor
        Key tensor ``[B, Sk, E]``.
    value : Tensor
        Value tensor ``[B, Sk, E]``.
    num_heads : int
        Number of attention heads.
    in_proj_weight : Tensor
        Combined QKV projection weight ``[3*E, E]``.
    in_proj_bias : Tensor
        Combined QKV projection bias ``[3*E]``.
    out_proj_weight : Tensor
        Output projection weight ``[E, E]``.
    out_proj_bias : Tensor
        Output projection bias ``[E]``.
    attn_mask : Tensor or None, optional
        Additive attention mask. Default: ``None``.
    dropout_p : float, optional
        Dropout probability on attention weights. Default: ``0.0``.
    training : bool, optional
        If ``True``, apply dropout. Default: ``False``.
    need_weights : bool, optional
        If ``True``, return attention weights. Default: ``False``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(output, attn_weights)`` where ``output`` is ``[B, Sq, E]`` and
        ``attn_weights`` is ``[B, H, Sq, Sk]`` (or empty if not requested).
    """
    return _nn.functional_multi_head_attention_forward(
        query, key, value, num_heads,
        in_proj_weight, in_proj_bias,
        out_proj_weight, out_proj_bias,
        attn_mask, dropout_p, training, need_weights)


def normalize(input: Variable, p: float = 2.0, dim: int = 1,
              eps: float = 1e-12) -> Variable:
    """Apply L_p normalization along a dimension.

    Divides the input by its L_p norm along the given dimension.

    Parameters
    ----------
    input : Variable
        Input tensor.
    p : float, optional
        Exponent for the norm. Default: ``2.0`` (L2).
    dim : int, optional
        Dimension to reduce. Default: ``1``.
    eps : float, optional
        Small value to avoid division by zero. Default: ``1e-12``.

    Returns
    -------
    Variable
        Normalized tensor of same shape as *input*.
    """
    return _nn.functional_normalize(input, p, dim, eps)


def pad(input: Variable, pad: Sequence[int], mode: str = "constant",
        value: float = 0.0) -> Variable:
    """Pad a tensor.

    Parameters
    ----------
    input : Variable
        Input tensor.
    pad : sequence of int
        Padding sizes in reverse-dimension order: ``(left, right)`` for 1D,
        ``(left, right, top, bottom)`` for 2D, etc.
    mode : str, optional
        Padding mode: ``'constant'``. Default: ``'constant'``.
    value : float, optional
        Fill value for constant padding. Default: ``0.0``.

    Returns
    -------
    Variable
        Padded tensor.
    """
    return _nn.functional_pad(input, list(pad), mode, value)


# ---------------------------------------------------------------------------
# Composed tensor operations
# ---------------------------------------------------------------------------

def diff(input, n=1, dim=-1):
    """Compute the n-th finite difference along the given dimension.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    n : int, optional
        Number of times to recursively compute differences (default: 1).
    dim : int, optional
        Dimension along which to compute differences (default: -1).

    Returns
    -------
    Tensor
        Tensor with size reduced by *n* along *dim*.
    """
    return _core.diff(input, n, dim)


def logaddexp(a, b):
    """Numerically stable ``log(exp(a) + exp(b))``.

    Parameters
    ----------
    a, b : Tensor
        Input tensors (must be broadcastable).

    Returns
    -------
    Tensor
    """
    return _core.logaddexp(a, b)


def logaddexp2(a, b):
    """Numerically stable ``log2(2**a + 2**b)``.

    Parameters
    ----------
    a, b : Tensor
        Input tensors (must be broadcastable).

    Returns
    -------
    Tensor
    """
    return _core.logaddexp2(a, b)


def xlogy(x, y):
    """Compute ``x * log(y)`` with the convention ``0 * log(y) = 0``.

    Parameters
    ----------
    x, y : Tensor
        Input tensors (must be broadcastable).

    Returns
    -------
    Tensor
    """
    return _core.xlogy(x, y)


def cov(input, correction=1, fweights=None, aweights=None):
    """Compute the sample covariance matrix.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(N, M)`` where *N* is the number of variables
        and *M* is the number of observations. A 1-D input is treated as a
        single variable with *N* observations.
    correction : int, optional
        Bessel's correction (default: ``1`` for unbiased estimate).
    fweights : Tensor or None, optional
        Integer frequency weights of length *M*.
    aweights : Tensor or None, optional
        Float analytic (importance) weights of length *M*.

    Returns
    -------
    Tensor
        Covariance matrix of shape ``(N, N)``.
    """
    from .. import tenzor_core as _c
    fw = fweights if fweights is not None else _c.Tensor()
    aw = aweights if aweights is not None else _c.Tensor()
    return _core.cov(input, correction, fw, aw)


def corrcoef(input):
    """Compute the Pearson correlation coefficient matrix.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(N, M)``.

    Returns
    -------
    Tensor
        Correlation coefficient matrix of shape ``(N, N)`` with ones on
        the diagonal.
    """
    return _core.corrcoef(input)


def tensordot(a, b, dims=2):
    """Generalized tensor contraction (like ``numpy.tensordot``).

    Parameters
    ----------
    a, b : Tensor
        Input tensors.
    dims : int or tuple of (list[int], list[int])
        If an int, contract the last *dims* dimensions of *a* with the first
        *dims* dimensions of *b*. If a tuple of two lists, contract the
        specified dimensions.

    Returns
    -------
    Tensor
    """
    if isinstance(dims, int):
        return _core.tensordot(a, b, dims)
    else:
        dims_a, dims_b = dims
        return _core.tensordot(a, b, list(dims_a), list(dims_b))


def dct(input, type=2, n=None, dim=-1, norm="backward"):
    """Compute the Discrete Cosine Transform.

    Parameters
    ----------
    input : Tensor
        Input signal.
    type : int
        DCT type (1, 2, 3, or 4). Default: 2.
    n : int or None
        Signal length. Default: input size along dim.
    dim : int
        Dimension along which to compute. Default: -1.
    norm : str
        Normalization mode: "backward", "forward", or "ortho". Default: "backward".

    Returns
    -------
    Tensor
        DCT coefficients.
    """
    return _core.fft.dct(input, type, n, dim, norm)


def idct(input, type=2, n=None, dim=-1, norm="backward"):
    """Compute the Inverse Discrete Cosine Transform.

    Parameters
    ----------
    input : Tensor
        DCT coefficients.
    type : int
        DCT type (1, 2, 3, or 4). Default: 2.
    n : int or None
        Signal length. Default: input size along dim.
    dim : int
        Dimension along which to compute. Default: -1.
    norm : str
        Normalization mode: "backward", "forward", or "ortho". Default: "backward".

    Returns
    -------
    Tensor
        Reconstructed signal.
    """
    return _core.fft.idct(input, type, n, dim, norm)


def mel_scale(spectrogram, n_mels=128, f_min=0.0, f_max=0.0, sample_rate=16000):
    """Apply a mel-frequency filterbank to a spectrogram.

    Converts a linear-frequency spectrogram into a mel-frequency spectrogram
    using triangular mel-spaced filter banks (HTK formula).

    Parameters
    ----------
    spectrogram : Tensor
        Input tensor of shape ``(..., n_freqs, time_frames)``.
    n_mels : int
        Number of mel bands (default: 128).
    f_min : float
        Minimum frequency in Hz (default: 0.0).
    f_max : float
        Maximum frequency in Hz (default: 0.0 means ``sample_rate / 2``).
    sample_rate : int
        Audio sample rate in Hz (default: 16000).

    Returns
    -------
    Tensor
        Mel spectrogram of shape ``(..., n_mels, time_frames)``.
    """
    return _core.fft.mel_scale(spectrogram, n_mels, f_min, f_max, sample_rate)


def mfcc(waveform, sample_rate=16000, n_mfcc=40, n_mels=128,
         n_fft=400, hop_length=160, f_min=0.0, f_max=0.0):
    """Compute Mel-Frequency Cepstral Coefficients (MFCC).

    Computes MFCCs from a raw waveform by computing a power spectrogram,
    applying mel-scale filterbank, taking log, and applying DCT.

    Parameters
    ----------
    waveform : Tensor
        Input waveform tensor of shape ``(..., signal_length)``.
    sample_rate : int
        Audio sample rate in Hz (default: 16000).
    n_mfcc : int
        Number of MFCC coefficients to return (default: 40).
    n_mels : int
        Number of mel bands (default: 128).
    n_fft : int
        FFT window size (default: 400).
    hop_length : int
        Hop length for STFT (default: 160).
    f_min : float
        Minimum frequency in Hz (default: 0.0).
    f_max : float
        Maximum frequency in Hz (default: 0.0 means ``sample_rate / 2``).

    Returns
    -------
    Tensor
        MFCC tensor of shape ``(..., n_mfcc, time_frames)``.
    """
    return _core.fft.mfcc(waveform, sample_rate, n_mfcc, n_mels,
                           n_fft, hop_length, f_min, f_max)


def lobpcg(A, X0, k, B=None, max_iter=100, tol=1e-6):
    """LOBPCG: find k smallest eigenvalues/vectors of a large sparse symmetric matrix.

    Locally Optimal Block Preconditioned Conjugate Gradient method.

    Parameters
    ----------
    A : Tensor
        Symmetric matrix of shape ``(N, N)``.
    X0 : Tensor
        Initial guess for eigenvectors of shape ``(N, k)``.
    k : int
        Number of eigenvalues to find.
    B : Tensor, optional
        Preconditioner matrix (defaults to identity).
    max_iter : int
        Maximum iterations (default: 100).
    tol : float
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    tuple of (Tensor, Tensor)
        (eigenvalues of shape ``(k,)``, eigenvectors of shape ``(N, k)``).
    """
    from .. import tenzor_core as _c
    b_tensor = B if B is not None else _c.Tensor()
    return _core.linalg.lobpcg(A, X0, k, b_tensor, max_iter, tol)


# ============================================================================
# Audit item E.8 — functions declared in functional.pyi that previously
# had no runtime implementation.  Each is a thin wrapper around the
# corresponding nn::Module class so the autograd path is preserved.
# ============================================================================

def relu6(input: Variable) -> Variable:
    """Apply ReLU6: ``min(max(0, x), 6)`` element-wise."""
    return _nn.ReLU6()(input)


def prelu(input: Variable, weight: Variable) -> Variable:
    """Parametric ReLU: ``max(0, x) + a * min(0, x)`` with learnable ``a``.

    Uses the supplied ``weight`` tensor as the per-channel slope.
    """
    # PReLU expects the slope tensor to be a Parameter; bind into a fresh
    # nn.PReLU instance and overwrite its weight before invoking forward.
    num_parameters = int(weight.tensor().numel())
    layer = _nn.PReLU(num_parameters=num_parameters)
    # Replace the layer's weight with the caller-supplied one so gradient
    # flow lands on the caller's Variable rather than the throwaway layer.
    layer.weight.tensor().copy_(weight.tensor())
    return layer(input)


def conv1d(input: Variable, weight: Variable, bias=None,
           stride: int = 1, padding: int = 0, dilation: int = 1,
           groups: int = 1) -> Variable:
    """Functional 1-D convolution (audit E.8 / Q.12).

    Routes through the C++ autograd-aware free function so gradients
    flow back into the caller's ``weight``/``bias`` Variables instead
    of a throwaway ``nn.Conv1d`` layer's internal parameters.
    """
    return _nn.functional_conv1d(input, weight, bias,
                                 int(stride), int(padding), int(dilation),
                                 int(groups))


def conv3d(input: Variable, weight: Variable, bias=None,
           stride=1, padding=0, dilation=1, groups: int = 1) -> Variable:
    """Functional 3-D convolution (audit E.8 / Q.12).

    Routes through the C++ autograd-aware free function so gradients
    flow back into the caller's ``weight``/``bias`` Variables instead
    of a throwaway ``nn.Conv3d`` layer's internal parameters.
    """
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    return _nn.functional_conv3d(input, weight, bias,
                                 tuple(int(v) for v in stride),
                                 tuple(int(v) for v in padding),
                                 tuple(int(v) for v in dilation),
                                 int(groups))


def conv_transpose2d(input: Variable, weight: Variable, bias=None,
                     stride=1, padding=0, output_padding=0,
                     groups: int = 1, dilation: int = 1) -> Variable:
    """Functional 2-D transposed convolution (audit E.8 / Q.12).

    Routes through the C++ autograd-aware free function so gradients
    flow back into the caller's ``weight``/``bias`` Variables instead
    of a throwaway ``nn.ConvTranspose2d`` layer's internal parameters.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return _nn.functional_conv_transpose2d(
        input, weight, bias,
        tuple(int(v) for v in stride),
        tuple(int(v) for v in padding),
        tuple(int(v) for v in output_padding),
        int(groups),
        tuple(int(v) for v in dilation))


def max_pool1d(input: Variable, kernel_size: int,
               stride=None, padding: int = 0) -> Variable:
    """Functional 1-D max pooling (audit E.8)."""
    if stride is None or stride == -1:
        stride = kernel_size
    return _nn.MaxPool1d(kernel_size, stride=stride, padding=padding)(input)


def max_pool3d(input: Variable, kernel_size,
               stride=None, padding=0) -> Variable:
    """Functional 3-D max pooling (audit E.8)."""
    if stride is None or stride == -1:
        stride = kernel_size
    return _nn.MaxPool3d(kernel_size, stride=stride, padding=padding)(input)


def avg_pool1d(input: Variable, kernel_size: int,
               stride=None, padding: int = 0,
               count_include_pad: bool = True) -> Variable:
    """Functional 1-D average pooling (audit E.8)."""
    if stride is None or stride == -1:
        stride = kernel_size
    return _nn.AvgPool1d(kernel_size, stride=stride, padding=padding,
                          count_include_pad=count_include_pad)(input)


def avg_pool3d(input: Variable, kernel_size,
               stride=None, padding=0,
               count_include_pad: bool = True) -> Variable:
    """Functional 3-D average pooling (audit E.8)."""
    if stride is None or stride == -1:
        stride = kernel_size
    return _nn.AvgPool3d(kernel_size, stride=stride, padding=padding,
                          count_include_pad=count_include_pad)(input)


def adaptive_avg_pool1d(input: Variable, output_size: int) -> Variable:
    """Functional adaptive 1-D average pooling (audit E.8)."""
    return _nn.AdaptiveAvgPool1d(output_size)(input)


def adaptive_avg_pool3d(input: Variable, output_size) -> Variable:
    """Functional adaptive 3-D average pooling (audit E.8)."""
    return _nn.AdaptiveAvgPool3d(output_size)(input)


def adaptive_max_pool1d(input: Variable, output_size: int) -> Variable:
    """Functional adaptive 1-D max pooling (audit E.8)."""
    return _nn.AdaptiveMaxPool1d(output_size)(input)


def adaptive_max_pool3d(input: Variable, output_size) -> Variable:
    """Functional adaptive 3-D max pooling (audit E.8)."""
    return _nn.AdaptiveMaxPool3d(output_size)(input)


def dropout2d(input: Variable, p: float = 0.5,
              training: bool = True) -> Variable:
    """Functional 2D dropout — zeros entire channels with probability p
    during training, scales the survivors by 1/(1-p) (audit E.8).

    Channels are the second dimension of an (N, C, H, W) input.
    """
    if not training or p == 0.0:
        return input
    import tenzor as tz
    shape = list(input.tensor().shape())
    if len(shape) < 4:
        raise ValueError(
            "dropout2d expects a 4D input (N, C, H, W); got shape " + str(shape))
    # Bernoulli mask over channels: shape (N, C, 1, 1) so it broadcasts.
    n, c = shape[0], shape[1]
    mask = (tz.rand([n, c, 1, 1], dtype=input.tensor().dtype()) >= p).to(input.tensor().dtype())
    scale = 1.0 / (1.0 - p)
    # Q.13 / J.3: multiply at Variable level so autograd records the
    # dropout mask as a constant cotangent multiplier (grad_fn survives).
    scaled_mask = Variable(mask * scale, requires_grad=False)
    return tz.mul(input, scaled_mask)


def alpha_dropout(input: Variable, p: float = 0.5,
                  training: bool = True) -> Variable:
    """Alpha Dropout: maintains zero mean / unit variance under SELU
    activations (audit E.8).

    Follows the formulation from torch.nn.functional.alpha_dropout —
    elements survive with probability 1-p; killed elements are mapped to
    a fixed constant `α'`, then the result is affinely rescaled so the
    SELU-equivariant statistics are preserved.
    """
    if not training or p == 0.0:
        return input
    import tenzor as tz
    alpha = -1.7580993408473766  # SELU's negative saturation value
    # Bernoulli keep-mask.
    shape = list(input.tensor().shape())
    keep = (tz.rand(shape, dtype=input.tensor().dtype()) >= p).to(input.tensor().dtype())
    # Affine constants: a * (mask * x + (1 - mask) * alpha) + b.
    a = ((1 - p) * (1 + p * alpha * alpha)) ** -0.5
    b = -a * alpha * p
    x_dtype = input.tensor().dtype()
    x_shape = list(input.tensor().shape())
    # The closed-form is:
    #   out = (a * keep) * x  +  (a * (1 - keep) * alpha + b)
    # The first term is the only one that depends on the input. Route the
    # multiplication through tz.mul so autograd records the dropout mask
    # as a constant cotangent multiplier (Q.13 / J.3).
    one_minus_keep = tz.full(x_shape, 1.0, dtype=x_dtype) - keep
    mul_coeff = Variable(a * keep, requires_grad=False)
    bias_term = Variable(a * (one_minus_keep * alpha) + b, requires_grad=False)
    return tz.add(tz.mul(input, mul_coeff), bias_term)


def focal_loss(input: Variable, target: Variable, alpha: float = 0.25,
               gamma: float = 2.0, reduction: str = "mean") -> Variable:
    """Focal loss for binary classification (Lin et al. 2017)
    (audit E.8 / J.3).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Assumes ``input`` contains logits and ``target`` contains 0/1
    labels (or class probabilities) matching ``input.shape``.

    Audit J.3: implemented as a pure Variable-level composition. The
    previous implementation extracted raw Tensors from ``input`` and
    wrapped the result back into a Variable, severing the autograd graph
    and silently flowing zero gradients into the model.
    """
    import tenzor as tz
    # Ensure target is a Variable so * / - / + dispatch the
    # autograd-aware overloads (and accept a plain Tensor for convenience).
    if not isinstance(target, tz.Variable):
        target = tz.Variable(target, requires_grad=False)

    p = sigmoid(input)                       # autograd-aware sigmoid
    # p_t = p     if target == 1
    #     = 1-p   if target == 0
    # alpha_t = alpha    if target == 1
    #         = 1-alpha  if target == 0
    one_minus_target = 1.0 - target
    p_t = p * target + (1.0 - p) * one_minus_target
    alpha_t = alpha * target + (1.0 - alpha) * one_minus_target

    # (1 - p_t)^gamma — Variable.__pow__ currently bypasses autograd, so
    # route through the autograd-aware tz.pow(Variable, float) overload.
    one_minus_pt = 1.0 - p_t
    focal_weight = alpha_t * tz.pow(one_minus_pt, float(gamma))
    log_pt = tz.log(p_t)                     # autograd-aware log
    loss = -(focal_weight * log_pt)

    if reduction == "mean":
        return tz.mean(loss)
    if reduction == "sum":
        return tz.sum(loss)
    return loss


def dice_loss(input: Variable, target: Variable, smooth: float = 1.0,
              reduction: str = "mean") -> Variable:
    """Sorensen-Dice loss for segmentation (audit E.8 / J.3).

    DL = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth).
    ``input`` is treated as logits; a sigmoid is applied internally so
    gradients flow back through the activation.

    Spatial dimensions (everything after the leading batch axis) are
    reduced when computing the per-sample intersection / union; the
    ``reduction`` argument then folds the resulting per-sample dice
    score.

    Audit J.3: implemented as a pure Variable-level composition. The
    previous implementation summed raw Tensors and re-wrapped the result,
    so ``loss.backward()`` produced zero gradients on ``input``.
    """
    import tenzor as tz
    if not isinstance(target, tz.Variable):
        target = tz.Variable(target, requires_grad=False)

    p = sigmoid(input)                       # autograd-aware sigmoid
    ndim = len(p.tensor().shape)
    # Sum over every axis after the batch dim so we get a per-sample
    # dice; for a 1-D / 0-D input fall back to a single global reduction.
    spatial_dims = list(range(1, ndim)) if ndim > 1 else []

    if spatial_dims:
        # tz.sum(Variable, dim=int) only takes a single axis; iterate
        # from the trailing axis to the first so earlier indices stay
        # valid (sum() is order-independent so the math is unchanged).
        intersection = p * target
        union_p = p
        union_t = target
        for d in reversed(spatial_dims):
            intersection = tz.sum(intersection, dim=d)
            union_p = tz.sum(union_p, dim=d)
            union_t = tz.sum(union_t, dim=d)
        union = union_p + union_t
    else:
        intersection = tz.sum(p * target)
        union = tz.sum(p) + tz.sum(target)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice

    if reduction == "mean":
        return tz.mean(loss)
    if reduction == "sum":
        return tz.sum(loss)
    return loss


def margin_ranking_loss(input1: Variable, input2: Variable, target: Variable,
                        margin: float = 0.0,
                        reduction: str = "mean") -> Variable:
    """Functional margin ranking loss (audit E.8).

    Wraps tenzor_core.nn.margin_ranking_loss.
    """
    return _nn.margin_ranking_loss(input1, input2, target, margin=margin, reduction=reduction)


__all__ = [
    # Activations
    "relu",
    "relu6",
    "prelu",
    # Conv (1D/3D/transposed — 2D already listed below)
    "conv1d",
    "conv3d",
    "conv_transpose2d",
    # Pooling (1D/3D + adaptive 1D/3D — 2D already listed below)
    "max_pool1d",
    "max_pool3d",
    "avg_pool1d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool3d",
    # Dropout variants
    "dropout2d",
    "alpha_dropout",
    # Losses
    "focal_loss",
    "dice_loss",
    "margin_ranking_loss",
    "leaky_relu",
    "elu",
    "gelu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "selu",
    "swish",
    "mish",
    "hardswish",
    "hardsigmoid",
    "glu",
    "softplus",
    # Losses
    "mse_loss",
    "l1_loss",
    "cross_entropy",
    "nll_loss",
    "bce_loss",
    "kl_div",
    "huber_loss",
    "smooth_l1_loss",
    # Functional operations
    "dropout",
    "linear",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "batch_norm",
    "layer_norm",
    "group_norm",
    "instance_norm",
    "rms_norm",
    "interpolate",
    "grid_sample",
    "affine_grid",
    "checkpoint",
    "embedding",
    "binary_cross_entropy_with_logits",
    # Additional functional operations
    "cosine_similarity",
    "conv2d",
    "scaled_dot_product_attention",
    "normalize",
    "pad",
    # Indexing / vision
    "one_hot",
    "unfold",
    "fold",
    # Gradient clipping
    "clip_grad_norm_",
    "clip_grad_value_",
    # Composed tensor operations
    "diff",
    "logaddexp",
    "logaddexp2",
    "xlogy",
    "tensordot",
    # Statistical operations
    "cov",
    "corrcoef",
    # Audio / spectral operations
    "mel_scale",
    "mfcc",
    # Sparse eigenvalue solver
    "lobpcg",
    # Audit item E.9 — these were implemented but missing from __all__.
    "channel_shuffle",
    "cosine_embedding_loss",
    "dct",
    "deformable_conv2d",
    "gaussian_nll_loss",
    "hinge_embedding_loss",
    "idct",
    "multi_head_attention_forward",
    "multi_label_soft_margin_loss",
    "multi_margin_loss",
    "pairwise_distance",
    "pdist",
    "poisson_nll_loss",
    "soft_margin_loss",
    "triplet_margin_loss",
]
