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

from .. import tenzor_core as _core

_nn = _core.nn


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(input):
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


def leaky_relu(input, negative_slope=0.01):
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


def elu(input, alpha=1.0):
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


def gelu(input):
    """Apply the Gaussian Error Linear Unit function element-wise.

    Parameters
    ----------
    input : Variable
        Input tensor.

    Returns
    -------
    Variable
        GELU activation applied element-wise.

    Example
    -------
    >>> y = F.gelu(x)
    """
    return _nn.gelu(input)


def sigmoid(input):
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


def tanh(input):
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


def softmax(input, dim=-1):
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


def log_softmax(input, dim=-1):
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


def selu(input):
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


def swish(input):
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


def mish(input):
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


def hardswish(input):
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


def hardsigmoid(input):
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


def glu(input, dim=-1):
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


def softplus(input, beta=1.0):
    """Apply the Softplus function element-wise.

    Computes ``(1/beta) * log(1 + exp(beta * x))``.

    Parameters
    ----------
    input : Variable
        Input tensor.
    beta : float, optional
        Scaling factor.  Default: ``1.0``.

    Returns
    -------
    Variable
        Softplus activation applied element-wise.

    Example
    -------
    >>> y = F.softplus(x, beta=1.0)
    """
    return _nn.softplus(input, beta)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(input, target, reduction="mean"):
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
    return _nn.mse_loss(input, target, reduction)


def l1_loss(input, target, reduction="mean"):
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
    return _nn.l1_loss(input, target, reduction)


def cross_entropy(input, target, reduction="mean"):
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

    Returns
    -------
    Variable
        Loss value (scalar if reduction is ``'mean'`` or ``'sum'``).

    Example
    -------
    >>> loss = F.cross_entropy(logits, labels)
    """
    return _nn.cross_entropy(input, target, reduction)


def nll_loss(input, target, reduction="mean"):
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

    Returns
    -------
    Variable
        Loss value (scalar if reduction is ``'mean'`` or ``'sum'``).

    Example
    -------
    >>> log_probs = F.log_softmax(logits, dim=-1)
    >>> loss = F.nll_loss(log_probs, labels)
    """
    return _nn.nll_loss(input, target, reduction)


def bce_loss(input, target, reduction="mean"):
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
    return _nn.bce_loss(input, target, reduction)


def kl_div(input, target, reduction="mean", log_target=False):
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


def huber_loss(input, target, delta=1.0, reduction="mean"):
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


def smooth_l1_loss(input, target, reduction="mean", beta=1.0):
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
    return _nn.smooth_l1_loss(input, target, reduction, beta)


# ---------------------------------------------------------------------------
# Functional operations (stateless wrappers)
# ---------------------------------------------------------------------------

def dropout(input, p=0.5, training=True):
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


def linear(input, weight, bias=None):
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


def max_pool2d(input, kernel_size, stride=None, padding=0):
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


def avg_pool2d(input, kernel_size, stride=None, padding=0):
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


def adaptive_avg_pool2d(input, output_size):
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


def adaptive_max_pool2d(input, output_size):
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


def batch_norm(input, num_features, training=True, momentum=0.1, eps=1e-5):
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


def layer_norm(input, normalized_shape, eps=1e-5):
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


def group_norm(input, num_groups, num_channels, eps=1e-5):
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


def instance_norm(input, num_features, eps=1e-5, affine=False):
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


def rms_norm(input, normalized_shape, eps=1e-6):
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


def interpolate(input, size, mode='bilinear', align_corners=False):
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


def embedding(input, weight, padding_idx=-1):
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


def binary_cross_entropy_with_logits(input, target, reduction="mean"):
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
    return _nn.functional_binary_cross_entropy_with_logits(input, target, reduction)


# ---------------------------------------------------------------------------
# Gradient clipping utilities
# ---------------------------------------------------------------------------

def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
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


def clip_grad_value_(parameters, clip_value):
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


__all__ = [
    # Activations
    "relu",
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
    "embedding",
    "binary_cross_entropy_with_logits",
    # Gradient clipping
    "clip_grad_norm_",
    "clip_grad_value_",
]
