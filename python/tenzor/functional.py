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

# Activation functions
relu = _nn.relu
leaky_relu = _nn.leaky_relu
elu = _nn.elu
gelu = _nn.gelu
sigmoid = _nn.sigmoid
tanh = _nn.tanh
softmax = _nn.softmax
log_softmax = _nn.log_softmax
selu = _nn.selu
swish = _nn.swish
mish = _nn.mish
hardswish = _nn.hardswish
hardsigmoid = _nn.hardsigmoid
glu = _nn.glu

# Loss functions
mse_loss = _nn.mse_loss
l1_loss = _nn.l1_loss
cross_entropy = _nn.cross_entropy
nll_loss = _nn.nll_loss
bce_loss = _nn.bce_loss

# Functional operations (stateless wrappers)
dropout = _nn.functional_dropout
linear = _nn.functional_linear
max_pool2d = _nn.functional_max_pool2d
avg_pool2d = _nn.functional_avg_pool2d
batch_norm = _nn.functional_batch_norm
layer_norm = _nn.functional_layer_norm
group_norm = _nn.functional_group_norm
interpolate = _nn.functional_interpolate
embedding = _nn.functional_embedding
binary_cross_entropy_with_logits = _nn.functional_binary_cross_entropy_with_logits

# Gradient clipping utilities
clip_grad_norm_ = _nn.clip_grad_norm_
clip_grad_value_ = _nn.clip_grad_value_

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
    # Losses
    "mse_loss",
    "l1_loss",
    "cross_entropy",
    "nll_loss",
    "bce_loss",
    # Functional operations
    "dropout",
    "linear",
    "max_pool2d",
    "avg_pool2d",
    "batch_norm",
    "layer_norm",
    "group_norm",
    "interpolate",
    "embedding",
    "binary_cross_entropy_with_logits",
    # Gradient clipping
    "clip_grad_norm_",
    "clip_grad_value_",
]
