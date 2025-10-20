Neural Network Module (tenzor.nn)
==================================

The ``tenzor.nn`` module contains neural network layers, activation functions,
loss functions, and utilities for building deep learning models.

.. currentmodule:: tenzor.nn

Module Base Class
-----------------

.. autoclass:: Module
   :members:
   :undoc-members:
   :show-inheritance:

Containers
----------

.. autoclass:: Sequential
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ModuleList
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ModuleDict
   :members:
   :undoc-members:
   :show-inheritance:

Linear Layers
-------------

.. autoclass:: Linear
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Bilinear
   :members:
   :undoc-members:
   :show-inheritance:

Convolutional Layers
--------------------

.. autoclass:: Conv1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Conv2d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Conv3d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConvTranspose1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConvTranspose2d
   :members:
   :undoc-members:
   :show-inheritance:

Pooling Layers
--------------

.. autoclass:: MaxPool1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MaxPool2d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AvgPool1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AvgPool2d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaptiveAvgPool1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaptiveAvgPool2d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaptiveMaxPool1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaptiveMaxPool2d
   :members:
   :undoc-members:
   :show-inheritance:

Normalization Layers
--------------------

.. autoclass:: BatchNorm1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BatchNorm2d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LayerNorm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GroupNorm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: InstanceNorm1d
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: InstanceNorm2d
   :members:
   :undoc-members:
   :show-inheritance:

Recurrent Layers
----------------

.. autoclass:: RNN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LSTM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GRU
   :members:
   :undoc-members:
   :show-inheritance:

Transformer Layers
------------------

.. autoclass:: MultiheadAttention
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TransformerEncoderLayer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TransformerDecoderLayer
   :members:
   :undoc-members:
   :show-inheritance:

Activation Functions
--------------------

Module Classes
^^^^^^^^^^^^^^

.. autoclass:: ReLU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LeakyReLU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PReLU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ELU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SELU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GELU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Sigmoid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Tanh
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Softmax
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LogSoftmax
   :members:
   :undoc-members:
   :show-inheritance:

Functional API
^^^^^^^^^^^^^^

.. autofunction:: relu
.. autofunction:: leaky_relu
.. autofunction:: elu
.. autofunction:: selu
.. autofunction:: gelu
.. autofunction:: sigmoid
.. autofunction:: tanh
.. autofunction:: softmax
.. autofunction:: log_softmax

Dropout Layers
--------------

.. autoclass:: Dropout
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Dropout2d
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
--------------

.. autoclass:: Loss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MSELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CrossEntropyLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BCELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BCEWithLogitsLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NLLLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: L1Loss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SmoothL1Loss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HuberLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: KLDivLoss
   :members:
   :undoc-members:
   :show-inheritance:

Embedding Layers
----------------

.. autoclass:: Embedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EmbeddingBag
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. autofunction:: init_weights
.. autofunction:: count_parameters
