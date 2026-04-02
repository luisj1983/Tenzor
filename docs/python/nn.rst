Neural Network Module
=====================

Layers, loss functions, and containers for building neural networks.

Module Base Class
-----------------

.. py:class:: tenzor.nn.Module

   Base class for all neural network modules. Subclass and implement
   ``forward()`` to define custom layers.

Layers
------

- ``tenzor.nn.Linear(in_features, out_features, bias=True)``
- ``tenzor.nn.Conv1d``, ``Conv2d``, ``Conv3d``
- ``tenzor.nn.ConvTranspose1d``, ``ConvTranspose2d``, ``ConvTranspose3d``
- ``tenzor.nn.BatchNorm1d``, ``BatchNorm2d``
- ``tenzor.nn.LayerNorm``, ``GroupNorm``, ``InstanceNorm1d``, ``InstanceNorm2d``
- ``tenzor.nn.Dropout``, ``Dropout2d``, ``AlphaDropout``
- ``tenzor.nn.Embedding``
- ``tenzor.nn.LSTM``, ``GRU``, ``RNN``, ``LSTMCell``, ``GRUCell``, ``RNNCell``
- ``tenzor.nn.MultiheadAttention``
- ``tenzor.nn.TransformerEncoderLayer``, ``TransformerDecoderLayer``
- ``tenzor.nn.TransformerEncoder``, ``TransformerDecoder``

Activation Functions
--------------------

- ``tenzor.nn.ReLU``, ``LeakyReLU``, ``ELU``, ``SELU``
- ``tenzor.nn.Sigmoid``, ``Tanh``
- ``tenzor.nn.GELU``, ``Mish``, ``Swish``
- ``tenzor.nn.Softmax``, ``LogSoftmax``

Loss Functions
--------------

- ``tenzor.nn.MSELoss``, ``L1Loss``, ``SmoothL1Loss``, ``HuberLoss``
- ``tenzor.nn.CrossEntropyLoss``, ``NLLLoss``, ``BCELoss``, ``BCEWithLogitsLoss``
- ``tenzor.nn.KLDivLoss``, ``FocalLoss``, ``DiceLoss``
- ``tenzor.nn.CTCLoss``, ``MarginRankingLoss``
- ``tenzor.nn.SoftMarginLoss``, ``HingeEmbeddingLoss``
- ``tenzor.nn.PoissonNLLLoss``, ``CosineEmbeddingLoss``
- ``tenzor.nn.TripletMarginLoss``, ``MultiLabelSoftMarginLoss``
- ``tenzor.nn.MultiMarginLoss``, ``GaussianNLLLoss``

Containers
----------

- ``tenzor.nn.Sequential(layers)``
- ``tenzor.nn.ModuleList(modules)``
- ``tenzor.nn.ModuleDict(modules)``
