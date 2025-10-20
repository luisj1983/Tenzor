Python API Reference
====================

This section contains the complete API reference for the Tenzor Python package.

.. toctree::
   :maxdepth: 2
   :caption: API Modules:

   tensor
   nn
   optim
   autograd
   torch_interop
   quantization
   onnx

Core Tensor API
---------------

.. automodule:: tenzor
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

Tensor Creation
^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   tenzor.tensor
   tenzor.empty
   tenzor.zeros
   tenzor.ones
   tenzor.full
   tenzor.randn
   tenzor.rand
   tenzor.randint
   tenzor.arange
   tenzor.linspace
   tenzor.eye
   tenzor.from_numpy

Tensor Operations
^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   tenzor.add
   tenzor.sub
   tenzor.mul
   tenzor.div
   tenzor.matmul
   tenzor.pow
   tenzor.sqrt
   tenzor.exp
   tenzor.log
   tenzor.sin
   tenzor.cos
   tenzor.tan
   tenzor.tanh
   tenzor.sigmoid

Reduction Operations
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   tenzor.sum
   tenzor.mean
   tenzor.max
   tenzor.min
   tenzor.argmax
   tenzor.argmin

Concatenation
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   tenzor.cat
   tenzor.stack
   tenzor.split
   tenzor.chunk

Data Types and Devices
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tenzor.DType
   :members:
   :undoc-members:

.. autoclass:: tenzor.DeviceType
   :members:
   :undoc-members:

.. autoclass:: tenzor.Device
   :members:
   :undoc-members:
   :show-inheritance:

Tensor Class
^^^^^^^^^^^^

.. autoclass:: tenzor.Tensor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __add__, __sub__, __mul__, __truediv__, __matmul__, __getitem__, __setitem__
