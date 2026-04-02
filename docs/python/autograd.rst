Autograd
========

Automatic differentiation engine.

Variable
--------

.. py:class:: tenzor.Variable(tensor, requires_grad=False)

   Wraps a Tensor with gradient tracking.

   - ``data`` -- The underlying Tensor
   - ``grad`` -- Accumulated gradient (after backward)
   - ``requires_grad`` -- Whether gradients are computed
   - ``backward()`` -- Compute gradients via backpropagation

Custom Functions
----------------

.. py:class:: tenzor.autograd.Function

   Base class for custom differentiable operations.
   Subclass and implement ``forward`` and ``backward`` as ``@staticmethod``.

   .. py:classmethod:: apply(*inputs)

      Apply the function and register it in the autograd graph.

Context Managers
----------------

- ``tenzor.no_grad()`` -- Disable gradient computation
- ``tenzor.enable_grad()`` -- Enable gradient computation
- ``tenzor.inference_mode()`` -- Disable gradient and version tracking
- ``tenzor.detect_anomaly()`` -- Enable NaN/Inf detection in backward
