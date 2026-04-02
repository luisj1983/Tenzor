Tensor
======

The core data structure for multi-dimensional arrays.

.. py:class:: tenzor.Tensor

   A multi-dimensional array with automatic backend dispatch.

   **Properties:**

   - ``shape`` -- Tensor dimensions as a list of integers
   - ``ndim`` -- Number of dimensions
   - ``dtype`` -- Element data type
   - ``device`` -- Device location (cpu, cuda:0, etc.)
   - ``numel`` -- Total number of elements
   - ``names`` -- Dimension names (None if unnamed, experimental)

   **Creation Functions:**

   .. py:function:: tenzor.zeros(shape, dtype=float32, device="cpu")
   .. py:function:: tenzor.ones(shape, dtype=float32, device="cpu")
   .. py:function:: tenzor.randn(shape, dtype=float32, device="cpu")
   .. py:function:: tenzor.rand(shape, dtype=float32, device="cpu")
   .. py:function:: tenzor.full(shape, fill_value, dtype=float32, device="cpu")
   .. py:function:: tenzor.empty(shape, dtype=float32, device="cpu")
   .. py:function:: tenzor.eye(n, dtype=float32, device="cpu")
   .. py:function:: tenzor.arange(start, end, step, dtype=float32)
   .. py:function:: tenzor.linspace(start, end, steps, dtype=float32)

   **Named Dimensions (experimental):**

   .. py:method:: Tensor.rename(*names)

      Return a view with named dimensions.

   .. py:method:: Tensor.has_names()

      Check if tensor has named dimensions.

   .. py:method:: Tensor.dim_index(name)

      Find dimension index by name.
