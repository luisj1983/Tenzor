Sparse Tensors
==============

Efficient storage and computation for tensors with many zero elements.

Formats
-------

- **COO** (Coordinate): Store indices and values separately
- **CSR** (Compressed Sparse Row): Efficient for row-based access
- **CSC** (Compressed Sparse Column): Efficient for column-based access
- **BSR** (Block Sparse Row): Block-structured sparsity

Creation
--------

.. code-block:: python

   import tenzor as tz

   # From indices and values (COO)
   indices = tz.tensor([[0, 1, 2], [0, 1, 2]], dtype=tz.dtype.int64)
   values = tz.tensor([1.0, 2.0, 3.0])
   s = tz.sparse.sparse_coo(indices, values, [3, 3])

   # From dense tensor
   d = tz.eye(3)
   s_csr = tz.sparse.to_sparse_csr(d)
   s_csc = tz.sparse.to_sparse_csc(d)

Operations
----------

- ``tz.sparse.spmm(sparse, dense)`` -- Sparse-dense matrix multiply
- ``tz.sparse.spmv(sparse, vector)`` -- Sparse-dense matrix-vector multiply
- ``tz.sparse.add(sparse, dense)`` -- Sparse + dense addition
- ``tz.sparse.sparse_add(a, b)`` -- Sparse + sparse addition
- ``tz.sparse.mul(sparse, scalar)`` -- Scalar multiplication

Format Conversion
-----------------

.. code-block:: python

   s.to_coo()     # Convert to COO
   s.to_csr()     # Convert to CSR
   s.to_csc()     # Convert to CSC
   s.to_bsr((2,2))  # Convert to BSR with 2x2 blocks
   s.to_dense()   # Convert to dense tensor
   s.transpose()  # Transpose 2D sparse tensor
