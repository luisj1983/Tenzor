"""Shared Python-side parity tolerances.

Mirrors the C++ constants in tests/backend_parity/parity_tolerances.hpp so
Python tests don't drift from the C++ parity bar. Import here rather than
hard-coding per-file.
"""

# Elementwise arithmetic / activations (float32): single accumulate, modest
# ULP error from non-associative FP sums.
ELEMWISE_RTOL = 1e-5
ELEMWISE_ATOL = 1e-5

# Matmul / linear layers: loose tolerance because cuBLAS, MIOpen, and Vulkan
# compute shaders may order the inner accumulation differently from a
# reference C loop.
MATMUL_RTOL = 1e-4
MATMUL_ATOL = 1e-4

# Reductions (sum/mean over all elements): tree reduction shape varies per
# backend, so tolerance must absorb small reorder error.
REDUCTION_RTOL = 1e-5
REDUCTION_ATOL = 1e-5

# Full NN layer forward: inherits matmul + activation tolerance budgets.
NN_RTOL = 1e-4
NN_ATOL = 1e-4
