/**
 * @file sparse.cpp
 * @brief CPU sparse tensor kernel helpers.
 *
 * This file provides the dense_to_sparse and sparse_to_dense conversion
 * functions that can be called from the core library's sparse ops.
 * The actual MKL-accelerated SpMV/SpMM kernels live in src/sparse/sparse_ops.cpp
 * since they are part of the core library (which also links MKL).
 *
 * This file is intentionally minimal — it exists to be compiled as part of the
 * CPU backend shared library and makes no registrations to the dispatch table.
 * The sparse operations use a different dispatch path (SparseTensor, not Tensor)
 * and are called directly via tenzor::sparse::spmm/spmv.
 */

// Intentionally empty — all CPU sparse logic is in src/sparse/sparse_ops.cpp
// which has direct access to MKL via the core library's MKL linkage.
//
// This file is kept as a placeholder for future CPU-backend-specific sparse
// kernels that may need to register via the dispatch table (e.g., if SparseTensor
// dispatch is unified with Tensor dispatch in the future).
