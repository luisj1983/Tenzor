/**
 * @file sparse_ops.hpp
 * @brief Operations on sparse tensors
 *
 * Sparse-dense and sparse-sparse operations including matrix multiply,
 * matrix-vector multiply, and element-wise operations.
 */

#pragma once

#include "sparse_tensor.hpp"

namespace tenzor {
namespace sparse {

/**
 * @brief Sparse-dense matrix multiplication.
 *
 * @param sparse Sparse matrix (M, K)
 * @param dense Dense matrix (K, N)
 * @return Dense result matrix (M, N)
 */
auto spmm(const SparseTensor& sparse, const Tensor& dense) -> Tensor;

/**
 * @brief Sparse-dense matrix-vector multiplication.
 *
 * @param sparse Sparse matrix (M, K)
 * @param vec Dense vector (K,)
 * @return Dense result vector (M,)
 */
auto spmv(const SparseTensor& sparse, const Tensor& vec) -> Tensor;

/**
 * @brief Sparse-dense addition.
 *
 * @param sparse Sparse tensor
 * @param dense Dense tensor
 * @return Dense result tensor
 */
auto add(const SparseTensor& sparse, const Tensor& dense) -> Tensor;

/**
 * @brief Sparse-sparse addition.
 */
auto add(const SparseTensor& a, const SparseTensor& b) -> SparseTensor;

/**
 * @brief Scalar multiplication of sparse tensor.
 */
auto mul(const SparseTensor& sparse, double scalar) -> SparseTensor;

/**
 * @brief Sparse-sparse matrix multiplication (SpGEMM).
 *
 * Computes C = A @ B where both A and B are sparse matrices.
 * Result is returned in CSR format.
 *
 * CPU: symbolic + numeric two-phase algorithm.
 * CUDA: uses cusparseSpGEMM when available.
 * ROCm: uses rocsparse_spgemm when available.
 *
 * @param a Sparse matrix (M, K)
 * @param b Sparse matrix (K, N)
 * @return Sparse result matrix (M, N) in CSR format
 */
auto spgemm(const SparseTensor& a, const SparseTensor& b) -> SparseTensor;

/**
 * @brief Sparse triangular solve.
 *
 * Solves L @ x = b (lower triangular) or U @ x = b (upper triangular)
 * where L/U is a sparse triangular matrix and b is a dense vector/matrix.
 *
 * CPU: forward/backward substitution.
 * CUDA: uses cusparseSpSV_solve when available.
 * ROCm: uses rocsparse_csrsv_solve when available.
 *
 * @param L Sparse triangular matrix (N, N)
 * @param b Dense right-hand side vector (N,) or matrix (N, K)
 * @param upper If true, treat L as upper triangular (default: false = lower)
 * @return Dense solution tensor with same shape as b
 */
auto sparse_triangular_solve(const SparseTensor& L, const Tensor& b, bool upper = false) -> Tensor;

} // namespace sparse
} // namespace tenzor
