/**
 * @file sparse_ops.hpp
 * @brief Operations on sparse tensors
 *
 * Sparse-dense and sparse-sparse operations including matrix multiply,
 * matrix-vector multiply, and element-wise operations.
 */

#pragma once

#include "sparse_tensor.hpp"

#include <optional>

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

/**
 * @brief Sampled dense-dense matrix multiplication.
 *
 * Computes C = mask * (A @ B^T) element-wise, where `mask` provides the
 * sparsity pattern (CSR): only entries at (i, j) where mask[i, j] is
 * non-zero are actually computed and stored. Returns a sparse matrix
 * with the same sparsity pattern as `mask` but with values filled from
 * the A @ B^T dot products.
 *
 * SDDMM is the bottleneck of many graph neural networks and attention
 * layers (e.g. FlashAttention's score computation with a pre-defined
 * pattern mask). This CPU implementation walks the mask's row pointer
 * and column index, computing one dot product per non-zero. Each dot
 * product is A[i, :] · B[j, :].
 *
 * @param mask Sparse mask (CSR) of shape (M, N)
 * @param A Dense matrix of shape (M, K)
 * @param B Dense matrix of shape (N, K) — note: read row-wise as B[j, :]
 * @return Sparse result (CSR) with mask's pattern and dot-product values
 */
auto sddmm(const SparseTensor& mask, const Tensor& A, const Tensor& B) -> SparseTensor;

/**
 * @brief Sparse softmax over non-zero values per row.
 *
 * For CSR format: applies softmax independently to each row's non-zero values.
 * Masked positions (zeros/unstored entries) are treated as -inf and do not
 * participate in the softmax normalization.
 *
 * @param sparse Input sparse tensor (CSR format, 2D)
 * @return Sparse tensor with same pattern, values replaced by softmax output
 */
auto sparse_softmax(const SparseTensor& sparse) -> SparseTensor;

/**
 * @brief Sparse log-softmax over non-zero values per row.
 *
 * Same as sparse_softmax but returns log-probabilities. Numerically stable
 * via the log-sum-exp trick over each row's non-zero values.
 *
 * @param sparse Input sparse tensor (CSR format, 2D)
 * @return Sparse tensor with same pattern, values replaced by log-softmax output
 */
auto sparse_log_softmax(const SparseTensor& sparse) -> SparseTensor;

/**
 * @brief Dense-weight linear via sparse SpMM (free function).
 *
 * Computes y = spmm(from_dense(weight), xᵀ)ᵀ + bias — the EXACT (lossless)
 * equivalent of x @ weightᵀ routed through the CSR SpMM path (from_dense keeps
 * every non-zero, so no values are dropped). Used by the JIT SparsePass
 * interpreter to execute SparseMatMul nodes retagged from Linear.
 *
 * @param input  Float activation, rank >= 2, last dim == weight.shape[1].
 * @param weight Float weight, [out_features, in_features].
 * @param bias   Optional bias, [out_features].
 * @return Output, input.shape[:-1] + [out_features].
 */
auto sparse_matmul_dynamic(const Tensor& input, const Tensor& weight,
                           const std::optional<Tensor>& bias) -> Tensor;

} // namespace sparse
} // namespace tenzor
