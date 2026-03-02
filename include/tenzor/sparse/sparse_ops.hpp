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

} // namespace sparse
} // namespace tenzor
