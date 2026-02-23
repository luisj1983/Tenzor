/**
 * @file linalg.hpp
 * @brief Linear algebra operations
 *
 * Provides matrix decomposition and solve operations using LAPACK backends.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include <tuple>

namespace tenzor {
namespace linalg {

/**
 * @brief Compute matrix determinant.
 *
 * @param A Input square matrix (..., N, N)
 * @return Determinant value(s) (...)
 */
auto det(const Tensor& A) -> Tensor;

/**
 * @brief Compute matrix inverse.
 *
 * @param A Input square matrix (..., N, N)
 * @return Inverse matrix (..., N, N)
 * @throws std::runtime_error if matrix is singular
 */
auto inv(const Tensor& A) -> Tensor;

/**
 * @brief Solve linear system AX = B.
 *
 * @param A Coefficient matrix (..., N, N)
 * @param B Right-hand side matrix (..., N, K)
 * @return Solution matrix X (..., N, K)
 */
auto solve(const Tensor& A, const Tensor& B) -> Tensor;

/**
 * @brief Compute Cholesky decomposition.
 *
 * Returns lower-triangular matrix L such that A = L @ L^T.
 *
 * @param A Symmetric positive-definite matrix (..., N, N)
 * @param upper If true, return upper-triangular factor (default: false)
 * @return Cholesky factor (..., N, N)
 */
auto cholesky(const Tensor& A, bool upper = false) -> Tensor;

/**
 * @brief Compute matrix norm.
 *
 * @param A Input tensor
 * @param ord Norm order ("fro" for Frobenius, "1", "2", "inf", or "nuc")
 * @return Norm value
 */
auto norm(const Tensor& A, const std::string& ord = "fro") -> Tensor;

} // namespace linalg
} // namespace tenzor
