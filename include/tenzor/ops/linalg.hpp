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

/**
 * @brief Compute sign and log of absolute determinant.
 *
 * More numerically stable than computing det directly for large matrices.
 *
 * @param A Input square matrix (..., N, N)
 * @return Tuple of (sign, logabsdet) tensors
 */
auto slogdet(const Tensor& A) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Compute Singular Value Decomposition.
 *
 * Factorizes A = U @ diag(S) @ V^T.
 *
 * @param A Input matrix (..., M, N)
 * @param full_matrices If true, compute full U (M,M) and Vh (N,N). If false, compute reduced (M,K) and (K,N) where K=min(M,N).
 * @return Tuple of (U, S, Vh) tensors
 */
auto svd(const Tensor& A, bool full_matrices = true) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Compute QR decomposition.
 *
 * Factorizes A = Q @ R where Q is orthogonal and R is upper-triangular.
 *
 * @param A Input matrix (..., M, N)
 * @return Tuple of (Q, R) tensors
 */
auto qr(const Tensor& A) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix.
 *
 * @param A Symmetric matrix (..., N, N)
 * @return Tuple of (eigenvalues, eigenvectors) where eigenvalues shape is (..., N)
 */
auto eigh(const Tensor& A) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Compute eigenvalues of a symmetric/Hermitian matrix.
 *
 * @param A Symmetric matrix (..., N, N)
 * @return Eigenvalues tensor (..., N)
 */
auto eigvalsh(const Tensor& A) -> Tensor;

} // namespace linalg
} // namespace tenzor
