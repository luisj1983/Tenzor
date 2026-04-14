/**
 * @file linalg.hpp
 * @brief Linear algebra operations
 *
 * Provides matrix decomposition and solve operations using LAPACK backends.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include <string>
#include <tuple>
#include <vector>

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

/**
 * @brief Compute eigenvalues and eigenvectors of a general (non-symmetric) matrix.
 *
 * Uses LAPACKE ?geev to compute the eigendecomposition of a general real matrix.
 * Eigenvalues may be complex even for real input matrices (complex conjugate pairs).
 * The eigenvalues are returned as two separate tensors (real and imaginary parts)
 * rather than complex tensors, matching the LAPACK output convention.
 *
 * @param A Input square matrix (..., N, N). Supports Float32, Float64, Float16,
 *          and BFloat16 (the latter two are upcast to Float32 internally).
 * @return Tuple of (eigenvalues_real, eigenvalues_imag, eigenvectors) where:
 *         - eigenvalues_real: Real parts of eigenvalues (..., N)
 *         - eigenvalues_imag: Imaginary parts of eigenvalues (..., N)
 *         - eigenvectors: Right eigenvectors (..., N, N)
 *
 * @note For real eigenvalues, the corresponding imaginary part is zero.
 *       Complex eigenvalues always appear in conjugate pairs.
 * @note Float16 and BFloat16 inputs are upcast to Float32 for computation,
 *       then results are downcast back to the original dtype.
 *
 * @throws std::invalid_argument if matrix is not at least 2D or not square
 * @throws std::runtime_error if eigendecomposition fails
 */
auto eig(const Tensor& A) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Compute the n-th power of a square matrix.
 *
 * Uses binary exponentiation: O(log n) matrix multiplications.
 * Supports n >= 0. For n < 0, computes inverse then exponentiates.
 *
 * @param A Square matrix (or batch of square matrices)
 * @param n Exponent (integer)
 * @return A^n
 *
 * @throws std::invalid_argument if A is not square
 */
auto matrix_power(const Tensor& A, int64_t n) -> Tensor;

/**
 * @brief Compute LU decomposition with partial pivoting.
 *
 * Factorizes A = P @ L @ U where P is a permutation matrix,
 * L is lower-triangular with unit diagonal, and U is upper-triangular.
 *
 * @param A Input square matrix (..., N, N)
 * @return Tuple of (L, U, pivots) where L is (..., N, N), U is (..., N, N),
 *         and pivots is (..., N) containing pivot indices (1-based LAPACK convention)
 */
auto lu(const Tensor& A) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Solve linear system using pre-computed LU factors.
 *
 * Given LU factors from lu(), solves AX = B efficiently.
 *
 * @param LU_data Packed LU factors (output of LAPACKE_?getrf, (..., N, N))
 * @param pivots Pivot indices from LU factorization (..., N)
 * @param B Right-hand side matrix (..., N, K)
 * @return Solution matrix X (..., N, K)
 */
auto lu_solve(const Tensor& LU_data, const Tensor& pivots,
              const Tensor& B) -> Tensor;

/**
 * @brief Solve a (possibly over- or under-determined) least-squares problem.
 *
 * Computes argmin_X ||A @ X - B||_2 using LAPACKE_?gels (QR/LQ based).
 * For overdetermined A (M >= N), X is the least-squares solution. For
 * underdetermined A (M < N), X is the minimum-norm solution.
 *
 * @param A Coefficient matrix (M, N). Supports Float32/Float64.
 * @param B Right-hand side; either (M,) for a single RHS or (M, K) for
 *          multiple RHS. Must have the same dtype as A.
 * @return Tuple (solution, residuals) where:
 *         - solution: (N,) or (N, K) least-squares / minimum-norm solution.
 *         - residuals: Sum of squared residuals per RHS. Shape is (K,) for
 *           2D B, (,) for 1D B. Only populated when M > N and A has full
 *           column rank; otherwise empty.
 *
 * @throws std::invalid_argument if shapes are inconsistent.
 */
auto lstsq(const Tensor& A, const Tensor& B) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Compute the Moore-Penrose pseudoinverse of A.
 *
 * Uses SVD under the hood: given A = U @ diag(s) @ Vh, returns
 * pinv(A) = V @ diag(s_inv) @ U^H, where s_inv[i] = 1/s[i] for
 * singular values larger than `rcond * max(s)`, and 0 otherwise.
 * Small singular values are thresholded out so a rank-deficient A
 * still yields a stable pseudoinverse.
 *
 * @param A Input matrix (M, N). Supports Float32/Float64.
 * @param rcond Cutoff for small singular values, relative to max(s).
 *              Default 1e-15 matches numpy.linalg.pinv.
 * @return Pseudoinverse (N, M).
 */
auto pinv(const Tensor& A, double rcond = 1e-15) -> Tensor;

/**
 * @brief Compute the matrix exponential exp(A).
 *
 * Uses Higham's scaling-and-squaring algorithm with the Padé-13
 * approximation (Higham, "The Scaling and Squaring Method for the
 * Matrix Exponential Revisited", 2005). This matches SciPy's
 * scipy.linalg.expm and torch.linalg.matrix_exp behaviour for
 * Float32/Float64 square matrices.
 *
 * Cost: O(N^3) matrix operations, with the scaling parameter chosen
 * so that ||A|| / 2^s lies in the Padé-13 region of convergence.
 *
 * @param A Square matrix (N, N). Supports Float32/Float64.
 * @return exp(A) as an (N, N) tensor with the same dtype as A.
 *
 * @throws std::invalid_argument if A is not square.
 */
auto matrix_exp(const Tensor& A) -> Tensor;

// =========================================================================
// New linear algebra operations for PyTorch parity
// =========================================================================

/**
 * @brief Solve a triangular linear system AX = B.
 *
 * @param A Triangular coefficient matrix (..., N, N)
 * @param B Right-hand side matrix (..., N, K)
 * @param upper If true, A is upper-triangular; otherwise lower-triangular (default: true)
 * @param unitriangular If true, assume A has unit diagonal (default: false)
 * @return Solution matrix X (..., N, K)
 */
auto solve_triangular(const Tensor& A, const Tensor& B, bool upper = true, bool unitriangular = false) -> Tensor;

/// Outer product: result[i,j] = a[i] * b[j]
auto outer(const Tensor& a, const Tensor& b) -> Tensor;

/// Generalized inner product (sum product over last dim of a, last dim of b)
auto inner(const Tensor& a, const Tensor& b) -> Tensor;

/// Conjugate dot product for complex tensors
auto vdot(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Return only the singular values of a matrix.
 *
 * Equivalent to `std::get<1>(svd(A, false))`.
 *
 * @param A Input matrix (..., M, N)
 * @return Singular values tensor (..., min(M,N))
 */
auto svdvals(const Tensor& A) -> Tensor;

/**
 * @brief Return only the eigenvalues of a general (non-symmetric) matrix.
 *
 * Returns a pair (real_parts, imag_parts) since eigenvalues of a real
 * non-symmetric matrix may be complex.
 *
 * @param A Input square matrix (..., N, N)
 * @return Tuple of (eigenvalues_real, eigenvalues_imag) tensors, each (..., N)
 */
auto eigvals(const Tensor& A) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Condition number of a matrix.
 *
 * For p="2" (default): max(svdvals) / min(svdvals).
 * For p="fro": norm(A, "fro") * norm(inv(A), "fro").
 *
 * @param A Input square matrix (..., N, N)
 * @param p Norm type: "2" (default) or "fro"
 * @return Condition number scalar
 */
auto cond(const Tensor& A, const std::string& p = "2") -> Tensor;

/**
 * @brief Numerical matrix rank via SVD.
 *
 * Counts singular values above a tolerance. Default tolerance is
 * max(M,N) * eps * max_singular_value.
 *
 * @param A Input matrix (..., M, N)
 * @param tol Tolerance (negative means use default)
 * @return Rank as an integer tensor
 */
auto matrix_rank(const Tensor& A, double tol = -1.0) -> Tensor;

/**
 * @brief Optimized chain of matrix multiplications.
 *
 * Uses dynamic programming to determine the optimal parenthesization
 * that minimizes the total number of scalar multiplications, then
 * executes the chain in that order.
 *
 * @param tensors Vector of 2-D matrices to multiply
 * @return Product of all matrices
 * @throws std::invalid_argument if fewer than 2 tensors or shapes are incompatible
 */
auto multi_dot(const std::vector<Tensor>& tensors) -> Tensor;

/// Embed a batch of vectors as batch diagonal matrices
auto diag_embed(const Tensor& input, int64_t offset = 0, int64_t dim1 = -2, int64_t dim2 = -1) -> Tensor;

/// Create diagonal matrix from flat input
auto diagflat(const Tensor& input, int64_t offset = 0) -> Tensor;

/**
 * @brief Generate orthogonal matrix Q from Householder reflectors.
 *
 * Given the output of a QR factorization (the packed reflectors and tau),
 * computes the full orthogonal matrix Q = H(0) * H(1) * ... * H(k-1).
 * Uses LAPACKE_?orgqr.
 *
 * @param input Matrix containing Householder reflectors (..., M, N)
 * @param tau Scalar factors of the reflectors (..., K) where K = min(M, N)
 * @return Orthogonal matrix Q (..., M, N)
 */
auto householder_product(const Tensor& input, const Tensor& tau) -> Tensor;

/**
 * @brief Compute LDL^T factorization of a symmetric indefinite matrix.
 *
 * Factorizes A = L @ D @ L^T using LAPACKE_?sytrf (Bunch-Kaufman pivoting).
 *
 * @param A Symmetric matrix (..., N, N)
 * @return Tuple of (LD, pivots) where LD contains the packed L and D factors
 *         (..., N, N) and pivots is (..., N)
 */
auto ldl_factor(const Tensor& A) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Solve linear system using pre-computed LDL^T factors.
 *
 * Given LDL^T factors from ldl_factor(), solves AX = B.
 *
 * @param LD Packed LDL^T factors (..., N, N)
 * @param pivots Pivot indices from LDL factorization (..., N)
 * @param B Right-hand side matrix (..., N, K)
 * @return Solution matrix X (..., N, K)
 */
auto ldl_solve(const Tensor& LD, const Tensor& pivots, const Tensor& B) -> Tensor;

/**
 * @brief Compute vector norm along specified dimensions.
 *
 * Supports arbitrary p-norms. Special cases:
 * - ord=inf: max(abs(x))
 * - ord=-inf: min(abs(x))
 * - ord=0: count of nonzero elements
 * - ord=p: sum(abs(x)^p)^(1/p)
 *
 * @param input Input tensor
 * @param ord Norm order (default 2.0)
 * @param dim Dimensions to reduce over (empty = all dims)
 * @param keepdim Whether to keep reduced dimensions (default false)
 * @return Norm value(s)
 */
auto vector_norm(const Tensor& input, double ord = 2.0,
                 std::vector<int64_t> dim = {}, bool keepdim = false) -> Tensor;

/**
 * @brief Compute matrix norm.
 *
 * For ord=2: largest singular value (spectral norm).
 * For ord=1: max absolute column sum.
 * For ord=-1: min absolute column sum.
 * For ord=inf: max absolute row sum.
 * For ord=-inf: min absolute row sum.
 *
 * @param input Input matrix (..., M, N)
 * @param ord Norm order (default 2.0)
 * @return Norm value(s)
 */
auto matrix_norm(const Tensor& input, double ord = 2.0) -> Tensor;

/**
 * @brief Compute dot product along a dimension.
 *
 * Computes sum(a * b, dim) — the dot product of corresponding vectors
 * along the specified dimension.
 *
 * @param a First input tensor
 * @param b Second input tensor (must be broadcastable with a)
 * @param dim Dimension along which to compute the dot product (default -1)
 * @return Result tensor with dim reduced
 */
auto vecdot(const Tensor& a, const Tensor& b, int64_t dim = -1) -> Tensor;

/**
 * @brief Compute inverse of a positive-definite matrix given its Cholesky factor.
 *
 * Given L from A = L @ L^T (or U from A = U^T @ U), returns A^{-1}.
 *
 * @param L Cholesky factor (..., N, N)
 * @param upper If true, L is upper-triangular (default: false)
 * @return Inverse matrix (..., N, N)
 */
auto cholesky_inverse(const Tensor& L, bool upper = false) -> Tensor;

/**
 * @brief Compute the generalized tensor inverse.
 *
 * Reshapes the first `ind` dimensions and the remaining dimensions into a 2D
 * matrix, inverts it, then reshapes back.
 *
 * @param input Input tensor whose first ind dimensions' product equals the remaining dimensions' product
 * @param ind Number of leading dimensions to flatten into rows (default: 2)
 * @return Inverse tensor
 */
auto tensorinv(const Tensor& input, int64_t ind = 2) -> Tensor;

/**
 * @brief Solve the tensor equation A x = B for x.
 *
 * Reshapes A and B into 2D, solves the linear system, then reshapes back.
 *
 * @param A Coefficient tensor
 * @param B Right-hand side tensor
 * @return Solution tensor x
 */
auto tensorsolve(const Tensor& A, const Tensor& B) -> Tensor;

/**
 * @brief Apply the orthogonal matrix Q from QR factorization to another matrix.
 *
 * Computes Q @ other (or Q^T @ other, or other @ Q, or other @ Q^T) where Q is
 * represented by Householder reflectors stored in input and their scalar factors tau.
 *
 * @param input Matrix containing Householder reflectors (..., M, N)
 * @param tau Scalar factors of the reflectors (..., K)
 * @param other Matrix to multiply (..., M, K) or (..., K, M)
 * @param left If true, compute Q @ other; if false, compute other @ Q (default: true)
 * @param transpose If true, use Q^T instead of Q (default: false)
 * @return Result of the multiplication
 */
auto ormqr(const Tensor& input, const Tensor& tau, const Tensor& other,
           bool left = true, bool transpose = false) -> Tensor;

/**
 * @brief Compute the raw QR factorization (GEQRF).
 *
 * Returns the Householder reflector matrix and tau vector, which is more
 * memory-efficient than the full QR decomposition when Q is not needed explicitly.
 *
 * @param input Input matrix (..., M, N)
 * @return Tuple of (result, tau) where result contains packed Householder reflectors
 *         (..., M, N) and tau is the scalar factors (..., min(M,N))
 */
auto geqrf(const Tensor& input) -> std::tuple<Tensor, Tensor>;

} // namespace linalg
} // namespace tenzor
