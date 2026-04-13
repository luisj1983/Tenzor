/**
 * @file advanced.hpp
 * @brief Advanced tensor operations (Phase 6 additions)
 *
 * Provides advanced tensor operations including topk, sort, unique,
 * cumulative operations, and expand functionality.
 */

#pragma once

#include <span>
#include <string>
#include <tuple>
#include <vector>
#include "../core/tensor.hpp"

namespace tenzor {

/**
 * @defgroup tensor_advanced Advanced Operations
 * @brief Advanced tensor manipulation and analysis operations
 * @{
 */

/**
 * @brief Find top k largest elements along dimension.
 *
 * Returns both values and indices of the k largest elements.
 *
 * @param input Input tensor
 * @param k Number of top elements
 * @param dim Dimension along which to find top-k
 * @param largest If true, find largest elements; if false, find smallest
 * @param sorted If true, return sorted results
 * @return Tuple of (values, indices)
 *
 * @code
 * auto [values, indices] = topk(input, 5, 0, true);
 * @endcode
 */
auto topk(const Tensor& input,
          int64_t k,
          int64_t dim = -1,
          bool largest = true,
          bool sorted = true) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Sort tensor along dimension.
 *
 * Returns both sorted values and indices.
 *
 * @param input Input tensor
 * @param dim Dimension to sort along
 * @param descending Sort in descending order
 * @return Tuple of (sorted_values, indices)
 *
 * @code
 * auto [sorted, indices] = sort(input, 0, false);
 * @endcode
 */
auto sort(const Tensor& input,
          int64_t dim = -1,
          bool descending = false) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Find unique elements in tensor.
 *
 * Returns unique elements, optionally with inverse indices and counts.
 *
 * @param input Input tensor (will be flattened)
 * @param sorted If true, return sorted unique elements
 * @param return_inverse If true, return inverse indices
 * @param return_counts If true, return counts for each unique element
 * @return Tuple of (unique, inverse_indices, counts)
 *         - inverse_indices is empty if return_inverse=false
 *         - counts is empty if return_counts=false
 *
 * @code
 * auto [unique_vals, inverse, counts] =
 *     unique(input, true, true, true);
 * @endcode
 */
auto unique(const Tensor& input,
            bool sorted = true,
            bool return_inverse = false,
            bool return_counts = false)
    -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Cumulative sum along dimension.
 *
 * Computes cumulative sum: output[i] = input[0] + input[1] + ... + input[i]
 *
 * @param input Input tensor
 * @param dim Dimension to compute cumsum
 * @return Tensor with cumulative sums
 *
 * @code
 * auto t = Tensor({5}, DType::Float32, Device::cpu());
 * auto cs = cumsum(t, 0);  // Cumulative sum along dimension 0
 * @endcode
 */
auto cumsum(const Tensor& input, int64_t dim) -> Tensor;

/**
 * @brief Cumulative product along dimension.
 *
 * Computes cumulative product: output[i] = input[0] * input[1] * ... * input[i]
 *
 * @param input Input tensor
 * @param dim Dimension to compute cumprod
 * @return Tensor with cumulative products
 *
 * @code
 * auto t = Tensor({5}, DType::Float32, Device::cpu());
 * auto cp = cumprod(t, 0);  // Cumulative product along dimension 0
 * @endcode
 */
auto cumprod(const Tensor& input, int64_t dim) -> Tensor;

/**
 * @brief Log-cumulative-sum-exp along a dimension (numerically stable).
 *
 * Computes: output[i] = log(sum(exp(input[0:i+1]))) using a running
 * max to maintain numerical stability.
 *
 * @param input Input tensor (float types only)
 * @param dim Dimension along which to compute
 * @return Tensor with log-cumulative-sum-exp values
 *
 * @code
 * auto t = Tensor({5}, DType::Float32, Device::cpu());
 * auto lcse = logcumsumexp(t, 0);
 * @endcode
 */
auto logcumsumexp(const Tensor& input, int64_t dim) -> Tensor;

/**
 * @brief Einstein summation convention.
 *
 * Performs tensor contractions, permutations, and reductions specified by
 * a subscript string following NumPy/PyTorch einsum notation.
 *
 * @param equation Subscript string (e.g., "ij,jk->ik" for matrix multiply)
 * @param tensors Input tensors
 * @return Result tensor
 *
 * @code
 * auto c = einsum("ij,jk->ik", {a, b});     // Matrix multiply
 * auto tr = einsum("ii->", {a});              // Trace
 * auto d = einsum("bij,bjk->bik", {a, b});   // Batch matmul
 * @endcode
 */
auto einsum(const std::string& equation,
            std::span<const Tensor> tensors) -> Tensor;

/**
 * @brief Median of tensor elements along a dimension.
 *
 * @param input Input tensor
 * @param dim Dimension along which to compute median
 * @param keepdim Whether to retain reduced dimension
 * @return Tuple of (values, indices) tensors
 */
auto median(const Tensor& input,
            int64_t dim = -1,
            bool keepdim = false) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Mode (most frequent value) along a dimension.
 *
 * @param input Input tensor
 * @param dim Dimension along which to compute mode
 * @param keepdim Whether to retain reduced dimension
 * @return Tuple of (values, indices) tensors
 */
auto mode(const Tensor& input,
          int64_t dim = -1,
          bool keepdim = false) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Assign each element to a bucket based on sorted boundaries.
 *
 * For each element in input, finds the bucket index via binary search
 * in the sorted boundaries tensor.
 *
 * @param input Input tensor of values to bucketize
 * @param boundaries 1-D sorted tensor of bucket boundaries
 * @param right If true, use right-closed intervals (default: false, left-closed)
 * @return Int64 tensor of bucket indices with same shape as input
 */
auto bucketize(const Tensor& input, const Tensor& boundaries, bool right = false) -> Tensor;

/// @name Matrix Construction Operations
/// @{

/**
 * @brief Kronecker product of two tensors.
 *
 * For 2-D tensors A (m×n) and B (p×q), returns (m*p × n*q) tensor where
 * each element a_ij is replaced by a_ij * B.
 *
 * @param a First tensor (must be 2-D)
 * @param b Second tensor (must be 2-D)
 * @return Kronecker product tensor
 */
auto kron(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Create a block diagonal matrix from a sequence of tensors.
 *
 * Each tensor is placed along the diagonal of the result, with zeros elsewhere.
 *
 * @param tensors Sequence of 2-D tensors
 * @return Block diagonal matrix
 */
auto block_diag(std::span<const Tensor> tensors) -> Tensor;

/**
 * @brief Generate a Vandermonde matrix.
 *
 * Column k of the output is x^k (or x^(N-1-k) if increasing=false).
 *
 * @param x 1-D tensor of values
 * @param N Number of columns (default: x.size(0))
 * @param increasing If true, columns are [1, x, x², ...]; if false [x^(N-1), ..., x, 1]
 * @return Vandermonde matrix of shape (len(x), N)
 */
auto vander(const Tensor& x, int64_t N = -1, bool increasing = false) -> Tensor;

/**
 * @brief Cartesian product of 1-D tensors.
 *
 * @param tensors Sequence of 1-D tensors
 * @return 2-D tensor where each row is one element of the Cartesian product
 */
auto cartesian_prod(std::span<const Tensor> tensors) -> Tensor;

/**
 * @brief All r-length combinations of elements from the input tensor.
 *
 * @param input 1-D tensor
 * @param r Combination length
 * @param with_replacement If true, allow repeated elements
 * @return 2-D tensor of shape (num_combinations, r)
 */
auto combinations(const Tensor& input, int64_t r, bool with_replacement = false) -> Tensor;

/// @}

/**
 * @brief Generalized tensor contraction (like numpy.tensordot).
 *
 * Contracts tensors a and b over the specified dimensions.
 * Implementation: permute + reshape to 2D + matmul + reshape back.
 *
 * @param a First tensor
 * @param b Second tensor
 * @param dims_a Dimensions of a to contract over
 * @param dims_b Dimensions of b to contract over
 * @return Contracted result tensor
 *
 * @code
 * // Matrix multiply via tensordot
 * auto c = tensordot(a, b, {1}, {0});  // contract dim 1 of a with dim 0 of b
 * @endcode
 */
auto tensordot(const Tensor& a, const Tensor& b,
               std::vector<int64_t> dims_a,
               std::vector<int64_t> dims_b) -> Tensor;

/**
 * @brief Generalized tensor contraction with integer dims.
 *
 * Contracts the last N dims of a with the first N dims of b.
 *
 * @param a First tensor
 * @param b Second tensor
 * @param dims Number of dimensions to contract (default: 2)
 * @return Contracted result tensor
 */
auto tensordot(const Tensor& a, const Tensor& b, int64_t dims = 2) -> Tensor;

/** @} */ // end of tensor_advanced group

} // namespace tenzor

namespace tenzor {
namespace ops {
using tenzor::cumsum;
using tenzor::cumprod;
using tenzor::logcumsumexp;
} // namespace ops
} // namespace tenzor
