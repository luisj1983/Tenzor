/**
 * @file advanced.hpp
 * @brief Advanced tensor operations (Phase 6 additions)
 *
 * Provides advanced tensor operations including topk, sort, unique,
 * cumulative operations, and expand functionality.
 */

#pragma once

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

/** @} */ // end of tensor_advanced group

} // namespace tenzor
