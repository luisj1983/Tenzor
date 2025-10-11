/**
 * @file indexing.hpp
 * @brief Tensor indexing and selection operations
 *
 * Provides advanced indexing, slicing, masking, and conditional
 * selection operations for tensors.
 */

#pragma once

#include "../core/tensor.hpp"

namespace tenzor {

/**
 * @defgroup tensor_indexing Indexing and Selection Operations
 * @brief Advanced indexing and element selection
 * @{
 */

/**
 * @brief Slice tensor along dimension.
 * @param dim Dimension to slice
 * @param start Start index
 * @param end End index (exclusive)
 * @param step Step size
 */
auto slice(const Tensor& input,
          int64_t dim,
          int64_t start,
          int64_t end,
          int64_t step = 1) -> Tensor;

/** @brief Select elements along dimension using index tensor. */
auto index_select(const Tensor& input,
                 int64_t dim,
                 const Tensor& index) -> Tensor;

/** @brief Gather elements along dimension using index tensor. */
auto gather(const Tensor& input,
           int64_t dim,
           const Tensor& index) -> Tensor;

/** @brief Scatter source elements into input using index tensor. */
auto scatter(const Tensor& input,
            int64_t dim,
            const Tensor& index,
            const Tensor& src) -> Tensor;

/** @brief Select elements where mask is true. */
auto masked_select(const Tensor& input, const Tensor& mask) -> Tensor;

/** @brief Fill elements with value where mask is true. */
auto masked_fill(const Tensor& input, const Tensor& mask, float value) -> Tensor;

/**
 * @brief Conditional element selection.
 * @param condition Boolean tensor
 * @param x Values where condition is true
 * @param y Values where condition is false
 */
auto where(const Tensor& condition,
          const Tensor& x,
          const Tensor& y) -> Tensor;

/** @brief Take elements from flattened tensor using indices. */
auto take(const Tensor& input, const Tensor& index) -> Tensor;

/** @brief Put source elements into flattened tensor at indices. */
auto put(const Tensor& input, const Tensor& index, const Tensor& source) -> Tensor;

/** @} */ // end of tensor_indexing group

} // namespace tenzor
