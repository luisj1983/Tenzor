/**
 * @file transform.hpp
 * @brief Tensor shape transformation operations
 *
 * Provides operations for reshaping, transposing, concatenating,
 * and otherwise manipulating tensor shapes and layouts.
 */

#pragma once

#include <vector>
#include "../core/tensor.hpp"

namespace tenzor {

/**
 * @defgroup tensor_transform Shape Transformation Operations
 * @brief Operations for reshaping and manipulating tensor layouts
 * @{
 */

/** @brief Reshape tensor (may copy if not contiguous). */
auto reshape(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

/** @brief Zero-copy reshape (requires contiguous tensor). */
auto view(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

/** @brief Transpose two dimensions. */
auto transpose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;

/** @brief Permute dimensions to new order. */
auto permute(const Tensor& input, std::vector<int64_t> dims) -> Tensor;

/** @brief Remove dimensions of size 1. */
auto squeeze(const Tensor& input, std::optional<int64_t> dim = std::nullopt) -> Tensor;

/** @brief Add dimension of size 1. */
auto unsqueeze(const Tensor& input, int64_t dim) -> Tensor;

/**
 * @brief Flatten tensor dimensions.
 * @param start_dim First dimension to flatten
 * @param end_dim Last dimension to flatten (-1 = last)
 */
auto flatten(const Tensor& input, int64_t start_dim = 0, int64_t end_dim = -1) -> Tensor;

/**
 * @brief Concatenate tensors along dimension.
 * @param tensors Tensors to concatenate
 * @param dim Dimension for concatenation
 */
auto cat(std::span<const Tensor> tensors, int64_t dim = 0) -> Tensor;

/** @brief Stack tensors along new dimension. */
auto stack(std::span<const Tensor> tensors, int64_t dim = 0) -> Tensor;

/** @brief Split tensor into chunks of specific size. */
auto split(const Tensor& input, int64_t split_size, int64_t dim = 0) -> std::vector<Tensor>;

/** @brief Split tensor into specific number of chunks. */
auto chunk(const Tensor& input, int64_t chunks, int64_t dim = 0) -> std::vector<Tensor>;

/** @brief Repeat tensor elements. */
auto repeat(const Tensor& input, std::vector<int64_t> repeats) -> Tensor;

/** @brief Tile tensor by repeating along dimensions. */
auto tile(const Tensor& input, std::vector<int64_t> reps) -> Tensor;

/** @brief Broadcast tensor to new shape. */
auto expand(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

/** @brief Return contiguous copy if needed. */
auto contiguous(const Tensor& input) -> Tensor;

/** @} */ // end of tensor_transform group

} // namespace tenzor
