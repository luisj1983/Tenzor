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

/** @brief Concatenate tensors from initializer list. */
inline auto cat(std::initializer_list<Tensor> tensors, int64_t dim = 0) -> Tensor {
    std::vector<Tensor> vec(tensors);
    return cat(std::span<const Tensor>(vec), dim);
}

/** @brief Concatenate tensors from vector. */
inline auto cat(const std::vector<Tensor>& tensors, int64_t dim = 0) -> Tensor {
    return cat(std::span<const Tensor>(tensors), dim);
}

/** @brief Stack tensors along new dimension. */
auto stack(std::span<const Tensor> tensors, int64_t dim = 0) -> Tensor;

/** @brief Split tensor into chunks of specific size. */
auto split(const Tensor& input, int64_t split_size, int64_t dim = 0) -> std::vector<Tensor>;

/** @brief Split tensor into chunks with specified sizes. */
auto split_with_sizes(const Tensor& input, const std::vector<int64_t>& split_sizes, int64_t dim = 0) -> std::vector<Tensor>;

/** @brief Split tensor into specific number of chunks. */
auto chunk(const Tensor& input, int64_t chunks, int64_t dim = 0) -> std::vector<Tensor>;

/** @brief Repeat tensor elements. */
auto repeat(const Tensor& input, std::vector<int64_t> repeats) -> Tensor;

/** @brief Tile tensor by repeating along dimensions. */
auto tile(const Tensor& input, std::vector<int64_t> reps) -> Tensor;

/** @brief Broadcast tensor to new shape. */
auto expand(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

/** @brief Broadcast tensor to target shape (alias for expand). */
inline auto broadcast_to(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    return expand(input, std::move(shape));
}

/** @brief Return contiguous copy if needed. */
auto contiguous(const Tensor& input) -> Tensor;

/**
 * @brief Roll tensor elements along a dimension (circular shift).
 * @param input Input tensor
 * @param shifts Number of positions to roll (positive = right, negative = left)
 * @param dim Dimension to roll along
 * @return Rolled tensor
 */
auto roll(const Tensor& input, int64_t shifts, int64_t dim) -> Tensor;

/// @name Triangular and Diagonal Operations
/// @{

/**
 * @brief Extract upper triangular part of a matrix.
 * @param input Input tensor (at least 2D)
 * @param diagonal Offset from main diagonal (0 = main, positive = above, negative = below)
 * @return Tensor with elements below the diagonal zeroed out
 */
auto triu(const Tensor& input, int64_t diagonal = 0) -> Tensor;

/**
 * @brief Extract lower triangular part of a matrix.
 * @param input Input tensor (at least 2D)
 * @param diagonal Offset from main diagonal (0 = main, positive = above, negative = below)
 * @return Tensor with elements above the diagonal zeroed out
 */
auto tril(const Tensor& input, int64_t diagonal = 0) -> Tensor;

/**
 * @brief Extract diagonal or construct diagonal matrix.
 *
 * If input is 1D, returns a 2D square matrix with input on the diagonal.
 * If input is 2D, returns a 1D tensor of diagonal elements.
 * @param input Input tensor (1D or 2D)
 * @param diagonal Offset from main diagonal
 */
auto diag(const Tensor& input, int64_t diagonal = 0) -> Tensor;

/**
 * @brief Sum of diagonal elements (trace of matrix).
 * @param input Input tensor (at least 2D)
 * @return Scalar tensor with trace value
 */
auto trace(const Tensor& input) -> Tensor;

/// @}
/// @name Reversal Operations
/// @{

/**
 * @brief Reverse tensor along specified dimensions.
 * @param input Input tensor
 * @param dims Dimensions to flip
 * @return Tensor with reversed elements along specified dims
 */
auto flip(const Tensor& input, std::vector<int64_t> dims) -> Tensor;

/**
 * @brief Move dimensions of a tensor to new positions.
 *
 * Computes a permutation that moves source[i] to destination[i] and calls permute().
 *
 * @param input Input tensor
 * @param source Source dimension(s)
 * @param destination Target position(s)
 * @return Permuted tensor (view, no copy)
 */
auto movedim(const Tensor& input, std::vector<int64_t> source, std::vector<int64_t> destination) -> Tensor;

/**
 * @brief Swap two dimensions of a tensor. Alias for transpose(input, dim0, dim1).
 */
auto swapaxes(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;

/// @}
/// @name Repeat Interleave
/// @{

/**
 * @brief Repeat each element of the tensor a given number of times along a dimension.
 *
 * If dim is not specified, the input is flattened first, then repeated along dim 0.
 * Each element is repeated `repeats` times consecutively.
 *
 * @param input Input tensor
 * @param repeats Number of times to repeat each element
 * @param dim Dimension along which to repeat (nullopt = flatten first)
 * @return Tensor with repeated elements
 */
auto repeat_interleave(const Tensor& input, int64_t repeats,
                       std::optional<int64_t> dim = std::nullopt) -> Tensor;

/**
 * @brief Repeat each element of the tensor by a per-element repeat count.
 *
 * If dim is not specified, the input is flattened first, then repeated along dim 0.
 * Element i is repeated repeats[i] times.
 *
 * @param input Input tensor
 * @param repeats 1D tensor of per-element repeat counts
 * @param dim Dimension along which to repeat (nullopt = flatten first)
 * @return Tensor with repeated elements
 */
auto repeat_interleave(const Tensor& input, const Tensor& repeats,
                       std::optional<int64_t> dim = std::nullopt) -> Tensor;

/// @}

// =========================================================================
// New shape/transform operations for PyTorch parity
// =========================================================================

/// Flatten to 1D (alias for reshape({-1}))
auto ravel(const Tensor& input) -> Tensor;

/// Reshape a dimension into multiple dimensions (inverse of flatten)
auto unflatten(const Tensor& input, int64_t dim, std::vector<int64_t> sizes) -> Tensor;

/// Horizontal stack: cat along dim 1 (or dim 0 for 1D)
auto hstack(const std::vector<Tensor>& tensors) -> Tensor;

/// Vertical stack: cat along dim 0
auto vstack(const std::vector<Tensor>& tensors) -> Tensor;

/// Depth stack: cat along dim 2
auto dstack(const std::vector<Tensor>& tensors) -> Tensor;

/// Split tensor into sections along dim (generalized split)
auto tensor_split(const Tensor& input, int64_t sections, int64_t dim = 0) -> std::vector<Tensor>;

/// Split tensor at given indices along dim
auto tensor_split(const Tensor& input, std::vector<int64_t> indices, int64_t dim = 0) -> std::vector<Tensor>;

/// Horizontal split
auto hsplit(const Tensor& input, int64_t sections) -> std::vector<Tensor>;

/// Vertical split
auto vsplit(const Tensor& input, int64_t sections) -> std::vector<Tensor>;

/// Depth split
auto dsplit(const Tensor& input, int64_t sections) -> std::vector<Tensor>;

/// Remove a dim and return slices as a vector
auto unbind(const Tensor& input, int64_t dim = 0) -> std::vector<Tensor>;

/// Rotate 90 degrees k times in the plane (dims[0], dims[1])
auto rot90(const Tensor& input, int64_t k = 1, std::vector<int64_t> dims = {0, 1}) -> Tensor;

/// Indices where tensor is nonzero (N x ndim result)
auto argwhere(const Tensor& input) -> Tensor;

/// Flip left-right (flip along dim 1)
auto fliplr(const Tensor& input) -> Tensor;

/// Flip up-down (flip along dim 0)
auto flipud(const Tensor& input) -> Tensor;

/// @}

/**
 * @brief Rearranges elements in a tensor of shape (*, C*r^2, H, W) to (*, C, H*r, W*r).
 *
 * Used in sub-pixel convolution for super-resolution.
 * @param input Tensor of shape (*, C*r^2, H, W)
 * @param upscale_factor Upscale factor r
 * @return Tensor of shape (*, C, H*r, W*r)
 */
auto pixel_shuffle(const Tensor& input, int64_t upscale_factor) -> Tensor;

/**
 * @brief Inverse of pixel_shuffle. Rearranges (*, C, H*r, W*r) to (*, C*r^2, H, W).
 */
auto pixel_unshuffle(const Tensor& input, int64_t downscale_factor) -> Tensor;

/**
 * @brief Rearrange channels by dividing into groups and transposing.
 *
 * Used in ShuffleNet architectures. Input (N, C, H, W) -> reshape to
 * (N, groups, C/groups, H, W) -> transpose dims 1,2 -> flatten back to (N, C, H, W).
 *
 * @param input 4D tensor of shape (N, C, H, W)
 * @param groups Number of groups (must divide C evenly)
 * @return Tensor with shuffled channels, same shape as input
 */
auto channel_shuffle(const Tensor& input, int64_t groups) -> Tensor;

/** @} */ // end of tensor_transform group

} // namespace tenzor

namespace tenzor {
namespace ops {
// Convenience namespace alias for common operations
using tenzor::unsqueeze;
using tenzor::cat;
using tenzor::reshape;
using tenzor::transpose;
using tenzor::permute;
using tenzor::squeeze;
using tenzor::flatten;
using tenzor::stack;
using tenzor::roll;
using tenzor::triu;
using tenzor::tril;
using tenzor::diag;
using tenzor::trace;
using tenzor::flip;
} // namespace ops
} // namespace tenzor
