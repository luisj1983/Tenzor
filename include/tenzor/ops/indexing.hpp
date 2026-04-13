/**
 * @file indexing.hpp
 * @brief Tensor indexing and selection operations
 *
 * Provides advanced indexing, slicing, masking, and conditional
 * selection operations for tensors.
 */

#pragma once

#include "../core/tensor.hpp"

#include <optional>
#include <variant>
#include <vector>

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

/** @brief Scatter-add: accumulate source elements into input at index positions. */
auto scatter_add(const Tensor& input,
                int64_t dim,
                const Tensor& index,
                const Tensor& src) -> Tensor;

/**
 * @brief Scatter with reduction: scatter source into input at index positions with a reduction.
 *
 * For each element in src, the value is reduced into the output (a clone of input) at the
 * position determined by the index tensor along the given dimension. Supported reductions:
 * "sum", "prod", "mean", "amax", "amin".
 *
 * @param input Destination tensor (cloned before modification)
 * @param dim Dimension along which to scatter
 * @param index Int64 tensor of indices (same shape as src)
 * @param src Source tensor
 * @param reduce Reduction operation: "sum", "prod", "mean", "amax", "amin"
 * @param include_self If true (default), include the existing values in input in the reduction.
 *                     If false, positions that receive scattered values are initialized to the
 *                     reduction identity before applying the reduction.
 * @return New tensor with scattered reductions applied
 */
auto scatter_reduce(const Tensor& input, int64_t dim, const Tensor& index,
                    const Tensor& src, const std::string& reduce,
                    bool include_self = true) -> Tensor;

/** @brief Accumulate source into self at index positions: self[index[i]] += source[i] along dim. */
auto index_add(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor;

/** @brief Copy source into self at index positions: self[index[i]] = source[i] along dim. */
auto index_copy(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor;

/** @brief Fill self at index positions with a scalar value along dim. */
auto index_fill(const Tensor& input, int64_t dim, const Tensor& index, float value) -> Tensor;

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

/**
 * @brief Return indices of non-zero elements.
 *
 * Returns a 2D tensor where each row contains the indices of a non-zero element.
 * For a 1D tensor, returns indices as a (N, 1) tensor.
 * For a multi-dimensional tensor, each row is [dim0_idx, dim1_idx, ...].
 *
 * @param input Input tensor
 * @return Tensor of indices (num_nonzero, ndim) in Int64 format
 */
auto nonzero(const Tensor& input) -> Tensor;

/**
 * @brief Select single index along dimension.
 * @param input Input tensor
 * @param dim Dimension to select from
 * @param index Index to select
 * @return Tensor with selected slice (dimension removed)
 */
auto select(const Tensor& input, int64_t dim, int64_t index) -> Tensor;

/**
 * @brief Narrow (slice) tensor along a dimension.
 * @param input Input tensor
 * @param dim Dimension to narrow
 * @param start Start index
 * @param length Length of the narrow
 * @return Narrowed tensor (view sharing storage)
 */
auto narrow(const Tensor& input, int64_t dim, int64_t start, int64_t length) -> Tensor;

/**
 * @brief NumPy-style advanced (fancy) indexing.
 *
 * Selects elements using integer index tensors. Each non-nullopt index
 * corresponds to one dimension. All non-null index tensors must be
 * broadcastable to a common shape. The output shape is formed by replacing
 * the indexed dimensions with the broadcast shape, with non-indexed
 * (nullopt) dimensions passing through.
 *
 * @param input   Source tensor
 * @param indices Vector of optional index tensors (Int32 or Int64).
 *                nullopt entries act as full-slice (`:`) for that dimension.
 * @return Gathered tensor
 */
auto index(const Tensor& input,
           const std::vector<std::optional<Tensor>>& indices) -> Tensor;

/**
 * @brief In-place advanced indexing assignment.
 *
 * Scatters values into input at positions specified by index tensors.
 * Semantics match NumPy `a[indices] = values`.
 *
 * @param input   Destination tensor (modified in-place)
 * @param indices Vector of optional index tensors (Int32 or Int64)
 * @param values  Values to scatter (must be broadcastable to the indexed shape)
 */
void index_put(Tensor& input,
               const std::vector<std::optional<Tensor>>& indices,
               const Tensor& values);

// ============================================================================
// Extended indexing types: Ellipsis, NewAxis, boolean mask support
// ============================================================================

/** @brief Sentinel type representing `...` (ellipsis) in indexing */
struct Ellipsis {};

/** @brief Sentinel type representing `None`/newaxis in indexing */
struct NewAxis {};

/** @brief Global Ellipsis constant for use in indexing expressions */
inline constexpr Ellipsis ellipsis{};

/** @brief Global NewAxis constant for use in indexing expressions */
inline constexpr NewAxis newaxis{};

/**
 * @brief Extended index element supporting Tensor, Ellipsis, NewAxis, and full-slice.
 *
 * Usage:
 * @code
 * auto result = index_extended(t, {ellipsis, Tensor(idx)});    // t[..., idx]
 * auto result = index_extended(t, {newaxis, nullopt});          // t[None, :]
 * auto result = index_extended(t, {bool_mask});                 // t[mask]
 * @endcode
 */
using IndexElement = std::variant<Tensor, Ellipsis, NewAxis, std::nullopt_t>;

/**
 * @brief Advanced indexing with ellipsis, newaxis, and boolean mask support.
 *
 * Extends the basic index() function with:
 * - Ellipsis (...): Expands to fill remaining dimensions with full-slices
 * - NewAxis (None): Inserts a new dimension of size 1
 * - Boolean Tensor: Converts to integer indices via nonzero()
 *
 * @param input Source tensor
 * @param indices Vector of IndexElement (Tensor, Ellipsis, NewAxis, or nullopt)
 * @return Indexed tensor with appropriate shape
 */
auto index_extended(const Tensor& input,
                    const std::vector<IndexElement>& indices) -> Tensor;

/**
 * @brief Create a one-hot encoded tensor.
 *
 * Converts a tensor of class indices to a one-hot encoded tensor.
 * The input is expected to contain integer class indices in [0, num_classes).
 *
 * @param input   Tensor of class indices (any integer dtype)
 * @param num_classes Total number of classes. If -1, inferred as max(input)+1.
 * @return One-hot tensor of shape (*input.shape(), num_classes) with dtype Float32
 *
 * @code
 * Tensor labels({4}, DType::Int64, Device::cpu());  // e.g. [0, 2, 1, 3]
 * Tensor oh = one_hot(labels, 4);  // shape (4, 4), identity-like
 * @endcode
 */
auto one_hot(const Tensor& input, int64_t num_classes = -1) -> Tensor;

/**
 * @brief Count occurrences of each value in an integer tensor.
 *
 * @param input 1D tensor of non-negative integers
 * @param weights Optional same-length float tensor of weights
 * @param minlength Minimum output size (default 0)
 * @return 1D tensor of size max(max(input)+1, minlength)
 *
 * @code
 * auto idx = Tensor({6}, DType::Int64, Device::cpu());
 * auto counts = bincount(idx);
 * @endcode
 */
auto bincount(const Tensor& input,
              const std::optional<Tensor>& weights = std::nullopt,
              int64_t minlength = 0) -> Tensor;

/**
 * @brief Reduce source into input at specified indices along a dimension.
 *
 * For each element in source, reduces into output (a clone of input) at
 * the position given by index along dim. This is a convenience wrapper
 * around scatter_reduce with PyTorch index_reduce_ semantics.
 *
 * @param input Destination tensor (cloned before modification)
 * @param dim Dimension along which to index
 * @param index 1D Int64 index tensor
 * @param source Source tensor (same shape as input except along dim)
 * @param reduce Reduction: "sum", "prod", "mean", "amax", "amin"
 * @param include_self Include existing input values in reduction (default true)
 * @return New tensor with reductions applied
 */
auto index_reduce(const Tensor& input, int64_t dim, const Tensor& index,
                  const Tensor& source, const std::string& reduce,
                  bool include_self = true) -> Tensor;

/**
 * @brief Gather values from input along dim using indices.
 *
 * Like PyTorch's torch.take_along_dim. For each position in indices,
 * gathers input[..., indices[position], ...] along the specified dim.
 *
 * @param input Source tensor
 * @param indices Index tensor (Int64), same shape as input except along dim
 * @param dim Dimension to gather along
 * @return Tensor with same shape as indices
 */
auto take_along_dim(const Tensor& input, const Tensor& indices, int64_t dim) -> Tensor;

/**
 * @brief Scatter source values into positions where mask is true.
 *
 * Like PyTorch's torch.Tensor.masked_scatter_. Iterates through the mask
 * in order; for each true position, writes the next value from source.
 *
 * @param input Destination tensor (cloned before modification)
 * @param mask Boolean mask tensor (same shape as input)
 * @param source 1D tensor of values to scatter
 * @return New tensor with masked positions filled from source
 */
auto masked_scatter(const Tensor& input, const Tensor& mask, const Tensor& source) -> Tensor;

/** @} */ // end of tensor_indexing group

} // namespace tenzor

namespace tenzor {
namespace ops {
// Convenience namespace alias for common operations
using tenzor::nonzero;
using tenzor::index_select;
using tenzor::select;
using tenzor::gather;
using tenzor::scatter;
using tenzor::scatter_reduce;
using tenzor::masked_select;
using tenzor::masked_fill;
using tenzor::where;
using tenzor::slice;
using tenzor::index;
using tenzor::index_put;
using tenzor::one_hot;
using tenzor::bincount;
using tenzor::index_reduce;
using tenzor::take_along_dim;
using tenzor::masked_scatter;
} // namespace ops
} // namespace tenzor
