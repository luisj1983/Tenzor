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
using tenzor::masked_select;
using tenzor::masked_fill;
using tenzor::where;
using tenzor::slice;
using tenzor::index;
using tenzor::index_put;
} // namespace ops
} // namespace tenzor
