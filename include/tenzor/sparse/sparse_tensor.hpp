/**
 * @file sparse_tensor.hpp
 * @brief Sparse tensor class supporting COO and CSR layouts
 *
 * SparseTensor is a separate class from Tensor, optimized for storing
 * data with a high proportion of zeros. Supports COO (Coordinate) and
 * CSR (Compressed Sparse Row) storage formats.
 */

#pragma once

#include "../core/tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace tenzor {

/**
 * @brief Sparse storage layout format.
 */
enum class SparseLayout {
    COO,    ///< Coordinate format: indices (sparse_dim, nnz) + values (nnz, *dense_dims)
    CSR     ///< Compressed Sparse Row: crow_indices (nrows+1) + col_indices (nnz) + values (nnz, *dense_dims)
};

/**
 * @brief Sparse tensor with COO or CSR storage.
 *
 * Unlike dense Tensor, SparseTensor stores only non-zero elements and their
 * positions. This is memory-efficient for matrices with >90% zeros.
 *
 * COO format stores:
 * - indices: Int64 tensor of shape (sparse_dim, nnz)
 * - values: tensor of shape (nnz, *dense_dims)
 *
 * CSR format stores:
 * - crow_indices: Int64 tensor of shape (nrows + 1)
 * - col_indices: Int64 tensor of shape (nnz)
 * - values: tensor of shape (nnz, *dense_dims)
 */
class SparseTensor {
public:
    /**
     * @brief Create a COO sparse tensor.
     *
     * @param indices Index tensor of shape (sparse_dim, nnz)
     * @param values Value tensor of shape (nnz, *dense_dims)
     * @param shape Full dense shape of the sparse tensor
     * @return COO sparse tensor
     */
    static auto sparse_coo(const Tensor& indices, const Tensor& values,
                           std::vector<int64_t> shape) -> SparseTensor;

    /**
     * @brief Create a CSR sparse tensor (2D only).
     *
     * @param crow_indices Compressed row indices of shape (nrows + 1)
     * @param col_indices Column indices of shape (nnz)
     * @param values Value tensor of shape (nnz,)
     * @param shape Dense shape {nrows, ncols}
     * @return CSR sparse tensor
     */
    static auto sparse_csr(const Tensor& crow_indices, const Tensor& col_indices,
                           const Tensor& values, std::vector<int64_t> shape) -> SparseTensor;

    // Accessors
    auto layout() const -> SparseLayout { return layout_; }
    auto shape() const -> const std::vector<int64_t>& { return shape_; }
    auto dtype() const -> DType { return values_.dtype(); }
    auto device() const -> Device { return values_.device(); }
    auto nnz() const -> int64_t { return nnz_; }
    auto sparse_dim() const -> int64_t { return sparse_dim_; }
    auto dense_dim() const -> int64_t { return dense_dim_; }
    auto is_coalesced() const -> bool { return coalesced_; }

    // COO accessors
    auto indices() const -> const Tensor& { return indices_; }
    auto values() const -> const Tensor& { return values_; }

    // CSR accessors
    auto crow_indices() const -> const Tensor& { return crow_indices_; }
    auto col_indices() const -> const Tensor& { return col_indices_; }

    /**
     * @brief Convert to dense tensor.
     */
    auto to_dense() const -> Tensor;

    /**
     * @brief Convert to COO format.
     */
    auto to_coo() const -> SparseTensor;

    /**
     * @brief Convert to CSR format (2D only).
     */
    auto to_csr() const -> SparseTensor;

    /**
     * @brief Coalesce COO tensor: sort indices and merge duplicate entries.
     */
    auto coalesce() const -> SparseTensor;

    /**
     * @brief Transfer to different device.
     */
    auto to(Device device) const -> SparseTensor;

private:
    SparseTensor() = default;

    SparseLayout layout_;
    std::vector<int64_t> shape_;
    int64_t nnz_ = 0;
    int64_t sparse_dim_ = 0;
    int64_t dense_dim_ = 0;
    bool coalesced_ = false;

    // COO storage
    Tensor indices_;   // shape: (sparse_dim, nnz)
    Tensor values_;    // shape: (nnz, *dense_dims)

    // CSR storage (in addition to values_)
    Tensor crow_indices_;   // shape: (nrows + 1)
    Tensor col_indices_;    // shape: (nnz)
};

// Free functions
auto to_sparse(const Tensor& dense) -> SparseTensor;
auto to_sparse_csr(const Tensor& dense) -> SparseTensor;

} // namespace tenzor
