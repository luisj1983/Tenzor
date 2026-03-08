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
    CSR,    ///< Compressed Sparse Row: crow_indices (nrows+1) + col_indices (nnz) + values (nnz, *dense_dims)
    CSC,    ///< Compressed Sparse Column: ccol_indices (ncols+1) + row_indices (nnz) + values (nnz, *dense_dims)
    BSR     ///< Block Sparse Row: bsr_row_ptr (nblockrows+1) + bsr_col_ind (nnzb) + values (nnzb, block_h, block_w)
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

    /**
     * @brief Create a CSC sparse tensor (2D only).
     *
     * @param ccol_indices Compressed column indices of shape (ncols + 1)
     * @param row_indices Row indices of shape (nnz)
     * @param values Value tensor of shape (nnz,)
     * @param shape Dense shape {nrows, ncols}
     * @return CSC sparse tensor
     */
    static auto sparse_csc(const Tensor& ccol_indices, const Tensor& row_indices,
                           const Tensor& values, std::vector<int64_t> shape) -> SparseTensor;

    /**
     * @brief Create a BSR sparse tensor (2D only).
     *
     * @param bsr_row_ptr Block row pointers of shape (nblockrows + 1)
     * @param bsr_col_ind Block column indices of shape (nnzb)
     * @param values Block values of shape (nnzb, block_h, block_w)
     * @param shape Dense shape {nrows, ncols}
     * @param block_size Block dimensions {block_h, block_w}
     * @return BSR sparse tensor
     */
    static auto sparse_bsr(const Tensor& bsr_row_ptr, const Tensor& bsr_col_ind,
                           const Tensor& values, std::vector<int64_t> shape,
                           std::pair<int64_t, int64_t> block_size) -> SparseTensor;

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

    // CSC accessors
    auto ccol_indices() const -> const Tensor& { return ccol_indices_; }
    auto row_indices() const -> const Tensor& { return row_indices_; }

    // BSR accessors
    auto bsr_row_ptr() const -> const Tensor& { return bsr_row_ptr_; }
    auto bsr_col_ind() const -> const Tensor& { return bsr_col_ind_; }
    auto block_size() const -> std::pair<int64_t, int64_t> { return block_size_; }

    /**
     * @brief Create sparse tensor from dense tensor (non-zero elements only).
     *
     * @param dense Input dense tensor
     * @param layout Target sparse layout (default: COO)
     * @return Sparse tensor containing only non-zero elements
     */
    static auto from_dense(const Tensor& dense,
                           SparseLayout layout = SparseLayout::COO) -> SparseTensor;

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
     * @brief Convert to CSC format (2D only).
     */
    auto to_csc() const -> SparseTensor;

    /**
     * @brief Convert to BSR format (2D only).
     * @param block_size Block dimensions {block_h, block_w}
     */
    auto to_bsr(std::pair<int64_t, int64_t> block_size) const -> SparseTensor;

    /**
     * @brief Transpose a 2D sparse tensor (swap rows and columns).
     *
     * For COO: swaps row and column indices.
     * For CSR: converts to COO, transposes, converts back to CSR.
     * For CSC: reinterprets as CSR with swapped dimensions.
     *
     * @return Transposed sparse tensor with shape {ncols, nrows}
     * @throws std::runtime_error if tensor is not 2D
     */
    auto transpose() const -> SparseTensor;

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

    // CSC storage (in addition to values_)
    Tensor ccol_indices_;   // shape: (ncols + 1)
    Tensor row_indices_;    // shape: (nnz)

    // BSR storage (in addition to values_)
    Tensor bsr_row_ptr_;    // shape: (nblockrows + 1)
    Tensor bsr_col_ind_;    // shape: (nnzb)
    std::pair<int64_t, int64_t> block_size_{0, 0};
};

// Free functions
auto to_sparse(const Tensor& dense) -> SparseTensor;
auto to_sparse_csr(const Tensor& dense) -> SparseTensor;
auto to_sparse_csc(const Tensor& dense) -> SparseTensor;

} // namespace tenzor
