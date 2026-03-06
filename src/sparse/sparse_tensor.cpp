#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tenzor {

auto SparseTensor::sparse_coo(const Tensor& indices, const Tensor& values,
                               std::vector<int64_t> shape) -> SparseTensor {
    if (indices.ndim() != 2) {
        throw std::runtime_error("sparse_coo: indices must be 2D (sparse_dim, nnz)");
    }
    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("sparse_coo: indices must be Int64");
    }

    int64_t sparse_dim = indices.shape()[0];
    int64_t nnz = indices.shape()[1];

    // Validate sparse_dim matches shape
    if (sparse_dim != static_cast<int64_t>(shape.size())) {
        throw std::runtime_error("sparse_coo: indices.shape()[0] (" +
            std::to_string(sparse_dim) + ") must match len(shape) (" +
            std::to_string(shape.size()) + ")");
    }

    // Validate nnz consistency between indices and values
    if (values.numel() > 0 && values.shape()[0] != nnz) {
        throw std::runtime_error("sparse_coo: values.shape()[0] (" +
            std::to_string(values.shape()[0]) + ") must match indices.shape()[1] (" +
            std::to_string(nnz) + ")");
    }

    // Bounds-check indices on CPU
    if (indices.device().type == Device::Type::CPU && nnz > 0) {
        auto* idx_ptr = indices.data<int64_t>();
        for (int64_t d = 0; d < sparse_dim; ++d) {
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t idx = idx_ptr[d * nnz + i];
                if (idx < 0 || idx >= shape[d]) {
                    throw std::runtime_error("sparse_coo: index " + std::to_string(idx) +
                        " out of bounds for dimension " + std::to_string(d) +
                        " with size " + std::to_string(shape[d]));
                }
            }
        }
    }

    SparseTensor s;
    s.layout_ = SparseLayout::COO;
    s.shape_ = std::move(shape);
    s.indices_ = indices;
    s.values_ = values;
    s.nnz_ = nnz;
    s.sparse_dim_ = sparse_dim;
    s.dense_dim_ = values.ndim() > 1 ? values.ndim() - 1 : 0;
    s.coalesced_ = false;
    return s;
}

auto SparseTensor::sparse_csr(const Tensor& crow_indices, const Tensor& col_indices,
                               const Tensor& values, std::vector<int64_t> shape) -> SparseTensor {
    if (shape.size() != 2) {
        throw std::runtime_error("sparse_csr: only 2D tensors supported");
    }
    if (crow_indices.dtype() != DType::Int64 || col_indices.dtype() != DType::Int64) {
        throw std::runtime_error("sparse_csr: indices must be Int64");
    }

    int64_t nrows = shape[0];
    int64_t ncols = shape[1];
    int64_t nnz = values.numel();

    // Validate crow_indices length
    if (crow_indices.shape()[0] != nrows + 1) {
        throw std::runtime_error("sparse_csr: crow_indices length (" +
            std::to_string(crow_indices.shape()[0]) + ") must be nrows+1 (" +
            std::to_string(nrows + 1) + ")");
    }

    // Validate col_indices and values consistency
    if (col_indices.shape()[0] != nnz) {
        throw std::runtime_error("sparse_csr: col_indices length (" +
            std::to_string(col_indices.shape()[0]) + ") must match values length (" +
            std::to_string(nnz) + ")");
    }

    // Bounds-check on CPU
    if (crow_indices.device().type == Device::Type::CPU && nnz > 0) {
        auto* crow_ptr = crow_indices.data<int64_t>();
        auto* col_ptr = col_indices.data<int64_t>();

        // Monotonicity check on crow_indices
        for (int64_t i = 0; i < nrows; ++i) {
            if (crow_ptr[i] > crow_ptr[i + 1]) {
                throw std::runtime_error("sparse_csr: crow_indices must be monotonically non-decreasing");
            }
        }
        if (crow_ptr[0] != 0) {
            throw std::runtime_error("sparse_csr: crow_indices[0] must be 0");
        }
        if (crow_ptr[nrows] != nnz) {
            throw std::runtime_error("sparse_csr: crow_indices[-1] (" +
                std::to_string(crow_ptr[nrows]) + ") must equal nnz (" +
                std::to_string(nnz) + ")");
        }

        // Column bounds check
        for (int64_t i = 0; i < nnz; ++i) {
            if (col_ptr[i] < 0 || col_ptr[i] >= ncols) {
                throw std::runtime_error("sparse_csr: col_index " + std::to_string(col_ptr[i]) +
                    " out of bounds for ncols=" + std::to_string(ncols));
            }
        }
    }

    SparseTensor s;
    s.layout_ = SparseLayout::CSR;
    s.shape_ = std::move(shape);
    s.crow_indices_ = crow_indices;
    s.col_indices_ = col_indices;
    s.values_ = values;
    s.nnz_ = nnz;
    s.sparse_dim_ = 2;
    s.dense_dim_ = 0;
    s.coalesced_ = true;  // CSR is always sorted by row
    return s;
}

auto SparseTensor::from_dense(const Tensor& dense, SparseLayout layout) -> SparseTensor {
    auto dense_cont = dense.contiguous().to(Device::cpu());
    auto shape = std::vector<int64_t>(dense.shape().begin(), dense.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t numel = dense_cont.numel();

    // Count non-zero elements
    std::vector<int64_t> nz_flat_indices;
    nz_flat_indices.reserve(numel / 4); // heuristic

    // Scan for non-zeros (using Float32/Float64 comparison)
    auto scan_nonzeros = [&]<typename T>(const T* data) {
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != T(0)) {
                nz_flat_indices.push_back(i);
            }
        }
    };

    switch (dense_cont.dtype()) {
        case DType::Float32: scan_nonzeros(dense_cont.data<float>()); break;
        case DType::Float64: scan_nonzeros(dense_cont.data<double>()); break;
        case DType::Int32:   scan_nonzeros(dense_cont.data<int32_t>()); break;
        case DType::Int64:   scan_nonzeros(dense_cont.data<int64_t>()); break;
        case DType::Int8:    scan_nonzeros(dense_cont.data<int8_t>()); break;
        case DType::UInt8:   scan_nonzeros(dense_cont.data<uint8_t>()); break;
        default:
            throw std::runtime_error("from_dense: unsupported dtype");
    }

    int64_t nnz = static_cast<int64_t>(nz_flat_indices.size());

    // Build COO indices (ndim x nnz) and values (nnz)
    Tensor indices({ndim, nnz}, DType::Int64, Device::cpu());
    Tensor values({nnz}, dense_cont.dtype(), Device::cpu());

    auto* idx_ptr = indices.data<int64_t>();

    // Convert flat indices to multi-dimensional indices
    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    for (int64_t j = 0; j < nnz; ++j) {
        int64_t flat = nz_flat_indices[j];
        for (int64_t d = 0; d < ndim; ++d) {
            idx_ptr[d * nnz + j] = flat / strides[d];
            flat %= strides[d];
        }
    }

    // Copy non-zero values
    auto copy_values = [&]<typename T>(const T* src, T* dst) {
        for (int64_t j = 0; j < nnz; ++j) {
            dst[j] = src[nz_flat_indices[j]];
        }
    };

    switch (dense_cont.dtype()) {
        case DType::Float32: copy_values(dense_cont.data<float>(), values.data<float>()); break;
        case DType::Float64: copy_values(dense_cont.data<double>(), values.data<double>()); break;
        case DType::Int32:   copy_values(dense_cont.data<int32_t>(), values.data<int32_t>()); break;
        case DType::Int64:   copy_values(dense_cont.data<int64_t>(), values.data<int64_t>()); break;
        case DType::Int8:    copy_values(dense_cont.data<int8_t>(), values.data<int8_t>()); break;
        case DType::UInt8:   copy_values(dense_cont.data<uint8_t>(), values.data<uint8_t>()); break;
        default: break;
    }

    // Move back to original device if needed
    if (dense.device() != Device::cpu()) {
        indices = indices.to(dense.device());
        values = values.to(dense.device());
    }

    auto result = sparse_coo(indices, values, shape);

    // Convert to requested layout if not COO
    if (layout == SparseLayout::CSR) return result.to_csr();
    if (layout == SparseLayout::CSC) return result.to_csc();

    return result;
}

auto SparseTensor::to_dense() const -> Tensor {
    auto result = zeros(shape_, values_.dtype(), values_.device());

    if (layout_ == SparseLayout::COO) {
        auto idx = indices_.contiguous();
        auto vals = values_.contiguous();
        auto* idx_ptr = idx.data<int64_t>();
        int64_t nnz_count = nnz_;

        if (sparse_dim_ == 2 && shape_.size() == 2) {
            // 2D COO -> Dense
            int64_t ncols = shape_[1];
            if (vals.dtype() == DType::Float32) {
                auto* v = vals.data<float>();
                auto* r = result.data<float>();
                for (int64_t i = 0; i < nnz_count; ++i) {
                    int64_t row = idx_ptr[i];
                    int64_t col = idx_ptr[nnz_count + i];
                    r[row * ncols + col] += v[i];
                }
            } else if (vals.dtype() == DType::Float64) {
                auto* v = vals.data<double>();
                auto* r = result.data<double>();
                for (int64_t i = 0; i < nnz_count; ++i) {
                    int64_t row = idx_ptr[i];
                    int64_t col = idx_ptr[nnz_count + i];
                    r[row * ncols + col] += v[i];
                }
            }
        }
    } else if (layout_ == SparseLayout::CSR) {
        auto crow = crow_indices_.contiguous();
        auto col = col_indices_.contiguous();
        auto vals = values_.contiguous();
        auto* crow_ptr = crow.data<int64_t>();
        auto* col_ptr = col.data<int64_t>();
        int64_t nrows = shape_[0];
        int64_t ncols = shape_[1];

        if (vals.dtype() == DType::Float32) {
            auto* v = vals.data<float>();
            auto* r = result.data<float>();
            for (int64_t row = 0; row < nrows; ++row) {
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    r[row * ncols + col_ptr[j]] += v[j];
                }
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* v = vals.data<double>();
            auto* r = result.data<double>();
            for (int64_t row = 0; row < nrows; ++row) {
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    r[row * ncols + col_ptr[j]] += v[j];
                }
            }
        }
    } else if (layout_ == SparseLayout::CSC) {
        auto ccol = ccol_indices_.contiguous();
        auto row = row_indices_.contiguous();
        auto vals = values_.contiguous();
        auto* ccol_ptr = ccol.data<int64_t>();
        auto* row_ptr = row.data<int64_t>();
        int64_t ncols = shape_[1];

        if (vals.dtype() == DType::Float32) {
            auto* v = vals.data<float>();
            auto* r = result.data<float>();
            for (int64_t col = 0; col < ncols; ++col) {
                for (int64_t j = ccol_ptr[col]; j < ccol_ptr[col + 1]; ++j) {
                    r[row_ptr[j] * ncols + col] += v[j];
                }
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* v = vals.data<double>();
            auto* r = result.data<double>();
            for (int64_t col = 0; col < ncols; ++col) {
                for (int64_t j = ccol_ptr[col]; j < ccol_ptr[col + 1]; ++j) {
                    r[row_ptr[j] * ncols + col] += v[j];
                }
            }
        }
    } else if (layout_ == SparseLayout::BSR) {
        auto rp = bsr_row_ptr_.contiguous();
        auto ci = bsr_col_ind_.contiguous();
        auto vals = values_.contiguous();
        auto* rp_ptr = rp.data<int64_t>();
        auto* ci_ptr = ci.data<int64_t>();
        auto [bh, bw] = block_size_;
        int64_t nrows = shape_[0];
        int64_t ncols = shape_[1];
        int64_t nblockrows = (nrows + bh - 1) / bh;

        if (vals.dtype() == DType::Float32) {
            auto* v = vals.data<float>();
            auto* r = result.data<float>();
            for (int64_t br = 0; br < nblockrows; ++br) {
                for (int64_t j = rp_ptr[br]; j < rp_ptr[br + 1]; ++j) {
                    int64_t bc = ci_ptr[j];
                    for (int64_t bi = 0; bi < bh; ++bi) {
                        for (int64_t bj = 0; bj < bw; ++bj) {
                            int64_t row = br * bh + bi;
                            int64_t col = bc * bw + bj;
                            if (row < nrows && col < ncols) {
                                r[row * ncols + col] += v[j * bh * bw + bi * bw + bj];
                            }
                        }
                    }
                }
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* v = vals.data<double>();
            auto* r = result.data<double>();
            for (int64_t br = 0; br < nblockrows; ++br) {
                for (int64_t j = rp_ptr[br]; j < rp_ptr[br + 1]; ++j) {
                    int64_t bc = ci_ptr[j];
                    for (int64_t bi = 0; bi < bh; ++bi) {
                        for (int64_t bj = 0; bj < bw; ++bj) {
                            int64_t row = br * bh + bi;
                            int64_t col = bc * bw + bj;
                            if (row < nrows && col < ncols) {
                                r[row * ncols + col] += v[j * bh * bw + bi * bw + bj];
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

auto SparseTensor::to_coo() const -> SparseTensor {
    if (layout_ == SparseLayout::COO) return *this;

    if (layout_ == SparseLayout::CSR) {
        // CSR -> COO
        auto crow = crow_indices_.contiguous();
        auto col = col_indices_.contiguous();
        auto* crow_ptr = crow.data<int64_t>();
        auto* col_ptr = col.data<int64_t>();
        int64_t nrows = shape_[0];

        auto row_indices = Tensor({nnz_}, DType::Int64, values_.device());
        auto* row_ptr = row_indices.data<int64_t>();
        for (int64_t row = 0; row < nrows; ++row) {
            for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                row_ptr[j] = row;
            }
        }

        auto indices = Tensor({2, nnz_}, DType::Int64, values_.device());
        auto* idx_ptr = indices.data<int64_t>();
        std::memcpy(idx_ptr, row_ptr, nnz_ * sizeof(int64_t));
        std::memcpy(idx_ptr + nnz_, col_ptr, nnz_ * sizeof(int64_t));

        return sparse_coo(indices, values_, shape_);
    } else if (layout_ == SparseLayout::CSC) {
        // CSC -> COO
        auto ccol = ccol_indices_.contiguous();
        auto row = row_indices_.contiguous();
        auto* ccol_ptr = ccol.data<int64_t>();
        auto* row_ptr = row.data<int64_t>();
        int64_t ncols = shape_[1];

        auto col_indices = Tensor({nnz_}, DType::Int64, values_.device());
        auto* col_ptr = col_indices.data<int64_t>();
        for (int64_t col = 0; col < ncols; ++col) {
            for (int64_t j = ccol_ptr[col]; j < ccol_ptr[col + 1]; ++j) {
                col_ptr[j] = col;
            }
        }

        auto indices = Tensor({2, nnz_}, DType::Int64, values_.device());
        auto* idx_ptr = indices.data<int64_t>();
        std::memcpy(idx_ptr, row_ptr, nnz_ * sizeof(int64_t));
        std::memcpy(idx_ptr + nnz_, col_ptr, nnz_ * sizeof(int64_t));

        return sparse_coo(indices, values_, shape_);
    } else if (layout_ == SparseLayout::BSR) {
        // BSR -> COO: expand blocks into individual elements
        // Easier to go via dense for correctness
        return to_sparse(to_dense());
    }

    throw std::runtime_error("to_coo: unsupported layout");
}

auto SparseTensor::to_csr() const -> SparseTensor {
    if (layout_ == SparseLayout::CSR) return *this;
    if (shape_.size() != 2) {
        throw std::runtime_error("to_csr: only 2D sparse tensors supported");
    }

    // COO -> CSR: sort by row, build crow_indices
    auto coo = coalesce();
    auto idx = coo.indices_.contiguous();
    auto vals = coo.values_.contiguous();
    auto* idx_ptr = idx.data<int64_t>();
    int64_t nrows = shape_[0];
    int64_t coalesced_nnz = coo.nnz();

    auto crow = Tensor({nrows + 1}, DType::Int64, values_.device());
    auto col = Tensor({coalesced_nnz}, DType::Int64, values_.device());
    auto* crow_ptr = crow.data<int64_t>();
    auto* col_ptr = col.data<int64_t>();

    std::memset(crow_ptr, 0, (nrows + 1) * sizeof(int64_t));

    // Count elements per row
    for (int64_t i = 0; i < coalesced_nnz; ++i) {
        crow_ptr[idx_ptr[i] + 1]++;
    }
    // Prefix sum
    for (int64_t i = 0; i < nrows; ++i) {
        crow_ptr[i + 1] += crow_ptr[i];
    }
    // Fill col_indices
    for (int64_t i = 0; i < coalesced_nnz; ++i) {
        col_ptr[i] = idx_ptr[coalesced_nnz + i];
    }

    return sparse_csr(crow, col, vals, shape_);
}

auto SparseTensor::coalesce() const -> SparseTensor {
    if (coalesced_ || layout_ != SparseLayout::COO) return *this;
    if (nnz_ == 0) {
        SparseTensor result = *this;
        result.coalesced_ = true;
        return result;
    }

    auto idx = indices_.contiguous();
    auto vals = values_.contiguous();
    auto* idx_ptr = idx.data<int64_t>();

    // Compute compound (linearized row-major) key for each element.
    // key[i] = idx[0,i] * stride[0] + idx[1,i] * stride[1] + ...
    // This converts the per-element multi-dim comparison in the sort into
    // a single int64_t comparison, turning O(sparse_dim) per compare into O(1).
    //
    // Compute dimension strides (row-major): stride[d] = product of shape[d+1..end]
    std::vector<int64_t> strides(sparse_dim_);
    if (sparse_dim_ > 0) {
        strides[sparse_dim_ - 1] = 1;
        for (int64_t d = sparse_dim_ - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape_[d + 1];
        }
    }

    // Build compound keys
    std::vector<int64_t> keys(nnz_);
    for (int64_t i = 0; i < nnz_; ++i) {
        int64_t key = 0;
        for (int64_t d = 0; d < sparse_dim_; ++d) {
            key += idx_ptr[d * nnz_ + i] * strides[d];
        }
        keys[i] = key;
    }

    // Create sort permutation by compound key (O(1) comparison per pair)
    std::vector<int64_t> perm(nnz_);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        return keys[a] < keys[b];
    });

    // Single-pass merge: walk sorted permutation, detect duplicates via key equality.
    // Pre-allocate output vectors sized to nnz_ (worst case = no duplicates).
    std::vector<int64_t> out_indices(sparse_dim_ * nnz_);
    std::vector<int64_t> group_starts;
    std::vector<int64_t> group_ends;
    group_starts.reserve(nnz_);
    group_ends.reserve(nnz_);

    int64_t new_nnz = 0;
    for (int64_t i = 0; i < nnz_;) {
        int64_t key_i = keys[perm[i]];
        int64_t j = i + 1;
        while (j < nnz_ && keys[perm[j]] == key_i) {
            ++j;
        }
        // perm[i..j) are duplicates — store the index tuple from perm[i]
        for (int64_t d = 0; d < sparse_dim_; ++d) {
            out_indices[d * nnz_ + new_nnz] = idx_ptr[d * nnz_ + perm[i]];
        }
        group_starts.push_back(i);
        group_ends.push_back(j);
        ++new_nnz;
        i = j;
    }

    // Build new indices tensor (compact from pre-allocated buffer)
    auto new_indices = Tensor({sparse_dim_, new_nnz}, DType::Int64, values_.device());
    auto* ni_ptr = new_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim_; ++d) {
        std::memcpy(ni_ptr + d * new_nnz,
                    out_indices.data() + d * nnz_,
                    new_nnz * sizeof(int64_t));
    }

    // Build new values (sum duplicates)
    auto new_values = zeros({new_nnz}, vals.dtype(), vals.device());
    if (vals.dtype() == DType::Float32) {
        auto* vp = vals.data<float>();
        auto* nvp = new_values.data<float>();
        for (int64_t g = 0; g < new_nnz; ++g) {
            float sum = 0;
            for (int64_t k = group_starts[g]; k < group_ends[g]; ++k) {
                sum += vp[perm[k]];
            }
            nvp[g] = sum;
        }
    } else if (vals.dtype() == DType::Float64) {
        auto* vp = vals.data<double>();
        auto* nvp = new_values.data<double>();
        for (int64_t g = 0; g < new_nnz; ++g) {
            double sum = 0;
            for (int64_t k = group_starts[g]; k < group_ends[g]; ++k) {
                sum += vp[perm[k]];
            }
            nvp[g] = sum;
        }
    }

    auto result = sparse_coo(new_indices, new_values, shape_);
    result.coalesced_ = true;
    return result;
}

auto SparseTensor::to(Device device) const -> SparseTensor {
    SparseTensor result = *this;
    result.values_ = values_.to(device);
    switch (layout_) {
        case SparseLayout::COO:
            result.indices_ = indices_.to(device);
            break;
        case SparseLayout::CSR:
            result.crow_indices_ = crow_indices_.to(device);
            result.col_indices_ = col_indices_.to(device);
            break;
        case SparseLayout::CSC:
            result.ccol_indices_ = ccol_indices_.to(device);
            result.row_indices_ = row_indices_.to(device);
            break;
        case SparseLayout::BSR:
            result.bsr_row_ptr_ = bsr_row_ptr_.to(device);
            result.bsr_col_ind_ = bsr_col_ind_.to(device);
            break;
    }
    return result;
}

auto SparseTensor::sparse_csc(const Tensor& ccol_indices, const Tensor& row_indices,
                               const Tensor& values, std::vector<int64_t> shape) -> SparseTensor {
    if (shape.size() != 2) {
        throw std::runtime_error("sparse_csc: only 2D tensors supported");
    }
    if (ccol_indices.dtype() != DType::Int64 || row_indices.dtype() != DType::Int64) {
        throw std::runtime_error("sparse_csc: indices must be Int64");
    }

    int64_t nrows = shape[0];
    int64_t ncols = shape[1];
    int64_t nnz = values.numel();

    if (ccol_indices.shape()[0] != ncols + 1) {
        throw std::runtime_error("sparse_csc: ccol_indices length (" +
            std::to_string(ccol_indices.shape()[0]) + ") must be ncols+1 (" +
            std::to_string(ncols + 1) + ")");
    }
    if (row_indices.shape()[0] != nnz) {
        throw std::runtime_error("sparse_csc: row_indices length must match values length");
    }

    // Bounds-check on CPU
    if (ccol_indices.device().type == Device::Type::CPU && nnz > 0) {
        auto* ccol_ptr = ccol_indices.data<int64_t>();
        auto* row_ptr = row_indices.data<int64_t>();

        for (int64_t i = 0; i < ncols; ++i) {
            if (ccol_ptr[i] > ccol_ptr[i + 1]) {
                throw std::runtime_error("sparse_csc: ccol_indices must be monotonically non-decreasing");
            }
        }
        if (ccol_ptr[0] != 0 || ccol_ptr[ncols] != nnz) {
            throw std::runtime_error("sparse_csc: ccol_indices[0] must be 0 and ccol_indices[-1] must equal nnz");
        }

        for (int64_t i = 0; i < nnz; ++i) {
            if (row_ptr[i] < 0 || row_ptr[i] >= nrows) {
                throw std::runtime_error("sparse_csc: row_index out of bounds");
            }
        }
    }

    SparseTensor s;
    s.layout_ = SparseLayout::CSC;
    s.shape_ = std::move(shape);
    s.ccol_indices_ = ccol_indices;
    s.row_indices_ = row_indices;
    s.values_ = values;
    s.nnz_ = nnz;
    s.sparse_dim_ = 2;
    s.dense_dim_ = 0;
    s.coalesced_ = true;
    return s;
}

auto SparseTensor::sparse_bsr(const Tensor& bsr_row_ptr, const Tensor& bsr_col_ind,
                               const Tensor& values, std::vector<int64_t> shape,
                               std::pair<int64_t, int64_t> block_size) -> SparseTensor {
    if (shape.size() != 2) {
        throw std::runtime_error("sparse_bsr: only 2D tensors supported");
    }
    if (bsr_row_ptr.dtype() != DType::Int64 || bsr_col_ind.dtype() != DType::Int64) {
        throw std::runtime_error("sparse_bsr: indices must be Int64");
    }

    auto [bh, bw] = block_size;
    if (bh <= 0 || bw <= 0) {
        throw std::runtime_error("sparse_bsr: block_size must be positive");
    }

    int64_t nrows = shape[0];
    int64_t ncols = shape[1];
    int64_t nblockrows = (nrows + bh - 1) / bh;
    int64_t nblockcols = (ncols + bw - 1) / bw;
    int64_t nnzb = bsr_col_ind.shape()[0];

    if (bsr_row_ptr.shape()[0] != nblockrows + 1) {
        throw std::runtime_error("sparse_bsr: bsr_row_ptr length must be nblockrows+1");
    }
    if (values.ndim() != 3 || values.shape()[1] != bh || values.shape()[2] != bw) {
        throw std::runtime_error("sparse_bsr: values must have shape (nnzb, block_h, block_w)");
    }
    if (values.shape()[0] != nnzb) {
        throw std::runtime_error("sparse_bsr: values.shape()[0] must match bsr_col_ind length");
    }

    // Bounds-check on CPU
    if (bsr_row_ptr.device().type == Device::Type::CPU && nnzb > 0) {
        auto* rp = bsr_row_ptr.data<int64_t>();
        auto* ci = bsr_col_ind.data<int64_t>();

        if (rp[0] != 0 || rp[nblockrows] != nnzb) {
            throw std::runtime_error("sparse_bsr: bsr_row_ptr[0] must be 0, bsr_row_ptr[-1] must equal nnzb");
        }
        for (int64_t i = 0; i < nblockrows; ++i) {
            if (rp[i] > rp[i + 1]) {
                throw std::runtime_error("sparse_bsr: bsr_row_ptr must be monotonically non-decreasing");
            }
        }
        for (int64_t i = 0; i < nnzb; ++i) {
            if (ci[i] < 0 || ci[i] >= nblockcols) {
                throw std::runtime_error("sparse_bsr: block column index out of bounds");
            }
        }
    }

    SparseTensor s;
    s.layout_ = SparseLayout::BSR;
    s.shape_ = std::move(shape);
    s.bsr_row_ptr_ = bsr_row_ptr;
    s.bsr_col_ind_ = bsr_col_ind;
    s.values_ = values;
    s.nnz_ = nnzb * bh * bw;  // Total non-zero elements (including block fill)
    s.sparse_dim_ = 2;
    s.dense_dim_ = 0;
    s.coalesced_ = true;
    s.block_size_ = block_size;
    return s;
}

auto SparseTensor::to_csc() const -> SparseTensor {
    if (layout_ == SparseLayout::CSC) return *this;
    if (shape_.size() != 2) {
        throw std::runtime_error("to_csc: only 2D sparse tensors supported");
    }

    // Convert to COO first, then build CSC
    auto coo = to_coo();
    auto idx = coo.indices_.contiguous();
    auto vals = coo.values_.contiguous();
    auto* idx_ptr = idx.data<int64_t>();
    int64_t ncols = shape_[1];
    int64_t coo_nnz = coo.nnz();

    // Sort by column (then by row within each column)
    std::vector<int64_t> perm(coo_nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        int64_t col_a = idx_ptr[coo_nnz + a];
        int64_t col_b = idx_ptr[coo_nnz + b];
        if (col_a != col_b) return col_a < col_b;
        return idx_ptr[a] < idx_ptr[b];
    });

    // Build ccol_indices
    auto ccol = Tensor({ncols + 1}, DType::Int64, values_.device());
    auto row = Tensor({coo_nnz}, DType::Int64, values_.device());
    auto* ccol_ptr = ccol.data<int64_t>();
    auto* row_ptr = row.data<int64_t>();

    std::memset(ccol_ptr, 0, (ncols + 1) * sizeof(int64_t));

    // Count per-column
    for (int64_t i = 0; i < coo_nnz; ++i) {
        ccol_ptr[idx_ptr[coo_nnz + perm[i]] + 1]++;
    }
    // Prefix sum
    for (int64_t c = 0; c < ncols; ++c) {
        ccol_ptr[c + 1] += ccol_ptr[c];
    }
    // Fill row_indices in sorted order
    for (int64_t i = 0; i < coo_nnz; ++i) {
        row_ptr[i] = idx_ptr[perm[i]];
    }

    // Reorder values
    auto new_vals = zeros({coo_nnz}, vals.dtype(), vals.device());
    if (vals.dtype() == DType::Float32) {
        auto* vp = vals.data<float>();
        auto* nvp = new_vals.data<float>();
        for (int64_t i = 0; i < coo_nnz; ++i) nvp[i] = vp[perm[i]];
    } else if (vals.dtype() == DType::Float64) {
        auto* vp = vals.data<double>();
        auto* nvp = new_vals.data<double>();
        for (int64_t i = 0; i < coo_nnz; ++i) nvp[i] = vp[perm[i]];
    }

    return sparse_csc(ccol, row, new_vals, shape_);
}

auto SparseTensor::to_bsr(std::pair<int64_t, int64_t> block_size) const -> SparseTensor {
    if (layout_ == SparseLayout::BSR && block_size_ == block_size) return *this;
    if (shape_.size() != 2) {
        throw std::runtime_error("to_bsr: only 2D sparse tensors supported");
    }

    auto [bh, bw] = block_size;
    int64_t nrows = shape_[0];
    int64_t ncols = shape_[1];
    int64_t nblockrows = (nrows + bh - 1) / bh;
    int64_t nblockcols = (ncols + bw - 1) / bw;

    // Convert to dense first, then extract blocks
    auto dense = to_dense();
    auto cont = dense.contiguous();

    // Find non-zero blocks
    std::vector<int64_t> block_rows, block_cols;
    std::vector<std::vector<float>> block_vals_f32;
    std::vector<std::vector<double>> block_vals_f64;

    bool is_f32 = (cont.dtype() == DType::Float32);

    for (int64_t br = 0; br < nblockrows; ++br) {
        for (int64_t bc = 0; bc < nblockcols; ++bc) {
            // Check if block is non-zero
            bool has_nonzero = false;
            for (int64_t i = 0; i < bh && !has_nonzero; ++i) {
                for (int64_t j = 0; j < bw && !has_nonzero; ++j) {
                    int64_t r = br * bh + i;
                    int64_t c = bc * bw + j;
                    if (r < nrows && c < ncols) {
                        if (is_f32) {
                            if (cont.data<float>()[r * ncols + c] != 0.0f) has_nonzero = true;
                        } else {
                            if (cont.data<double>()[r * ncols + c] != 0.0) has_nonzero = true;
                        }
                    }
                }
            }

            if (has_nonzero) {
                block_rows.push_back(br);
                block_cols.push_back(bc);

                if (is_f32) {
                    std::vector<float> blk(bh * bw, 0.0f);
                    for (int64_t i = 0; i < bh; ++i) {
                        for (int64_t j = 0; j < bw; ++j) {
                            int64_t r = br * bh + i;
                            int64_t c = bc * bw + j;
                            if (r < nrows && c < ncols) {
                                blk[i * bw + j] = cont.data<float>()[r * ncols + c];
                            }
                        }
                    }
                    block_vals_f32.push_back(std::move(blk));
                } else {
                    std::vector<double> blk(bh * bw, 0.0);
                    for (int64_t i = 0; i < bh; ++i) {
                        for (int64_t j = 0; j < bw; ++j) {
                            int64_t r = br * bh + i;
                            int64_t c = bc * bw + j;
                            if (r < nrows && c < ncols) {
                                blk[i * bw + j] = cont.data<double>()[r * ncols + c];
                            }
                        }
                    }
                    block_vals_f64.push_back(std::move(blk));
                }
            }
        }
    }

    int64_t nnzb = static_cast<int64_t>(block_cols.size());

    // Build bsr_row_ptr
    auto row_ptr = Tensor({nblockrows + 1}, DType::Int64, cont.device());
    auto col_ind = Tensor({nnzb}, DType::Int64, cont.device());
    auto* rp = row_ptr.data<int64_t>();
    auto* ci = col_ind.data<int64_t>();

    std::memset(rp, 0, (nblockrows + 1) * sizeof(int64_t));
    for (int64_t i = 0; i < nnzb; ++i) {
        rp[block_rows[i] + 1]++;
    }
    for (int64_t i = 0; i < nblockrows; ++i) {
        rp[i + 1] += rp[i];
    }
    for (int64_t i = 0; i < nnzb; ++i) {
        ci[i] = block_cols[i];
    }

    // Build values tensor
    DType vdt = cont.dtype();
    auto values = Tensor({nnzb, bh, bw}, vdt, cont.device());
    if (is_f32) {
        auto* vp = values.data<float>();
        for (int64_t i = 0; i < nnzb; ++i) {
            std::memcpy(vp + i * bh * bw, block_vals_f32[i].data(), bh * bw * sizeof(float));
        }
    } else {
        auto* vp = values.data<double>();
        for (int64_t i = 0; i < nnzb; ++i) {
            std::memcpy(vp + i * bh * bw, block_vals_f64[i].data(), bh * bw * sizeof(double));
        }
    }

    return sparse_bsr(row_ptr, col_ind, values, shape_, block_size);
}

// Free functions

auto to_sparse(const Tensor& dense) -> SparseTensor {
    if (dense.ndim() != 2) {
        throw std::runtime_error("to_sparse: only 2D tensors supported");
    }

    auto cont = dense.contiguous();
    int64_t nrows = cont.shape()[0];
    int64_t ncols = cont.shape()[1];

    // Find nonzero positions
    std::vector<int64_t> row_idx, col_idx;
    std::vector<float> vals_f32;
    std::vector<double> vals_f64;

    if (cont.dtype() == DType::Float32) {
        auto* ptr = cont.data<float>();
        for (int64_t r = 0; r < nrows; ++r) {
            for (int64_t c = 0; c < ncols; ++c) {
                float v = ptr[r * ncols + c];
                if (v != 0.0f) {
                    row_idx.push_back(r);
                    col_idx.push_back(c);
                    vals_f32.push_back(v);
                }
            }
        }
        int64_t nnz = static_cast<int64_t>(vals_f32.size());
        auto indices = Tensor({2, nnz}, DType::Int64, cont.device());
        auto values = Tensor({nnz}, DType::Float32, cont.device());
        auto* ip = indices.data<int64_t>();
        auto* vp = values.data<float>();
        std::memcpy(ip, row_idx.data(), nnz * sizeof(int64_t));
        std::memcpy(ip + nnz, col_idx.data(), nnz * sizeof(int64_t));
        std::memcpy(vp, vals_f32.data(), nnz * sizeof(float));
        return SparseTensor::sparse_coo(indices, values, {nrows, ncols});
    } else if (cont.dtype() == DType::Float64) {
        auto* ptr = cont.data<double>();
        for (int64_t r = 0; r < nrows; ++r) {
            for (int64_t c = 0; c < ncols; ++c) {
                double v = ptr[r * ncols + c];
                if (v != 0.0) {
                    row_idx.push_back(r);
                    col_idx.push_back(c);
                    vals_f64.push_back(v);
                }
            }
        }
        int64_t nnz = static_cast<int64_t>(vals_f64.size());
        auto indices = Tensor({2, nnz}, DType::Int64, cont.device());
        auto values = Tensor({nnz}, DType::Float64, cont.device());
        auto* ip = indices.data<int64_t>();
        auto* vp = values.data<double>();
        std::memcpy(ip, row_idx.data(), nnz * sizeof(int64_t));
        std::memcpy(ip + nnz, col_idx.data(), nnz * sizeof(int64_t));
        std::memcpy(vp, vals_f64.data(), nnz * sizeof(double));
        return SparseTensor::sparse_coo(indices, values, {nrows, ncols});
    }
    throw std::runtime_error("to_sparse: unsupported dtype");
}

auto to_sparse_csr(const Tensor& dense) -> SparseTensor {
    return to_sparse(dense).to_csr();
}

auto to_sparse_csc(const Tensor& dense) -> SparseTensor {
    return to_sparse(dense).to_csc();
}

} // namespace tenzor
