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
    }

    return result;
}

auto SparseTensor::to_coo() const -> SparseTensor {
    if (layout_ == SparseLayout::COO) return *this;

    // CSR -> COO
    auto crow = crow_indices_.contiguous();
    auto col = col_indices_.contiguous();
    auto* crow_ptr = crow.data<int64_t>();
    auto* col_ptr = col.data<int64_t>();
    int64_t nrows = shape_[0];

    // Build row indices from crow
    auto row_indices = Tensor({nnz_}, DType::Int64, values_.device());
    auto* row_ptr = row_indices.data<int64_t>();
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
            row_ptr[j] = row;
        }
    }

    // Combine into indices tensor (2, nnz)
    auto indices = Tensor({2, nnz_}, DType::Int64, values_.device());
    auto* idx_ptr = indices.data<int64_t>();
    std::memcpy(idx_ptr, row_ptr, nnz_ * sizeof(int64_t));
    std::memcpy(idx_ptr + nnz_, col_ptr, nnz_ * sizeof(int64_t));

    return sparse_coo(indices, values_, shape_);
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

    // Create sort permutation by row-major order of indices
    std::vector<int64_t> perm(nnz_);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        for (int64_t d = 0; d < sparse_dim_; ++d) {
            int64_t ia = idx_ptr[d * nnz_ + a];
            int64_t ib = idx_ptr[d * nnz_ + b];
            if (ia != ib) return ia < ib;
        }
        return false;
    });

    // Merge duplicates
    std::vector<int64_t> new_indices_data;
    new_indices_data.reserve(sparse_dim_ * nnz_);

    // Track unique positions
    std::vector<std::vector<int64_t>> unique_idx_cols(sparse_dim_);
    std::vector<int64_t> merge_groups;

    auto get_idx = [&](int64_t elem, int64_t d) -> int64_t {
        return idx_ptr[d * nnz_ + elem];
    };

    int64_t new_nnz = 0;
    for (int64_t i = 0; i < nnz_;) {
        int64_t j = i + 1;
        while (j < nnz_) {
            bool same = true;
            for (int64_t d = 0; d < sparse_dim_; ++d) {
                if (get_idx(perm[j], d) != get_idx(perm[i], d)) {
                    same = false;
                    break;
                }
            }
            if (!same) break;
            ++j;
        }
        // perm[i..j) are duplicates
        for (int64_t d = 0; d < sparse_dim_; ++d) {
            unique_idx_cols[d].push_back(get_idx(perm[i], d));
        }
        merge_groups.push_back(i);
        merge_groups.push_back(j);
        ++new_nnz;
        i = j;
    }

    // Build new indices
    auto new_indices = Tensor({sparse_dim_, new_nnz}, DType::Int64, values_.device());
    auto* ni_ptr = new_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim_; ++d) {
        for (int64_t i = 0; i < new_nnz; ++i) {
            ni_ptr[d * new_nnz + i] = unique_idx_cols[d][i];
        }
    }

    // Build new values (sum duplicates)
    auto new_values = zeros({new_nnz}, vals.dtype(), vals.device());
    if (vals.dtype() == DType::Float32) {
        auto* vp = vals.data<float>();
        auto* nvp = new_values.data<float>();
        for (int64_t g = 0; g < new_nnz; ++g) {
            int64_t start = merge_groups[g * 2];
            int64_t end = merge_groups[g * 2 + 1];
            float sum = 0;
            for (int64_t k = start; k < end; ++k) {
                sum += vp[perm[k]];
            }
            nvp[g] = sum;
        }
    } else if (vals.dtype() == DType::Float64) {
        auto* vp = vals.data<double>();
        auto* nvp = new_values.data<double>();
        for (int64_t g = 0; g < new_nnz; ++g) {
            int64_t start = merge_groups[g * 2];
            int64_t end = merge_groups[g * 2 + 1];
            double sum = 0;
            for (int64_t k = start; k < end; ++k) {
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
    if (layout_ == SparseLayout::COO) {
        result.indices_ = indices_.to(device);
    } else {
        result.crow_indices_ = crow_indices_.to(device);
        result.col_indices_ = col_indices_.to(device);
    }
    return result;
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

} // namespace tenzor
