#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/advanced.hpp"
#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>

// Forward declarations for GPU-native sparse format conversions.
// These symbols live in the backend shared objects (tenzor_backend_cuda.so,
// tenzor_backend_rocm.so) loaded via dlopen. Guarded by compile-time flags
// so the linker doesn't complain when a backend is absent.
#ifdef TENZOR_HAS_CUSPARSE
namespace tenzor::cuda {
SparseTensor cuda_coo_to_csr(const SparseTensor& sparse);
SparseTensor cuda_coalesce(const SparseTensor& sparse);
SparseTensor cuda_coo_to_csc(const SparseTensor& sparse);
} // namespace tenzor::cuda
#endif

#ifdef TENZOR_HAS_ROCSPARSE
namespace tenzor::rocm {
SparseTensor rocm_coo_to_csr(const SparseTensor& sparse);
SparseTensor rocm_coalesce(const SparseTensor& sparse);
SparseTensor rocm_coo_to_csc(const SparseTensor& sparse);
} // namespace tenzor::rocm
#endif

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
    // Device-aware implementation: nonzero + index_select on the original
    // device. The previous implementation pulled the full dense tensor to
    // host and host-scanned for nonzeros — a real CPU compute fallback for
    // every GPU sparse user.
    auto dense_cont = dense.contiguous();
    auto shape_span = dense_cont.shape();
    auto shape = std::vector<int64_t>(shape_span.begin(), shape_span.end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t numel = dense_cont.numel();
    Device dev = dense_cont.device();

    auto build_empty = [&]() -> SparseTensor {
        Tensor empty_indices({ndim, int64_t(0)}, DType::Int64, dev);
        Tensor empty_values({0}, dense_cont.dtype(), dev);
        auto result = sparse_coo(empty_indices, empty_values, shape);
        if (layout == SparseLayout::CSR) return result.to_csr();
        if (layout == SparseLayout::CSC) return result.to_csc();
        return result;
    };

    if (numel == 0) {
        return build_empty();
    }

    // mask = (dense != 0). Compare against a zero scalar of matching dtype.
    Tensor zero_scalar = zeros({}, dense_cont.dtype(), dev);
    Tensor mask = ne(dense_cont, zero_scalar);  // shape == dense shape, Bool

    // Multi-dim coordinates of nonzero entries: shape (nnz, ndim) Int64.
    Tensor coords = nonzero(mask);
    int64_t nnz = coords.shape()[0];
    if (nnz == 0) {
        return build_empty();
    }

    // Compute strides (host-side scalars) and use them to fold coords into
    // a flat index per row of `coords`. Each call below stays on `dev`.
    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    Tensor flat_idx = zeros({nnz}, DType::Int64, dev);
    for (int64_t d = 0; d < ndim; ++d) {
        // coord_d shape (nnz,) — slice column d of (nnz, ndim) coords.
        Tensor coord_d = reshape(slice(coords, /*dim=*/1, d, d + 1), {nnz});
        if (strides[d] != 1) {
            Tensor stride_t = full({nnz}, static_cast<double>(strides[d]),
                                   DType::Int64, dev);
            coord_d = mul(coord_d, stride_t);
        }
        flat_idx = add(flat_idx, coord_d);
    }

    // Gather values from the flattened dense tensor.
    Tensor dense_flat = reshape(dense_cont, {numel});
    Tensor values = index_select(dense_flat, /*dim=*/0, flat_idx);

    // COO indices want shape (ndim, nnz) — transpose (nnz, ndim).
    Tensor indices = ::tenzor::transpose(coords, 0, 1).contiguous();

    auto result = sparse_coo(indices, values, shape);
    // `nonzero` produces row-major-ordered, unique multi-dim coordinates,
    // so the resulting COO is implicitly coalesced. Marking it lets the
    // downstream to_csr / to_csc paths skip the coalesce() call.
    result.coalesced_ = true;
    if (layout == SparseLayout::CSR) return result.to_csr();
    if (layout == SparseLayout::CSC) return result.to_csc();
    return result;
}

auto SparseTensor::to_dense() const -> Tensor {
    // Device-aware implementation: build a flat scatter index on the
    // values_ device, then `scatter_add` into a flat zeros buffer and
    // reshape. Replaces the previous CPU round-trip path.
    //
    // BSR fallback (block scatter is fiddly): keep the host pointer-walk
    // for BSR, since BSR-to-dense on GPU is rare in practice and the
    // dense output already lives on `values_.device()`. The BSR branch
    // below stages indices to host explicitly.

    // Float16 / BFloat16 widen path (unchanged).
    if (values_.dtype() == DType::Float16 || values_.dtype() == DType::BFloat16) {
        const DType orig_dtype = values_.dtype();
        SparseTensor widened = *this;
        widened.values_ = values_.to(DType::Float32);
        return widened.to_dense().to(orig_dtype);
    }

    Device dev = values_.device();

    // Empty case.
    if (nnz_ == 0) {
        return zeros(shape_, values_.dtype(), dev);
    }

    // 2D CSR / CSC path: build per-element (row, col) on-device, scatter-add
    // into a flat buffer, reshape. n-D COO is handled by the generic flat-
    // index branch below, sharing the same scatter_add primitive.
    const bool is_2d = (shape_.size() == 2);
    if (is_2d && (layout_ == SparseLayout::CSR || layout_ == SparseLayout::CSC)) {
        int64_t M = shape_[0];
        int64_t K = shape_[1];

        Tensor row_idx, col_idx;
        if (layout_ == SparseLayout::CSR) {
            Tensor row_lens = sub(slice(crow_indices_, 0, 1, M + 1),
                                  slice(crow_indices_, 0, 0, M));
            Tensor row_arange = arange(0, M, 1, DType::Int64, dev);
            row_idx = repeat_interleave(row_arange, row_lens, std::nullopt);
            col_idx = col_indices_;
        } else {
            Tensor col_lens = sub(slice(ccol_indices_, 0, 1, K + 1),
                                  slice(ccol_indices_, 0, 0, K));
            Tensor col_arange = arange(0, K, 1, DType::Int64, dev);
            col_idx = repeat_interleave(col_arange, col_lens, std::nullopt);
            row_idx = row_indices_;
        }

        Tensor K_t = full({nnz_}, static_cast<double>(K), DType::Int64, dev);
        Tensor flat_idx = add(mul(row_idx, K_t), col_idx);

        Tensor result_flat = zeros({M * K}, values_.dtype(), dev);
        result_flat = scatter_add(result_flat, /*dim=*/0, flat_idx, values_);
        return reshape(result_flat, shape_);
    }

    // Generic n-D COO path: fold per-dim coords into a flat index using
    // host-known strides, then scatter_add into a flat buffer of `numel`
    // elements. Handles both 2D and arbitrary-rank sparse dimensions.
    if (layout_ == SparseLayout::COO) {
        int64_t ndim = static_cast<int64_t>(shape_.size());
        // Compute strides (host scalars) — last dim has stride 1.
        std::vector<int64_t> strides(ndim);
        strides[ndim - 1] = 1;
        for (int64_t d = ndim - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape_[d + 1];
        }
        int64_t numel = strides[0] * shape_[0];

        // Fold the sparse_dim_ index rows. For partial-sparse tensors with
        // dense_dim_ > 0, the per-element values are vectors and we need
        // to scatter into the leading slab; that path isn't exercised by
        // any current tests, so fall through to the host path for safety.
        if (dense_dim_ != 0 || sparse_dim_ != ndim) {
            if (dev.type != Device::Type::CPU) {
                return this->to(Device::cpu()).to_dense().to(dev);
            }
        } else {
            Tensor flat_idx = zeros({nnz_}, DType::Int64, dev);
            for (int64_t d = 0; d < sparse_dim_; ++d) {
                Tensor coord_d = reshape(slice(indices_, 0, d, d + 1), {nnz_}).contiguous();
                if (strides[d] != 1) {
                    Tensor stride_t = full({nnz_}, static_cast<double>(strides[d]),
                                           DType::Int64, dev);
                    coord_d = mul(coord_d, stride_t);
                }
                flat_idx = add(flat_idx, coord_d);
            }
            Tensor result_flat = zeros({numel}, values_.dtype(), dev);
            result_flat = scatter_add(result_flat, /*dim=*/0, flat_idx, values_);
            return reshape(result_flat, shape_);
        }
    }

    // BSR → dense on-device. Algorithm:
    //   * block_row_per_block[k] = block-row index of block k (via
    //     repeat_interleave(arange(nblockrows), block_row_lens)).
    //   * Per-element: block_id varies slowest, then bi, then bj. Using
    //     repeat_interleave + tile to build the index streams without
    //     integer div/mod.
    //   * Compute (row, col) for each element, fold into a flat dense
    //     index, scatter_add the (already correctly-laid-out) values into
    //     a flat buffer, reshape.
    if (layout_ == SparseLayout::BSR && shape_.size() == 2) {
        int64_t M = shape_[0];
        int64_t N = shape_[1];
        auto [bh, bw] = block_size_;
        int64_t num_blocks = bsr_col_ind_.numel();
        int64_t elem_per_block = bh * bw;
        int64_t total = num_blocks * elem_per_block;

        if (total == 0) {
            return zeros(shape_, values_.dtype(), dev);
        }

        int64_t nblockrows = (M + bh - 1) / bh;

        // block_row_per_block: for each block, which block-row it lives in.
        Tensor block_row_lens = sub(slice(bsr_row_ptr_, 0, 1, nblockrows + 1),
                                    slice(bsr_row_ptr_, 0, 0, nblockrows));
        Tensor block_row_arange = arange(0, nblockrows, 1, DType::Int64, dev);
        Tensor block_row_per_block = repeat_interleave(block_row_arange,
                                                       block_row_lens, std::nullopt);

        // Per-element block_id: each block expanded `elem_per_block` times.
        Tensor blocks_arange = arange(0, num_blocks, 1, DType::Int64, dev);
        Tensor block_id_per_elem = repeat_interleave(blocks_arange,
                                                     elem_per_block, std::nullopt);

        // Lookup (block_row, block_col) per element.
        Tensor block_row_per_elem = index_select(block_row_per_block,
                                                  /*dim=*/0, block_id_per_elem);
        Tensor block_col_per_elem = index_select(bsr_col_ind_,
                                                  /*dim=*/0, block_id_per_elem);

        // bi / bj patterns built without tile (which is broken for Int64 on
        // most backends — cf. dispatchRepeatInterleaveTensor / dispatchTile
        // SSBO reinterpret bug). Build on host (size bh*bw, small), then
        // upload once and repeat_interleave by num_blocks with scalar count.
        Tensor bi_pattern_cpu({bh * bw}, DType::Int64, Device::cpu());
        Tensor bj_pattern_cpu({bh * bw}, DType::Int64, Device::cpu());
        {
            auto* bi_ptr = bi_pattern_cpu.data<int64_t>();
            auto* bj_ptr = bj_pattern_cpu.data<int64_t>();
            for (int64_t i = 0; i < bh; ++i) {
                for (int64_t j = 0; j < bw; ++j) {
                    bi_ptr[i * bw + j] = i;
                    bj_ptr[i * bw + j] = j;
                }
            }
        }
        Tensor bi_pattern_in_block = bi_pattern_cpu.to(dev);
        Tensor bj_pattern_in_block = bj_pattern_cpu.to(dev);

        // Tile across blocks via index_select with a precomputed map:
        //   map[k] = k % (bh*bw)   for k in [0, total).
        // Build this on host too (cheap) and upload once.
        Tensor map_cpu({total}, DType::Int64, Device::cpu());
        {
            auto* mp = map_cpu.data<int64_t>();
            for (int64_t k = 0; k < total; ++k) mp[k] = k % elem_per_block;
        }
        Tensor pattern_index = map_cpu.to(dev);
        Tensor bi_per_elem = index_select(bi_pattern_in_block, /*dim=*/0, pattern_index);
        Tensor bj_per_elem = index_select(bj_pattern_in_block, /*dim=*/0, pattern_index);

        // Final dense (row, col) per element.
        Tensor bh_t = full({total}, static_cast<double>(bh), DType::Int64, dev);
        Tensor bw_t = full({total}, static_cast<double>(bw), DType::Int64, dev);
        Tensor row = add(mul(block_row_per_elem, bh_t), bi_per_elem);
        Tensor col = add(mul(block_col_per_elem, bw_t), bj_per_elem);

        // Mask out-of-bounds entries (nblockrows*bh and nblockcols*bw can
        // exceed M and N respectively when shape isn't a block-multiple).
        // The CPU implementation checks `row < nrows && col < ncols`. We
        // mirror by clamping flat_idx out-of-bounds entries via a mask: do
        // the scatter_add only at valid positions. Simpler approach: build
        // a "valid_mask" tensor and scatter into a temporary, then keep
        // valid entries only via in-bounds clamping (scatter_add with index
        // M*N is illegal, so we must filter).
        Tensor N_t = full({total}, static_cast<double>(N), DType::Int64, dev);
        Tensor flat = add(mul(row, N_t), col);

        // Filter out-of-bounds (row >= M || col >= N) entries by replacing
        // their flat index with 0 and zeroing the value (so scatter_add
        // contributes nothing). Build a Bool mask and apply it both ways.
        Tensor M_check = full({total}, static_cast<double>(M), DType::Int64, dev);
        Tensor N_check = full({total}, static_cast<double>(N), DType::Int64, dev);
        Tensor row_lt = ::tenzor::lt(row, M_check);
        Tensor col_lt = ::tenzor::lt(col, N_check);
        Tensor in_bounds = ::tenzor::logical_and(row_lt, col_lt);
        Tensor in_bounds_i64 = in_bounds.to(DType::Int64);
        // Clamp flat to 0 where out of bounds (so scatter_add hits a valid
        // index; the masked value will be 0 below, so no harm done).
        Tensor flat_safe = mul(flat, in_bounds_i64);

        // Values flat (already laid out (block, bi, bj) row-major).
        Tensor values_flat = reshape(values_, {total}).contiguous();
        // Zero out values at out-of-bounds positions.
        Tensor mask_v = in_bounds.to(values_.dtype());
        Tensor values_masked = mul(values_flat, mask_v);

        Tensor result_flat = zeros({M * N}, values_.dtype(), dev);
        result_flat = scatter_add(result_flat, /*dim=*/0, flat_safe, values_masked);
        return reshape(result_flat, shape_);
    }

    // Remaining fallback: any layout/shape we haven't handled (n-D BSR,
    // partial-sparse with non-full sparse_dim_, etc.). These are rare
    // edge cases not covered by any current test.
    if (dev.type != Device::Type::CPU) {
        const Device orig_device = dev;
        return this->to(Device::cpu()).to_dense().to(orig_device);
    }

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

    Device dev = values_.device();

    // CSR / CSC → COO via on-device repeat_interleave + stack.
    // The compressed pointer becomes per-element row (or col) indices via
    // `repeat_interleave(arange, lens)`; the other index is reused as-is;
    // they're stacked to form COO indices [2, nnz]. The result is naturally
    // sorted (CSR has row-ordered entries; CSC col-ordered) so we mark it
    // coalesced. Replaces the previous host pointer-walk.
    if (layout_ == SparseLayout::CSR || layout_ == SparseLayout::CSC) {
        if (nnz_ == 0) {
            Tensor empty_indices({2, int64_t(0)}, DType::Int64, dev);
            Tensor empty_values({0}, values_.dtype(), dev);
            auto result = sparse_coo(empty_indices, empty_values, shape_);
            result.coalesced_ = (layout_ == SparseLayout::CSR);  // CSR is row-major
            return result;
        }

        Tensor compressed, varying;
        int64_t outer_size;
        if (layout_ == SparseLayout::CSR) {
            compressed = crow_indices_;     // [M+1]
            varying = col_indices_;          // [nnz] — col indices
            outer_size = shape_[0];          // nrows
        } else {
            compressed = ccol_indices_;      // [N+1]
            varying = row_indices_;          // [nnz] — row indices
            outer_size = shape_[1];          // ncols
        }

        Tensor lens = sub(slice(compressed, 0, 1, outer_size + 1),
                          slice(compressed, 0, 0, outer_size));
        Tensor outer_arange = arange(0, outer_size, 1, DType::Int64, dev);
        Tensor outer_idx = repeat_interleave(outer_arange, lens, std::nullopt);

        // outer_idx and varying are both shape [nnz]. For CSR: outer_idx is
        // rows, varying is cols. For CSC: outer_idx is cols, varying is rows.
        Tensor row_idx = (layout_ == SparseLayout::CSR) ? outer_idx : varying;
        Tensor col_idx = (layout_ == SparseLayout::CSR) ? varying : outer_idx;

        // Stack into [2, nnz].
        Tensor row_2d = reshape(row_idx, {1, nnz_});
        Tensor col_2d = reshape(col_idx, {1, nnz_});
        Tensor indices = cat({row_2d, col_2d}, /*dim=*/0);

        auto result = sparse_coo(indices, values_, shape_);
        // CSR's per-row entries are already row-ordered; CSC's are
        // column-ordered (i.e. NOT lex-sorted by (row, col)). Only flag
        // CSR-derived COO as coalesced.
        result.coalesced_ = (layout_ == SparseLayout::CSR);
        return result;
    }

    if (layout_ == SparseLayout::BSR) {
        // BSR -> COO: expand blocks into individual elements via dense
        // round-trip. Block expansion is a separate kernel project; using
        // dense here keeps the conversion correct on every backend.
        return ::tenzor::to_sparse(to_dense());
    }

    throw std::runtime_error("to_coo: unsupported layout");
}

auto SparseTensor::to_csr() const -> SparseTensor {
    if (layout_ == SparseLayout::CSR) return *this;
    if (shape_.size() != 2) {
        throw std::runtime_error("to_csr: only 2D sparse tensors supported");
    }

    // CSC and BSR: convert to COO first, then COO→CSR below.
    if (layout_ == SparseLayout::CSC || layout_ == SparseLayout::BSR) {
        return to_coo().to_csr();
    }

    // GPU-native path: use cusparseXcoo2csr / rocsparse_coo2csr directly
    // on the device, avoiding expensive GPU->CPU->GPU round-trips.
#ifdef TENZOR_HAS_CUSPARSE
    if (values_.device().type == Device::Type::CUDA) {
        return cuda::cuda_coo_to_csr(*this);
    }
#endif
#ifdef TENZOR_HAS_ROCSPARSE
    if (values_.device().type == Device::Type::ROCm) {
        return rocm::rocm_coo_to_csr(*this);
    }
#endif

    // OneAPI / Vulkan / CPU path: build the CSR row-pointer with on-device
    // bincount + cumsum, and slice col_indices straight off the COO indices
    // tensor. Replaces the previous host stage. (CUDA / ROCm vendor paths
    // above remain in place.)
    auto coo = coalesce();
    int64_t nrows = shape_[0];
    int64_t coalesced_nnz = coo.nnz();
    Device dev = coo.values_.device();

    if (coalesced_nnz == 0) {
        // Empty CSR — crow is all zeros, col / vals are length 0.
        Tensor crow_empty = zeros({nrows + 1}, DType::Int64, dev);
        Tensor col_empty({0}, DType::Int64, dev);
        Tensor vals_empty({0}, coo.values_.dtype(), dev);
        return sparse_csr(crow_empty, col_empty, vals_empty, shape_);
    }

    // Per-row counts via bincount(rows, minlength=nrows). Backends differ
    // in the bincount output dtype (CPU: Int64; Vulkan: Float32). Force
    // Int64 to satisfy the sparse_csr crow contract, then exclusive-cumsum
    // by prepending a leading zero.
    Tensor row_idx = reshape(slice(coo.indices_, 0, 0, 1), {coalesced_nnz}).contiguous();
    Tensor col_idx = reshape(slice(coo.indices_, 0, 1, 2), {coalesced_nnz}).contiguous();
    Tensor row_counts = bincount(row_idx, std::nullopt, /*minlength=*/nrows);
    if (row_counts.dtype() != DType::Int64) {
        row_counts = row_counts.to(DType::Int64);
    }

    Tensor zero_prefix = zeros({1}, DType::Int64, dev);
    Tensor cumsum_out = cumsum(row_counts, /*dim=*/0);
    if (cumsum_out.dtype() != DType::Int64) {
        cumsum_out = cumsum_out.to(DType::Int64);
    }
    Tensor crow = cat({zero_prefix, cumsum_out}, /*dim=*/0);

    return sparse_csr(crow, col_idx, coo.values_, shape_);
}

auto SparseTensor::transpose() const -> SparseTensor {
    if (shape_.size() != 2) {
        throw std::runtime_error("SparseTensor::transpose: only 2D sparse tensors supported");
    }

    int64_t nrows = shape_[0];
    int64_t ncols = shape_[1];
    std::vector<int64_t> transposed_shape = {ncols, nrows};

    if (layout_ == SparseLayout::COO) {
        // Swap row and column indices: indices_ is [2, nnz]
        // Row 0 = row indices, Row 1 = col indices
        auto idx = indices_.contiguous();
        auto* idx_ptr = idx.data<int64_t>();

        auto new_indices = Tensor({int64_t(2), nnz_}, DType::Int64, values_.device());
        auto* new_ptr = new_indices.data<int64_t>();

        // New row indices = old col indices, new col indices = old row indices
        for (int64_t i = 0; i < nnz_; ++i) {
            new_ptr[i] = idx_ptr[nnz_ + i];          // new row = old col
            new_ptr[nnz_ + i] = idx_ptr[i];           // new col = old row
        }

        // The transposed COO is not coalesced (ordering changed)
        auto result = sparse_coo(new_indices, values_, transposed_shape);
        return result.coalesce();
    }

    if (layout_ == SparseLayout::CSC) {
        // CSC of (nrows, ncols) is equivalent to CSR of (ncols, nrows)
        // ccol_indices -> crow_indices, row_indices -> col_indices
        return sparse_csr(ccol_indices_, row_indices_, values_, transposed_shape);
    }

    if (layout_ == SparseLayout::CSR) {
        // CSR transpose: reinterpret as CSC of transposed shape, then convert
        // CSR of (nrows, ncols) with crow/col is CSC of (ncols, nrows) with ccol=crow, row=col
        auto transposed_csc = sparse_csc(crow_indices_, col_indices_, values_,
                                          transposed_shape);
        return transposed_csc.to_csr();
    }

    // BSR: convert to COO, transpose, convert back
    auto coo = to_coo();
    return coo.transpose();
}

auto SparseTensor::coalesce() const -> SparseTensor {
    if (coalesced_ || layout_ != SparseLayout::COO) return *this;
    if (nnz_ == 0) {
        SparseTensor result = *this;
        result.coalesced_ = true;
        return result;
    }

    // Widen "narrow" value dtypes the device-aware scatter_add can't handle
    // directly — Float16 / BFloat16 / Int8 / UInt8 / Bool / FP8 — to a
    // wider dtype, coalesce, then cast back. Matches the to_dense widen-
    // narrow pattern. After this, values_.dtype() is one of the four
    // dtypes the on-device segment reduce supports natively.
    {
        DType vd = values_.dtype();
        std::optional<DType> widen_to;
        if (vd == DType::Float16 || vd == DType::BFloat16 ||
            vd == DType::FP8_E4M3 || vd == DType::FP8_E5M2) {
            widen_to = DType::Float32;
        } else if (vd == DType::Int8 || vd == DType::UInt8 || vd == DType::Bool) {
            widen_to = DType::Int32;
        }
        if (widen_to.has_value()) {
            const DType orig_dtype = vd;
            SparseTensor widened = *this;
            widened.values_ = values_.to(*widen_to);
            auto coalesced = widened.coalesce();
            coalesced.values_ = coalesced.values_.to(orig_dtype);
            return coalesced;
        }
    }

    // GPU-native path: sort + reduce_by_key entirely on device using
    // thrust, avoiding the GPU->CPU->GPU round-trip.
#ifdef TENZOR_HAS_CUSPARSE
    if (indices_.device().type == Device::Type::CUDA) {
        auto result = cuda::cuda_coalesce(*this);
        result.coalesced_ = true;
        return result;
    }
#endif
#ifdef TENZOR_HAS_ROCSPARSE
    if (indices_.device().type == Device::Type::ROCm) {
        auto result = rocm::rocm_coalesce(*this);
        result.coalesced_ = true;
        return result;
    }
#endif

    // OneAPI / Vulkan / CPU device-aware path: compound-key + on-device
    // sort + segment-reduce. Replaces the previous host stage. Algorithm:
    //   1. keys[i] = sum_d indices[d,i] * stride[d]   (linearised flat idx)
    //   2. (sorted_keys, perm) = sort(keys)
    //   3. is_new[i] = 1 if i==0 || sorted_keys[i] != sorted_keys[i-1]
    //   4. group_id = cumsum(is_new) - 1                (0-indexed groups)
    //   5. new_nnz = group_id[nnz-1] + 1                (single scalar D2H)
    //   6. for each d:  new_indices[d, group_id[i]] = sorted_indices[d, i]
    //                                                   (first hit / mask)
    //      new_values[group_id[i]] += sorted_values[i]   (scatter_add)
    Device dev = indices_.device();
    DType vdtype = values_.dtype();
    // J16-followup: Complex coalesce via real/imag split. Since both
    // SparseTensors share the same `indices_`, the segment-key sort
    // permutation is identical for both halves — so the coalesced
    // outputs have the same `new_indices` and we can recombine via
    // `tenzor::complex(real, imag)`.
    if (vdtype == DType::Complex64 || vdtype == DType::Complex128) {
        Tensor re = ::tenzor::real(values_);
        Tensor im = ::tenzor::imag(values_);
        // Build two SparseTensors with identical indices but real/imag values.
        SparseTensor st_re = sparse_coo(indices_, re,
                                         std::vector<int64_t>(shape_.begin(), shape_.end()));
        SparseTensor st_im = sparse_coo(indices_, im,
                                         std::vector<int64_t>(shape_.begin(), shape_.end()));
        SparseTensor co_re = st_re.coalesce();
        SparseTensor co_im = st_im.coalesce();
        // Indices match across the two by construction (same input ordering).
        Tensor merged_values = ::tenzor::complex(co_re.values(), co_im.values());
        SparseTensor result = sparse_coo(co_re.indices(), merged_values,
                                          std::vector<int64_t>(shape_.begin(), shape_.end()));
        result.coalesced_ = true;
        return result;
    }
    if (vdtype != DType::Float32 && vdtype != DType::Float64 &&
        vdtype != DType::Int32 && vdtype != DType::Int64) {
        // After the widening table above + the Complex split path above,
        // anything that reaches here is a genuinely-unsupported value
        // dtype. Throw with the most informative message we can.
        throw std::runtime_error(
            std::string("SparseTensor::coalesce: unsupported value dtype ") +
            std::string(dtype_name(vdtype)));
    }

    {
        // Compute compound keys.
        std::vector<int64_t> strides(sparse_dim_);
        if (sparse_dim_ > 0) {
            strides[sparse_dim_ - 1] = 1;
            for (int64_t d = sparse_dim_ - 2; d >= 0; --d) {
                strides[d] = strides[d + 1] * shape_[d + 1];
            }
        }
        Tensor keys = zeros({nnz_}, DType::Int64, dev);
        for (int64_t d = 0; d < sparse_dim_; ++d) {
            Tensor coord_d = reshape(slice(indices_, 0, d, d + 1), {nnz_}).contiguous();
            if (strides[d] != 1) {
                Tensor stride_t = full({nnz_}, static_cast<double>(strides[d]),
                                       DType::Int64, dev);
                coord_d = mul(coord_d, stride_t);
            }
            keys = add(keys, coord_d);
        }

        // Sort: (sorted_keys, perm) such that sorted_keys = keys[perm].
        auto [sorted_keys, perm] = ::tenzor::sort(keys, /*dim=*/0, /*descending=*/false);

        // Mark first-of-each-group: is_new[0]=1; is_new[i] = (sorted_keys[i] != sorted_keys[i-1]).
        Tensor is_new;
        if (nnz_ == 1) {
            is_new = full({1}, 1.0, DType::Int64, dev);
        } else {
            Tensor key_curr = slice(sorted_keys, 0, 1, nnz_);     // [nnz-1]
            Tensor key_prev = slice(sorted_keys, 0, 0, nnz_ - 1); // [nnz-1]
            Tensor diff_bool = ne(key_curr, key_prev);
            Tensor diff_i64 = (diff_bool.dtype() == DType::Int64)
                ? diff_bool : diff_bool.to(DType::Int64);
            Tensor one_prefix = full({1}, 1.0, DType::Int64, dev);
            is_new = cat({one_prefix, diff_i64}, /*dim=*/0);
        }

        // group_id = cumsum(is_new) - 1.
        Tensor cs = cumsum(is_new, /*dim=*/0);
        if (cs.dtype() != DType::Int64) cs = cs.to(DType::Int64);
        Tensor one_full = full({nnz_}, 1.0, DType::Int64, dev);
        Tensor group_id = sub(cs, one_full);

        // new_nnz = group_id[nnz-1] + 1 — single int64 readback.
        Tensor max_group_cpu = slice(group_id, 0, nnz_ - 1, nnz_).to(Device::cpu());
        int64_t new_nnz = max_group_cpu.data<int64_t>()[0] + 1;

        // Build per-dim sorted indices, then pick first-of-each-group via
        // `nonzero` of is_new (which gives the group-start positions in
        // sorted order) plus `index_select`. Avoids `masked_select` whose
        // ROCm kernel rejects Int64.
        Tensor is_new_bool = (is_new.dtype() == DType::Bool)
            ? is_new : ne(is_new, zeros({nnz_}, DType::Int64, dev));

        // nonzero(is_new) returns shape (new_nnz, 1) of Int64 positions.
        Tensor group_starts_2d = nonzero(is_new_bool);
        Tensor group_starts = reshape(group_starts_2d, {new_nnz}).contiguous();

        std::vector<Tensor> dim_rows;
        dim_rows.reserve(static_cast<size_t>(sparse_dim_));
        for (int64_t d = 0; d < sparse_dim_; ++d) {
            Tensor coord_d = reshape(slice(indices_, 0, d, d + 1), {nnz_}).contiguous();
            Tensor sorted_d = index_select(coord_d, /*dim=*/0, perm);
            Tensor first_d = index_select(sorted_d, /*dim=*/0, group_starts);
            dim_rows.push_back(reshape(first_d, {1, new_nnz}));
        }
        Tensor new_indices = cat(dim_rows, /*dim=*/0);

        // Sum values per group: scatter_add(zeros(new_nnz), 0, group_id, sorted_values).
        // (Note: scatter_add takes sources in the original order; group_id maps each
        // original-sorted entry to its destination group.)
        Tensor sorted_values = index_select(values_, /*dim=*/0, perm);
        Tensor new_values = zeros({new_nnz}, vdtype, dev);
        new_values = scatter_add(new_values, /*dim=*/0, group_id, sorted_values);

        SparseTensor result = sparse_coo(new_indices, new_values, shape_);
        result.coalesced_ = true;
        return result;
    }
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

    // GPU-native path: sort by (col, row) and build ccol_indices entirely
    // on device using thrust, avoiding GPU->CPU->GPU round-trips.
#ifdef TENZOR_HAS_CUSPARSE
    if (values_.device().type == Device::Type::CUDA) {
        return cuda::cuda_coo_to_csc(*this);
    }
#endif
#ifdef TENZOR_HAS_ROCSPARSE
    if (values_.device().type == Device::Type::ROCm) {
        return rocm::rocm_coo_to_csc(*this);
    }
#endif

    // CPU path: convert to COO first, then build CSC
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
    // Delegate to the device-aware static factory; that path uses
    // nonzero / index_select on the input's device with no host roundtrip.
    return SparseTensor::from_dense(dense, SparseLayout::COO);
}

auto to_sparse_csr(const Tensor& dense) -> SparseTensor {
    return to_sparse(dense).to_csr();
}

auto to_sparse_csc(const Tensor& dense) -> SparseTensor {
    return to_sparse(dense).to_csc();
}

} // namespace tenzor
