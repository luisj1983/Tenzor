#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstring>
#include <stdexcept>

// Forward declarations for CUDA sparse kernels (defined in kernels/sparse.cu)
#ifdef TENZOR_HAS_CUSPARSE
namespace tenzor {
namespace cuda {
Tensor cuda_spmm_kernel(const SparseTensor& sparse, const Tensor& dense);
Tensor cuda_spmv_kernel(const SparseTensor& sparse, const Tensor& vec);
} // namespace cuda
} // namespace tenzor
#endif

namespace tenzor {
namespace sparse {

// ============================================================================
// Helper: cast unsupported dtypes to a compute dtype for spmm/spmv
// ============================================================================

namespace {

/// Determine the compute dtype for a given input dtype.
/// Float16/BFloat16/Int32 -> Float32, Int64 -> Float64.
/// Float32/Float64 are returned as-is.
DType compute_dtype_for(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Float64:
            return dtype;
        case DType::Float16:
        case DType::BFloat16:
        case DType::Int32:
            return DType::Float32;
        case DType::Int64:
            return DType::Float64;
        default:
            throw std::runtime_error("sparse ops: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
    }
}

/// Return true if sparse/dense are on CUDA and cuSPARSE is available.
[[maybe_unused]] bool should_use_cuda(const SparseTensor& sparse, const Tensor& dense) {
#ifdef TENZOR_HAS_CUSPARSE
    return (dense.device().type == Device::Type::CUDA ||
            sparse.device().type == Device::Type::CUDA);
#else
    (void)sparse;
    (void)dense;
    return false;
#endif
}

[[maybe_unused]] bool should_use_cuda_vec(const SparseTensor& sparse, const Tensor& vec) {
#ifdef TENZOR_HAS_CUSPARSE
    return (vec.device().type == Device::Type::CUDA ||
            sparse.device().type == Device::Type::CUDA);
#else
    (void)sparse;
    (void)vec;
    return false;
#endif
}

} // anonymous namespace

// ============================================================================
// spmm: Sparse-Dense Matrix Multiplication
// ============================================================================

auto spmm(const SparseTensor& sparse, const Tensor& dense) -> Tensor {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("spmm: both inputs must be 2D");
    }
    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("spmm: inner dimensions must match");
    }

    DType orig_dtype = dense.dtype();
    DType comp_dtype = compute_dtype_for(orig_dtype);

    // Cast dense and sparse values to compute dtype if needed
    Tensor dense_compute = (orig_dtype != comp_dtype) ? dense.to(comp_dtype) : dense;
    SparseTensor sparse_compute = sparse;
    if (sparse.dtype() != comp_dtype) {
        // Rebuild the sparse tensor with cast values
        auto new_vals = sparse.values().to(comp_dtype);
        auto shape_vec = std::vector<int64_t>(sp_shape.begin(), sp_shape.end());
        if (sparse.layout() == SparseLayout::COO) {
            sparse_compute = SparseTensor::sparse_coo(sparse.indices(), new_vals, shape_vec);
        } else {
            sparse_compute = SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(), new_vals, shape_vec);
        }
    }

#ifdef TENZOR_HAS_CUSPARSE
    if (should_use_cuda(sparse_compute, dense_compute)) {
        auto result = cuda::cuda_spmm_kernel(sparse_compute, dense_compute);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }
#endif

    // CPU path
    auto result = zeros({M, N}, comp_dtype, dense_compute.device());
    auto dense_c = dense_compute.contiguous();

    if (sparse_compute.layout() == SparseLayout::COO) {
        auto coo = sparse_compute.is_coalesced() ? sparse_compute : sparse_compute.coalesce();
        auto idx = coo.indices().contiguous();
        auto vals = coo.values().contiguous();
        auto* idx_ptr = idx.data<int64_t>();
        int64_t nnz = coo.nnz();

        if (comp_dtype == DType::Float32) {
            auto* v = vals.data<float>();
            auto* d = dense_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t row = idx_ptr[i];
                int64_t col = idx_ptr[nnz + i];
                float val = v[i];
                for (int64_t j = 0; j < N; ++j) {
                    r[row * N + j] += val * d[col * N + j];
                }
            }
        } else if (comp_dtype == DType::Float64) {
            auto* v = vals.data<double>();
            auto* d = dense_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t row = idx_ptr[i];
                int64_t col = idx_ptr[nnz + i];
                double val = v[i];
                for (int64_t j = 0; j < N; ++j) {
                    r[row * N + j] += val * d[col * N + j];
                }
            }
        }
    } else {
        // CSR format
        auto crow = sparse_compute.crow_indices().contiguous();
        auto col = sparse_compute.col_indices().contiguous();
        auto vals = sparse_compute.values().contiguous();
        auto* crow_ptr = crow.data<int64_t>();
        auto* col_ptr = col.data<int64_t>();

        if (comp_dtype == DType::Float32) {
            auto* v = vals.data<float>();
            auto* d = dense_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t row = 0; row < M; ++row) {
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    int64_t c = col_ptr[j];
                    float val = v[j];
                    for (int64_t n = 0; n < N; ++n) {
                        r[row * N + n] += val * d[c * N + n];
                    }
                }
            }
        } else if (comp_dtype == DType::Float64) {
            auto* v = vals.data<double>();
            auto* d = dense_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t row = 0; row < M; ++row) {
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    int64_t c = col_ptr[j];
                    double val = v[j];
                    for (int64_t n = 0; n < N; ++n) {
                        r[row * N + n] += val * d[c * N + n];
                    }
                }
            }
        }
    }

    // Cast result back to original dtype if needed
    return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
}

// ============================================================================
// spmv: Sparse-Dense Matrix-Vector Multiplication
// ============================================================================

auto spmv(const SparseTensor& sparse, const Tensor& vec) -> Tensor {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("spmv: sparse must be 2D, vec must be 1D");
    }
    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("spmv: dimensions must match");
    }

    DType orig_dtype = vec.dtype();
    DType comp_dtype = compute_dtype_for(orig_dtype);

    // Cast to compute dtype if needed
    Tensor vec_compute = (orig_dtype != comp_dtype) ? vec.to(comp_dtype) : vec;
    SparseTensor sparse_compute = sparse;
    if (sparse.dtype() != comp_dtype) {
        auto new_vals = sparse.values().to(comp_dtype);
        auto shape_vec = std::vector<int64_t>(sp_shape.begin(), sp_shape.end());
        if (sparse.layout() == SparseLayout::COO) {
            sparse_compute = SparseTensor::sparse_coo(sparse.indices(), new_vals, shape_vec);
        } else {
            sparse_compute = SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(), new_vals, shape_vec);
        }
    }

#ifdef TENZOR_HAS_CUSPARSE
    if (should_use_cuda_vec(sparse_compute, vec_compute)) {
        auto result = cuda::cuda_spmv_kernel(sparse_compute, vec_compute);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }
#endif

    // CPU path
    auto result = zeros({M}, comp_dtype, vec_compute.device());
    auto vec_c = vec_compute.contiguous();

    if (sparse_compute.layout() == SparseLayout::COO) {
        auto coo = sparse_compute.is_coalesced() ? sparse_compute : sparse_compute.coalesce();
        auto idx = coo.indices().contiguous();
        auto vals = coo.values().contiguous();
        auto* idx_ptr = idx.data<int64_t>();
        int64_t nnz = coo.nnz();

        if (comp_dtype == DType::Float32) {
            auto* v = vals.data<float>();
            auto* x = vec_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t i = 0; i < nnz; ++i) {
                r[idx_ptr[i]] += v[i] * x[idx_ptr[nnz + i]];
            }
        } else if (comp_dtype == DType::Float64) {
            auto* v = vals.data<double>();
            auto* x = vec_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                r[idx_ptr[i]] += v[i] * x[idx_ptr[nnz + i]];
            }
        }
    } else {
        auto crow = sparse_compute.crow_indices().contiguous();
        auto col = sparse_compute.col_indices().contiguous();
        auto vals = sparse_compute.values().contiguous();
        auto* crow_ptr = crow.data<int64_t>();
        auto* col_ptr = col.data<int64_t>();

        if (comp_dtype == DType::Float32) {
            auto* v = vals.data<float>();
            auto* x = vec_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t row = 0; row < M; ++row) {
                float sum = 0;
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += v[j] * x[col_ptr[j]];
                }
                r[row] = sum;
            }
        } else if (comp_dtype == DType::Float64) {
            auto* v = vals.data<double>();
            auto* x = vec_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t row = 0; row < M; ++row) {
                double sum = 0;
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += v[j] * x[col_ptr[j]];
                }
                r[row] = sum;
            }
        }
    }

    return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
}

// ============================================================================
// add: Sparse + Dense
// ============================================================================

auto add(const SparseTensor& sparse, const Tensor& dense) -> Tensor {
    auto result = sparse.to_dense();
    auto dense_c = dense.contiguous();
    auto result_c = result.contiguous();

    int64_t n = result_c.numel();
    if (n != dense_c.numel()) {
        throw std::runtime_error("sparse::add: shape mismatch");
    }

    if (result_c.dtype() == DType::Float32) {
        auto* r = result_c.data<float>();
        auto* d = dense_c.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    } else if (result_c.dtype() == DType::Float64) {
        auto* r = result_c.data<double>();
        auto* d = dense_c.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    }

    return result_c;
}

// ============================================================================
// add: Sparse + Sparse
// ============================================================================

auto add(const SparseTensor& a, const SparseTensor& b) -> SparseTensor {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("sparse::add: shape mismatch");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("sparse::add: dtype mismatch - a is " +
            std::string(dtype_name(a.dtype())) + ", b is " +
            std::string(dtype_name(b.dtype())));
    }

    // Convert both to COO, concatenate indices and values
    auto a_coo = a.to_coo();
    auto b_coo = b.to_coo();

    int64_t nnz_a = a_coo.nnz();
    int64_t nnz_b = b_coo.nnz();
    int64_t total_nnz = nnz_a + nnz_b;
    int64_t sparse_dim = a_coo.sparse_dim();

    if (total_nnz == 0) {
        auto indices = Tensor({sparse_dim, int64_t(0)}, DType::Int64, a.device());
        auto values = Tensor({int64_t(0)}, a.dtype(), a.device());
        return SparseTensor::sparse_coo(indices, values, std::vector<int64_t>(a.shape().begin(), a.shape().end()));
    }

    // Concatenate indices
    auto new_indices = Tensor({sparse_dim, total_nnz}, DType::Int64, a.device());
    auto* ni_ptr = new_indices.data<int64_t>();

    if (nnz_a > 0) {
        auto a_idx = a_coo.indices().contiguous();
        auto* a_ptr = a_idx.data<int64_t>();
        for (int64_t d = 0; d < sparse_dim; ++d) {
            std::memcpy(ni_ptr + d * total_nnz, a_ptr + d * nnz_a, nnz_a * sizeof(int64_t));
        }
    }
    if (nnz_b > 0) {
        auto b_idx = b_coo.indices().contiguous();
        auto* b_ptr = b_idx.data<int64_t>();
        for (int64_t d = 0; d < sparse_dim; ++d) {
            std::memcpy(ni_ptr + d * total_nnz + nnz_a, b_ptr + d * nnz_b, nnz_b * sizeof(int64_t));
        }
    }

    // Concatenate values
    auto new_values = Tensor({total_nnz}, a.dtype(), a.device());
    if (a.dtype() == DType::Float32) {
        auto* nv = new_values.data<float>();
        if (nnz_a > 0) {
            auto av = a_coo.values().contiguous();
            std::memcpy(nv, av.data<float>(), nnz_a * sizeof(float));
        }
        if (nnz_b > 0) {
            auto bv = b_coo.values().contiguous();
            std::memcpy(nv + nnz_a, bv.data<float>(), nnz_b * sizeof(float));
        }
    } else if (a.dtype() == DType::Float64) {
        auto* nv = new_values.data<double>();
        if (nnz_a > 0) {
            auto av = a_coo.values().contiguous();
            std::memcpy(nv, av.data<double>(), nnz_a * sizeof(double));
        }
        if (nnz_b > 0) {
            auto bv = b_coo.values().contiguous();
            std::memcpy(nv + nnz_a, bv.data<double>(), nnz_b * sizeof(double));
        }
    }

    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    return SparseTensor::sparse_coo(new_indices, new_values, shape_vec).coalesce();
}

// ============================================================================
// mul: Sparse * Scalar
// ============================================================================

auto mul(const SparseTensor& sparse, double scalar) -> SparseTensor {
    if (sparse.layout() == SparseLayout::COO) {
        auto vals = sparse.values().contiguous();
        int64_t nnz = sparse.nnz();

        auto new_values = Tensor({nnz}, vals.dtype(), vals.device());
        if (vals.dtype() == DType::Float32) {
            auto* src = vals.data<float>();
            auto* dst = new_values.data<float>();
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * s;
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* src = vals.data<double>();
            auto* dst = new_values.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * scalar;
            }
        }

        auto shape_vec = std::vector<int64_t>(sparse.shape().begin(), sparse.shape().end());
        auto result = SparseTensor::sparse_coo(sparse.indices(), new_values, shape_vec);
        return result;
    } else {
        // CSR
        auto vals = sparse.values().contiguous();
        int64_t nnz = sparse.nnz();

        auto new_values = Tensor({nnz}, vals.dtype(), vals.device());
        if (vals.dtype() == DType::Float32) {
            auto* src = vals.data<float>();
            auto* dst = new_values.data<float>();
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * s;
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* src = vals.data<double>();
            auto* dst = new_values.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * scalar;
            }
        }

        auto shape_vec = std::vector<int64_t>(sparse.shape().begin(), sparse.shape().end());
        return SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(), new_values, shape_vec);
    }
}

} // namespace sparse
} // namespace tenzor
