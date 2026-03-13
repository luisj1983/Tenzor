#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// MKL sparse BLAS for accelerated SpMV/SpMM
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

// Forward declarations for CUDA sparse kernels (defined in kernels/sparse.cu)
#ifdef TENZOR_HAS_CUSPARSE
namespace tenzor {
namespace cuda {
Tensor cuda_spmm_kernel(const SparseTensor& sparse, const Tensor& dense);
Tensor cuda_spmv_kernel(const SparseTensor& sparse, const Tensor& vec);
} // namespace cuda
} // namespace tenzor
#endif

// Forward declarations for ROCm sparse kernels (defined in kernels/sparse.hip.cpp)
#ifdef TENZOR_HAS_ROCSPARSE
namespace tenzor {
namespace rocm {
Tensor rocm_spmm_kernel(const SparseTensor& sparse, const Tensor& dense);
Tensor rocm_spmv_kernel(const SparseTensor& sparse, const Tensor& vec);
} // namespace rocm
} // namespace tenzor
#endif

// Forward declarations for OneAPI sparse kernels (defined in oneapi/kernels/sparse.cpp)
#ifdef TENZOR_HAS_ONEMKL
namespace tenzor {
namespace oneapi {
Tensor oneapi_spmm_kernel(const SparseTensor& sparse, const Tensor& dense);
Tensor oneapi_spmv_kernel(const SparseTensor& sparse, const Tensor& vec);
} // namespace oneapi
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

/// Return true if sparse/dense are on ROCm and rocSPARSE is available.
[[maybe_unused]] bool should_use_rocm(const SparseTensor& sparse, const Tensor& dense) {
#ifdef TENZOR_HAS_ROCSPARSE
    return (dense.device().type == Device::Type::ROCm ||
            sparse.device().type == Device::Type::ROCm);
#else
    (void)sparse;
    (void)dense;
    return false;
#endif
}

[[maybe_unused]] bool should_use_rocm_vec(const SparseTensor& sparse, const Tensor& vec) {
#ifdef TENZOR_HAS_ROCSPARSE
    return (vec.device().type == Device::Type::ROCm ||
            sparse.device().type == Device::Type::ROCm);
#else
    (void)sparse;
    (void)vec;
    return false;
#endif
}

/// Return true if sparse/dense are on Vulkan.
[[maybe_unused]] bool should_use_vulkan(const SparseTensor& sparse, const Tensor& dense) {
    return (dense.device().type == Device::Type::Vulkan ||
            sparse.device().type == Device::Type::Vulkan);
}

[[maybe_unused]] bool should_use_vulkan_vec(const SparseTensor& sparse, const Tensor& vec) {
    return (vec.device().type == Device::Type::Vulkan ||
            sparse.device().type == Device::Type::Vulkan);
}

/// Return true if sparse/dense are on OneAPI/SYCL and oneMKL sparse is available.
[[maybe_unused]] bool should_use_oneapi(const SparseTensor& sparse, const Tensor& dense) {
#ifdef TENZOR_HAS_ONEMKL
    return (dense.device().type == Device::Type::OneAPI ||
            sparse.device().type == Device::Type::OneAPI);
#else
    (void)sparse;
    (void)dense;
    return false;
#endif
}

[[maybe_unused]] bool should_use_oneapi_vec(const SparseTensor& sparse, const Tensor& vec) {
#ifdef TENZOR_HAS_ONEMKL
    return (vec.device().type == Device::Type::OneAPI ||
            sparse.device().type == Device::Type::OneAPI);
#else
    (void)sparse;
    (void)vec;
    return false;
#endif
}

// ============================================================================
// MKL Sparse Helpers
// ============================================================================

#ifdef TENZOR_USE_MKL

/// RAII wrapper for MKL sparse matrix handle.
struct MklSparseGuard {
    sparse_matrix_t handle = nullptr;
    explicit MklSparseGuard(sparse_matrix_t h) : handle(h) {}
    ~MklSparseGuard() { if (handle) mkl_sparse_destroy(handle); }
    MklSparseGuard(const MklSparseGuard&) = delete;
    MklSparseGuard& operator=(const MklSparseGuard&) = delete;
};

/// Convert Int64 indices to MKL_INT.
/// With MKL_ILP64, MKL_INT is long long (64-bit), so this is a reinterpret.
std::vector<MKL_INT> to_mkl_int(const int64_t* src, int64_t n) {
    std::vector<MKL_INT> dst(n);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = static_cast<MKL_INT>(src[i]);
    }
    return dst;
}

/// Create MKL Float32 CSR handle from SparseTensor.
sparse_matrix_t create_mkl_csr_f32(const SparseTensor& sparse, MKL_INT nrows, MKL_INT ncols,
                                     std::vector<MKL_INT>& crow_buf,
                                     std::vector<MKL_INT>& col_buf) {
    auto crow = sparse.crow_indices().contiguous();
    auto col = sparse.col_indices().contiguous();
    auto vals = sparse.values().contiguous();

    crow_buf = to_mkl_int(crow.data<int64_t>(), nrows + 1);
    col_buf = to_mkl_int(col.data<int64_t>(), sparse.nnz());

    sparse_matrix_t handle = nullptr;
    sparse_status_t status = mkl_sparse_s_create_csr(
        &handle,
        SPARSE_INDEX_BASE_ZERO,
        nrows, ncols,
        crow_buf.data(),
        crow_buf.data() + 1,
        col_buf.data(),
        const_cast<float*>(vals.data<float>())
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_s_create_csr failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    return handle;
}

/// Create MKL Float64 CSR handle from SparseTensor.
sparse_matrix_t create_mkl_csr_f64(const SparseTensor& sparse, MKL_INT nrows, MKL_INT ncols,
                                     std::vector<MKL_INT>& crow_buf,
                                     std::vector<MKL_INT>& col_buf) {
    auto crow = sparse.crow_indices().contiguous();
    auto col = sparse.col_indices().contiguous();
    auto vals = sparse.values().contiguous();

    crow_buf = to_mkl_int(crow.data<int64_t>(), nrows + 1);
    col_buf = to_mkl_int(col.data<int64_t>(), sparse.nnz());

    sparse_matrix_t handle = nullptr;
    sparse_status_t status = mkl_sparse_d_create_csr(
        &handle,
        SPARSE_INDEX_BASE_ZERO,
        nrows, ncols,
        crow_buf.data(),
        crow_buf.data() + 1,
        col_buf.data(),
        const_cast<double*>(vals.data<double>())
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_create_csr failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    return handle;
}

/// MKL SpMV for Float32 CSR.
Tensor mkl_csr_spmv_f32(const SparseTensor& sparse, const Tensor& vec,
                          int64_t M, int64_t K) {
    std::vector<MKL_INT> crow_buf, col_buf;
    auto handle = create_mkl_csr_f32(sparse, M, K, crow_buf, col_buf);
    MklSparseGuard guard(handle);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto result = zeros({M}, DType::Float32, Device::cpu());
    float alpha = 1.0f, beta = 0.0f;
    sparse_status_t status = mkl_sparse_s_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha, handle, descr,
        vec.data<float>(),
        beta,
        result.data<float>()
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_s_mv failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    return result;
}

/// MKL SpMV for Float64 CSR.
Tensor mkl_csr_spmv_f64(const SparseTensor& sparse, const Tensor& vec,
                          int64_t M, int64_t K) {
    std::vector<MKL_INT> crow_buf, col_buf;
    auto handle = create_mkl_csr_f64(sparse, M, K, crow_buf, col_buf);
    MklSparseGuard guard(handle);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto result = zeros({M}, DType::Float64, Device::cpu());
    double alpha = 1.0, beta = 0.0;
    sparse_status_t status = mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha, handle, descr,
        vec.data<double>(),
        beta,
        result.data<double>()
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_mv failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    return result;
}

/// MKL SpMM for Float32 CSR.
Tensor mkl_csr_spmm_f32(const SparseTensor& sparse, const Tensor& dense,
                          int64_t M, int64_t K, int64_t N) {
    std::vector<MKL_INT> crow_buf, col_buf;
    auto handle = create_mkl_csr_f32(sparse, M, K, crow_buf, col_buf);
    MklSparseGuard guard(handle);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto result = zeros({M, N}, DType::Float32, Device::cpu());
    float alpha = 1.0f, beta = 0.0f;
    sparse_status_t status = mkl_sparse_s_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha, handle, descr,
        SPARSE_LAYOUT_ROW_MAJOR,
        dense.data<float>(),
        static_cast<MKL_INT>(N),   // columns of B
        static_cast<MKL_INT>(N),   // leading dimension of B (row-major: ldB = N)
        beta,
        result.data<float>(),
        static_cast<MKL_INT>(N)    // leading dimension of C
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_s_mm failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    return result;
}

/// MKL SpMM for Float64 CSR.
Tensor mkl_csr_spmm_f64(const SparseTensor& sparse, const Tensor& dense,
                          int64_t M, int64_t K, int64_t N) {
    std::vector<MKL_INT> crow_buf, col_buf;
    auto handle = create_mkl_csr_f64(sparse, M, K, crow_buf, col_buf);
    MklSparseGuard guard(handle);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto result = zeros({M, N}, DType::Float64, Device::cpu());
    double alpha = 1.0, beta = 0.0;
    sparse_status_t status = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha, handle, descr,
        SPARSE_LAYOUT_ROW_MAJOR,
        dense.data<double>(),
        static_cast<MKL_INT>(N),
        static_cast<MKL_INT>(N),
        beta,
        result.data<double>(),
        static_cast<MKL_INT>(N)
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_mm failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    return result;
}

#endif // TENZOR_USE_MKL

// ============================================================================
// Fallback (non-MKL) scalar implementations
// ============================================================================

/// Fallback SpMV: CSR format, y = A * x
template<typename T>
void fallback_csr_spmv(const int64_t* crow_ptr, const int64_t* col_ptr,
                        const T* vals, const T* x, T* y,
                        int64_t nrows) {
    #pragma omp parallel for schedule(static) if(nrows > 128)
    for (int64_t row = 0; row < nrows; ++row) {
        T sum = T(0);
        for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
            sum += vals[j] * x[col_ptr[j]];
        }
        y[row] = sum;
    }
}

/// Fallback SpMM: CSR format, C = A * B where B is (K, N) row-major
template<typename T>
void fallback_csr_spmm(const int64_t* crow_ptr, const int64_t* col_ptr,
                        const T* vals, const T* B, T* C,
                        int64_t M, int64_t N) {
    #pragma omp parallel for schedule(static) if(M > 64)
    for (int64_t row = 0; row < M; ++row) {
        T* c_row = C + row * N;
        std::memset(c_row, 0, N * sizeof(T));
        for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
            int64_t col = col_ptr[j];
            T val = vals[j];
            const T* b_row = B + col * N;
            for (int64_t n = 0; n < N; ++n) {
                c_row[n] += val * b_row[n];
            }
        }
    }
}

/// Fallback COO SpMV: y = A * x (cannot be parallelized due to race on y)
template<typename T>
void fallback_coo_spmv(const int64_t* idx_ptr, const T* vals, const T* x, T* y,
                        int64_t nnz) {
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t row = idx_ptr[i];
        int64_t col = idx_ptr[nnz + i];
        y[row] += vals[i] * x[col];
    }
}

/// Fallback COO SpMM: C = A * B (cannot be parallelized due to race on C)
template<typename T>
void fallback_coo_spmm(const int64_t* idx_ptr, const T* vals, const T* B, T* C,
                        int64_t nnz, int64_t N) {
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t row = idx_ptr[i];
        int64_t col = idx_ptr[nnz + i];
        T val = vals[i];
        for (int64_t n = 0; n < N; ++n) {
            C[row * N + n] += val * B[col * N + n];
        }
    }
}

// ============================================================================
// Internal CPU SpMM/SpMV dispatchers
// ============================================================================

/// CPU SpMM implementation: tries MKL for CSR, falls back to scalar loops.
Tensor cpu_spmm(const SparseTensor& sparse, const Tensor& dense,
                int64_t M, int64_t K, int64_t N) {
    DType dtype = dense.dtype();
    auto dense_c = dense.contiguous();

    // For CSR, try MKL first
    if (sparse.layout() == SparseLayout::CSR) {
#ifdef TENZOR_USE_MKL
        if (dtype == DType::Float32) {
            return mkl_csr_spmm_f32(sparse, dense_c, M, K, N);
        } else if (dtype == DType::Float64) {
            return mkl_csr_spmm_f64(sparse, dense_c, M, K, N);
        }
#endif
        // Fallback scalar CSR
        auto result = zeros({M, N}, dtype, Device::cpu());
        auto crow = sparse.crow_indices().contiguous();
        auto col = sparse.col_indices().contiguous();
        auto vals = sparse.values().contiguous();

        if (dtype == DType::Float32) {
            fallback_csr_spmm<float>(crow.data<int64_t>(), col.data<int64_t>(),
                                      vals.data<float>(), dense_c.data<float>(),
                                      result.data<float>(), M, N);
        } else if (dtype == DType::Float64) {
            fallback_csr_spmm<double>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<double>(), dense_c.data<double>(),
                                       result.data<double>(), M, N);
        }
        return result;
    }

    if (sparse.layout() == SparseLayout::COO) {
        // For COO with MKL: convert to CSR first, then use MKL
#ifdef TENZOR_USE_MKL
        if (dtype == DType::Float32 || dtype == DType::Float64) {
            auto csr = sparse.to_csr();
            if (dtype == DType::Float32) {
                return mkl_csr_spmm_f32(csr, dense_c, M, K, N);
            } else {
                return mkl_csr_spmm_f64(csr, dense_c, M, K, N);
            }
        }
#endif
        // Fallback scalar COO
        auto coo = sparse.is_coalesced() ? sparse : sparse.coalesce();
        auto result = zeros({M, N}, dtype, Device::cpu());
        auto idx = coo.indices().contiguous();
        auto vals = coo.values().contiguous();
        int64_t nnz = coo.nnz();

        if (dtype == DType::Float32) {
            fallback_coo_spmm<float>(idx.data<int64_t>(), vals.data<float>(),
                                      dense_c.data<float>(), result.data<float>(), nnz, N);
        } else if (dtype == DType::Float64) {
            fallback_coo_spmm<double>(idx.data<int64_t>(), vals.data<double>(),
                                       dense_c.data<double>(), result.data<double>(), nnz, N);
        }
        return result;
    }

    // CSC/BSR: convert to CSR and recurse
    auto csr = sparse.to_csr();
    return cpu_spmm(csr, dense, M, K, N);
}

/// CPU SpMV implementation: tries MKL for CSR, falls back to scalar loops.
Tensor cpu_spmv(const SparseTensor& sparse, const Tensor& vec,
                int64_t M, int64_t K) {
    DType dtype = vec.dtype();
    auto vec_c = vec.contiguous();

    // For CSR, try MKL first
    if (sparse.layout() == SparseLayout::CSR) {
#ifdef TENZOR_USE_MKL
        if (dtype == DType::Float32) {
            return mkl_csr_spmv_f32(sparse, vec_c, M, K);
        } else if (dtype == DType::Float64) {
            return mkl_csr_spmv_f64(sparse, vec_c, M, K);
        }
#endif
        // Fallback scalar CSR
        auto result = zeros({M}, dtype, Device::cpu());
        auto crow = sparse.crow_indices().contiguous();
        auto col = sparse.col_indices().contiguous();
        auto vals = sparse.values().contiguous();

        if (dtype == DType::Float32) {
            fallback_csr_spmv<float>(crow.data<int64_t>(), col.data<int64_t>(),
                                      vals.data<float>(), vec_c.data<float>(),
                                      result.data<float>(), M);
        } else if (dtype == DType::Float64) {
            fallback_csr_spmv<double>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<double>(), vec_c.data<double>(),
                                       result.data<double>(), M);
        }
        return result;
    }

    if (sparse.layout() == SparseLayout::COO) {
        // For COO with MKL: convert to CSR first
#ifdef TENZOR_USE_MKL
        if (dtype == DType::Float32 || dtype == DType::Float64) {
            auto csr = sparse.to_csr();
            if (dtype == DType::Float32) {
                return mkl_csr_spmv_f32(csr, vec_c, M, K);
            } else {
                return mkl_csr_spmv_f64(csr, vec_c, M, K);
            }
        }
#endif
        // Fallback scalar COO
        auto coo = sparse.is_coalesced() ? sparse : sparse.coalesce();
        auto result = zeros({M}, dtype, Device::cpu());
        auto idx = coo.indices().contiguous();
        auto vals = coo.values().contiguous();
        int64_t nnz = coo.nnz();

        if (dtype == DType::Float32) {
            fallback_coo_spmv<float>(idx.data<int64_t>(), vals.data<float>(),
                                      vec_c.data<float>(), result.data<float>(), nnz);
        } else if (dtype == DType::Float64) {
            fallback_coo_spmv<double>(idx.data<int64_t>(), vals.data<double>(),
                                       vec_c.data<double>(), result.data<double>(), nnz);
        }
        return result;
    }

    // CSC/BSR: convert to CSR and recurse
    auto csr = sparse.to_csr();
    return cpu_spmv(csr, vec, M, K);
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

#ifdef TENZOR_HAS_ROCSPARSE
    if (should_use_rocm(sparse_compute, dense_compute)) {
        auto result = rocm::rocm_spmm_kernel(sparse_compute, dense_compute);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }
#endif

#ifdef TENZOR_HAS_ONEMKL
    if (should_use_oneapi(sparse_compute, dense_compute)) {
        auto result = oneapi::oneapi_spmm_kernel(sparse_compute, dense_compute);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }
#endif

    // Vulkan path: dispatch via OpId table (backend loaded dynamically)
    if (should_use_vulkan(sparse_compute, dense_compute)) {
        auto csr = sparse_compute.to_csr();
        std::vector<Tensor> inputs = {csr.crow_indices(), csr.col_indices(), csr.values(), dense_compute};
        OpAttributes attrs;
        attrs.set(AttrKey::M, M);
        attrs.set(AttrKey::K, K);
        attrs.set(AttrKey::N, N);
        auto& table = DispatchTableRegistry::get_table(Device::Type::Vulkan);
        auto results = table.dispatch(OpId::SparseSpMM, inputs, attrs);
        auto result = results[0];
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }

    // CPU path: MKL-accelerated with scalar fallback
    auto result = cpu_spmm(sparse_compute, dense_compute, M, K, N);

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

#ifdef TENZOR_HAS_ROCSPARSE
    if (should_use_rocm_vec(sparse_compute, vec_compute)) {
        auto result = rocm::rocm_spmv_kernel(sparse_compute, vec_compute);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }
#endif

#ifdef TENZOR_HAS_ONEMKL
    if (should_use_oneapi_vec(sparse_compute, vec_compute)) {
        auto result = oneapi::oneapi_spmv_kernel(sparse_compute, vec_compute);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }
#endif

    // Vulkan path: dispatch via OpId table
    if (should_use_vulkan_vec(sparse_compute, vec_compute)) {
        auto csr = sparse_compute.to_csr();
        std::vector<Tensor> inputs = {csr.crow_indices(), csr.col_indices(), csr.values(), vec_compute};
        OpAttributes attrs;
        attrs.set(AttrKey::M, M);
        attrs.set(AttrKey::K, K);
        auto& table = DispatchTableRegistry::get_table(Device::Type::Vulkan);
        auto results = table.dispatch(OpId::SparseSpMV, inputs, attrs);
        auto result = results[0];
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    }

    // CPU path: MKL-accelerated with scalar fallback
    auto result = cpu_spmv(sparse_compute, vec_compute, M, K);

    return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
}

// ============================================================================
// add: Sparse + Dense
// ============================================================================

auto add(const SparseTensor& sparse, const Tensor& dense) -> Tensor {
    // Promote dtypes if they don't match
    DType common_dtype = promote_types(sparse.dtype(), dense.dtype());
    if (sparse.dtype() != common_dtype || dense.dtype() != common_dtype) {
        auto sparse_promoted = (sparse.dtype() != common_dtype)
            ? SparseTensor::sparse_coo(sparse.indices(), sparse.values().to(common_dtype),
                std::vector<int64_t>(sparse.shape().begin(), sparse.shape().end()))
            : sparse;
        auto dense_promoted = (dense.dtype() != common_dtype) ? dense.to(common_dtype) : dense;
        return add(sparse_promoted, dense_promoted);
    }

    // Vulkan path: dispatch via OpId table
    if (should_use_vulkan(sparse, dense)) {
        auto csr = sparse.to_csr();
        auto sp_shape = sparse.shape();
        std::vector<Tensor> inputs = {csr.crow_indices(), csr.col_indices(), csr.values(), dense};
        OpAttributes attrs;
        attrs.set(AttrKey::M, sp_shape[0]);
        attrs.set(AttrKey::K, sp_shape.size() > 1 ? sp_shape[1] : int64_t(1));
        auto& table = DispatchTableRegistry::get_table(Device::Type::Vulkan);
        auto results = table.dispatch(OpId::SparseAdd, inputs, attrs);
        return results[0];
    }

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
        #pragma omp parallel for schedule(static) if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    } else if (result_c.dtype() == DType::Float64) {
        auto* r = result_c.data<double>();
        auto* d = dense_c.data<double>();
        #pragma omp parallel for schedule(static) if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    } else if (result_c.dtype() == DType::Int32) {
        auto* r = result_c.data<int32_t>();
        auto* d = dense_c.data<int32_t>();
        #pragma omp parallel for schedule(static) if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    } else if (result_c.dtype() == DType::Int64) {
        auto* r = result_c.data<int64_t>();
        auto* d = dense_c.data<int64_t>();
        #pragma omp parallel for schedule(static) if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    } else {
        throw std::runtime_error("sparse::add: unsupported dtype " +
            std::string(dtype_name(result_c.dtype())));
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
    auto vals = sparse.values().contiguous();
    int64_t nnz = sparse.nnz();

    auto new_values = Tensor({nnz}, vals.dtype(), vals.device());
    if (vals.dtype() == DType::Float32) {
        auto* src = vals.data<float>();
        auto* dst = new_values.data<float>();
        float s = static_cast<float>(scalar);
        #pragma omp parallel for schedule(static) if(nnz > 65536)
        for (int64_t i = 0; i < nnz; ++i) {
            dst[i] = src[i] * s;
        }
    } else if (vals.dtype() == DType::Float64) {
        auto* src = vals.data<double>();
        auto* dst = new_values.data<double>();
        #pragma omp parallel for schedule(static) if(nnz > 65536)
        for (int64_t i = 0; i < nnz; ++i) {
            dst[i] = src[i] * scalar;
        }
    } else if (vals.dtype() == DType::Int32) {
        auto* src = vals.data<int32_t>();
        auto* dst = new_values.data<int32_t>();
        int32_t s = static_cast<int32_t>(scalar);
        #pragma omp parallel for schedule(static) if(nnz > 65536)
        for (int64_t i = 0; i < nnz; ++i) {
            dst[i] = src[i] * s;
        }
    } else if (vals.dtype() == DType::Int64) {
        auto* src = vals.data<int64_t>();
        auto* dst = new_values.data<int64_t>();
        int64_t s = static_cast<int64_t>(scalar);
        #pragma omp parallel for schedule(static) if(nnz > 65536)
        for (int64_t i = 0; i < nnz; ++i) {
            dst[i] = src[i] * s;
        }
    } else {
        throw std::runtime_error("sparse::mul: unsupported dtype " +
            std::string(dtype_name(vals.dtype())));
    }

    auto shape_vec = std::vector<int64_t>(sparse.shape().begin(), sparse.shape().end());
    if (sparse.layout() == SparseLayout::COO) {
        return SparseTensor::sparse_coo(sparse.indices(), new_values, shape_vec);
    } else {
        return SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(), new_values, shape_vec);
    }
}

} // namespace sparse
} // namespace tenzor
