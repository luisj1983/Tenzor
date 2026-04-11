#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
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
// Available with or without rocSPARSE (native HIP fallback when rocSPARSE absent)
#ifdef TENZOR_ROCM_BACKEND
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

#ifdef TENZOR_ROCM_BACKEND
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

#ifdef TENZOR_ROCM_BACKEND
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

// ============================================================================
// spgemm: Sparse-Sparse Matrix Multiplication (C = A @ B)
// ============================================================================

namespace {

/// CPU SpGEMM: two-phase symbolic + numeric algorithm.
/// Both inputs must be in CSR format. Returns CSR result.
template<typename T>
SparseTensor cpu_spgemm_typed(const SparseTensor& a, const SparseTensor& b,
                               int64_t M, int64_t K, int64_t N) {
    auto a_crow = a.crow_indices().contiguous();
    auto a_col = a.col_indices().contiguous();
    auto a_vals = a.values().contiguous();
    auto b_crow = b.crow_indices().contiguous();
    auto b_col = b.col_indices().contiguous();
    auto b_vals = b.values().contiguous();

    const int64_t* a_row_ptr = a_crow.data<int64_t>();
    const int64_t* a_col_idx = a_col.data<int64_t>();
    const T* a_data = a_vals.data<T>();
    const int64_t* b_row_ptr = b_crow.data<int64_t>();
    const int64_t* b_col_idx = b_col.data<int64_t>();
    const T* b_data = b_vals.data<T>();

    // Phase 1: Symbolic — count non-zeros per row in C
    std::vector<int64_t> c_row_ptr(M + 1, 0);
    std::vector<std::vector<std::pair<int64_t, T>>> row_entries(M);

    for (int64_t i = 0; i < M; ++i) {
        // Use a map to accumulate column entries for row i
        std::unordered_map<int64_t, T> acc;
        for (int64_t ja = a_row_ptr[i]; ja < a_row_ptr[i + 1]; ++ja) {
            int64_t k = a_col_idx[ja];
            T a_val = a_data[ja];
            for (int64_t jb = b_row_ptr[k]; jb < b_row_ptr[k + 1]; ++jb) {
                int64_t col = b_col_idx[jb];
                acc[col] += a_val * b_data[jb];
            }
        }
        // Sort by column index for CSR ordering
        row_entries[i].reserve(acc.size());
        for (auto& [col, val] : acc) {
            if (val != T(0)) {
                row_entries[i].emplace_back(col, val);
            }
        }
        std::sort(row_entries[i].begin(), row_entries[i].end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        c_row_ptr[i + 1] = static_cast<int64_t>(row_entries[i].size());
    }

    // Prefix sum to get row pointers
    for (int64_t i = 0; i < M; ++i) {
        c_row_ptr[i + 1] += c_row_ptr[i];
    }
    int64_t nnz_c = c_row_ptr[M];

    // Phase 2: Build CSR arrays
    auto crow_tensor = Tensor({M + 1}, DType::Int64, Device::cpu());
    auto col_tensor = Tensor({nnz_c}, DType::Int64, Device::cpu());
    auto val_tensor = Tensor({nnz_c}, a.dtype(), Device::cpu());

    std::memcpy(crow_tensor.data<int64_t>(), c_row_ptr.data(), (M + 1) * sizeof(int64_t));

    int64_t* c_col = col_tensor.data<int64_t>();
    T* c_data = val_tensor.data<T>();

    for (int64_t i = 0; i < M; ++i) {
        int64_t offset = c_row_ptr[i];
        for (size_t j = 0; j < row_entries[i].size(); ++j) {
            c_col[offset + static_cast<int64_t>(j)] = row_entries[i][j].first;
            c_data[offset + static_cast<int64_t>(j)] = row_entries[i][j].second;
        }
    }

    return SparseTensor::sparse_csr(crow_tensor, col_tensor, val_tensor, {M, N});
}

} // anonymous namespace

auto spgemm(const SparseTensor& a, const SparseTensor& b) -> SparseTensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("spgemm: both inputs must be 2D sparse matrices");
    }
    int64_t M = a_shape[0];
    int64_t K = a_shape[1];
    int64_t N = b_shape[1];
    if (K != b_shape[0]) {
        throw std::runtime_error("spgemm: inner dimensions must match (A is " +
            std::to_string(M) + "x" + std::to_string(K) + ", B is " +
            std::to_string(b_shape[0]) + "x" + std::to_string(N) + ")");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("spgemm: dtype mismatch");
    }

    // Convert to CSR for efficient row-based iteration
    auto a_csr = a.to_csr();
    auto b_csr = b.to_csr();

    // GPU SpGEMM is not yet implemented. The top-level dispatch is CPU-only:
    // we transfer inputs to CPU if needed and bring the result back to the
    // original device. A proper implementation requires routing through the
    // OpId::SparseSpMM-style dispatch table with cuSPARSE / rocSPARSE-native
    // kernels (cusparseSpGEMM_workEstimation → cusparseSpGEMM_compute, or
    // rocsparse_spgemm with a cached workspace). Tracked as a follow-up.
    Device target_device = a.device();
    SparseTensor a_compute = (target_device.type != Device::Type::CPU)
        ? a_csr.to(Device::cpu()) : a_csr;
    SparseTensor b_compute = (target_device.type != Device::Type::CPU)
        ? b_csr.to(Device::cpu()) : b_csr;

    auto finish = [&](SparseTensor r) {
        return (target_device.type != Device::Type::CPU) ? r.to(target_device) : r;
    };

    if (a.dtype() == DType::Float32) {
        return finish(cpu_spgemm_typed<float>(a_compute, b_compute, M, K, N));
    }
    if (a.dtype() == DType::Float64) {
        return finish(cpu_spgemm_typed<double>(a_compute, b_compute, M, K, N));
    }
    throw std::runtime_error("spgemm: unsupported dtype " +
        std::string(dtype_name(a.dtype())));
}

// ============================================================================
// sparse_triangular_solve: Solve L*x = b or U*x = b
// ============================================================================

namespace {

/// CPU forward substitution for lower triangular: L*x = b
template<typename T>
void cpu_forward_substitution(const int64_t* row_ptr, const int64_t* col_idx,
                               const T* vals, const T* b_data, T* x_data,
                               int64_t N) {
    for (int64_t i = 0; i < N; ++i) {
        T sum = b_data[i];
        T diag = T(0);
        for (int64_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int64_t col = col_idx[j];
            if (col < i) {
                sum -= vals[j] * x_data[col];
            } else if (col == i) {
                diag = vals[j];
            }
        }
        if (diag == T(0)) {
            throw std::runtime_error("sparse_triangular_solve: zero diagonal element at row " +
                std::to_string(i));
        }
        x_data[i] = sum / diag;
    }
}

/// CPU backward substitution for upper triangular: U*x = b
template<typename T>
void cpu_backward_substitution(const int64_t* row_ptr, const int64_t* col_idx,
                                const T* vals, const T* b_data, T* x_data,
                                int64_t N) {
    for (int64_t i = N - 1; i >= 0; --i) {
        T sum = b_data[i];
        T diag = T(0);
        for (int64_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int64_t col = col_idx[j];
            if (col > i) {
                sum -= vals[j] * x_data[col];
            } else if (col == i) {
                diag = vals[j];
            }
        }
        if (diag == T(0)) {
            throw std::runtime_error("sparse_triangular_solve: zero diagonal element at row " +
                std::to_string(i));
        }
        x_data[i] = sum / diag;
    }
}

/// CPU triangular solve for a single RHS vector
template<typename T>
Tensor cpu_sparse_trsv(const SparseTensor& L, const Tensor& b, bool upper, int64_t N) {
    auto L_crow = L.crow_indices().contiguous();
    auto L_col = L.col_indices().contiguous();
    auto L_vals = L.values().contiguous();
    auto b_c = b.contiguous();

    auto result = Tensor({N}, b.dtype(), Device::cpu());
    const T* b_data = b_c.data<T>();
    T* x_data = result.data<T>();

    if (upper) {
        cpu_backward_substitution<T>(L_crow.data<int64_t>(), L_col.data<int64_t>(),
                                      L_vals.data<T>(), b_data, x_data, N);
    } else {
        cpu_forward_substitution<T>(L_crow.data<int64_t>(), L_col.data<int64_t>(),
                                     L_vals.data<T>(), b_data, x_data, N);
    }
    return result;
}

/// CPU triangular solve for multiple RHS columns (N x K)
template<typename T>
Tensor cpu_sparse_trsm(const SparseTensor& L, const Tensor& B, bool upper, int64_t N, int64_t K) {
    auto L_crow = L.crow_indices().contiguous();
    auto L_col = L.col_indices().contiguous();
    auto L_vals = L.values().contiguous();
    auto B_c = B.contiguous();

    auto result = Tensor({N, K}, B.dtype(), Device::cpu());
    const T* B_data = B_c.data<T>();
    T* X_data = result.data<T>();

    // Solve column by column
    for (int64_t k = 0; k < K; ++k) {
        // Extract column k of B into a temporary buffer
        std::vector<T> b_col(N);
        for (int64_t i = 0; i < N; ++i) {
            b_col[i] = B_data[i * K + k];
        }
        std::vector<T> x_col(N);
        if (upper) {
            cpu_backward_substitution<T>(L_crow.data<int64_t>(), L_col.data<int64_t>(),
                                          L_vals.data<T>(), b_col.data(), x_col.data(), N);
        } else {
            cpu_forward_substitution<T>(L_crow.data<int64_t>(), L_col.data<int64_t>(),
                                         L_vals.data<T>(), b_col.data(), x_col.data(), N);
        }
        for (int64_t i = 0; i < N; ++i) {
            X_data[i * K + k] = x_col[i];
        }
    }
    return result;
}

} // anonymous namespace

auto sparse_triangular_solve(const SparseTensor& L, const Tensor& b, bool upper) -> Tensor {
    auto L_shape = L.shape();
    if (L_shape.size() != 2 || L_shape[0] != L_shape[1]) {
        throw std::runtime_error("sparse_triangular_solve: L must be a square 2D sparse matrix");
    }
    int64_t N = L_shape[0];

    if (b.ndim() == 1) {
        if (b.shape()[0] != N) {
            throw std::runtime_error("sparse_triangular_solve: dimension mismatch");
        }
    } else if (b.ndim() == 2) {
        if (b.shape()[0] != N) {
            throw std::runtime_error("sparse_triangular_solve: dimension mismatch");
        }
    } else {
        throw std::runtime_error("sparse_triangular_solve: b must be 1D or 2D");
    }

    // Convert to CSR for row-based access
    auto L_csr = L.to_csr();

    // GPU triangular solve is not yet implemented. The top-level dispatch is
    // CPU-only: we transfer inputs to CPU and bring the result back. A proper
    // implementation requires cusparseSpSV_analysis+solve or
    // rocsparse_csrsv_analysis+solve with cached descriptors. Tracked as a
    // follow-up.
    Device target_device = b.device();
    SparseTensor L_compute = L_csr;
    Tensor b_compute = b;
    if (target_device.type != Device::Type::CPU) {
        L_compute = L_csr.to(Device::cpu());
        b_compute = b.to(Device::cpu());
    }

    auto finish = [&](Tensor result) {
        if (target_device.type != Device::Type::CPU) {
            result = result.to(target_device);
        }
        return result;
    };

    // CPU path (used whether inputs came from CPU or GPU)
    if (b_compute.ndim() == 1) {
        if (b.dtype() == DType::Float32) {
            return finish(cpu_sparse_trsv<float>(L_compute, b_compute, upper, N));
        } else if (b.dtype() == DType::Float64) {
            return finish(cpu_sparse_trsv<double>(L_compute, b_compute, upper, N));
        } else {
            throw std::runtime_error("sparse_triangular_solve: unsupported dtype " +
                std::string(dtype_name(b.dtype())));
        }
    } else {
        int64_t K = b_compute.shape()[1];
        if (b.dtype() == DType::Float32) {
            return finish(cpu_sparse_trsm<float>(L_compute, b_compute, upper, N, K));
        } else if (b.dtype() == DType::Float64) {
            return finish(cpu_sparse_trsm<double>(L_compute, b_compute, upper, N, K));
        } else {
            throw std::runtime_error("sparse_triangular_solve: unsupported dtype " +
                std::string(dtype_name(b.dtype())));
        }
    }
}

// ============================================================================
// SDDMM — Sampled Dense-Dense Matrix Multiplication (Phase 6.2)
// ============================================================================

namespace {

template <typename T>
auto cpu_sddmm_csr(const Tensor& mask_row_ptr,
                   const Tensor& mask_col_ind,
                   const Tensor& A,
                   const Tensor& B,
                   int64_t M,
                   int64_t N,
                   int64_t K) -> Tensor {
    const auto nnz = mask_col_ind.numel();
    Tensor values({nnz}, A.dtype(), A.device());

    const auto* rp = mask_row_ptr.data<int64_t>();
    const auto* ci = mask_col_ind.data<int64_t>();
    const auto* a  = A.data<T>();
    const auto* b  = B.data<T>();
    auto* v        = values.data<T>();

    // For each row i, walk its non-zeros and compute A[i,:] . B[j,:]
    // (treating B as row-major with B[j,k] = b[j*K + k]).
    #pragma omp parallel for if(M > 64)
    for (int64_t i = 0; i < M; ++i) {
        const int64_t start = rp[i];
        const int64_t end   = rp[i + 1];
        const T* a_row = a + i * K;
        for (int64_t p = start; p < end; ++p) {
            const int64_t j = ci[p];
            const T* b_row = b + j * K;
            T acc = T{0};
            for (int64_t k = 0; k < K; ++k) {
                acc += a_row[k] * b_row[k];
            }
            v[p] = acc;
        }
    }
    return values;
}

} // anonymous namespace

auto sddmm(const SparseTensor& mask, const Tensor& A, const Tensor& B) -> SparseTensor {
    // Shape validation.
    auto mask_shape = mask.shape();
    if (mask_shape.size() != 2) {
        throw std::runtime_error("sddmm: mask must be 2D (M, N)");
    }
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("sddmm: A and B must be 2D");
    }
    const int64_t M = mask_shape[0];
    const int64_t N = mask_shape[1];
    if (a_shape[0] != M) {
        throw std::runtime_error("sddmm: A.shape[0] must equal mask.shape[0]");
    }
    if (b_shape[0] != N) {
        throw std::runtime_error("sddmm: B.shape[0] must equal mask.shape[1] "
                                 "(B is read row-wise: B[j, :] is the j-th "
                                 "output column vector)");
    }
    if (a_shape[1] != b_shape[1]) {
        throw std::runtime_error("sddmm: A.shape[1] must equal B.shape[1] "
                                 "(shared reduction dimension K)");
    }
    if (A.dtype() != B.dtype()) {
        throw std::runtime_error("sddmm: A and B must have the same dtype");
    }

    // Normalize to CSR for uniform access.
    SparseTensor csr_mask = mask;
    if (csr_mask.layout() != SparseLayout::CSR) {
        csr_mask = csr_mask.to_csr();
    }

    const int64_t K = a_shape[1];
    Tensor values;
    if (A.dtype() == DType::Float32) {
        values = cpu_sddmm_csr<float>(csr_mask.crow_indices(),
                                      csr_mask.col_indices(),
                                      A.contiguous(), B.contiguous(),
                                      M, N, K);
    } else if (A.dtype() == DType::Float64) {
        values = cpu_sddmm_csr<double>(csr_mask.crow_indices(),
                                       csr_mask.col_indices(),
                                       A.contiguous(), B.contiguous(),
                                       M, N, K);
    } else {
        throw std::runtime_error("sddmm: unsupported dtype " +
                                 std::string(dtype_name(A.dtype())));
    }

    // Build the result CSR using the same row_ptr / col_ind as the mask
    // but with the freshly-computed dot-product values.
    return SparseTensor::sparse_csr(csr_mask.crow_indices(),
                                    csr_mask.col_indices(),
                                    values,
                                    {M, N});
}

} // namespace sparse
} // namespace tenzor
