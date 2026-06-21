#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include <algorithm>
#include <complex>
#include <cstring>
#include <limits>
#include <optional>
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

// GPU sparse kernels (CUDA/ROCm/OneAPI/Vulkan) live in their respective backend
// shared objects (e.g. tenzor_backend_cuda.so), loaded dynamically at runtime.
// sparse_ops.cpp compiles into tenzor_core.so, so it cannot call those kernels
// directly — the symbols would be unresolved at tenzor_core link time. Instead,
// every GPU sparse op routes through the OpId dispatch table (the
// dispatch_gpu_spmm/spmv/add lambdas in each top-level op below). The kernels
// register into that table from src/backends/<be>/*_kernel_registry.cpp, on the
// correct side of the dynamic boundary.
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
    // Wave G: native support for half-precision, integer, and complex
    // sparse. Float16/BFloat16 widen at-register to F32 inside the kernel
    // via sparse_acc_type traits; Int32/Int64 use native integer
    // arithmetic; Complex64/Complex128 use the matching complex template
    // type (the fallback_csr_spm* templates are generic on T).
    switch (dtype) {
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
        case DType::Int32:
        case DType::Int64:
        case DType::Complex64:
        case DType::Complex128:
            return dtype;
        default:
            throw std::runtime_error("sparse ops: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
    }
}

/// Extract a SparseTensor as CSR components (crow_indices, col_indices, values)
/// for dispatch-table routing. SparseTensor::to_csr() dispatches to GPU-native
/// conversion (cusparseXcoo2csr / rocsparse_coo2csr) when on CUDA/ROCm, and
/// falls back to the CPU histogram + prefix-sum path otherwise.
struct CsrComponents {
    Tensor crow;
    Tensor col;
    Tensor values;
};

[[maybe_unused]] CsrComponents extract_csr_on_device(const SparseTensor& sparse) {
    // SparseTensor::to_csr() now dispatches to GPU-native conversions when
    // the tensor is on CUDA/ROCm, so we can call it directly for all devices
    // without forcing a CPU round-trip.
    const SparseTensor csr = sparse.to_csr();
    return {csr.crow_indices(), csr.col_indices(), csr.values()};
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
/// NOTE: the build uses the LP64 interface (cmake/FindMKL.cmake: intel_lp64),
/// so MKL_INT is a 32-bit int. Callers MUST verify the indices/dimensions fit
/// (see mkl_index_fits) before taking the MKL path — values exceeding
/// INT32_MAX would be silently truncated here, producing wrong results.
std::vector<MKL_INT> to_mkl_int(const int64_t* src, int64_t n) {
    std::vector<MKL_INT> dst(n);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = static_cast<MKL_INT>(src[i]);
    }
    return dst;
}

/// True iff every index/dimension magnitude fits in MKL_INT (32-bit under
/// LP64). When false the caller must use the int64-correct scalar fallback,
/// otherwise to_mkl_int / dimension casts would truncate.
inline bool mkl_index_fits(const SparseTensor& sparse, int64_t M, int64_t K, int64_t N) {
    const int64_t mx = static_cast<int64_t>(std::numeric_limits<MKL_INT>::max());
    if (M > mx || K > mx || N > mx) return false;
    if (M + 1 > mx) return false;          // crow_indices length
    if (sparse.nnz() > mx) return false;   // col_indices / values length
    return true;
}

/// Create MKL Float32 CSR handle from SparseTensor.
sparse_matrix_t create_mkl_csr_f32(const SparseTensor& sparse, MKL_INT nrows, MKL_INT ncols,
                                     std::vector<MKL_INT>& crow_buf,
                                     std::vector<MKL_INT>& col_buf,
                                     Tensor& vals_keepalive) {
    auto crow = sparse.crow_indices().contiguous();
    auto col = sparse.col_indices().contiguous();
    // MKL stores the values pointer without copying, so the backing tensor must
    // outlive the handle. Hand it back to the caller via vals_keepalive instead
    // of letting a function-local tensor free its storage on return.
    vals_keepalive = sparse.values().contiguous();

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
        const_cast<float*>(vals_keepalive.data<float>())
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
                                     std::vector<MKL_INT>& col_buf,
                                     Tensor& vals_keepalive) {
    auto crow = sparse.crow_indices().contiguous();
    auto col = sparse.col_indices().contiguous();
    // MKL stores the values pointer without copying, so the backing tensor must
    // outlive the handle. Hand it back to the caller via vals_keepalive instead
    // of letting a function-local tensor free its storage on return.
    vals_keepalive = sparse.values().contiguous();

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
        const_cast<double*>(vals_keepalive.data<double>())
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
    Tensor vals_keepalive;
    auto handle = create_mkl_csr_f32(sparse, M, K, crow_buf, col_buf, vals_keepalive);
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
    Tensor vals_keepalive;
    auto handle = create_mkl_csr_f64(sparse, M, K, crow_buf, col_buf, vals_keepalive);
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
    Tensor vals_keepalive;
    auto handle = create_mkl_csr_f32(sparse, M, K, crow_buf, col_buf, vals_keepalive);
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
    Tensor vals_keepalive;
    auto handle = create_mkl_csr_f64(sparse, M, K, crow_buf, col_buf, vals_keepalive);
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

// Wave G: per-element-T accumulator selection. Half-precision (F16/BF16)
// accumulate in float (per-element widen — single instruction, NOT a
// tensor-wide widen-narrow). Integer and complex use T directly.
template<typename T> struct sparse_acc_type { using type = T; };
template<> struct sparse_acc_type<Float16>   { using type = float; };
template<> struct sparse_acc_type<BFloat16>  { using type = float; };

/// Fallback SpMV: CSR format, y = A * x
template<typename T>
void fallback_csr_spmv(const int64_t* crow_ptr, const int64_t* col_ptr,
                        const T* vals, const T* x, T* y,
                        int64_t nrows) {
    using Acc = typename sparse_acc_type<T>::type;
    #pragma omp parallel for schedule(static) if(nrows > 128)
    for (int64_t row = 0; row < nrows; ++row) {
        Acc sum = Acc(0);
        for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
            sum += static_cast<Acc>(vals[j]) * static_cast<Acc>(x[col_ptr[j]]);
        }
        y[row] = static_cast<T>(sum);
    }
}

/// Fallback SpMM: CSR format, C = A * B where B is (K, N) row-major
template<typename T>
void fallback_csr_spmm(const int64_t* crow_ptr, const int64_t* col_ptr,
                        const T* vals, const T* B, T* C,
                        int64_t M, int64_t N) {
    using Acc = typename sparse_acc_type<T>::type;
    if constexpr (std::is_same_v<T, Acc>) {
        // Fast path for F32/F64/int/complex — direct accumulation in T.
        #pragma omp parallel for schedule(static) if(M > 64)
        for (int64_t row = 0; row < M; ++row) {
            T* c_row = C + row * N;
            // std::fill is correct for complex<T> (memset on complex is
            // bit-pattern-correct on IEEE-754 but the warning is fair).
            std::fill(c_row, c_row + N, T{});
            for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                int64_t col = col_ptr[j];
                T val = vals[j];
                const T* b_row = B + col * N;
                for (int64_t n = 0; n < N; ++n) {
                    c_row[n] += val * b_row[n];
                }
            }
        }
    } else {
        // Half-precision path: accumulate each row in an F32 scratch buffer and
        // narrow to T at end-of-row. Allocate ONE scratch buffer per thread
        // outside the row loop (mirroring the COO SpMM half-path), zeroing it at
        // the start of each row — rather than heap-allocating a std::vector per
        // row, which would be M allocations of size N on the hot path.
        #pragma omp parallel
        {
            std::vector<Acc> acc(static_cast<size_t>(N), Acc(0));
            #pragma omp for schedule(static)
            for (int64_t row = 0; row < M; ++row) {
                T* c_row = C + row * N;
                for (int64_t n = 0; n < N; ++n) acc[static_cast<size_t>(n)] = Acc(0);
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    int64_t col = col_ptr[j];
                    Acc val = static_cast<Acc>(vals[j]);
                    const T* b_row = B + col * N;
                    for (int64_t n = 0; n < N; ++n) {
                        acc[static_cast<size_t>(n)] += val * static_cast<Acc>(b_row[n]);
                    }
                }
                for (int64_t n = 0; n < N; ++n) c_row[n] = static_cast<T>(acc[static_cast<size_t>(n)]);
            }
        }
    }
}

/// Fallback COO SpMV: y = A * x (cannot be parallelized due to race on y)
template<typename T>
void fallback_coo_spmv(const int64_t* idx_ptr, const T* vals, const T* x, T* y,
                        int64_t nnz, int64_t M = 0) {
    using Acc = typename sparse_acc_type<T>::type;
    // Wave G2 (deferred → landed): native parallel COO SpMV via row-partition.
    // When the COO is row-sorted (the callers coalesce before dispatching here),
    // we can pre-compute row_start[r] in a single O(nnz) scan and then run an
    // OpenMP parallel-for over rows — no atomics, no false sharing on `y`.
    //
    // M = 0 disables parallelization (caller didn't supply row count); falls
    // back to the serial path that matches the pre-Wave-G2 behaviour.
    if (M > 0 && nnz >= 4096) {
        // M1 fix: previously the row_start derivation only recorded the FIRST
        // occurrence of each row; for unsorted COO this silently dropped
        // subsequent entries that fell after entries of a later row. Now we
        // verify the COO is non-decreasing (the caller coalesce contract);
        // if not, fall back to the serial path which handles unsorted COO
        // correctly.
        bool is_row_sorted = true;
        int64_t prev_row = -1;
        for (int64_t i = 0; i < nnz; ++i) {
            int64_t r = idx_ptr[i];
            if (r < prev_row) { is_row_sorted = false; break; }
            prev_row = r;
        }
        if (is_row_sorted) {
            // valid_end marks one-past the last in-range entry. Because the COO
            // is row-sorted, out-of-range entries with r<0 sit at the front and
            // r>=M at the tail; valid_end excludes the trailing r>=M block so the
            // last valid row's sweep does not read out-of-range columns (matching
            // the serial path, which skips every out-of-range entry).
            int64_t valid_end = nnz;
            std::vector<int64_t> row_start(static_cast<size_t>(M) + 1, nnz);
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t r = idx_ptr[i];
                if (r < 0 || r >= M) continue;
                if (row_start[static_cast<size_t>(r)] == nnz) {
                    row_start[static_cast<size_t>(r)] = i;
                }
                valid_end = i + 1;
            }
            // The sentinel row_start[M] bounds the last valid row's sweep; clamp
            // it to valid_end so trailing r>=M entries are not swept.
            row_start[static_cast<size_t>(M)] = valid_end;
            // Fill forward: rows with no entries inherit the next row's start.
            for (int64_t r = M - 1; r >= 0; --r) {
                if (row_start[static_cast<size_t>(r)] == nnz) {
                    row_start[static_cast<size_t>(r)] = row_start[static_cast<size_t>(r) + 1];
                }
            }
            #pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < M; ++r) {
                int64_t start = row_start[static_cast<size_t>(r)];
                int64_t end   = row_start[static_cast<size_t>(r) + 1];
                if constexpr (std::is_same_v<T, Acc>) {
                    T acc = T{};
                    for (int64_t i = start; i < end; ++i) {
                        int64_t col = idx_ptr[nnz + i];
                        acc += vals[i] * x[col];
                    }
                    y[r] = acc;
                } else {
                    Acc acc = Acc{};
                    for (int64_t i = start; i < end; ++i) {
                        int64_t col = idx_ptr[nnz + i];
                        acc += static_cast<Acc>(vals[i]) * static_cast<Acc>(x[col]);
                    }
                    y[r] = static_cast<T>(acc);
                }
            }
            return;
        }
        // Unsorted COO — fall through to the serial path below. (The
        // serial path correctly handles unsorted COO since it
        // accumulates `y[row] += ...` regardless of order.)
    }
    // Serial fallback (small nnz, or M=0 from a legacy caller).
    // Skip out-of-range rows to stay consistent with the parallel row-partition
    // path (which drops r<0 || r>=M). Otherwise the same bad input would be
    // silently dropped on one path and cause an out-of-bounds write here. Row
    // bounds are only known when the caller supplied a row count (M>0).
    if constexpr (std::is_same_v<T, Acc>) {
        for (int64_t i = 0; i < nnz; ++i) {
            int64_t row = idx_ptr[i];
            if (M > 0 && (row < 0 || row >= M)) continue;
            int64_t col = idx_ptr[nnz + i];
            y[row] += vals[i] * x[col];
        }
    } else {
        for (int64_t i = 0; i < nnz; ++i) {
            int64_t row = idx_ptr[i];
            if (M > 0 && (row < 0 || row >= M)) continue;
            int64_t col = idx_ptr[nnz + i];
            Acc cur = static_cast<Acc>(y[row]);
            cur += static_cast<Acc>(vals[i]) * static_cast<Acc>(x[col]);
            y[row] = static_cast<T>(cur);
        }
    }
}

/// Fallback COO SpMM: C = A * B (cannot be parallelized due to race on C)
template<typename T>
void fallback_coo_spmm(const int64_t* idx_ptr, const T* vals, const T* B, T* C,
                        int64_t nnz, int64_t N, int64_t M = 0) {
    using Acc = typename sparse_acc_type<T>::type;
    // Wave G2 (deferred → landed): native parallel COO SpMM via row-partition.
    // Same row_start construction as fallback_coo_spmv; parallelize over
    // result rows. Per-row accumulator is a small N-element scratch buffer
    // (stack-allocated for typical N ≤ 1024 via VLA fallback to heap).
    if (M > 0 && nnz >= 4096) {
        // M1 fix (SpMM variant): verify row-sorted COO before the
        // parallel row-partition path. Unsorted COO falls through to
        // the serial path which handles it correctly.
        bool is_row_sorted = true;
        int64_t prev_row = -1;
        for (int64_t i = 0; i < nnz; ++i) {
            int64_t r = idx_ptr[i];
            if (r < prev_row) { is_row_sorted = false; break; }
            prev_row = r;
        }
        if (is_row_sorted) {
            // valid_end excludes the trailing r>=M block (see fallback_coo_spmv)
            // so the last valid row's sweep stays in range, matching the serial
            // path which skips every out-of-range entry.
            int64_t valid_end = nnz;
            std::vector<int64_t> row_start(static_cast<size_t>(M) + 1, nnz);
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t r = idx_ptr[i];
                if (r < 0 || r >= M) continue;
                if (row_start[static_cast<size_t>(r)] == nnz) {
                    row_start[static_cast<size_t>(r)] = i;
                }
                valid_end = i + 1;
            }
            row_start[static_cast<size_t>(M)] = valid_end;
            for (int64_t r = M - 1; r >= 0; --r) {
                if (row_start[static_cast<size_t>(r)] == nnz) {
                    row_start[static_cast<size_t>(r)] = row_start[static_cast<size_t>(r) + 1];
                }
            }
            #pragma omp parallel
            {
                std::vector<Acc> scratch(static_cast<size_t>(N), Acc{});
                #pragma omp for schedule(static)
                for (int64_t r = 0; r < M; ++r) {
                    int64_t start = row_start[static_cast<size_t>(r)];
                    int64_t end   = row_start[static_cast<size_t>(r) + 1];
                    for (int64_t n = 0; n < N; ++n) scratch[static_cast<size_t>(n)] = Acc{};
                    for (int64_t i = start; i < end; ++i) {
                        int64_t col = idx_ptr[nnz + i];
                        Acc v = static_cast<Acc>(vals[i]);
                        const T* brow = B + col * N;
                        for (int64_t n = 0; n < N; ++n) {
                            scratch[static_cast<size_t>(n)] += v * static_cast<Acc>(brow[n]);
                        }
                    }
                    T* crow = C + r * N;
                    for (int64_t n = 0; n < N; ++n) {
                        crow[n] = static_cast<T>(scratch[static_cast<size_t>(n)]);
                    }
                }
            }
            return;
        }
        // Unsorted COO — fall through to serial path.
    }
    // Serial fallback. Skip out-of-range rows to match the parallel
    // row-partition path (which drops r<0 || r>=M); otherwise the same bad
    // input would be silently dropped on one path and cause an out-of-bounds
    // write here. Row bounds are only known when the caller supplied M>0.
    if constexpr (std::is_same_v<T, Acc>) {
        for (int64_t i = 0; i < nnz; ++i) {
            int64_t row = idx_ptr[i];
            if (M > 0 && (row < 0 || row >= M)) continue;
            int64_t col = idx_ptr[nnz + i];
            T val = vals[i];
            for (int64_t n = 0; n < N; ++n) {
                C[row * N + n] += val * B[col * N + n];
            }
        }
    } else {
        for (int64_t i = 0; i < nnz; ++i) {
            int64_t row = idx_ptr[i];
            if (M > 0 && (row < 0 || row >= M)) continue;
            int64_t col = idx_ptr[nnz + i];
            Acc val = static_cast<Acc>(vals[i]);
            for (int64_t n = 0; n < N; ++n) {
                Acc cur = static_cast<Acc>(C[row * N + n]);
                cur += val * static_cast<Acc>(B[col * N + n]);
                C[row * N + n] = static_cast<T>(cur);
            }
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
        if (mkl_index_fits(sparse, M, K, N)) {
            if (dtype == DType::Float32) {
                return mkl_csr_spmm_f32(sparse, dense_c, M, K, N);
            } else if (dtype == DType::Float64) {
                return mkl_csr_spmm_f64(sparse, dense_c, M, K, N);
            }
        }
        // else: indices exceed MKL_INT (LP64 32-bit) — fall through to the
        // int64-correct scalar CSR path below to avoid silent truncation.
#endif
        // Fallback scalar CSR — natively supports F16/BF16/Int32/Int64 via
        // sparse_acc_type<T> traits in the kernel templates.
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
        } else if (dtype == DType::Float16) {
            fallback_csr_spmm<Float16>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<Float16>(), dense_c.data<Float16>(),
                                       result.data<Float16>(), M, N);
        } else if (dtype == DType::BFloat16) {
            fallback_csr_spmm<BFloat16>(crow.data<int64_t>(), col.data<int64_t>(),
                                        vals.data<BFloat16>(), dense_c.data<BFloat16>(),
                                        result.data<BFloat16>(), M, N);
        } else if (dtype == DType::Int32) {
            fallback_csr_spmm<int32_t>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<int32_t>(), dense_c.data<int32_t>(),
                                       result.data<int32_t>(), M, N);
        } else if (dtype == DType::Int64) {
            fallback_csr_spmm<int64_t>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<int64_t>(), dense_c.data<int64_t>(),
                                       result.data<int64_t>(), M, N);
        } else if (dtype == DType::Complex64) {
            fallback_csr_spmm<std::complex<float>>(
                crow.data<int64_t>(), col.data<int64_t>(),
                vals.data<std::complex<float>>(),
                dense_c.data<std::complex<float>>(),
                result.data<std::complex<float>>(), M, N);
        } else if (dtype == DType::Complex128) {
            fallback_csr_spmm<std::complex<double>>(
                crow.data<int64_t>(), col.data<int64_t>(),
                vals.data<std::complex<double>>(),
                dense_c.data<std::complex<double>>(),
                result.data<std::complex<double>>(), M, N);
        } else {
            throw std::runtime_error("cpu_spmm CSR: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
        }
        return result;
    }

    if (sparse.layout() == SparseLayout::COO) {
        // For COO with MKL: convert to CSR first, then use MKL
#ifdef TENZOR_USE_MKL
        if ((dtype == DType::Float32 || dtype == DType::Float64) &&
            mkl_index_fits(sparse, M, K, N)) {
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
                                      dense_c.data<float>(), result.data<float>(), nnz, N, M);
        } else if (dtype == DType::Float64) {
            fallback_coo_spmm<double>(idx.data<int64_t>(), vals.data<double>(),
                                       dense_c.data<double>(), result.data<double>(), nnz, N, M);
        } else if (dtype == DType::Float16) {
            fallback_coo_spmm<Float16>(idx.data<int64_t>(), vals.data<Float16>(),
                                       dense_c.data<Float16>(), result.data<Float16>(), nnz, N, M);
        } else if (dtype == DType::BFloat16) {
            fallback_coo_spmm<BFloat16>(idx.data<int64_t>(), vals.data<BFloat16>(),
                                        dense_c.data<BFloat16>(), result.data<BFloat16>(), nnz, N, M);
        } else if (dtype == DType::Int32) {
            fallback_coo_spmm<int32_t>(idx.data<int64_t>(), vals.data<int32_t>(),
                                       dense_c.data<int32_t>(), result.data<int32_t>(), nnz, N, M);
        } else if (dtype == DType::Int64) {
            fallback_coo_spmm<int64_t>(idx.data<int64_t>(), vals.data<int64_t>(),
                                       dense_c.data<int64_t>(), result.data<int64_t>(), nnz, N, M);
        } else if (dtype == DType::Complex64) {
            fallback_coo_spmm<std::complex<float>>(
                idx.data<int64_t>(),
                vals.data<std::complex<float>>(),
                dense_c.data<std::complex<float>>(),
                result.data<std::complex<float>>(), nnz, N, M);
        } else if (dtype == DType::Complex128) {
            fallback_coo_spmm<std::complex<double>>(
                idx.data<int64_t>(),
                vals.data<std::complex<double>>(),
                dense_c.data<std::complex<double>>(),
                result.data<std::complex<double>>(), nnz, N, M);
        } else {
            throw std::runtime_error("cpu_spmm COO: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
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
        if (mkl_index_fits(sparse, M, K, /*N=*/1)) {
            if (dtype == DType::Float32) {
                return mkl_csr_spmv_f32(sparse, vec_c, M, K);
            } else if (dtype == DType::Float64) {
                return mkl_csr_spmv_f64(sparse, vec_c, M, K);
            }
        }
        // else: indices exceed MKL_INT (LP64 32-bit) — use scalar fallback.
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
        } else if (dtype == DType::Float16) {
            fallback_csr_spmv<Float16>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<Float16>(), vec_c.data<Float16>(),
                                       result.data<Float16>(), M);
        } else if (dtype == DType::BFloat16) {
            fallback_csr_spmv<BFloat16>(crow.data<int64_t>(), col.data<int64_t>(),
                                        vals.data<BFloat16>(), vec_c.data<BFloat16>(),
                                        result.data<BFloat16>(), M);
        } else if (dtype == DType::Int32) {
            fallback_csr_spmv<int32_t>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<int32_t>(), vec_c.data<int32_t>(),
                                       result.data<int32_t>(), M);
        } else if (dtype == DType::Int64) {
            fallback_csr_spmv<int64_t>(crow.data<int64_t>(), col.data<int64_t>(),
                                       vals.data<int64_t>(), vec_c.data<int64_t>(),
                                       result.data<int64_t>(), M);
        } else if (dtype == DType::Complex64) {
            // Wave G1 (deferred → landed): native complex CSR SpMV.
            fallback_csr_spmv<std::complex<float>>(
                crow.data<int64_t>(), col.data<int64_t>(),
                vals.data<std::complex<float>>(),
                vec_c.data<std::complex<float>>(),
                result.data<std::complex<float>>(), M);
        } else if (dtype == DType::Complex128) {
            fallback_csr_spmv<std::complex<double>>(
                crow.data<int64_t>(), col.data<int64_t>(),
                vals.data<std::complex<double>>(),
                vec_c.data<std::complex<double>>(),
                result.data<std::complex<double>>(), M);
        } else {
            throw std::runtime_error("cpu_spmv CSR: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
        }
        return result;
    }

    if (sparse.layout() == SparseLayout::COO) {
        // For COO with MKL: convert to CSR first
#ifdef TENZOR_USE_MKL
        if ((dtype == DType::Float32 || dtype == DType::Float64) &&
            mkl_index_fits(sparse, M, K, /*N=*/1)) {
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
                                      vec_c.data<float>(), result.data<float>(), nnz, M);
        } else if (dtype == DType::Float64) {
            fallback_coo_spmv<double>(idx.data<int64_t>(), vals.data<double>(),
                                       vec_c.data<double>(), result.data<double>(), nnz, M);
        } else if (dtype == DType::Float16) {
            fallback_coo_spmv<Float16>(idx.data<int64_t>(), vals.data<Float16>(),
                                       vec_c.data<Float16>(), result.data<Float16>(), nnz, M);
        } else if (dtype == DType::BFloat16) {
            fallback_coo_spmv<BFloat16>(idx.data<int64_t>(), vals.data<BFloat16>(),
                                        vec_c.data<BFloat16>(), result.data<BFloat16>(), nnz, M);
        } else if (dtype == DType::Int32) {
            fallback_coo_spmv<int32_t>(idx.data<int64_t>(), vals.data<int32_t>(),
                                       vec_c.data<int32_t>(), result.data<int32_t>(), nnz, M);
        } else if (dtype == DType::Int64) {
            fallback_coo_spmv<int64_t>(idx.data<int64_t>(), vals.data<int64_t>(),
                                       vec_c.data<int64_t>(), result.data<int64_t>(), nnz, M);
        } else if (dtype == DType::Complex64) {
            // Wave G1 (deferred → landed): native complex COO SpMV.
            fallback_coo_spmv<std::complex<float>>(
                idx.data<int64_t>(),
                vals.data<std::complex<float>>(),
                vec_c.data<std::complex<float>>(),
                result.data<std::complex<float>>(), nnz, M);
        } else if (dtype == DType::Complex128) {
            fallback_coo_spmv<std::complex<double>>(
                idx.data<int64_t>(),
                vals.data<std::complex<double>>(),
                vec_c.data<std::complex<double>>(),
                result.data<std::complex<double>>(), nnz, M);
        } else {
            throw std::runtime_error("cpu_spmv COO: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
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

    // Promote across both operands (PyTorch semantics) so a higher-precision
    // sparse operand is not silently downcast to the dense operand's dtype.
    // orig_dtype is the result dtype the returns below cast back to.
    DType orig_dtype = promote_types(sparse.dtype(), dense.dtype());
    DType comp_dtype = compute_dtype_for(orig_dtype);

    // Cast dense and sparse values to compute dtype if needed
    Tensor dense_compute = (dense.dtype() != comp_dtype) ? dense.to(comp_dtype) : dense;
    SparseTensor sparse_compute = sparse;
    if (sparse.dtype() != comp_dtype) {
        // Cast only the value buffer; the sparsity structure is unchanged.
        // with_values() preserves the layout and its index members for every
        // layout (COO/CSR/CSC/BSR) — the old CSR-only else-branch threw on
        // CSC/BSR inputs with default-constructed crow/col members.
        sparse_compute = sparse.with_values(sparse.values().to(comp_dtype));
    }

    // GPU paths — CUDA / ROCm / OneAPI / Vulkan all route through their
    // respective dispatch tables so the call crosses the backend .so
    // boundary. The direct `tenzor::cuda::*` / `tenzor::rocm::*` calls
    // that used to live here only worked from *inside* the backend .so
    // because the TENZOR_HAS_CUSPARSE / TENZOR_HAS_ROCSPARSE macros are
    // only defined there; from tenzor_core they were compiled out and
    // the function silently fell through to cpu_spmm on device pointers.
    auto dispatch_gpu_spmm = [&](Device::Type dev_type) -> std::optional<Tensor> {
        auto& table = DispatchTableRegistry::get_table(dev_type);
        if (!table.has_kernel(OpId::SparseSpMM)) return std::nullopt;
        // Move the sparse tensor to the target device before extracting CSR —
        // otherwise the CSR components stay on CPU and the backend kernel
        // receives a mix of CPU and device pointers, which either fails the
        // device-consistency check or crashes when dereferenced on-device.
        // Preserve the on-device operand's full Device (type AND index) so a
        // non-zero GPU index (e.g. cuda:1) is honoured; hardcoding index 0 would
        // send moved operands to cuda:0 while on-device operands stay on cuda:1,
        // handing the kernel pointers from two physical devices.
        Device target_dev = (sparse_compute.device().type == dev_type)
                                 ? sparse_compute.device()
                                 : dense_compute.device();
        SparseTensor sparse_on_dev = (sparse_compute.device() == target_dev)
                                         ? sparse_compute
                                         : sparse_compute.to(target_dev);
        auto ac = extract_csr_on_device(sparse_on_dev);
        Tensor crow = (ac.crow.device() == target_dev) ? ac.crow : ac.crow.to(target_dev);
        Tensor col  = (ac.col.device()  == target_dev) ? ac.col  : ac.col.to(target_dev);
        Tensor vals = (ac.values.device() == target_dev) ? ac.values : ac.values.to(target_dev);
        Tensor dense_on_dev = (dense_compute.device() == target_dev)
                                 ? dense_compute
                                 : dense_compute.to(target_dev);
        std::vector<Tensor> inputs = {crow, col, vals, dense_on_dev};
        OpAttributes attrs;
        attrs.set(AttrKey::M, M);
        attrs.set(AttrKey::K, K);
        attrs.set(AttrKey::N, N);
        auto result = table.dispatch_single(OpId::SparseSpMM, inputs, attrs);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    };

    if (sparse_compute.device().type == Device::Type::CUDA ||
        dense_compute.device().type == Device::Type::CUDA) {
        if (auto r = dispatch_gpu_spmm(Device::Type::CUDA)) return *r;
    }
    if (sparse_compute.device().type == Device::Type::ROCm ||
        dense_compute.device().type == Device::Type::ROCm) {
        if (auto r = dispatch_gpu_spmm(Device::Type::ROCm)) return *r;
    }
    if (sparse_compute.device().type == Device::Type::OneAPI ||
        dense_compute.device().type == Device::Type::OneAPI) {
        if (auto r = dispatch_gpu_spmm(Device::Type::OneAPI)) return *r;
    }
    if (sparse_compute.device().type == Device::Type::Vulkan ||
        dense_compute.device().type == Device::Type::Vulkan) {
        if (auto r = dispatch_gpu_spmm(Device::Type::Vulkan)) return *r;
    }

    // Refuse CPU fallback for GPU tensors (mirrors sparse::add). cpu_spmm calls
    // .data<float>()/.data<int64_t>() which dereference device pointers on the
    // host for a GPU tensor — crash/UB. If a GPU backend lacks the SparseSpMM
    // kernel, fail loudly instead of silently falling back to CPU.
    if (sparse_compute.device().type != Device::Type::CPU ||
        dense_compute.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "sparse::spmm: GPU tensor but no GPU SparseSpMM kernel matched — "
            "refusing CPU fallback (move tensors to CPU explicitly)");
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

    // Promote across both operands (see spmm) so a higher-precision sparse
    // operand is not silently downcast to the vector's dtype.
    DType orig_dtype = promote_types(sparse.dtype(), vec.dtype());
    DType comp_dtype = compute_dtype_for(orig_dtype);

    // Cast to compute dtype if needed
    Tensor vec_compute = (vec.dtype() != comp_dtype) ? vec.to(comp_dtype) : vec;
    SparseTensor sparse_compute = sparse;
    if (sparse.dtype() != comp_dtype) {
        // Cast only the value buffer; structure is preserved. with_values()
        // keeps the correct index members for every layout (COO/CSR/CSC/BSR),
        // unlike the old CSR-only branch which threw on CSC/BSR inputs.
        sparse_compute = sparse.with_values(sparse.values().to(comp_dtype));
    }

    // GPU path for spmv — symmetric to spmm above. See the long comment
    // on spmm() for why direct `cuda::*` calls don't work from here.
    auto dispatch_gpu_spmv = [&](Device::Type dev_type) -> std::optional<Tensor> {
        auto& table = DispatchTableRegistry::get_table(dev_type);
        if (!table.has_kernel(OpId::SparseSpMV)) return std::nullopt;
        // Move sparse to device first so CSR components don't stay on CPU.
        // See dispatch_gpu_spmm for the full rationale.
        // Preserve the on-device operand's full Device (type AND index) so a
        // non-zero GPU index (e.g. cuda:1) is honoured rather than collapsing
        // moved operands onto index 0. See dispatch_gpu_spmm.
        Device target_dev = (sparse_compute.device().type == dev_type)
                                 ? sparse_compute.device()
                                 : vec_compute.device();
        SparseTensor sparse_on_dev = (sparse_compute.device() == target_dev)
                                         ? sparse_compute
                                         : sparse_compute.to(target_dev);
        auto sc = extract_csr_on_device(sparse_on_dev);
        Tensor crow = (sc.crow.device() == target_dev) ? sc.crow : sc.crow.to(target_dev);
        Tensor col  = (sc.col.device()  == target_dev) ? sc.col  : sc.col.to(target_dev);
        Tensor vals = (sc.values.device() == target_dev) ? sc.values : sc.values.to(target_dev);
        Tensor vec_on_dev = (vec_compute.device() == target_dev)
                                ? vec_compute
                                : vec_compute.to(target_dev);
        std::vector<Tensor> inputs = {crow, col, vals, vec_on_dev};
        OpAttributes attrs;
        attrs.set(AttrKey::M, M);
        attrs.set(AttrKey::K, K);
        auto result = table.dispatch_single(OpId::SparseSpMV, inputs, attrs);
        return (orig_dtype != comp_dtype) ? result.to(orig_dtype) : result;
    };

    if (sparse_compute.device().type == Device::Type::CUDA ||
        vec_compute.device().type == Device::Type::CUDA) {
        if (auto r = dispatch_gpu_spmv(Device::Type::CUDA)) return *r;
    }
    if (sparse_compute.device().type == Device::Type::ROCm ||
        vec_compute.device().type == Device::Type::ROCm) {
        if (auto r = dispatch_gpu_spmv(Device::Type::ROCm)) return *r;
    }
    if (sparse_compute.device().type == Device::Type::OneAPI ||
        vec_compute.device().type == Device::Type::OneAPI) {
        if (auto r = dispatch_gpu_spmv(Device::Type::OneAPI)) return *r;
    }
    if (sparse_compute.device().type == Device::Type::Vulkan ||
        vec_compute.device().type == Device::Type::Vulkan) {
        if (auto r = dispatch_gpu_spmv(Device::Type::Vulkan)) return *r;
    }

    // Refuse CPU fallback for GPU tensors (mirrors sparse::add). cpu_spmv
    // dereferences device pointers on the host for a GPU tensor — crash/UB.
    if (sparse_compute.device().type != Device::Type::CPU ||
        vec_compute.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "sparse::spmv: GPU tensor but no GPU SparseSpMV kernel matched — "
            "refusing CPU fallback (move tensors to CPU explicitly)");
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
        // with_values() preserves the layout and its index members for every
        // layout (COO/CSR/CSC/BSR); sparse.indices() is only populated for COO,
        // so the old COO reconstruction crashed for CSR/CSC/BSR inputs.
        auto sparse_promoted = (sparse.dtype() != common_dtype)
            ? sparse.with_values(sparse.values().to(common_dtype))
            : sparse;
        auto dense_promoted = (dense.dtype() != common_dtype) ? dense.to(common_dtype) : dense;
        return add(sparse_promoted, dense_promoted);
    }

    // Float16 / BFloat16: the CPU path's element-wise accumulate only
    // handles Float32/Float64/Int. Densify the sparse operand then add in
    // Float32 and narrow. Both sides must share the target device — use
    // the dense operand's device (or sparse's if dense is on CPU) so GPU
    // dense tensors don't silently get added to CPU sparse densifications.
    if (common_dtype == DType::Float16 || common_dtype == DType::BFloat16) {
        // Reject shape-mismatched operands here too. tenzor::add broadcasts
        // NumPy-style, so without this guard a mismatched dense operand would
        // silently broadcast — accepting shapes that every other dtype rejects
        // via the CPU path's numel check below.
        const auto& sp_shape_chk = sparse.shape();
        auto dn_shape_chk = dense.shape();
        bool shape_eq = (static_cast<int64_t>(sp_shape_chk.size()) ==
                         static_cast<int64_t>(dn_shape_chk.size()));
        if (shape_eq) {
            for (size_t i = 0; i < sp_shape_chk.size(); ++i) {
                if (sp_shape_chk[i] != dn_shape_chk[i]) { shape_eq = false; break; }
            }
        }
        if (!shape_eq) {
            throw std::runtime_error("sparse::add: shape mismatch");
        }
        Device target_dev = (dense.device().type != Device::Type::CPU)
                                ? dense.device()
                                : sparse.device();
        auto dense_sparse_f32 = sparse.to_dense().to(DType::Float32).to(target_dev);
        auto dense_f32 = dense.to(DType::Float32).to(target_dev);
        return tenzor::add(dense_sparse_f32, dense_f32).to(common_dtype);
    }

    auto sp_shape = sparse.shape();

    // GPU path: route through dispatch table for any backend that has a
    // SparseAdd kernel registered. Symmetric to the spmm/spmv pattern.
    auto dispatch_gpu_add = [&](Device::Type dev_type) -> std::optional<Tensor> {
        auto& table = DispatchTableRegistry::get_table(dev_type);
        if (!table.has_kernel(OpId::SparseAdd)) return std::nullopt;
        // Move the sparse tensor to the target device before extracting CSR —
        // otherwise the CPU→CSR path may invoke ops that reject non-Float32/64
        // values (e.g. Float16 coalesce), and we'd produce CPU CSR components
        // that then have to be individually shipped to the device.
        // Preserve the on-device operand's full Device (type AND index) so a
        // non-zero GPU index (e.g. cuda:1) is honoured rather than collapsing
        // moved operands onto index 0. See dispatch_gpu_spmm.
        Device target_dev = (sparse.device().type == dev_type)
                                ? sparse.device()
                                : dense.device();
        SparseTensor sparse_on_dev = (sparse.device() == target_dev)
                                         ? sparse
                                         : sparse.to(target_dev);
        auto sc = extract_csr_on_device(sparse_on_dev);
        // Guard rail: ensure everything really is on the target device before
        // dispatching — a belt-and-braces check, as to_csr is device-native.
        Tensor crow = (sc.crow.device() == target_dev) ? sc.crow : sc.crow.to(target_dev);
        Tensor col  = (sc.col.device()  == target_dev) ? sc.col  : sc.col.to(target_dev);
        Tensor vals = (sc.values.device() == target_dev) ? sc.values : sc.values.to(target_dev);
        Tensor dense_on_dev = (dense.device() == target_dev) ? dense : dense.to(target_dev);
        std::vector<Tensor> inputs = {crow, col, vals, dense_on_dev};
        OpAttributes attrs;
        attrs.set(AttrKey::M, sp_shape[0]);
        attrs.set(AttrKey::K, sp_shape.size() > 1 ? sp_shape[1] : int64_t(1));
        return table.dispatch_single(OpId::SparseAdd, inputs, attrs);
    };

    for (auto dev_type : {Device::Type::CUDA, Device::Type::ROCm,
                          Device::Type::OneAPI, Device::Type::Vulkan}) {
        if (sparse.device().type == dev_type || dense.device().type == dev_type) {
            if (auto r = dispatch_gpu_add(dev_type)) return *r;
        }
    }

    // Refuse CPU fallback for GPU tensors.
    if (sparse.device().type != Device::Type::CPU ||
        dense.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "sparse::add: GPU tensor but no GPU SparseAdd kernel matched — "
            "refusing CPU fallback (move tensors to CPU explicitly)");
    }

    // CPU path: convert sparse to dense, then element-wise add.
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
    } else if (result_c.dtype() == DType::Float16 || result_c.dtype() == DType::BFloat16) {
        // Widen to Float32, sum, narrow back — these dtypes lack natural CPU
        // arithmetic support in the per-element loop.
        auto lo = result_c.dtype();
        auto r32 = result_c.to(DType::Float32).contiguous();
        auto d32 = dense_c.to(DType::Float32).contiguous();
        auto* r = r32.data<float>();
        auto* d = d32.data<float>();
        #pragma omp parallel for schedule(static) if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
        return r32.to(lo);
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

    // Concatenate the two COO tensors along the nnz axis and coalesce. Both the
    // index/value concatenation (tenzor::cat) and the coalesce dispatch to the
    // active backend, so this stays on-device for CUDA/ROCm (Vulkan/OneAPI fall
    // to coalesce's host path, consistent with the rest of the sparse subsystem).
    // No host pointer math, every dtype supported (cat covers int/half/complex).
    auto a_coo = a.to_coo();
    auto b_coo = b.to_coo();
    const int64_t nnz_a = a_coo.nnz();
    const int64_t nnz_b = b_coo.nnz();
    const int64_t sparse_dim = a_coo.sparse_dim();
    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());

    if (nnz_a + nnz_b == 0) {
        auto indices = Tensor({sparse_dim, int64_t(0)}, DType::Int64, a.device());
        auto values = Tensor({int64_t(0)}, a.dtype(), a.device());
        return SparseTensor::sparse_coo(indices, values, shape_vec);
    }
    if (nnz_a == 0) return b_coo.coalesce();
    if (nnz_b == 0) return a_coo.coalesce();

    Tensor new_indices = tenzor::cat({a_coo.indices(), b_coo.indices()}, /*dim=*/1);
    Tensor new_values  = tenzor::cat({a_coo.values(),  b_coo.values()},  /*dim=*/0);
    return SparseTensor::sparse_coo(new_indices, new_values, shape_vec).coalesce();
}

// ============================================================================
// mul: Sparse * Scalar
// ============================================================================

auto mul(const SparseTensor& sparse, double scalar) -> SparseTensor {
    auto vals = sparse.values().contiguous();

    // Scalar-multiply the value buffer with the standard dense elementwise op.
    // It dispatches to the active backend (CPU/CUDA/ROCm/Vulkan/OneAPI) and
    // covers every dtype, so there is no host pointer math on device memory.
    Tensor new_values = tenzor::mul(vals, scalar);

    // A scalar rescale preserves the index structure for every layout
    // (COO/CSR/CSC/BSR), so swap only the value buffer and keep the existing
    // index members intact. with_values() copies *this, so the coalesced flag
    // (and all layout metadata) is propagated automatically — the previous
    // CSR-only else-branch threw on CSC/BSR inputs whose crow/col members are
    // default-constructed.
    return sparse.with_values(new_values);
}

// ============================================================================
// spgemm: Sparse-Sparse Matrix Multiplication (C = A @ B)
// ============================================================================

namespace {

/// CPU SpGEMM: two-phase symbolic + numeric algorithm.
/// Both inputs must be in CSR format. Returns CSR result.
template<typename T>
SparseTensor cpu_spgemm_typed(const SparseTensor& a, const SparseTensor& b,
                               int64_t M, [[maybe_unused]] int64_t K, int64_t N) {
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

    // GPU path: dispatch through the backend's registered SpGEMM kernel.
    // Supports CUDA, ROCm, Vulkan, OneAPI.
    auto dev_type = a.device().type;
    if (dev_type != Device::Type::CPU) {
        // A and B must reside on the same device; otherwise the kernel would
        // receive CSR pointers from two different devices (consistency failure
        // / out-of-bounds device dereference). Move B onto A's device.
        SparseTensor b_on_dev = (b.device() == a.device()) ? b : b.to(a.device());
        auto ac = extract_csr_on_device(a);
        auto bc = extract_csr_on_device(b_on_dev);
        // Half-precision widen-narrow, matching the CPU path: backend SpGEMM
        // kernels are typically only instantiated for F32/F64, so widen the
        // F16/BF16 value buffers to F32 before dispatch and narrow the result
        // back. Without this, half-precision sparse×sparse on GPU throws while
        // the same op on CPU succeeds.
        const DType orig_dtype = a.dtype();
        const bool widen_half = (orig_dtype == DType::Float16 ||
                                 orig_dtype == DType::BFloat16);
        Tensor a_vals = widen_half ? ac.values.to(DType::Float32) : ac.values;
        Tensor b_vals = widen_half ? bc.values.to(DType::Float32) : bc.values;
        std::vector<Tensor> inputs = {
            ac.crow, ac.col, a_vals,
            bc.crow, bc.col, b_vals,
        };
        OpAttributes attrs;
        attrs.set(AttrKey::M, M);
        attrs.set(AttrKey::K, K);
        attrs.set(AttrKey::N, N);
        auto& table = DispatchTableRegistry::get_table(dev_type);
        if (table.has_kernel(OpId::SparseSpGEMM)) {
            auto results = table.dispatch(OpId::SparseSpGEMM, inputs, attrs);
            if (results.size() != 3) {
                throw std::runtime_error(
                    "sparse::spgemm: dispatch must return 3 tensors "
                    "(crow, col, values), got " + std::to_string(results.size()));
            }
            Tensor out_vals = widen_half ? results[2].to(orig_dtype) : results[2];
            return SparseTensor::sparse_csr(
                results[0], results[1], out_vals,
                std::vector<int64_t>{M, N});
        }
        throw std::runtime_error(
            "spgemm: no GPU kernel registered for device " +
            std::string(a.device().to_string()) +
            " (vendor sparse library may be missing)");
    }

    // CPU path: convert to CSR for efficient row-based iteration.
    if (a.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "spgemm: GPU tensor but no GPU kernel available — "
            "refusing CPU fallback (move tensors to CPU explicitly)");
    }
    auto a_csr = a.to_csr();
    auto b_csr = b.to_csr();

    if (a.dtype() == DType::Float32) {
        return cpu_spgemm_typed<float>(a_csr, b_csr, M, K, N);
    }
    if (a.dtype() == DType::Float64) {
        return cpu_spgemm_typed<double>(a_csr, b_csr, M, K, N);
    }
    // Audit J13: F16/BF16 widen-narrow. cpu_spgemm_typed is only specialized
    // for float/double; widen the value buffers to F32, run, then narrow.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig = a.dtype();
        auto widen_csr = [](const SparseTensor& s) {
            auto sh = std::vector<int64_t>(s.shape().begin(), s.shape().end());
            return SparseTensor::sparse_csr(
                s.crow_indices(), s.col_indices(),
                s.values().to(DType::Float32), sh);
        };
        SparseTensor a_w = widen_csr(a_csr);
        SparseTensor b_w = widen_csr(b_csr);
        SparseTensor c_w = cpu_spgemm_typed<float>(a_w, b_w, M, K, N);
        return SparseTensor::sparse_csr(
            c_w.crow_indices(), c_w.col_indices(),
            c_w.values().to(orig),
            std::vector<int64_t>{M, N});
    }
    // Integer and complex dtypes: cpu_spgemm_typed<T> is generic (T(0), acc[col]
    // += a_val*b_data, val != T(0) all valid for these T), so support the same
    // coverage as cpu_spmm / cpu_spmv for API consistency.
    if (a.dtype() == DType::Int32) {
        return cpu_spgemm_typed<int32_t>(a_csr, b_csr, M, K, N);
    }
    if (a.dtype() == DType::Int64) {
        return cpu_spgemm_typed<int64_t>(a_csr, b_csr, M, K, N);
    }
    if (a.dtype() == DType::Complex64) {
        return cpu_spgemm_typed<std::complex<float>>(a_csr, b_csr, M, K, N);
    }
    if (a.dtype() == DType::Complex128) {
        return cpu_spgemm_typed<std::complex<double>>(a_csr, b_csr, M, K, N);
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

    // Float16 / BFloat16 widen-narrow, matching spmm/spmv/spgemm/sddmm/softmax:
    // the substitution kernels (and vendor sparse libraries) lack half-precision
    // instantiations, so widen L's values and b to Float32, solve, then narrow
    // the result back. Without this, half-precision sparse triangular solve
    // throws while every other sparse op accepts it.
    if (b.dtype() == DType::Float16 || b.dtype() == DType::BFloat16) {
        const DType orig = b.dtype();
        auto L_sh = std::vector<int64_t>(L.shape().begin(), L.shape().end());
        SparseTensor L_w = SparseTensor::sparse_csr(
            L.to_csr().crow_indices(), L.to_csr().col_indices(),
            L.to_csr().values().to(DType::Float32), L_sh);
        Tensor b_w = b.to(DType::Float32);
        return sparse_triangular_solve(L_w, b_w, upper).to(orig);
    }

    if (L.device().type != b.device().type) {
        throw std::runtime_error(
            "sparse_triangular_solve: L and b must be on the same device (L on " +
            std::string(L.device().to_string()) + ", b on " +
            std::string(b.device().to_string()) + ")");
    }

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

    // GPU path: dispatch through the OpId table for any GPU device type.
    auto dev_type = b.device().type;
    if (dev_type != Device::Type::CPU) {
        // L and b must live on the same device: extract_csr_on_device(L)
        // returns components on L's own device, and the dispatch is selected by
        // b's device. If they differ the kernel would receive crow/col/values
        // on one device and b on another. Reconcile by moving both onto the
        // device chosen by b (the previous `b.to(b.device())` ternary was a
        // no-op and never transferred anything).
        // b is passed to the kernel as-is, so b's full Device (type AND index)
        // is the target. Moving L's CSR components onto b.device() preserves a
        // non-zero GPU index (e.g. cuda:1) and rules out the case where L sits on
        // cuda:0 while b sits on cuda:1 (the .type-only guard above would let
        // that through, handing the kernel pointers on two physical devices).
        Device target_dev = b.device();
        auto Lc = extract_csr_on_device(L);
        Tensor L_crow = (Lc.crow.device() == target_dev) ? Lc.crow : Lc.crow.to(target_dev);
        Tensor L_col  = (Lc.col.device()  == target_dev) ? Lc.col  : Lc.col.to(target_dev);
        Tensor L_vals = (Lc.values.device() == target_dev) ? Lc.values : Lc.values.to(target_dev);
        std::vector<Tensor> inputs = {L_crow, L_col, L_vals, b};
        OpAttributes attrs;
        attrs.set(AttrKey::N, N);
        attrs.set(AttrKey::Upper, upper);
        auto& table = DispatchTableRegistry::get_table(dev_type);
        const OpId op = (b.ndim() == 1) ? OpId::SparseTrsv : OpId::SparseTrsm;
        if (table.has_kernel(op)) {
            return table.dispatch_single(op, inputs, attrs);
        }
        throw std::runtime_error(
            "sparse_triangular_solve: no GPU kernel registered for device " +
            std::string(b.device().to_string()) +
            " (vendor sparse library may be missing)");
    }

    // CPU path: only when tensors are actually on CPU.
    auto L_csr = L.to_csr();

    if (b.ndim() == 1) {
        if (b.dtype() == DType::Float32) {
            return cpu_sparse_trsv<float>(L_csr, b, upper, N);
        } else if (b.dtype() == DType::Float64) {
            return cpu_sparse_trsv<double>(L_csr, b, upper, N);
        } else if (b.dtype() == DType::Complex64) {
            return cpu_sparse_trsv<std::complex<float>>(L_csr, b, upper, N);
        } else if (b.dtype() == DType::Complex128) {
            return cpu_sparse_trsv<std::complex<double>>(L_csr, b, upper, N);
        } else {
            throw std::runtime_error("sparse_triangular_solve: unsupported dtype " +
                std::string(dtype_name(b.dtype())));
        }
    } else {
        int64_t K = b.shape()[1];
        if (b.dtype() == DType::Float32) {
            return cpu_sparse_trsm<float>(L_csr, b, upper, N, K);
        } else if (b.dtype() == DType::Float64) {
            return cpu_sparse_trsm<double>(L_csr, b, upper, N, K);
        } else if (b.dtype() == DType::Complex64) {
            return cpu_sparse_trsm<std::complex<float>>(L_csr, b, upper, N, K);
        } else if (b.dtype() == DType::Complex128) {
            return cpu_sparse_trsm<std::complex<double>>(L_csr, b, upper, N, K);
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
                   [[maybe_unused]] int64_t N,
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

    // Float16/BFloat16: the CPU sddmm kernel below only instantiates float and
    // double. Widen the dense operands to Float32, compute, then narrow the
    // result values back to the original dtype (mirrors sparse_softmax).
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        const DType orig = A.dtype();
        auto res = sddmm(mask, A.to(DType::Float32), B.to(DType::Float32));
        std::vector<int64_t> shp(res.shape().begin(), res.shape().end());
        return SparseTensor::sparse_csr(res.crow_indices(), res.col_indices(),
                                        res.values().to(orig), shp);
    }

    // Normalize to CSR for uniform access.
    SparseTensor csr_mask = mask;
    if (csr_mask.layout() != SparseLayout::CSR) {
        csr_mask = csr_mask.to_csr();
    }

    // The sddmm kernel below is CPU-only: it calls `.data<T>()` and
    // dereferences in an OpenMP loop. When A/B are on a non-CPU device
    // that pointer is a device address; dereferencing it from the host
    // either segfaults or (on the CUDA driver) enters an error-retry loop
    // that never returns — which was surfaced as the "sddmm hangs on GPU"
    // bug by the multi-backend test migration. Move inputs to CPU for the
    // compute and stage the result back to whichever device the caller
    // gave us, so sddmm works on every backend (CPU kernel internally).
    Device src_device = A.device();
    Tensor A_cpu = (A.device() == Device::cpu()) ? A.contiguous() : A.to(Device::cpu()).contiguous();
    Tensor B_cpu = (B.device() == Device::cpu()) ? B.contiguous() : B.to(Device::cpu()).contiguous();
    Tensor crow_cpu = (csr_mask.crow_indices().device() == Device::cpu())
        ? csr_mask.crow_indices() : csr_mask.crow_indices().to(Device::cpu());
    Tensor col_cpu = (csr_mask.col_indices().device() == Device::cpu())
        ? csr_mask.col_indices() : csr_mask.col_indices().to(Device::cpu());

    const int64_t K = a_shape[1];
    Tensor values_cpu;
    if (A.dtype() == DType::Float32) {
        values_cpu = cpu_sddmm_csr<float>(crow_cpu, col_cpu, A_cpu, B_cpu, M, N, K);
    } else if (A.dtype() == DType::Float64) {
        values_cpu = cpu_sddmm_csr<double>(crow_cpu, col_cpu, A_cpu, B_cpu, M, N, K);
    } else {
        throw std::runtime_error("sddmm: unsupported dtype " +
                                 std::string(dtype_name(A.dtype())));
    }

    // Build the result CSR on the source device so callers don't have to
    // re-home it. The crow/col indices already live on src_device via
    // csr_mask; values is the only newly-computed tensor.
    Tensor values = (src_device == Device::cpu()) ? values_cpu : values_cpu.to(src_device);
    return SparseTensor::sparse_csr(csr_mask.crow_indices(),
                                    csr_mask.col_indices(),
                                    values,
                                    {M, N});
}

// ============================================================================
// Sparse Softmax
// ============================================================================

auto sparse_softmax(const SparseTensor& sparse) -> SparseTensor {
    if (sparse.layout() != SparseLayout::CSR) {
        throw std::runtime_error("sparse_softmax: only CSR layout is supported");
    }
    auto shape = sparse.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("sparse_softmax: input must be 2D");
    }

    // Refuse CPU fallback for GPU tensors (mirrors spmm/spmv/add). There is no
    // registered GPU sparse-softmax kernel; the host loop below dereferences
    // values()/crow_indices() via .data<T>(), which is UB on device memory.
    // Round-tripping silently through the host would be a hidden CPU fallback,
    // exactly what the rest of the sparse subsystem refuses — so fail loudly.
    if (sparse.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "sparse::sparse_softmax: GPU tensor but no GPU sparse-softmax kernel — "
            "refusing CPU fallback (move tensors to CPU explicitly)");
    }

    // Float16/BFloat16: widen values to Float32, softmax, narrow back.
    if (sparse.values().dtype() == DType::Float16 || sparse.values().dtype() == DType::BFloat16) {
        const DType orig = sparse.values().dtype();
        std::vector<int64_t> shp(sparse.shape().begin(), sparse.shape().end());
        auto widened = SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                                sparse.values().to(DType::Float32), shp);
        auto res = sparse_softmax(widened);
        return SparseTensor::sparse_csr(res.crow_indices(), res.col_indices(),
                                        res.values().to(orig), shp);
    }

    int64_t M = shape[0];
    auto crow = sparse.crow_indices();
    auto col = sparse.col_indices();
    auto vals = sparse.values();

    // CPU-only compute (GPU inputs were rejected above). Operate directly on
    // the host buffers — no device round-trip.
    auto crow_cpu = crow.contiguous();
    auto vals_cpu = vals.contiguous();

    int64_t nnz = vals_cpu.numel();
    Tensor out_vals = tenzor::zeros({nnz}, vals_cpu.dtype(), Device::cpu());

    const int64_t* row_ptr = crow_cpu.data<int64_t>();

    if (vals_cpu.dtype() == DType::Float32) {
        const float* v = vals_cpu.data<float>();
        float* o = out_vals.data<float>();

        for (int64_t i = 0; i < M; ++i) {
            int64_t start = row_ptr[i];
            int64_t end = row_ptr[i + 1];
            if (start == end) continue;

            // Find max for numerical stability
            float max_val = v[start];
            for (int64_t j = start + 1; j < end; ++j) {
                max_val = std::max(max_val, v[j]);
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int64_t j = start; j < end; ++j) {
                o[j] = std::exp(v[j] - max_val);
                sum_exp += o[j];
            }

            // Normalize
            float inv_sum = 1.0f / sum_exp;
            for (int64_t j = start; j < end; ++j) {
                o[j] *= inv_sum;
            }
        }
    } else if (vals_cpu.dtype() == DType::Float64) {
        const double* v = vals_cpu.data<double>();
        double* o = out_vals.data<double>();

        for (int64_t i = 0; i < M; ++i) {
            int64_t start = row_ptr[i];
            int64_t end = row_ptr[i + 1];
            if (start == end) continue;

            double max_val = v[start];
            for (int64_t j = start + 1; j < end; ++j) {
                max_val = std::max(max_val, v[j]);
            }

            double sum_exp = 0.0;
            for (int64_t j = start; j < end; ++j) {
                o[j] = std::exp(v[j] - max_val);
                sum_exp += o[j];
            }

            double inv_sum = 1.0 / sum_exp;
            for (int64_t j = start; j < end; ++j) {
                o[j] *= inv_sum;
            }
        }
    } else {
        throw std::runtime_error("sparse_softmax: unsupported dtype");
    }

    // Return the contiguous crow that was actually used for the compute and a
    // contiguous col, so value index j aligns positionally with column j and
    // sparse_csr validation reads row-major-contiguous memory. Returning the
    // original (possibly non-contiguous) crow/col would mis-align out_vals
    // (laid out in compacted element order) against the column indices.
    return SparseTensor::sparse_csr(crow_cpu, col.contiguous(), out_vals, shape);
}

// ============================================================================
// Sparse Log-Softmax
// ============================================================================

auto sparse_log_softmax(const SparseTensor& sparse) -> SparseTensor {
    if (sparse.layout() != SparseLayout::CSR) {
        throw std::runtime_error("sparse_log_softmax: only CSR layout is supported");
    }
    auto shape = sparse.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("sparse_log_softmax: input must be 2D");
    }

    // Refuse CPU fallback for GPU tensors (mirrors spmm/spmv/add). There is no
    // registered GPU sparse-log-softmax kernel; the host loop below dereferences
    // values()/crow_indices() via .data<T>(), which is UB on device memory.
    if (sparse.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "sparse::sparse_log_softmax: GPU tensor but no GPU sparse-softmax kernel — "
            "refusing CPU fallback (move tensors to CPU explicitly)");
    }

    // Float16/BFloat16: widen values to Float32, log-softmax, narrow back.
    if (sparse.values().dtype() == DType::Float16 || sparse.values().dtype() == DType::BFloat16) {
        const DType orig = sparse.values().dtype();
        std::vector<int64_t> shp(sparse.shape().begin(), sparse.shape().end());
        auto widened = SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                                sparse.values().to(DType::Float32), shp);
        auto res = sparse_log_softmax(widened);
        return SparseTensor::sparse_csr(res.crow_indices(), res.col_indices(),
                                        res.values().to(orig), shp);
    }

    int64_t M = shape[0];
    auto crow = sparse.crow_indices();
    auto col = sparse.col_indices();
    auto vals = sparse.values();

    // CPU-only compute (GPU inputs were rejected above). Operate directly on
    // the host buffers — no device round-trip.
    auto crow_cpu = crow.contiguous();
    auto vals_cpu = vals.contiguous();

    int64_t nnz = vals_cpu.numel();
    Tensor out_vals = tenzor::zeros({nnz}, vals_cpu.dtype(), Device::cpu());

    const int64_t* row_ptr = crow_cpu.data<int64_t>();

    if (vals_cpu.dtype() == DType::Float32) {
        const float* v = vals_cpu.data<float>();
        float* o = out_vals.data<float>();

        for (int64_t i = 0; i < M; ++i) {
            int64_t start = row_ptr[i];
            int64_t end = row_ptr[i + 1];
            if (start == end) continue;

            // Find max for numerical stability
            float max_val = v[start];
            for (int64_t j = start + 1; j < end; ++j) {
                max_val = std::max(max_val, v[j]);
            }

            // Compute log-sum-exp
            float sum_exp = 0.0f;
            for (int64_t j = start; j < end; ++j) {
                sum_exp += std::exp(v[j] - max_val);
            }
            float log_sum_exp = max_val + std::log(sum_exp);

            // log_softmax = x - log_sum_exp
            for (int64_t j = start; j < end; ++j) {
                o[j] = v[j] - log_sum_exp;
            }
        }
    } else if (vals_cpu.dtype() == DType::Float64) {
        const double* v = vals_cpu.data<double>();
        double* o = out_vals.data<double>();

        for (int64_t i = 0; i < M; ++i) {
            int64_t start = row_ptr[i];
            int64_t end = row_ptr[i + 1];
            if (start == end) continue;

            double max_val = v[start];
            for (int64_t j = start + 1; j < end; ++j) {
                max_val = std::max(max_val, v[j]);
            }

            double sum_exp = 0.0;
            for (int64_t j = start; j < end; ++j) {
                sum_exp += std::exp(v[j] - max_val);
            }
            double log_sum_exp = max_val + std::log(sum_exp);

            for (int64_t j = start; j < end; ++j) {
                o[j] = v[j] - log_sum_exp;
            }
        }
    } else {
        throw std::runtime_error("sparse_log_softmax: unsupported dtype");
    }

    // Return the contiguous crow that was actually used for the compute and a
    // contiguous col, so value index j aligns positionally with column j and
    // sparse_csr validation reads row-major-contiguous memory. Returning the
    // original (possibly non-contiguous) crow/col would mis-align out_vals
    // (laid out in compacted element order) against the column indices.
    return SparseTensor::sparse_csr(crow_cpu, col.contiguous(), out_vals, shape);
}

} // namespace sparse
} // namespace tenzor
