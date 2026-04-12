/**
 * @file sparse.hip.cpp
 * @brief ROCm kernels for sparse tensor operations using rocSPARSE.
 *
 * Provides GPU-accelerated implementations of:
 * - spmm (sparse-dense matrix multiplication) via rocsparse_spmm()
 * - spmv (sparse-dense matrix-vector multiplication) via rocsparse_spmv()
 *
 * Uses CSR format descriptors for rocSPARSE API compatibility.
 * Both COO and CSR inputs are supported; COO is converted to CSR internally.
 */

#ifdef TENZOR_HAS_ROCSPARSE

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"

#include "tenzor/backend/loader_fwd.hpp"

#include <rocsparse/rocsparse.h>
#include <hip/hip_runtime.h>
#include "../hip_buffer.hpp"
#include "../rocsparse_handle_pool.hpp"
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>

// Forward-declare zeros to avoid including creation.hpp
namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}

namespace tenzor {
namespace rocm {

namespace {

#define HIP_CHECK_SPARSE(call)                                                  \
    do {                                                                         \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("HIP error in sparse at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));     \
        }                                                                       \
    } while (0)

// ROCSPARSE_CHECK is provided by rocsparse_handle_pool.hpp

/// Convert span to vector (HIP compiler may not do implicit span->vector).
std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

// HipBuffer is provided by ../hip_buffer.hpp (in namespace tenzor::rocm)

/// RAII guard for rocSPARSE sparse matrix descriptor.
struct SpMatGuard {
    rocsparse_spmat_descr desc = nullptr;
    explicit SpMatGuard(rocsparse_spmat_descr d) : desc(d) {}
    ~SpMatGuard() { if (desc) rocsparse_destroy_spmat_descr(desc); }
    SpMatGuard(const SpMatGuard&) = delete;
    SpMatGuard& operator=(const SpMatGuard&) = delete;
};

/// RAII guard for rocSPARSE dense matrix descriptor.
struct DnMatGuard {
    rocsparse_dnmat_descr desc = nullptr;
    explicit DnMatGuard(rocsparse_dnmat_descr d) : desc(d) {}
    ~DnMatGuard() { if (desc) rocsparse_destroy_dnmat_descr(desc); }
    DnMatGuard(const DnMatGuard&) = delete;
    DnMatGuard& operator=(const DnMatGuard&) = delete;
};

/// RAII guard for rocSPARSE dense vector descriptor.
struct DnVecGuard {
    rocsparse_dnvec_descr desc = nullptr;
    explicit DnVecGuard(rocsparse_dnvec_descr d) : desc(d) {}
    ~DnVecGuard() { if (desc) rocsparse_destroy_dnvec_descr(desc); }
    DnVecGuard(const DnVecGuard&) = delete;
    DnVecGuard& operator=(const DnVecGuard&) = delete;
};

/// RAII guard for rocSPARSE SpMV descriptor (v2 API).
struct SpMVDescrGuard {
    rocsparse_spmv_descr desc = nullptr;
    explicit SpMVDescrGuard(rocsparse_spmv_descr d) : desc(d) {}
    ~SpMVDescrGuard() { if (desc) rocsparse_destroy_spmv_descr(desc); }
    SpMVDescrGuard(const SpMVDescrGuard&) = delete;
    SpMVDescrGuard& operator=(const SpMVDescrGuard&) = delete;
};

/// Per-thread rocSPARSE handle — forwards to the shared pool.
inline rocsparse_handle get_rocsparse_handle() {
    return RocSPARSEHandlePool::get();
}

/// Get rocSPARSE data type from DType.
rocsparse_datatype get_rocsparse_data_type(DType dtype) {
    switch (dtype) {
        case DType::Float32: return rocsparse_datatype_f32_r;
        case DType::Float64: return rocsparse_datatype_f64_r;
        default:
            throw std::runtime_error("rocm_sparse: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
    }
}

/// HIP kernel: convert Int64 row indices to Int32.
__global__ void cast_i64_to_i32(const int64_t* __restrict__ src,
                                 int32_t* __restrict__ dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int32_t>(src[i]);
}

/// HIP kernel: convert Int32 crow_indices to Int64.
__global__ void cast_i32_to_i64(const int32_t* __restrict__ src,
                                 int64_t* __restrict__ dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int64_t>(src[i]);
}

/// HIP kernel: check if any int64 value overflows int32 range.
__global__ void check_i64_overflow(const int64_t* __restrict__ src,
                                    int64_t n, int* overflow_flag) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (src[i] > INT32_MAX || src[i] < INT32_MIN) {
            atomicOr(overflow_flag, 1);
        }
    }
}

/// Check that all int64 indices fit in int32. Throws std::overflow_error on failure.
static void verify_i64_fits_i32(const int64_t* d_data, int64_t n,
                                 hipStream_t stream = nullptr) {
    if (n == 0) return;
    HipBuffer flag_buf(sizeof(int));
    int* d_flag = flag_buf.as<int>();
    HIP_CHECK_SPARSE(hipMemsetAsync(d_flag, 0, sizeof(int), stream));

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipLaunchKernelGGL(check_i64_overflow, dim3(blocks), dim3(threads),
                       0, stream, d_data, n, d_flag);
    HIP_CHECK_SPARSE(hipGetLastError());

    int h_flag = 0;
    HIP_CHECK_SPARSE(hipMemcpyAsync(&h_flag, d_flag, sizeof(int),
                                     hipMemcpyDeviceToHost, stream));
    HIP_CHECK_SPARSE(hipStreamSynchronize(stream));

    if (h_flag != 0) {
        throw std::overflow_error(
            "rocm_sparse: int64 index value exceeds int32 range");
    }
}

/// Helper to build a CSR SparseTensor on GPU from a COO SparseTensor.
SparseTensor ensure_csr_on_gpu(const SparseTensor& sparse) {
    auto sp = (sparse.device().type != Device::Type::ROCm)
              ? sparse.to(Device::rocm())
              : sparse;

    if (sp.layout() == SparseLayout::COO) {
        auto sp_shape = sp.shape();
        int64_t nrows = sp_shape[0];
        int64_t ncols = sp_shape[1];
        int64_t nnz = sp.nnz();

        Tensor indices = sp.indices().contiguous();
        Tensor values = sp.values().contiguous();

        const int64_t* indices_ptr = indices.data<int64_t>();
        const int64_t* row_indices_ptr = indices_ptr;
        const int64_t* col_indices_ptr = indices_ptr + nnz;

        // rocSPARSE coo2csr requires Int32
        HipBuffer row_i32_buf(nnz * sizeof(int32_t));
        HipBuffer crow_i32_buf((nrows + 1) * sizeof(int32_t));

        int threads = 256;
        int blocks_nnz = static_cast<int>((nnz + threads - 1) / threads);
        verify_i64_fits_i32(row_indices_ptr, nnz);
        cast_i64_to_i32<<<blocks_nnz, threads>>>(row_indices_ptr, row_i32_buf.as<int32_t>(), nnz);
        HIP_CHECK_SPARSE(hipGetLastError());

        // Convert COO row indices to CSR row pointers on GPU
        if (nnz > static_cast<int64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("rocm_sparse: nnz exceeds int32 range for rocsparse_coo2csr");
        if (nrows > static_cast<int64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("rocm_sparse: nrows exceeds int32 range for rocsparse_coo2csr");
        rocsparse_handle handle = get_rocsparse_handle();
        ROCSPARSE_CHECK(rocsparse_coo2csr(
            handle, row_i32_buf.as<int32_t>(), static_cast<int>(nnz), static_cast<int>(nrows),
            crow_i32_buf.as<int32_t>(), rocsparse_index_base_zero));

        // Convert Int32 crow_indices back to Int64 on GPU
        Tensor crow_indices = zeros(std::vector<int64_t>{nrows + 1}, DType::Int64, Device::rocm());
        int blocks_crow = static_cast<int>((nrows + 1 + threads - 1) / threads);
        cast_i32_to_i64<<<blocks_crow, threads>>>(crow_i32_buf.as<int32_t>(), crow_indices.data<int64_t>(), nrows + 1);
        HIP_CHECK_SPARSE(hipGetLastError());

        // Copy col indices
        Tensor col_idx = zeros(std::vector<int64_t>{nnz}, DType::Int64, Device::rocm());
        HIP_CHECK_SPARSE(hipMemcpy(col_idx.data<int64_t>(), col_indices_ptr,
                                   nnz * sizeof(int64_t), hipMemcpyDeviceToDevice));

        return SparseTensor::sparse_csr(
            crow_indices, col_idx, values,
            std::vector<int64_t>{nrows, ncols});
    }
    return sp;
}

// CSR SparseAdd kernel: one thread per row, adds sparse values into dense output
template <typename T>
__global__ void csr_sparse_add_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    T* __restrict__ out_ptr,
    int64_t nrows, int64_t ncols)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        out_ptr[row * ncols + col_ptr[j]] += val_ptr[j];
    }
}

} // anonymous namespace

Tensor rocm_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("rocm_spmm: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("rocm_spmm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(dense.shape()[0]) + ")");
    }

    DType dtype = dense.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_spmm: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto csr = ensure_csr_on_gpu(sparse);
    int64_t nnz = csr.nnz();

    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous()
                     : dense.contiguous();

    auto result = zeros({M, N}, dtype, Device::rocm());

    rocsparse_handle handle = get_rocsparse_handle();
    rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    // Create CSR sparse matrix descriptor
    rocsparse_spmat_descr mat_sparse;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_sparse,
        M, K, nnz,
        const_cast<void*>(static_cast<const void*>(crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(col.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(vals.data_ptr())),
        rocsparse_indextype_i64,
        rocsparse_indextype_i64,
        rocsparse_index_base_zero,
        roc_dtype
    ));
    SpMatGuard sparse_guard(mat_sparse);

    // Dense matrix descriptor (row-major)
    rocsparse_dnmat_descr mat_dense;
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &mat_dense,
        K, N, N,
        const_cast<void*>(dense_gpu.data_ptr()),
        roc_dtype,
        rocsparse_order_row
    ));
    DnMatGuard dense_guard(mat_dense);

    rocsparse_dnmat_descr mat_result;
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &mat_result,
        M, N, N,
        result.data_ptr(),
        roc_dtype,
        rocsparse_order_row
    ));
    DnMatGuard result_guard(mat_result);

    // Determine buffer size
    size_t buffer_size = 0;
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_ptr = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f) : static_cast<void*>(&alpha_d);
    void* beta_ptr = (dtype == DType::Float32) ? static_cast<void*>(&beta_f) : static_cast<void*>(&beta_d);

    ROCSPARSE_CHECK(rocsparse_spmm(
        handle,
        rocsparse_operation_none,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        roc_dtype,
        rocsparse_spmm_alg_default,
        rocsparse_spmm_stage_buffer_size,
        &buffer_size,
        nullptr
    ));

    HipBuffer workspace(buffer_size);

    ROCSPARSE_CHECK(rocsparse_spmm(
        handle,
        rocsparse_operation_none,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        roc_dtype,
        rocsparse_spmm_alg_default,
        rocsparse_spmm_stage_preprocess,
        &buffer_size,
        workspace.ptr
    ));

    ROCSPARSE_CHECK(rocsparse_spmm(
        handle,
        rocsparse_operation_none,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        roc_dtype,
        rocsparse_spmm_alg_default,
        rocsparse_spmm_stage_compute,
        &buffer_size,
        workspace.ptr
    ));

    return result;
}

Tensor rocm_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("rocm_spmv: sparse must be 2D, vec must be 1D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("rocm_spmv: dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(vec.shape()[0]) + ")");
    }

    DType dtype = vec.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_spmv: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto csr = ensure_csr_on_gpu(sparse);
    int64_t nnz = csr.nnz();

    auto vec_gpu = (vec.device().type != Device::Type::ROCm)
                   ? vec.to(Device::rocm()).contiguous()
                   : vec.contiguous();

    auto result = zeros({M}, dtype, Device::rocm());

    rocsparse_handle handle = get_rocsparse_handle();
    rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    rocsparse_spmat_descr mat_sparse;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_sparse,
        M, K, nnz,
        const_cast<void*>(static_cast<const void*>(crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(col.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(vals.data_ptr())),
        rocsparse_indextype_i64,
        rocsparse_indextype_i64,
        rocsparse_index_base_zero,
        roc_dtype
    ));
    SpMatGuard sparse_guard(mat_sparse);

    rocsparse_dnvec_descr vec_x;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_x,
        K,
        const_cast<void*>(vec_gpu.data_ptr()),
        roc_dtype
    ));
    DnVecGuard vec_x_guard(vec_x);

    rocsparse_dnvec_descr vec_y;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_y,
        M,
        result.data_ptr(),
        roc_dtype
    ));
    DnVecGuard vec_y_guard(vec_y);

    // Set up v2 SpMV descriptor
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_ptr = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f) : static_cast<void*>(&alpha_d);
    void* beta_ptr = (dtype == DType::Float32) ? static_cast<void*>(&beta_f) : static_cast<void*>(&beta_d);

    rocsparse_spmv_descr spmv_descr;
    ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&spmv_descr));
    SpMVDescrGuard spmv_guard(spmv_descr);

    rocsparse_operation op = rocsparse_operation_none;
    rocsparse_spmv_alg alg = rocsparse_spmv_alg_default;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_operation, &op, sizeof(op), nullptr));
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr));
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_scalar_datatype, &roc_dtype, sizeof(roc_dtype), nullptr));
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_compute_datatype, &roc_dtype, sizeof(roc_dtype), nullptr));

    // Analysis stage: get buffer size and run analysis
    size_t analysis_buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(
        handle, spmv_descr, mat_sparse, vec_x, vec_y,
        rocsparse_v2_spmv_stage_analysis, &analysis_buffer_size, nullptr));

    HipBuffer analysis_workspace(analysis_buffer_size);

    ROCSPARSE_CHECK(rocsparse_v2_spmv(
        handle, spmv_descr, alpha_ptr, mat_sparse, vec_x, beta_ptr, vec_y,
        rocsparse_v2_spmv_stage_analysis, analysis_buffer_size, analysis_workspace.ptr, nullptr));

    // Compute stage: get buffer size and run compute
    size_t compute_buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(
        handle, spmv_descr, mat_sparse, vec_x, vec_y,
        rocsparse_v2_spmv_stage_compute, &compute_buffer_size, nullptr));

    HipBuffer compute_workspace(compute_buffer_size);

    ROCSPARSE_CHECK(rocsparse_v2_spmv(
        handle, spmv_descr, alpha_ptr, mat_sparse, vec_x, beta_ptr, vec_y,
        rocsparse_v2_spmv_stage_compute, compute_buffer_size, compute_workspace.ptr, nullptr));

    return result;
}

// ============================================================================
// SpGEMM — sparse × sparse → sparse (CSR × CSR → CSR) via rocsparse_spgemm
// ============================================================================

/// RAII guard for rocSPARSE csr2csr compression descriptors — not used here
/// but the same pattern applies for any rocSPARSE handle.

SparseTensor rocm_spgemm_kernel(const SparseTensor& a, const SparseTensor& b) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("rocm_spgemm: both inputs must be 2D");
    }
    const int64_t M = a_shape[0];
    const int64_t K = a_shape[1];
    const int64_t N = b_shape[1];
    if (K != b_shape[0]) {
        throw std::runtime_error("rocm_spgemm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(b_shape[0]) + ")");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("rocm_spgemm: dtype mismatch");
    }
    const DType dtype = a.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_spgemm: only Float32/Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto a_csr = ensure_csr_on_gpu(a);
    auto b_csr = ensure_csr_on_gpu(b);
    const int64_t nnz_a = a_csr.nnz();
    const int64_t nnz_b = b_csr.nnz();

    auto a_crow = a_csr.crow_indices().contiguous();
    auto a_col  = a_csr.col_indices().contiguous();
    auto a_vals = a_csr.values().contiguous();
    auto b_crow = b_csr.crow_indices().contiguous();
    auto b_col  = b_csr.col_indices().contiguous();
    auto b_vals = b_csr.values().contiguous();

    rocsparse_handle handle = get_rocsparse_handle();
    const rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    // A descriptor.
    rocsparse_spmat_descr mat_a;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_a, M, K, nnz_a,
        const_cast<void*>(static_cast<const void*>(a_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(a_col.data<int64_t>())),
        const_cast<void*>(a_vals.data_ptr()),
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard a_guard(mat_a);

    // B descriptor.
    rocsparse_spmat_descr mat_b;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_b, K, N, nnz_b,
        const_cast<void*>(static_cast<const void*>(b_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(b_col.data<int64_t>())),
        const_cast<void*>(b_vals.data_ptr()),
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard b_guard(mat_b);

    // C descriptor: start with an empty CSR with a pre-allocated row ptr
    // buffer. rocsparse_spgemm nnz stage fills it in; compute stage fills
    // col_ind / values once they are allocated.
    auto c_crow = zeros(std::vector<int64_t>{M + 1}, DType::Int64, Device::rocm());
    rocsparse_spmat_descr mat_c;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_c, M, N, 0,
        c_crow.data<int64_t>(), nullptr, nullptr,
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard c_guard(mat_c);

    // D descriptor is a nullptr alias for "no D matrix" — beta must be 0.
    // rocsparse_spgemm requires a valid D descriptor; reuse C for this
    // purpose since it will be read only when beta != 0.
    rocsparse_spmat_descr mat_d;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_d, M, N, 0,
        c_crow.data<int64_t>(), nullptr, nullptr,
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard d_guard(mat_d);

    const rocsparse_operation op_none = rocsparse_operation_none;

    float  alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0,  beta_d = 0.0;
    void* alpha = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f)
                                            : static_cast<void*>(&alpha_d);
    void* beta  = (dtype == DType::Float32) ? static_cast<void*>(&beta_f)
                                            : static_cast<void*>(&beta_d);

    // Stage 1: buffer size.
    size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_spgemm(
        handle, op_none, op_none, alpha, mat_a, mat_b, beta, mat_d, mat_c,
        roc_dtype, rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_buffer_size, &buffer_size, nullptr));
    HipBuffer workspace(buffer_size);

    // Stage 2: nnz. Fills in c_crow and computes the nnz of C.
    ROCSPARSE_CHECK(rocsparse_spgemm(
        handle, op_none, op_none, alpha, mat_a, mat_b, beta, mat_d, mat_c,
        roc_dtype, rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_nnz, &buffer_size, workspace.ptr));

    int64_t c_rows = 0, c_cols = 0, c_nnz = 0;
    ROCSPARSE_CHECK(rocsparse_spmat_get_size(mat_c, &c_rows, &c_cols, &c_nnz));

    auto c_col  = zeros(std::vector<int64_t>{c_nnz}, DType::Int64, Device::rocm());
    auto c_vals = zeros(std::vector<int64_t>{c_nnz}, dtype,         Device::rocm());

    ROCSPARSE_CHECK(rocsparse_csr_set_pointers(
        mat_c,
        c_crow.data<int64_t>(),
        c_col.data<int64_t>(),
        c_vals.data_ptr()));

    // Stage 3: compute.
    ROCSPARSE_CHECK(rocsparse_spgemm(
        handle, op_none, op_none, alpha, mat_a, mat_b, beta, mat_d, mat_c,
        roc_dtype, rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_compute, &buffer_size, workspace.ptr));

    HIP_CHECK_SPARSE(hipDeviceSynchronize());

    return SparseTensor::sparse_csr(
        c_crow, c_col, c_vals,
        std::vector<int64_t>{M, N});
}

// ============================================================================
// SpSV — triangular solve L*x = b (single RHS) via rocsparse_spsv
// ============================================================================

Tensor rocm_sparse_trsv_kernel(const SparseTensor& L, const Tensor& b, bool upper) {
    auto L_shape = L.shape();
    if (L_shape.size() != 2 || L_shape[0] != L_shape[1]) {
        throw std::runtime_error("rocm_sparse_trsv: L must be square 2D");
    }
    if (b.ndim() != 1) {
        throw std::runtime_error("rocm_sparse_trsv: b must be 1D");
    }
    const int64_t N = L_shape[0];
    if (b.shape()[0] != N) {
        throw std::runtime_error("rocm_sparse_trsv: dimension mismatch");
    }
    const DType dtype = b.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_sparse_trsv: only Float32/Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto L_csr = ensure_csr_on_gpu(L);
    const int64_t nnz = L_csr.nnz();
    auto L_crow = L_csr.crow_indices().contiguous();
    auto L_col  = L_csr.col_indices().contiguous();
    auto L_vals = L_csr.values().contiguous();

    auto b_gpu = (b.device().type != Device::Type::ROCm)
                   ? b.to(Device::rocm()).contiguous()
                   : b.contiguous();
    auto result = zeros(std::vector<int64_t>{N}, dtype, Device::rocm());

    rocsparse_handle handle = get_rocsparse_handle();
    const rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    // L matrix descriptor.
    rocsparse_spmat_descr mat_L;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_L, N, N, nnz,
        const_cast<void*>(static_cast<const void*>(L_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(L_col.data<int64_t>())),
        const_cast<void*>(L_vals.data_ptr()),
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard L_guard(mat_L);

    // Fill mode + diagonal type.
    rocsparse_fill_mode fill_mode = upper ? rocsparse_fill_mode_upper
                                          : rocsparse_fill_mode_lower;
    rocsparse_diag_type diag_type = rocsparse_diag_type_non_unit;
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        mat_L, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        mat_L, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)));

    // Dense vector descriptors.
    rocsparse_dnvec_descr vec_x;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_x, N, const_cast<void*>(b_gpu.data_ptr()), roc_dtype));
    DnVecGuard x_guard(vec_x);

    rocsparse_dnvec_descr vec_y;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_y, N, result.data_ptr(), roc_dtype));
    DnVecGuard y_guard(vec_y);

    float  alpha_f = 1.0f;
    double alpha_d = 1.0;
    void* alpha = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f)
                                            : static_cast<void*>(&alpha_d);

    // Stage 1: buffer size.
    size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, mat_L, vec_x, vec_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_buffer_size, &buffer_size, nullptr));
    HipBuffer workspace(buffer_size);

    // Stage 2: preprocess.
    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, mat_L, vec_x, vec_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_preprocess, &buffer_size, workspace.ptr));

    // Stage 3: compute.
    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, mat_L, vec_x, vec_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_compute, &buffer_size, workspace.ptr));

    HIP_CHECK_SPARSE(hipDeviceSynchronize());
    return result;
}

// Multi-RHS triangular solve: loops per column calling SpSV. As on the
// CUDA side, this is O(K) suboptimal — rocsparse has SpSM (multi-RHS)
// but caching descriptors and workspace across K calls is a follow-up.
Tensor rocm_sparse_trsm_kernel(const SparseTensor& L, const Tensor& B, bool upper) {
    auto L_shape = L.shape();
    if (L_shape.size() != 2 || L_shape[0] != L_shape[1]) {
        throw std::runtime_error("rocm_sparse_trsm: L must be square 2D");
    }
    if (B.ndim() != 2) {
        throw std::runtime_error("rocm_sparse_trsm: B must be 2D");
    }
    const int64_t N = L_shape[0];
    const int64_t K = B.shape()[1];
    if (B.shape()[0] != N) {
        throw std::runtime_error("rocm_sparse_trsm: dimension mismatch");
    }
    const DType dtype = B.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_sparse_trsm: only Float32/Float64 supported");
    }

    auto B_gpu = (B.device().type != Device::Type::ROCm)
                   ? B.to(Device::rocm()).contiguous()
                   : B.contiguous();
    auto X = zeros(std::vector<int64_t>{N, K}, dtype, Device::rocm());

    for (int64_t k = 0; k < K; ++k) {
        // Gather B[:, k] into a contiguous 1D buffer on GPU.
        Tensor b_col = zeros(std::vector<int64_t>{N}, dtype, Device::rocm());
        if (dtype == DType::Float32) {
            auto* dst = b_col.data<float>();
            const auto* src = B_gpu.data<float>();
            for (int64_t i = 0; i < N; ++i) {
                HIP_CHECK_SPARSE(hipMemcpy(
                    dst + i, src + i * K + k, sizeof(float),
                    hipMemcpyDeviceToDevice));
            }
        } else {
            auto* dst = b_col.data<double>();
            const auto* src = B_gpu.data<double>();
            for (int64_t i = 0; i < N; ++i) {
                HIP_CHECK_SPARSE(hipMemcpy(
                    dst + i, src + i * K + k, sizeof(double),
                    hipMemcpyDeviceToDevice));
            }
        }

        auto x_col = rocm_sparse_trsv_kernel(L, b_col, upper);

        // Scatter x_col back into X[:, k].
        if (dtype == DType::Float32) {
            auto* dst = X.data<float>();
            const auto* src = x_col.data<float>();
            for (int64_t i = 0; i < N; ++i) {
                HIP_CHECK_SPARSE(hipMemcpy(
                    dst + i * K + k, src + i, sizeof(float),
                    hipMemcpyDeviceToDevice));
            }
        } else {
            auto* dst = X.data<double>();
            const auto* src = x_col.data<double>();
            for (int64_t i = 0; i < N; ++i) {
                HIP_CHECK_SPARSE(hipMemcpy(
                    dst + i * K + k, src + i, sizeof(double),
                    hipMemcpyDeviceToDevice));
            }
        }
    }

    HIP_CHECK_SPARSE(hipDeviceSynchronize());
    return X;
}

// ============================================================================
// GPU-native sparse format conversions (called from SparseTensor methods)
// ============================================================================

SparseTensor rocm_coo_to_csr(const SparseTensor& sparse) {
    return ensure_csr_on_gpu(sparse);
}

SparseTensor rocm_coalesce(const SparseTensor& sparse) {
    // Coalesce COO on GPU using thrust (HIP-compatible) sort + reduce_by_key.
    if (sparse.layout() != SparseLayout::COO) return sparse;
    if (sparse.nnz() == 0) return sparse;

    auto sp_shape = sparse.shape();
    int64_t sparse_dim = static_cast<int64_t>(sp_shape.size());
    int64_t nnz = sparse.nnz();

    Tensor indices = sparse.indices().contiguous();
    Tensor values = sparse.values().contiguous();
    const int64_t* idx_ptr = indices.data<int64_t>();

    // Compute strides for linearized keys
    std::vector<int64_t> h_strides(sparse_dim);
    if (sparse_dim > 0) {
        h_strides[sparse_dim - 1] = 1;
        for (int64_t d = sparse_dim - 2; d >= 0; --d) {
            h_strides[d] = h_strides[d + 1] * sp_shape[d + 1];
        }
    }

    // Build linearized keys on GPU
    Tensor keys_t = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* keys_ptr = keys_t.data<int64_t>();

    for (int64_t d = 0; d < sparse_dim; ++d) {
        if (h_strides[d] == 0) continue;
        const int64_t* dim_idx = idx_ptr + d * nnz;
        int64_t stride = h_strides[d];
        auto keys_dptr = thrust::device_pointer_cast(keys_ptr);
        auto dim_dptr = thrust::device_pointer_cast(dim_idx);
        thrust::transform(thrust::hip::par,
                          keys_dptr, keys_dptr + nnz,
                          dim_dptr,
                          keys_dptr,
                          [stride] __device__ (int64_t k, int64_t idx_val) {
                              return k + idx_val * stride;
                          });
    }

    // Sort by key with permutation
    Tensor perm_t = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* perm_ptr = perm_t.data<int64_t>();
    thrust::sequence(thrust::hip::par, thrust::device_pointer_cast(perm_ptr),
                     thrust::device_pointer_cast(perm_ptr + nnz));
    thrust::sort_by_key(thrust::hip::par,
                        thrust::device_pointer_cast(keys_ptr),
                        thrust::device_pointer_cast(keys_ptr + nnz),
                        thrust::device_pointer_cast(perm_ptr));

    // Gather sorted indices and values
    Tensor sorted_indices = zeros({sparse_dim, nnz}, DType::Int64, Device::rocm());
    int64_t* si_ptr = sorted_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim; ++d) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(idx_ptr + d * nnz),
                       thrust::device_pointer_cast(si_ptr + d * nnz));
    }

    Tensor sorted_vals = zeros({nnz}, values.dtype(), Device::rocm());
    if (values.dtype() == DType::Float32) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<float>()),
                       thrust::device_pointer_cast(sorted_vals.data<float>()));
    } else if (values.dtype() == DType::Float64) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<double>()),
                       thrust::device_pointer_cast(sorted_vals.data<double>()));
    }

    // Reduce duplicates by key
    Tensor out_keys = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* ok_ptr = out_keys.data<int64_t>();
    Tensor out_vals = zeros({nnz}, values.dtype(), Device::rocm());
    int64_t new_nnz = 0;

    if (values.dtype() == DType::Float32) {
        auto end = thrust::reduce_by_key(
            thrust::hip::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<float>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<float>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    } else if (values.dtype() == DType::Float64) {
        auto end = thrust::reduce_by_key(
            thrust::hip::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<double>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<double>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    }

    // Decode linearized keys back to multi-dim indices
    Tensor new_indices = zeros({sparse_dim, new_nnz}, DType::Int64, Device::rocm());
    int64_t* ni_ptr = new_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim; ++d) {
        int64_t stride = h_strides[d];
        auto ok_dptr = thrust::device_pointer_cast(ok_ptr);
        auto ni_dptr = thrust::device_pointer_cast(ni_ptr + d * new_nnz);
        if (d < sparse_dim - 1) {
            int64_t next_stride = h_strides[d + 1];
            thrust::transform(thrust::hip::par,
                              ok_dptr, ok_dptr + new_nnz,
                              ni_dptr,
                              [stride, next_stride] __device__ (int64_t key) {
                                  return (key / stride) % (stride / next_stride);
                              });
        } else {
            thrust::transform(thrust::hip::par,
                              ok_dptr, ok_dptr + new_nnz,
                              ni_dptr,
                              [stride] __device__ (int64_t key) {
                                  return key % stride;
                              });
        }
    }

    Tensor final_vals = zeros({new_nnz}, values.dtype(), Device::rocm());
    if (values.dtype() == DType::Float32) {
        HIP_CHECK_SPARSE(hipMemcpy(final_vals.data<float>(), out_vals.data<float>(),
                                   new_nnz * sizeof(float), hipMemcpyDeviceToDevice));
    } else if (values.dtype() == DType::Float64) {
        HIP_CHECK_SPARSE(hipMemcpy(final_vals.data<double>(), out_vals.data<double>(),
                                   new_nnz * sizeof(double), hipMemcpyDeviceToDevice));
    }

    return SparseTensor::sparse_coo(new_indices, final_vals,
                                    std::vector<int64_t>(sp_shape.begin(), sp_shape.end()));
}

SparseTensor rocm_coo_to_csc(const SparseTensor& sparse) {
    if (sparse.layout() == SparseLayout::CSC) return sparse;

    auto sp_shape = sparse.shape();
    int64_t nrows = sp_shape[0];
    int64_t ncols = sp_shape[1];

    auto coo = rocm_coalesce(sparse);
    int64_t nnz = coo.nnz();

    Tensor indices = coo.indices().contiguous();
    Tensor values = coo.values().contiguous();
    const int64_t* idx_ptr = indices.data<int64_t>();
    const int64_t* row_ptr_src = idx_ptr;
    const int64_t* col_ptr_src = idx_ptr + nnz;

    // Sort by (col, row): key = col * nrows + row
    Tensor sort_keys = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* sk_ptr = sort_keys.data<int64_t>();
    {
        auto sk_dptr = thrust::device_pointer_cast(sk_ptr);
        auto col_dptr = thrust::device_pointer_cast(col_ptr_src);
        auto row_dptr = thrust::device_pointer_cast(row_ptr_src);
        thrust::transform(thrust::hip::par,
                          col_dptr, col_dptr + nnz,
                          row_dptr,
                          sk_dptr,
                          [nrows] __device__ (int64_t c, int64_t r) {
                              return c * nrows + r;
                          });
    }

    Tensor perm_t = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* perm_ptr = perm_t.data<int64_t>();
    thrust::sequence(thrust::hip::par, thrust::device_pointer_cast(perm_ptr),
                     thrust::device_pointer_cast(perm_ptr + nnz));
    thrust::sort_by_key(thrust::hip::par,
                        thrust::device_pointer_cast(sk_ptr),
                        thrust::device_pointer_cast(sk_ptr + nnz),
                        thrust::device_pointer_cast(perm_ptr));

    // Gather sorted row/col indices
    Tensor sorted_rows = zeros({nnz}, DType::Int64, Device::rocm());
    thrust::gather(thrust::hip::par,
                   thrust::device_pointer_cast(perm_ptr),
                   thrust::device_pointer_cast(perm_ptr + nnz),
                   thrust::device_pointer_cast(row_ptr_src),
                   thrust::device_pointer_cast(sorted_rows.data<int64_t>()));

    Tensor sorted_cols = zeros({nnz}, DType::Int64, Device::rocm());
    thrust::gather(thrust::hip::par,
                   thrust::device_pointer_cast(perm_ptr),
                   thrust::device_pointer_cast(perm_ptr + nnz),
                   thrust::device_pointer_cast(col_ptr_src),
                   thrust::device_pointer_cast(sorted_cols.data<int64_t>()));

    // Build ccol_indices using upper_bound
    Tensor ccol = zeros({ncols + 1}, DType::Int64, Device::rocm());
    int64_t* ccol_ptr = ccol.data<int64_t>();
    Tensor boundaries = zeros({ncols}, DType::Int64, Device::rocm());
    int64_t* bounds_ptr = boundaries.data<int64_t>();
    thrust::upper_bound(thrust::hip::par,
                        thrust::device_pointer_cast(sorted_cols.data<int64_t>()),
                        thrust::device_pointer_cast(sorted_cols.data<int64_t>() + nnz),
                        thrust::counting_iterator<int64_t>(0),
                        thrust::counting_iterator<int64_t>(ncols),
                        thrust::device_pointer_cast(bounds_ptr));
    HIP_CHECK_SPARSE(hipMemset(ccol_ptr, 0, sizeof(int64_t)));
    HIP_CHECK_SPARSE(hipMemcpy(ccol_ptr + 1, bounds_ptr,
                               ncols * sizeof(int64_t), hipMemcpyDeviceToDevice));

    // Gather sorted values
    Tensor sorted_vals = zeros({nnz}, values.dtype(), Device::rocm());
    if (values.dtype() == DType::Float32) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<float>()),
                       thrust::device_pointer_cast(sorted_vals.data<float>()));
    } else if (values.dtype() == DType::Float64) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<double>()),
                       thrust::device_pointer_cast(sorted_vals.data<double>()));
    }

    return SparseTensor::sparse_csc(ccol, sorted_rows, sorted_vals,
                                    std::vector<int64_t>{nrows, ncols});
}

Tensor rocm_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2)
        throw std::runtime_error("rocm_sparse_add: both inputs must be 2D");
    int64_t M = sp_shape[0], K = sp_shape[1];
    if (M != dense.shape()[0] || K != dense.shape()[1])
        throw std::runtime_error("rocm_sparse_add: shape mismatch");

    DType dtype = dense.dtype();
    auto csr = ensure_csr_on_gpu(sparse);
    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous() : dense.contiguous();
    auto result = dense_gpu.clone();

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int blocks = static_cast<int>((M + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            result.data<float>(), M, K);
    } else if (dtype == DType::Float64) {
        csr_sparse_add_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            result.data<double>(), M, K);
    } else {
        throw std::runtime_error("rocm_sparse_add: only Float32 and Float64 supported");
    }
    HIP_CHECK_SPARSE(hipGetLastError());
    return result;
}

} // namespace rocm
} // namespace tenzor

#else // !TENZOR_HAS_ROCSPARSE — native HIP CSR SpMM/SpMV fallbacks

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}

namespace tenzor {
namespace rocm {

namespace {

#define HIP_CHECK_SPARSE(call)                                                  \
    do {                                                                         \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("HIP error in sparse at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));     \
        }                                                                       \
    } while (0)

// CSR SpMV kernel: one thread per row, accumulates dot product
template <typename T>
__global__ void csr_spmv_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    const T* __restrict__ x_ptr,
    T* __restrict__ y_ptr,
    int64_t nrows)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    T sum = static_cast<T>(0);
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        sum += val_ptr[j] * x_ptr[col_ptr[j]];
    }
    y_ptr[row] = sum;
}

// CSR SpMM kernel: one thread per output element (row, col)
template <typename T>
__global__ void csr_spmm_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    const T* __restrict__ b_ptr,
    T* __restrict__ c_ptr,
    int64_t nrows, int64_t ncols_b)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t row = idx / ncols_b;
    int64_t col = idx % ncols_b;
    if (row >= nrows) return;

    T sum = static_cast<T>(0);
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        sum += val_ptr[j] * b_ptr[col_ptr[j] * ncols_b + col];
    }
    c_ptr[row * ncols_b + col] = sum;
}

// CSR SparseAdd kernel: one thread per row, adds sparse values into dense output
template <typename T>
__global__ void csr_sparse_add_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    T* __restrict__ out_ptr,
    int64_t nrows, int64_t ncols)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        out_ptr[row * ncols + col_ptr[j]] += val_ptr[j];
    }
}

/// Ensure SparseTensor is in CSR format on ROCm device.
SparseTensor ensure_csr_on_gpu(const SparseTensor& sparse) {
    auto sp = (sparse.device().type != Device::Type::ROCm)
              ? sparse.to(Device::rocm())
              : sparse;
    if (sp.layout() != SparseLayout::CSR) {
        throw std::runtime_error("rocm native sparse fallback requires CSR format");
    }
    return sp;
}

} // anonymous namespace

Tensor rocm_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("rocm_spmm: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("rocm_spmm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(dense.shape()[0]) + ")");
    }

    DType dtype = dense.dtype();
    auto csr = ensure_csr_on_gpu(sparse);

    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous()
                     : dense.contiguous();

    auto result = zeros({M, N}, dtype, Device::rocm());

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int64_t total = M * N;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_spmm_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            dense_gpu.data<float>(), result.data<float>(), M, N);
    } else if (dtype == DType::Float64) {
        csr_spmm_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            dense_gpu.data<double>(), result.data<double>(), M, N);
    } else {
        throw std::runtime_error("rocm_spmm: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    HIP_CHECK_SPARSE(hipGetLastError());

    return result;
}

Tensor rocm_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("rocm_spmv: sparse must be 2D, vec must be 1D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("rocm_spmv: dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(vec.shape()[0]) + ")");
    }

    DType dtype = vec.dtype();
    auto csr = ensure_csr_on_gpu(sparse);

    auto vec_gpu = (vec.device().type != Device::Type::ROCm)
                   ? vec.to(Device::rocm()).contiguous()
                   : vec.contiguous();

    auto result = zeros({M}, dtype, Device::rocm());

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int blocks = static_cast<int>((M + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_spmv_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            vec_gpu.data<float>(), result.data<float>(), M);
    } else if (dtype == DType::Float64) {
        csr_spmv_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            vec_gpu.data<double>(), result.data<double>(), M);
    } else {
        throw std::runtime_error("rocm_spmv: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    HIP_CHECK_SPARSE(hipGetLastError());

    return result;
}

Tensor rocm_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2)
        throw std::runtime_error("rocm_sparse_add: both inputs must be 2D");
    int64_t M = sp_shape[0], K = sp_shape[1];
    if (M != dense.shape()[0] || K != dense.shape()[1])
        throw std::runtime_error("rocm_sparse_add: shape mismatch");

    DType dtype = dense.dtype();
    auto csr = ensure_csr_on_gpu(sparse);
    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous() : dense.contiguous();
    auto result = dense_gpu.clone();

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int blocks = static_cast<int>((M + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            result.data<float>(), M, K);
    } else if (dtype == DType::Float64) {
        csr_sparse_add_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            result.data<double>(), M, K);
    } else {
        throw std::runtime_error("rocm_sparse_add: only Float32 and Float64 supported");
    }
    HIP_CHECK_SPARSE(hipGetLastError());
    return result;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCSPARSE
