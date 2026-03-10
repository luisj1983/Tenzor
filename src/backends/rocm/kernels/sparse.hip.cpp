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

#include <rocsparse/rocsparse.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

#define ROCSPARSE_CHECK(call)                                                   \
    do {                                                                         \
        rocsparse_status status = (call);                                       \
        if (status != rocsparse_status_success) {                               \
            throw std::runtime_error(                                           \
                std::string("rocSPARSE error at ") + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " - status " +                      \
                std::to_string(static_cast<int>(status)));                     \
        }                                                                       \
    } while (0)

/// Convert span to vector (HIP compiler may not do implicit span->vector).
std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

/// RAII wrapper for HIP device memory.
struct HipBuffer {
    void* ptr = nullptr;
    explicit HipBuffer(size_t bytes) {
        if (bytes > 0) HIP_CHECK_SPARSE(hipMalloc(&ptr, bytes));
    }
    ~HipBuffer() { if (ptr) hipFree(ptr); }
    HipBuffer(const HipBuffer&) = delete;
    HipBuffer& operator=(const HipBuffer&) = delete;
    template<typename T> T* as() { return static_cast<T*>(ptr); }
};

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

/// Per-thread rocSPARSE handle.
rocsparse_handle get_rocsparse_handle() {
    static thread_local rocsparse_handle handle = nullptr;
    if (!handle) {
        ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    }
    return handle;
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
        cast_i64_to_i32<<<blocks_nnz, threads>>>(row_indices_ptr, row_i32_buf.as<int32_t>(), nnz);
        HIP_CHECK_SPARSE(hipGetLastError());

        // Convert COO row indices to CSR row pointers on GPU
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

    HIP_CHECK_SPARSE(hipDeviceSynchronize());
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

    // Determine buffer size
    size_t buffer_size = 0;
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_ptr = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f) : static_cast<void*>(&alpha_d);
    void* beta_ptr = (dtype == DType::Float32) ? static_cast<void*>(&beta_f) : static_cast<void*>(&beta_d);

    ROCSPARSE_CHECK(rocsparse_spmv(
        handle,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        vec_x,
        beta_ptr,
        vec_y,
        roc_dtype,
        rocsparse_spmv_alg_default,
        rocsparse_spmv_stage_buffer_size,
        &buffer_size,
        nullptr
    ));

    HipBuffer workspace(buffer_size);

    ROCSPARSE_CHECK(rocsparse_spmv(
        handle,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        vec_x,
        beta_ptr,
        vec_y,
        roc_dtype,
        rocsparse_spmv_alg_default,
        rocsparse_spmv_stage_compute,
        &buffer_size,
        workspace.ptr
    ));

    HIP_CHECK_SPARSE(hipDeviceSynchronize());
    return result;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCSPARSE
