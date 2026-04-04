/**
 * @file sparse.cu
 * @brief CUDA kernels for sparse tensor operations using cuSPARSE.
 *
 * Provides GPU-accelerated implementations of:
 * - spmm (sparse-dense matrix multiplication) via cusparseSpMM()
 * - spmv (sparse-dense matrix-vector multiplication) via cusparseSpMV()
 *
 * Uses CSR format descriptors for cuSPARSE API compatibility.
 * Both COO and CSR inputs are supported; COO is converted to CSR internally.
 */

#ifdef TENZOR_HAS_CUSPARSE

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "../cusparse_handle_pool.hpp"

#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tenzor/ops/creation.hpp"

namespace tenzor {
namespace cuda {

namespace {

#ifndef CUDA_CHECK_SPARSE
#define CUDA_CHECK_SPARSE(call)                                                 \
    do {                                                                          \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            throw std::runtime_error(                                            \
                std::string("CUDA error in sparse at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));     \
        }                                                                        \
    } while (0)
#endif

/// Convert span to vector (nvcc doesn't support implicit span->vector conversion).
std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

/// RAII wrapper for CUDA device memory (exception-safe cudaMalloc/cudaFree).
struct CudaBuffer {
    void* ptr = nullptr;
    explicit CudaBuffer(size_t bytes) {
        if (bytes > 0) CUDA_CHECK_SPARSE(cudaMalloc(&ptr, bytes));
    }
    ~CudaBuffer() { if (ptr) cudaFree(ptr); }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    template<typename T> T* as() { return static_cast<T*>(ptr); }
};

/// RAII guard for cuSPARSE sparse matrix descriptor.
struct SpMatGuard {
    cusparseSpMatDescr_t desc = nullptr;
    explicit SpMatGuard(cusparseSpMatDescr_t d) : desc(d) {}
    ~SpMatGuard() { if (desc) cusparseDestroySpMat(desc); }
    SpMatGuard(const SpMatGuard&) = delete;
    SpMatGuard& operator=(const SpMatGuard&) = delete;
};

/// RAII guard for cuSPARSE dense matrix descriptor.
struct DnMatGuard {
    cusparseDnMatDescr_t desc = nullptr;
    explicit DnMatGuard(cusparseDnMatDescr_t d) : desc(d) {}
    ~DnMatGuard() { if (desc) cusparseDestroyDnMat(desc); }
    DnMatGuard(const DnMatGuard&) = delete;
    DnMatGuard& operator=(const DnMatGuard&) = delete;
};

/// RAII guard for cuSPARSE dense vector descriptor.
struct DnVecGuard {
    cusparseDnVecDescr_t desc = nullptr;
    explicit DnVecGuard(cusparseDnVecDescr_t d) : desc(d) {}
    ~DnVecGuard() { if (desc) cusparseDestroyDnVec(desc); }
    DnVecGuard(const DnVecGuard&) = delete;
    DnVecGuard& operator=(const DnVecGuard&) = delete;
};

/// Get cuSPARSE data type from DType.
cudaDataType get_cuda_data_type(DType dtype) {
    switch (dtype) {
        case DType::Float32: return CUDA_R_32F;
        case DType::Float64: return CUDA_R_64F;
        default:
            throw std::runtime_error("cuda_sparse: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
    }
}

/// CUDA kernel: convert Int64 row indices to Int32.
__global__ void cast_i64_to_i32(const int64_t* __restrict__ src,
                                 int32_t* __restrict__ dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int32_t>(src[i]);
}

/// CUDA kernel: convert Int32 crow_indices to Int64.
__global__ void cast_i32_to_i64(const int32_t* __restrict__ src,
                                 int64_t* __restrict__ dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int64_t>(src[i]);
}

/// Helper to build a CSR SparseTensor on GPU from a COO SparseTensor.
/// If already CSR, returns the input. Otherwise converts COO -> CSR on GPU
/// using cusparseXcoo2csr() — avoids CPU round-trips.
SparseTensor ensure_csr_on_gpu(const SparseTensor& sparse, cudaStream_t stream = 0) {
    // Move to GPU if on CPU
    auto sp = (sparse.device().type != Device::Type::CUDA)
              ? sparse.to(Device::cuda())
              : sparse;

    // Convert to CSR if in COO format — directly on GPU
    if (sp.layout() == SparseLayout::COO) {
        auto sp_shape = sp.shape();
        int64_t nrows = sp_shape[0];
        int64_t ncols = sp_shape[1];
        int64_t nnz = sp.nnz();

        // COO indices: [2, nnz] — row 0 = row indices, row 1 = col indices
        Tensor indices = sp.indices().contiguous();
        Tensor values = sp.values().contiguous();

        // indices is [2, nnz], laid out as [row0..rowN, col0..colN]
        const int64_t* indices_ptr = indices.data<int64_t>();
        const int64_t* row_indices_ptr = indices_ptr;
        const int64_t* col_indices_ptr = indices_ptr + nnz;

        // cusparseXcoo2csr requires Int32 — convert row indices on GPU
        CudaBuffer row_i32_buf(nnz * sizeof(int32_t));
        CudaBuffer crow_i32_buf((nrows + 1) * sizeof(int32_t));

        int threads = 256;
        int blocks_nnz = static_cast<int>((nnz + threads - 1) / threads);
        cast_i64_to_i32<<<blocks_nnz, threads, 0, stream>>>(row_indices_ptr, row_i32_buf.as<int32_t>(), nnz);
        CUDA_CHECK_SPARSE(cudaGetLastError());

        // Convert COO row indices to CSR row pointers on GPU
        cusparseHandle_t handle = CuSPARSEHandlePool::get(stream);
        CUSPARSE_CHECK(cusparseXcoo2csr(
            handle, row_i32_buf.as<int32_t>(), static_cast<int>(nnz), static_cast<int>(nrows),
            crow_i32_buf.as<int32_t>(), CUSPARSE_INDEX_BASE_ZERO));

        // Convert Int32 crow_indices back to Int64 on GPU
        Tensor crow_indices = zeros(std::vector<int64_t>{nrows + 1}, DType::Int64, Device::cuda());
        int blocks_crow = static_cast<int>((nrows + 1 + threads - 1) / threads);
        cast_i32_to_i64<<<blocks_crow, threads, 0, stream>>>(crow_i32_buf.as<int32_t>(), crow_indices.data<int64_t>(), nrows + 1);
        CUDA_CHECK_SPARSE(cudaGetLastError());

        // Copy col indices (already on GPU, just need a separate tensor)
        Tensor col_idx = zeros(std::vector<int64_t>{nnz}, DType::Int64, Device::cuda());
        CUDA_CHECK_SPARSE(cudaMemcpyAsync(col_idx.data<int64_t>(), col_indices_ptr,
                                          nnz * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));

        return SparseTensor::sparse_csr(
            crow_indices, col_idx, values,
            std::vector<int64_t>{nrows, ncols});
    }
    return sp;
}

} // anonymous namespace

Tensor cuda_spmm_kernel(const SparseTensor& sparse, const Tensor& dense, cudaStream_t stream = 0) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("cuda_spmm: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("cuda_spmm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(dense.shape()[0]) + ")");
    }

    DType dtype = dense.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("cuda_spmm: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    // Ensure sparse is in CSR format on GPU
    auto csr = ensure_csr_on_gpu(sparse, stream);
    int64_t nnz = csr.nnz();

    // Ensure dense is contiguous and on GPU
    auto dense_gpu = (dense.device().type != Device::Type::CUDA)
                     ? dense.to(Device::cuda()).contiguous()
                     : dense.contiguous();

    // Create output tensor
    auto result = zeros({M, N}, dtype, Device::cuda());

    // Get cuSPARSE handle
    cusparseHandle_t handle = CuSPARSEHandlePool::get(stream);
    cudaDataType cuda_dtype = get_cuda_data_type(dtype);

    // Get raw pointers
    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    // cuSPARSE uses int32 for indices in many APIs; we use int64 (CUSPARSE_INDEX_64I)
    cusparseSpMatDescr_t mat_sparse;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_sparse,
        M, K, nnz,
        const_cast<void*>(static_cast<const void*>(crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(col.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(vals.data_ptr())),
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO,
        cuda_dtype
    ));
    SpMatGuard sparse_guard(mat_sparse);

    // Dense matrix descriptor (column-major for cuSPARSE, but we use row-major)
    // cuSPARSE expects column-major dense matrices by default.
    // For row-major: we use CUSPARSE_ORDER_ROW
    cusparseDnMatDescr_t mat_dense;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_dense,
        K, N, N,  // rows, cols, leading dimension (row-major: ld = N)
        const_cast<void*>(dense_gpu.data_ptr()),
        cuda_dtype,
        CUSPARSE_ORDER_ROW
    ));
    DnMatGuard dense_guard(mat_dense);

    cusparseDnMatDescr_t mat_result;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_result,
        M, N, N,  // rows, cols, leading dimension
        result.data_ptr(),
        cuda_dtype,
        CUSPARSE_ORDER_ROW
    ));
    DnMatGuard result_guard(mat_result);

    // Determine buffer size
    size_t buffer_size = 0;
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_ptr = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f) : static_cast<void*>(&alpha_d);
    void* beta_ptr = (dtype == DType::Float32) ? static_cast<void*>(&beta_f) : static_cast<void*>(&beta_d);

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        cuda_dtype,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &buffer_size
    ));

    // Allocate workspace buffer (RAII — freed on scope exit or exception)
    CudaBuffer workspace(buffer_size);

    // Execute SpMM
    CUSPARSE_CHECK(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        cuda_dtype,
        CUSPARSE_SPMM_ALG_DEFAULT,
        workspace.ptr
    ));

    // Stream-synchronize to ensure computation is complete (avoids blocking other streams)
    CUDA_CHECK_SPARSE(cudaStreamSynchronize(stream));

    return result;
}

Tensor cuda_spmv_kernel(const SparseTensor& sparse, const Tensor& vec, cudaStream_t stream = 0) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("cuda_spmv: sparse must be 2D, vec must be 1D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("cuda_spmv: dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(vec.shape()[0]) + ")");
    }

    DType dtype = vec.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("cuda_spmv: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    // Ensure sparse is in CSR format on GPU
    auto csr = ensure_csr_on_gpu(sparse, stream);
    int64_t nnz = csr.nnz();

    // Ensure vec is contiguous and on GPU
    auto vec_gpu = (vec.device().type != Device::Type::CUDA)
                   ? vec.to(Device::cuda()).contiguous()
                   : vec.contiguous();

    // Create output tensor
    auto result = zeros({M}, dtype, Device::cuda());

    // Get cuSPARSE handle
    cusparseHandle_t handle = CuSPARSEHandlePool::get(stream);
    cudaDataType cuda_dtype = get_cuda_data_type(dtype);

    // Get raw pointers
    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    // Create sparse matrix descriptor
    cusparseSpMatDescr_t mat_sparse;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_sparse,
        M, K, nnz,
        const_cast<void*>(static_cast<const void*>(crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(col.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(vals.data_ptr())),
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO,
        cuda_dtype
    ));
    SpMatGuard sparse_guard(mat_sparse);

    // Create dense vector descriptors
    cusparseDnVecDescr_t vec_x;
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_x,
        K,
        const_cast<void*>(vec_gpu.data_ptr()),
        cuda_dtype
    ));
    DnVecGuard vec_x_guard(vec_x);

    cusparseDnVecDescr_t vec_y;
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_y,
        M,
        result.data_ptr(),
        cuda_dtype
    ));
    DnVecGuard vec_y_guard(vec_y);

    // Determine buffer size
    size_t buffer_size = 0;
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_ptr = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f) : static_cast<void*>(&alpha_d);
    void* beta_ptr = (dtype == DType::Float32) ? static_cast<void*>(&beta_f) : static_cast<void*>(&beta_d);

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr,
        mat_sparse,
        vec_x,
        beta_ptr,
        vec_y,
        cuda_dtype,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size
    ));

    // Allocate workspace buffer (RAII — freed on scope exit or exception)
    CudaBuffer workspace(buffer_size);

    // Execute SpMV
    CUSPARSE_CHECK(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr,
        mat_sparse,
        vec_x,
        beta_ptr,
        vec_y,
        cuda_dtype,
        CUSPARSE_SPMV_ALG_DEFAULT,
        workspace.ptr
    ));

    // Synchronize
    CUDA_CHECK_SPARSE(cudaStreamSynchronize(stream));

    return result;
}

} // namespace cuda
} // namespace tenzor

#else // !TENZOR_HAS_CUSPARSE — native CUDA CSR SpMM/SpMV fallbacks

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}

namespace tenzor {
namespace cuda {

namespace {

#define CUDA_CHECK_SPARSE_FALLBACK(call)                                        \
    do {                                                                         \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error in sparse at ") + __FILE__ + ":" +     \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));    \
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

/// Ensure SparseTensor is in CSR format on CUDA device.
SparseTensor ensure_csr_on_gpu(const SparseTensor& sparse) {
    auto sp = (sparse.device().type != Device::Type::CUDA)
              ? sparse.to(Device::cuda())
              : sparse;
    if (sp.layout() != SparseLayout::CSR) {
        throw std::runtime_error("cuda native sparse fallback requires CSR format");
    }
    return sp;
}

} // anonymous namespace

Tensor cuda_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("cuda_spmm: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("cuda_spmm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(dense.shape()[0]) + ")");
    }

    DType dtype = dense.dtype();
    auto csr = ensure_csr_on_gpu(sparse);

    auto dense_gpu = (dense.device().type != Device::Type::CUDA)
                     ? dense.to(Device::cuda()).contiguous()
                     : dense.contiguous();

    auto result = zeros({M, N}, dtype, Device::cuda());

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
        throw std::runtime_error("cuda_spmm: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    CUDA_CHECK_SPARSE_FALLBACK(cudaGetLastError());

    return result;
}

Tensor cuda_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("cuda_spmv: sparse must be 2D, vec must be 1D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("cuda_spmv: dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(vec.shape()[0]) + ")");
    }

    DType dtype = vec.dtype();
    auto csr = ensure_csr_on_gpu(sparse);

    auto vec_gpu = (vec.device().type != Device::Type::CUDA)
                   ? vec.to(Device::cuda()).contiguous()
                   : vec.contiguous();

    auto result = zeros({M}, dtype, Device::cuda());

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
        throw std::runtime_error("cuda_spmv: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    CUDA_CHECK_SPARSE_FALLBACK(cudaGetLastError());

    return result;
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUSPARSE
