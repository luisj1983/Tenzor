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

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>

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

// Two-argument overloads used by cuda_kernel_registry.cpp. The
// three-argument forms take an optional stream — sparse_ops.cpp can't
// reference cudaStream_t without pulling in CUDA headers, so the
// two-argument versions forward with stream=0 (default stream).
Tensor cuda_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) {
    return cuda_spmm_kernel(sparse, dense, /*stream=*/0);
}

Tensor cuda_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) {
    return cuda_spmv_kernel(sparse, vec, /*stream=*/0);
}

// ============================================================================
// SpGEMM — sparse × sparse → sparse (CSR × CSR → CSR)
// ============================================================================
//
// cuSPARSE generic SpGEMM API (available since CUDA 11.3). The call sequence
// is fixed: createDescr → workEstimation (2×, once for each internal buffer)
// → compute (2×) → copy result → destroyDescr.
//
// Workspace buffers are allocated per call. Caching by sparsity pattern is a
// natural follow-up (plan item referenced in sparse_ops.cpp) but requires
// invalidation logic; keep it simple for the first GPU integration.

/// RAII guard for cusparseSpGEMM descriptors.
struct SpGEMMDescrGuard {
    cusparseSpGEMMDescr_t desc = nullptr;
    SpGEMMDescrGuard() { CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&desc)); }
    ~SpGEMMDescrGuard() { if (desc) cusparseSpGEMM_destroyDescr(desc); }
    SpGEMMDescrGuard(const SpGEMMDescrGuard&) = delete;
    SpGEMMDescrGuard& operator=(const SpGEMMDescrGuard&) = delete;
};

SparseTensor cuda_spgemm_kernel(const SparseTensor& a, const SparseTensor& b,
                                 void* stream_opaque) {
    // sparse_ops.cpp can't include cuda_runtime.h, so the stream is passed
    // in as a void*. cudaStream_t is a typedef for `CUstream_st*` — a
    // pointer — so this cast is ABI-safe.
    cudaStream_t stream = static_cast<cudaStream_t>(stream_opaque);
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("cuda_spgemm: both inputs must be 2D");
    }
    const int64_t M = a_shape[0];
    const int64_t K = a_shape[1];
    const int64_t N = b_shape[1];
    if (K != b_shape[0]) {
        throw std::runtime_error("cuda_spgemm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(b_shape[0]) + ")");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("cuda_spgemm: dtype mismatch");
    }
    const DType dtype = a.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("cuda_spgemm: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    // Move both operands to CSR on GPU.
    auto a_csr = ensure_csr_on_gpu(a, stream);
    auto b_csr = ensure_csr_on_gpu(b, stream);
    const int64_t nnz_a = a_csr.nnz();
    const int64_t nnz_b = b_csr.nnz();

    auto a_crow = a_csr.crow_indices().contiguous();
    auto a_col  = a_csr.col_indices().contiguous();
    auto a_vals = a_csr.values().contiguous();
    auto b_crow = b_csr.crow_indices().contiguous();
    auto b_col  = b_csr.col_indices().contiguous();
    auto b_vals = b_csr.values().contiguous();

    cusparseHandle_t handle = CuSPARSEHandlePool::get(stream);
    const cudaDataType cuda_dtype = get_cuda_data_type(dtype);

    // A descriptor.
    cusparseSpMatDescr_t mat_a;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_a, M, K, nnz_a,
        const_cast<void*>(static_cast<const void*>(a_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(a_col.data<int64_t>())),
        const_cast<void*>(a_vals.data_ptr()),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
    SpMatGuard a_guard(mat_a);

    // B descriptor.
    cusparseSpMatDescr_t mat_b;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_b, K, N, nnz_b,
        const_cast<void*>(static_cast<const void*>(b_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(b_col.data<int64_t>())),
        const_cast<void*>(b_vals.data_ptr()),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
    SpMatGuard b_guard(mat_b);

    // C descriptor: start with an empty CSR placeholder; cuSPARSE will tell
    // us the required output nnz after the compute step and we allocate
    // then. The initial row-pointer buffer is (M+1) entries so the
    // descriptor is well-formed.
    auto c_crow = zeros(std::vector<int64_t>{M + 1}, DType::Int64, Device::cuda());
    cusparseSpMatDescr_t mat_c;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_c, M, N, 0,
        c_crow.data<int64_t>(),
        nullptr, nullptr,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
    SpMatGuard c_guard(mat_c);

    SpGEMMDescrGuard spgemm_desc;

    const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    float  alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0,  beta_d = 0.0;
    void* alpha = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f)
                                            : static_cast<void*>(&alpha_d);
    void* beta  = (dtype == DType::Float32) ? static_cast<void*>(&beta_f)
                                            : static_cast<void*>(&beta_d);

    // Phase 1: work estimation — determines buffer1 size.
    size_t buffer1_size = 0;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, opA, opB, alpha, mat_a, mat_b, beta, mat_c,
        cuda_dtype, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc.desc,
        &buffer1_size, nullptr));
    CudaBuffer buffer1(buffer1_size);
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, opA, opB, alpha, mat_a, mat_b, beta, mat_c,
        cuda_dtype, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc.desc,
        &buffer1_size, buffer1.ptr));

    // Phase 2: compute — determines buffer2 size and computes the
    // structure of C (still on cusparse-owned memory).
    size_t buffer2_size = 0;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, opA, opB, alpha, mat_a, mat_b, beta, mat_c,
        cuda_dtype, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc.desc,
        &buffer2_size, nullptr));
    CudaBuffer buffer2(buffer2_size);
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, opA, opB, alpha, mat_a, mat_b, beta, mat_c,
        cuda_dtype, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc.desc,
        &buffer2_size, buffer2.ptr));

    // Query the resulting C dimensions and nnz.
    int64_t c_rows = 0, c_cols = 0, c_nnz = 0;
    CUSPARSE_CHECK(cusparseSpMatGetSize(mat_c, &c_rows, &c_cols, &c_nnz));

    // Allocate user-owned C arrays. c_crow is already M+1 wide.
    auto c_col  = zeros(std::vector<int64_t>{c_nnz}, DType::Int64, Device::cuda());
    auto c_vals = zeros(std::vector<int64_t>{c_nnz}, dtype,         Device::cuda());

    // Point the descriptor at our allocated arrays, then copy the
    // cusparse-internal CSR arrays into them.
    CUSPARSE_CHECK(cusparseCsrSetPointers(
        mat_c,
        c_crow.data<int64_t>(),
        c_col.data<int64_t>(),
        c_vals.data_ptr()));
    CUSPARSE_CHECK(cusparseSpGEMM_copy(
        handle, opA, opB, alpha, mat_a, mat_b, beta, mat_c,
        cuda_dtype, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc.desc));

    CUDA_CHECK_SPARSE(cudaStreamSynchronize(stream));

    return SparseTensor::sparse_csr(
        c_crow, c_col, c_vals,
        std::vector<int64_t>{M, N});
}

// ============================================================================
// SpSV — triangular solve L*x = b  (or U*x = b) for a single RHS
// ============================================================================
//
// Uses cuSPARSE's generic SpSV (available since CUDA 11.3). Each call does
// its own analysis pass — descriptor caching keyed on sparsity pattern is a
// natural follow-up if triangular solve becomes a hot path.

struct SpSVDescrGuard {
    cusparseSpSVDescr_t desc = nullptr;
    SpSVDescrGuard() { CUSPARSE_CHECK(cusparseSpSV_createDescr(&desc)); }
    ~SpSVDescrGuard() { if (desc) cusparseSpSV_destroyDescr(desc); }
    SpSVDescrGuard(const SpSVDescrGuard&) = delete;
    SpSVDescrGuard& operator=(const SpSVDescrGuard&) = delete;
};

Tensor cuda_sparse_trsv_kernel(const SparseTensor& L, const Tensor& b,
                                bool upper, void* stream_opaque) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_opaque);
    auto L_shape = L.shape();
    if (L_shape.size() != 2 || L_shape[0] != L_shape[1]) {
        throw std::runtime_error("cuda_sparse_trsv: L must be square 2D");
    }
    if (b.ndim() != 1) {
        throw std::runtime_error("cuda_sparse_trsv: b must be 1D");
    }
    const int64_t N = L_shape[0];
    if (b.shape()[0] != N) {
        throw std::runtime_error("cuda_sparse_trsv: dimension mismatch");
    }
    const DType dtype = b.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("cuda_sparse_trsv: only Float32/Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto L_csr = ensure_csr_on_gpu(L, stream);
    const int64_t nnz = L_csr.nnz();
    auto L_crow = L_csr.crow_indices().contiguous();
    auto L_col  = L_csr.col_indices().contiguous();
    auto L_vals = L_csr.values().contiguous();

    auto b_gpu = (b.device().type != Device::Type::CUDA)
                   ? b.to(Device::cuda()).contiguous()
                   : b.contiguous();
    auto result = zeros(std::vector<int64_t>{N}, dtype, Device::cuda());

    cusparseHandle_t handle = CuSPARSEHandlePool::get(stream);
    const cudaDataType cuda_dtype = get_cuda_data_type(dtype);

    // L matrix descriptor.
    cusparseSpMatDescr_t mat_L;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_L, N, N, nnz,
        const_cast<void*>(static_cast<const void*>(L_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(L_col.data<int64_t>())),
        const_cast<void*>(L_vals.data_ptr()),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
    SpMatGuard L_guard(mat_L);

    // Fill mode (lower / upper) and diagonal type (non-unit — the solver
    // reads the diagonal entry from the matrix values).
    cusparseFillMode_t fill_mode = upper ? CUSPARSE_FILL_MODE_UPPER
                                          : CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        mat_L, CUSPARSE_SPMAT_FILL_MODE, &fill_mode, sizeof(fill_mode)));
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        mat_L, CUSPARSE_SPMAT_DIAG_TYPE, &diag_type, sizeof(diag_type)));

    // Dense vector descriptors for x (input b) and y (output result).
    cusparseDnVecDescr_t vec_x;
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_x, N, const_cast<void*>(b_gpu.data_ptr()), cuda_dtype));
    DnVecGuard x_guard(vec_x);

    cusparseDnVecDescr_t vec_y;
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_y, N, result.data_ptr(), cuda_dtype));
    DnVecGuard y_guard(vec_y);

    SpSVDescrGuard spsv_desc;

    float  alpha_f = 1.0f;
    double alpha_d = 1.0;
    void* alpha = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f)
                                            : static_cast<void*>(&alpha_d);

    // Analysis: cuSPARSE inspects the sparsity pattern and computes any
    // internal data structures needed for the solve. The workspace
    // returned here is consumed by both bufferSize and analysis.
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_L,
        vec_x, vec_y, cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT,
        spsv_desc.desc, &buffer_size));
    CudaBuffer workspace(buffer_size);
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_L,
        vec_x, vec_y, cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT,
        spsv_desc.desc, workspace.ptr));

    // Solve.
    CUSPARSE_CHECK(cusparseSpSV_solve(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_L,
        vec_x, vec_y, cuda_dtype, CUSPARSE_SPSV_ALG_DEFAULT,
        spsv_desc.desc));

    CUDA_CHECK_SPARSE(cudaStreamSynchronize(stream));
    return result;
}

// Triangular solve with multiple right-hand sides: solve L * X = B where
// B is (N, K). We loop per column calling SpSV because the existing CPU
// code does the same — cuSPARSE has a matrix variant (SpSM) we can add
// later if this becomes a bottleneck.
Tensor cuda_sparse_trsm_kernel(const SparseTensor& L, const Tensor& B,
                                bool upper, void* stream_opaque) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_opaque);
    auto L_shape = L.shape();
    if (L_shape.size() != 2 || L_shape[0] != L_shape[1]) {
        throw std::runtime_error("cuda_sparse_trsm: L must be square 2D");
    }
    if (B.ndim() != 2) {
        throw std::runtime_error("cuda_sparse_trsm: B must be 2D");
    }
    const int64_t N = L_shape[0];
    const int64_t K = B.shape()[1];
    if (B.shape()[0] != N) {
        throw std::runtime_error("cuda_sparse_trsm: dimension mismatch");
    }

    auto B_gpu = (B.device().type != Device::Type::CUDA)
                   ? B.to(Device::cuda()).contiguous()
                   : B.contiguous();
    auto X = zeros(std::vector<int64_t>{N, K}, B.dtype(), Device::cuda());

    // We can't take column views of a row-major matrix cheaply (a column
    // is strided by K), so extract each column to a contiguous buffer,
    // call SpSV, and write back. Each SpSV call re-runs analysis — this
    // is O(K) suboptimal and is flagged as a follow-up: caching the
    // analysis across calls for the same L would eliminate the overhead.
    auto B_t = B_gpu;  // alias
    for (int64_t k = 0; k < K; ++k) {
        // Extract B[:, k] into a dense 1-D tensor on GPU.
        Tensor b_col = zeros(std::vector<int64_t>{N}, B.dtype(), Device::cuda());
        if (B.dtype() == DType::Float32) {
            auto* dst = b_col.data<float>();
            const auto* src = B_t.data<float>();
            for (int64_t i = 0; i < N; ++i) {
                CUDA_CHECK_SPARSE(cudaMemcpyAsync(
                    dst + i, src + i * K + k, sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
            }
        } else if (B.dtype() == DType::Float64) {
            auto* dst = b_col.data<double>();
            const auto* src = B_t.data<double>();
            for (int64_t i = 0; i < N; ++i) {
                CUDA_CHECK_SPARSE(cudaMemcpyAsync(
                    dst + i, src + i * K + k, sizeof(double),
                    cudaMemcpyDeviceToDevice, stream));
            }
        } else {
            throw std::runtime_error("cuda_sparse_trsm: only Float32/Float64 supported");
        }

        auto x_col = cuda_sparse_trsv_kernel(L, b_col, upper, stream_opaque);

        // Scatter x_col back to X[:, k].
        if (B.dtype() == DType::Float32) {
            auto* dst = X.data<float>();
            const auto* src = x_col.data<float>();
            for (int64_t i = 0; i < N; ++i) {
                CUDA_CHECK_SPARSE(cudaMemcpyAsync(
                    dst + i * K + k, src + i, sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
            }
        } else {
            auto* dst = X.data<double>();
            const auto* src = x_col.data<double>();
            for (int64_t i = 0; i < N; ++i) {
                CUDA_CHECK_SPARSE(cudaMemcpyAsync(
                    dst + i * K + k, src + i, sizeof(double),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    CUDA_CHECK_SPARSE(cudaStreamSynchronize(stream));
    return X;
}

// ============================================================================
// GPU-native sparse format conversions (called from SparseTensor methods)
// ============================================================================

SparseTensor cuda_coo_to_csr(const SparseTensor& sparse) {
    return ensure_csr_on_gpu(sparse);
}

SparseTensor cuda_coalesce(const SparseTensor& sparse) {
    // Coalesce COO on GPU using thrust sort + scan.
    // The sparse tensor must already be on CUDA.
    if (sparse.layout() != SparseLayout::COO) return sparse;
    if (sparse.nnz() == 0) {
        SparseTensor result = sparse;
        // Can't set coalesced_ directly, but returning a zero-nnz tensor
        // that was already coalesced is fine.
        return result;
    }

    auto sp_shape = sparse.shape();
    int64_t sparse_dim = static_cast<int64_t>(sp_shape.size());
    int64_t nnz = sparse.nnz();

    Tensor indices = sparse.indices().contiguous();
    Tensor values = sparse.values().contiguous();

    // Compute linearized keys on GPU: key[i] = row[i] * ncols + col[i] (2D)
    // For general sparse_dim, key = sum(idx[d,i] * stride[d])
    // We do this with a simple kernel.
    // For 2D (the common case), key = idx[0,i] * shape[1] + idx[1,i]
    const int64_t* idx_ptr = indices.data<int64_t>();

    // Compute strides
    std::vector<int64_t> h_strides(sparse_dim);
    if (sparse_dim > 0) {
        h_strides[sparse_dim - 1] = 1;
        for (int64_t d = sparse_dim - 2; d >= 0; --d) {
            h_strides[d] = h_strides[d + 1] * sp_shape[d + 1];
        }
    }

    // Allocate keys on GPU
    Tensor keys_t = zeros({nnz}, DType::Int64, Device::cuda());
    int64_t* keys_ptr = keys_t.data<int64_t>();

    // Build keys: for each dimension, add idx[d,i] * stride[d] to keys[i]
    // Use cudaMemcpy to move strides to device, then a simple kernel
    for (int64_t d = 0; d < sparse_dim; ++d) {
        if (h_strides[d] == 0) continue;
        // keys[i] += idx_ptr[d * nnz + i] * stride
        // Simple approach: launch a kernel per dimension
        // We reuse cast kernels pattern — define an inline lambda via thrust
        const int64_t* dim_idx = idx_ptr + d * nnz;
        int64_t stride = h_strides[d];
        auto keys_dptr = thrust::device_pointer_cast(keys_ptr);
        auto dim_dptr = thrust::device_pointer_cast(dim_idx);
        thrust::transform(thrust::cuda::par,
                          keys_dptr, keys_dptr + nnz,
                          dim_dptr,
                          keys_dptr,
                          [stride] __device__ (int64_t k, int64_t idx_val) {
                              return k + idx_val * stride;
                          });
    }

    // Sort by key: create a permutation array and sort keys + perm together
    Tensor perm_t = zeros({nnz}, DType::Int64, Device::cuda());
    int64_t* perm_ptr = perm_t.data<int64_t>();
    thrust::sequence(thrust::cuda::par, thrust::device_pointer_cast(perm_ptr),
                     thrust::device_pointer_cast(perm_ptr + nnz));
    thrust::sort_by_key(thrust::cuda::par,
                        thrust::device_pointer_cast(keys_ptr),
                        thrust::device_pointer_cast(keys_ptr + nnz),
                        thrust::device_pointer_cast(perm_ptr));

    // Gather sorted indices and values using the permutation
    Tensor sorted_indices = zeros({sparse_dim, nnz}, DType::Int64, Device::cuda());
    int64_t* si_ptr = sorted_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim; ++d) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(idx_ptr + d * nnz),
                       thrust::device_pointer_cast(si_ptr + d * nnz));
    }

    // Gather values
    Tensor sorted_vals = zeros({nnz}, values.dtype(), Device::cuda());
    if (values.dtype() == DType::Float32) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<float>()),
                       thrust::device_pointer_cast(sorted_vals.data<float>()));
    } else if (values.dtype() == DType::Float64) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<double>()),
                       thrust::device_pointer_cast(sorted_vals.data<double>()));
    }

    // Detect unique keys and sum duplicates using thrust::reduce_by_key
    Tensor out_keys = zeros({nnz}, DType::Int64, Device::cuda());
    int64_t* ok_ptr = out_keys.data<int64_t>();

    // For values: reduce_by_key with sum
    Tensor out_vals = zeros({nnz}, values.dtype(), Device::cuda());
    int64_t new_nnz = 0;

    if (values.dtype() == DType::Float32) {
        auto end = thrust::reduce_by_key(
            thrust::cuda::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<float>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<float>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    } else if (values.dtype() == DType::Float64) {
        auto end = thrust::reduce_by_key(
            thrust::cuda::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<double>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<double>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    }

    // Reconstruct unique indices from unique keys
    // key = sum(idx[d] * stride[d]) — decode back to multi-dim indices
    Tensor new_indices = zeros({sparse_dim, new_nnz}, DType::Int64, Device::cuda());
    int64_t* ni_ptr = new_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim; ++d) {
        int64_t stride = h_strides[d];
        auto ok_dptr = thrust::device_pointer_cast(ok_ptr);
        auto ni_dptr = thrust::device_pointer_cast(ni_ptr + d * new_nnz);
        if (d < sparse_dim - 1) {
            int64_t next_stride = h_strides[d + 1];
            thrust::transform(thrust::cuda::par,
                              ok_dptr, ok_dptr + new_nnz,
                              ni_dptr,
                              [stride, next_stride] __device__ (int64_t key) {
                                  return (key / stride) % (stride / next_stride);
                              });
        } else {
            thrust::transform(thrust::cuda::par,
                              ok_dptr, ok_dptr + new_nnz,
                              ni_dptr,
                              [stride] __device__ (int64_t key) {
                                  return key % stride;  // stride == 1 for last dim
                              });
        }
    }

    // Trim output values to new_nnz
    // We need to slice — simplest is to copy
    Tensor final_vals = zeros({new_nnz}, values.dtype(), Device::cuda());
    if (values.dtype() == DType::Float32) {
        CUDA_CHECK_SPARSE(cudaMemcpy(final_vals.data<float>(), out_vals.data<float>(),
                                     new_nnz * sizeof(float), cudaMemcpyDeviceToDevice));
    } else if (values.dtype() == DType::Float64) {
        CUDA_CHECK_SPARSE(cudaMemcpy(final_vals.data<double>(), out_vals.data<double>(),
                                     new_nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    return SparseTensor::sparse_coo(new_indices, final_vals,
                                    std::vector<int64_t>(sp_shape.begin(), sp_shape.end()));
}

SparseTensor cuda_coo_to_csc(const SparseTensor& sparse) {
    if (sparse.layout() == SparseLayout::CSC) return sparse;

    auto sp_shape = sparse.shape();
    int64_t nrows = sp_shape[0];
    int64_t ncols = sp_shape[1];

    // First coalesce on GPU, then convert COO -> CSC
    auto coo = cuda_coalesce(sparse);
    int64_t nnz = coo.nnz();

    Tensor indices = coo.indices().contiguous();
    Tensor values = coo.values().contiguous();
    const int64_t* idx_ptr = indices.data<int64_t>();
    const int64_t* row_ptr_src = idx_ptr;
    const int64_t* col_ptr_src = idx_ptr + nnz;

    // Sort by (col, row) — create compound keys: col * nrows + row
    Tensor sort_keys = zeros({nnz}, DType::Int64, Device::cuda());
    int64_t* sk_ptr = sort_keys.data<int64_t>();
    {
        auto sk_dptr = thrust::device_pointer_cast(sk_ptr);
        auto col_dptr = thrust::device_pointer_cast(col_ptr_src);
        auto row_dptr = thrust::device_pointer_cast(row_ptr_src);
        thrust::transform(thrust::cuda::par,
                          col_dptr, col_dptr + nnz,
                          row_dptr,
                          sk_dptr,
                          [nrows] __device__ (int64_t c, int64_t r) {
                              return c * nrows + r;
                          });
    }

    // Create permutation and sort
    Tensor perm_t = zeros({nnz}, DType::Int64, Device::cuda());
    int64_t* perm_ptr = perm_t.data<int64_t>();
    thrust::sequence(thrust::cuda::par, thrust::device_pointer_cast(perm_ptr),
                     thrust::device_pointer_cast(perm_ptr + nnz));
    thrust::sort_by_key(thrust::cuda::par,
                        thrust::device_pointer_cast(sk_ptr),
                        thrust::device_pointer_cast(sk_ptr + nnz),
                        thrust::device_pointer_cast(perm_ptr));

    // Gather sorted row indices
    Tensor sorted_rows = zeros({nnz}, DType::Int64, Device::cuda());
    thrust::gather(thrust::cuda::par,
                   thrust::device_pointer_cast(perm_ptr),
                   thrust::device_pointer_cast(perm_ptr + nnz),
                   thrust::device_pointer_cast(row_ptr_src),
                   thrust::device_pointer_cast(sorted_rows.data<int64_t>()));

    // Gather sorted col indices (for building ccol_indices)
    Tensor sorted_cols = zeros({nnz}, DType::Int64, Device::cuda());
    thrust::gather(thrust::cuda::par,
                   thrust::device_pointer_cast(perm_ptr),
                   thrust::device_pointer_cast(perm_ptr + nnz),
                   thrust::device_pointer_cast(col_ptr_src),
                   thrust::device_pointer_cast(sorted_cols.data<int64_t>()));

    // Build ccol_indices using histogram + prefix sum
    // Count elements per column using thrust
    Tensor ccol = zeros({ncols + 1}, DType::Int64, Device::cuda());
    int64_t* ccol_ptr = ccol.data<int64_t>();
    // Use sorted_cols to compute histogram: ccol[col+1]++
    // Simplest: use thrust::upper_bound to find boundaries
    Tensor boundaries = zeros({ncols}, DType::Int64, Device::cuda());
    int64_t* bounds_ptr = boundaries.data<int64_t>();
    thrust::upper_bound(thrust::cuda::par,
                        thrust::device_pointer_cast(sorted_cols.data<int64_t>()),
                        thrust::device_pointer_cast(sorted_cols.data<int64_t>() + nnz),
                        thrust::counting_iterator<int64_t>(0),
                        thrust::counting_iterator<int64_t>(ncols),
                        thrust::device_pointer_cast(bounds_ptr));
    // ccol[0] = 0, ccol[i+1] = upper_bound result for column i
    CUDA_CHECK_SPARSE(cudaMemset(ccol_ptr, 0, sizeof(int64_t)));
    CUDA_CHECK_SPARSE(cudaMemcpy(ccol_ptr + 1, bounds_ptr,
                                 ncols * sizeof(int64_t), cudaMemcpyDeviceToDevice));

    // Gather sorted values
    Tensor sorted_vals = zeros({nnz}, values.dtype(), Device::cuda());
    if (values.dtype() == DType::Float32) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<float>()),
                       thrust::device_pointer_cast(sorted_vals.data<float>()));
    } else if (values.dtype() == DType::Float64) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<double>()),
                       thrust::device_pointer_cast(sorted_vals.data<double>()));
    }

    return SparseTensor::sparse_csc(ccol, sorted_rows, sorted_vals,
                                    std::vector<int64_t>{nrows, ncols});
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
