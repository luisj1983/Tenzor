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
#include "cuda_common.cuh"  // For TENZOR_CUDA_CHECK (K.3 hygiene)
#include "cuda_launch_utils.cuh"  // tenzor::cuda::CudaDeviceGuard

#include <cusparse.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuComplex.h>
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

// Defined later in this TU; declared here so ensure_csr_on_gpu (in the
// anonymous namespace below) can coalesce duplicate COO coordinates before
// coo2csr, matching the CPU reference's duplicate-summing semantics.
SparseTensor cuda_coalesce(const SparseTensor& sparse);

// Defined later in this TU (native CUDA CSR x CSR SpGEMM, no cuSPARSE
// dependency — always compiled). Declared here so cuda_spgemm_kernel below
// can fall back to it for Int32/Int64, which cuSPARSE's generic SpGEMM API
// does not support (F-096).
template <typename T>
auto spgemm_standalone_typed(
    const Tensor& a_crow, const Tensor& a_col, const Tensor& a_vals,
    const Tensor& b_crow, const Tensor& b_col, const Tensor& b_vals,
    int64_t M, int64_t K, int64_t N, cudaStream_t stream) -> std::vector<Tensor>;

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

// CudaBuffer (RAII cudaMalloc/cudaFree) is provided by cuda_launch_utils.cuh.

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
        case DType::Complex64: return CUDA_C_32F;
        case DType::Complex128: return CUDA_C_64F;
        default:
            throw std::runtime_error("cuda_sparse: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
    }
}

/// CUDA kernel: convert Int64 row indices to Int32.
__global__ void cast_i64_to_i32(const int64_t* __restrict__ src,
                                 int32_t* __restrict__ dst, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int32_t>(src[i]);
}

// ---------------------------------------------------------------------------
// F16/BF16 widen kernels — used by the spmm/spmv widen-narrow path to keep
// all temporary buffers stream-ordered (cudaMallocAsync) rather than allocated
// via Tensor::to() (which uses the synchronous default-stream allocator and
// races with concurrent streams operating on the same low-precision sparse
// matrix). See audit L.1.
// ---------------------------------------------------------------------------
__global__ void widen_f16_to_f32(const __half* __restrict__ src,
                                  float* __restrict__ dst, int64_t n) {
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        dst[i] = __half2float(src[i]);
}

__global__ void widen_bf16_to_f32(const __nv_bfloat16* __restrict__ src,
                                   float* __restrict__ dst, int64_t n) {
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        dst[i] = __bfloat162float(src[i]);
}

__global__ void narrow_f32_to_f16(const float* __restrict__ src,
                                   __half* __restrict__ dst, int64_t n) {
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        dst[i] = __float2half(src[i]);
}

__global__ void narrow_f32_to_bf16(const float* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst, int64_t n) {
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        dst[i] = __float2bfloat16(src[i]);
}

/// Widen a F16/BF16 GPU tensor to a freshly-allocated F32 tensor, with all
/// memory and compute strictly bound to `stream` via cudaMallocAsync /
/// cudaFreeAsync.  The returned Tensor owns the buffer via a custom deleter
/// that releases it back to the stream-ordered pool.  Per-call allocation;
/// no shared state — concurrent streams on the same source tensor never
/// share temp storage.  (Audit L.1.)
Tensor widen_to_f32_async(const Tensor& src, cudaStream_t stream) {
    const int64_t n = src.numel();
    float* d_buf = nullptr;
    if (n > 0) {
        TENZOR_CUDA_CHECK(cudaMallocAsync(&d_buf, n * sizeof(float), stream));
    }
    const int threads = 256;
    const int64_t blocks64 = (n + threads - 1) / threads;
    const unsigned blocks = static_cast<unsigned>(
        blocks64 > 2147483647LL ? 2147483647LL : blocks64);

    if (src.dtype() == DType::Float16) {
        if (n > 0) {
            widen_f16_to_f32<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __half*>(src.data_ptr()), d_buf, n);
            TENZOR_CUDA_CHECK(cudaGetLastError());
        }
    } else if (src.dtype() == DType::BFloat16) {
        if (n > 0) {
            widen_bf16_to_f32<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()), d_buf, n);
            TENZOR_CUDA_CHECK(cudaGetLastError());
        }
    } else {
        if (d_buf) cudaFreeAsync(d_buf, stream);
        throw std::runtime_error("widen_to_f32_async: unsupported src dtype");
    }

    // Capture stream by value in the deleter so the right pool reclaims it.
    auto deleter = [stream](void* p) {
        if (p) cudaFreeAsync(p, stream);
    };
    auto shape = src.shape();
    return Tensor::from_blob(d_buf,
                             std::vector<int64_t>(shape.begin(), shape.end()),
                             DType::Float32, Device::cuda(),
                             deleter);
}

/// Narrow an F32 GPU tensor back to F16 or BF16.  Allocates the destination
/// buffer with cudaMallocAsync(... stream) so it's safe to interleave with
/// other operations on the same stream; the returned Tensor's storage is
/// released via cudaFreeAsync.  (Audit L.1.)
Tensor narrow_from_f32_async(const Tensor& src_f32, DType target,
                              cudaStream_t stream) {
    const int64_t n = src_f32.numel();
    void* d_buf = nullptr;
    size_t elem_size = (target == DType::Float16) ? sizeof(__half) : sizeof(__nv_bfloat16);
    if (n > 0) {
        TENZOR_CUDA_CHECK(cudaMallocAsync(&d_buf, n * elem_size, stream));
    }
    const int threads = 256;
    const int64_t blocks64 = (n + threads - 1) / threads;
    const unsigned blocks = static_cast<unsigned>(
        blocks64 > 2147483647LL ? 2147483647LL : blocks64);

    if (target == DType::Float16) {
        if (n > 0) {
            narrow_f32_to_f16<<<blocks, threads, 0, stream>>>(
                src_f32.data<float>(), reinterpret_cast<__half*>(d_buf), n);
            TENZOR_CUDA_CHECK(cudaGetLastError());
        }
    } else if (target == DType::BFloat16) {
        if (n > 0) {
            narrow_f32_to_bf16<<<blocks, threads, 0, stream>>>(
                src_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(d_buf), n);
            TENZOR_CUDA_CHECK(cudaGetLastError());
        }
    } else {
        if (d_buf) cudaFreeAsync(d_buf, stream);
        throw std::runtime_error("narrow_from_f32_async: unsupported target dtype");
    }

    auto deleter = [stream](void* p) {
        if (p) cudaFreeAsync(p, stream);
    };
    auto shape = src_f32.shape();
    return Tensor::from_blob(d_buf,
                             std::vector<int64_t>(shape.begin(), shape.end()),
                             target, Device::cuda(),
                             deleter);
}

/// CUDA kernel: convert Int32 crow_indices to Int64.
__global__ void cast_i32_to_i64(const int32_t* __restrict__ src,
                                 int64_t* __restrict__ dst, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
        // Coalesce duplicate (row,col) entries first. cusparseXcoo2csr does not
        // merge duplicates, so an un-coalesced COO would yield a CSR with
        // repeated columns in a row and rocSPARSE/cuSPARSE would double-count or
        // produce undefined ordering — diverging from the CPU reference which
        // sums duplicates.
        if (!sp.is_coalesced()) {
            sp = cuda_coalesce(sp);
        }
        auto sp_shape = sp.shape();
        int64_t nrows = sp_shape[0];
        int64_t ncols = sp_shape[1];
        int64_t nnz = sp.nnz();

        // cusparseXcoo2csr's API takes int (32-bit) for nnz and the row count,
        // and the row-index buffer it consumes is int32. nnz, nrows, and every
        // row index (each < nrows) must therefore fit in int32; otherwise the
        // static_cast<int>(...) / cast_i64_to_i32 below would silently truncate
        // and produce a corrupted CSR row-pointer array with no diagnostic.
        // Reject oversized inputs up front rather than corrupt results.
        constexpr int64_t kInt32Max = static_cast<int64_t>(INT32_MAX);
        if (nnz > kInt32Max) {
            throw std::runtime_error(
                "cuda_sparse: nnz=" + std::to_string(nnz) +
                " exceeds int32 limit (" + std::to_string(kInt32Max) +
                ") required by cusparseXcoo2csr");
        }
        if (nrows > kInt32Max) {
            throw std::runtime_error(
                "cuda_sparse: row count=" + std::to_string(nrows) +
                " exceeds int32 limit (" + std::to_string(kInt32Max) +
                ") required by cusparseXcoo2csr");
        }

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
        unsigned blocks_nnz = static_cast<unsigned>((nnz + threads - 1) / threads);
        cast_i64_to_i32<<<blocks_nnz, threads, 0, stream>>>(row_indices_ptr, row_i32_buf.as<int32_t>(), nnz);
        CUDA_CHECK_SPARSE(cudaGetLastError());

        // Convert COO row indices to CSR row pointers on GPU
        cusparseHandle_t handle = CuSPARSEHandlePool::get(stream);
        CUSPARSE_CHECK(cusparseXcoo2csr(
            handle, row_i32_buf.as<int32_t>(), static_cast<int>(nnz), static_cast<int>(nrows),
            crow_i32_buf.as<int32_t>(), CUSPARSE_INDEX_BASE_ZERO));

        // Convert Int32 crow_indices back to Int64 on GPU
        Tensor crow_indices = zeros(std::vector<int64_t>{nrows + 1}, DType::Int64, Device::cuda());
        unsigned blocks_crow = static_cast<unsigned>((nrows + 1 + threads - 1) / threads);
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

    // Everything past the COO branch must already be CSR: the callers
    // (spmm/spmv/spgemm/trsv) read .crow_indices()/.col_indices() and hand them
    // to cusparseCreateCsr as an M+1-length row-pointer array. A CSC tensor's
    // compressed array is ccol (ncols+1 column pointers), so passing it through
    // as if it were CSR feeds cuSPARSE a malformed/wrong-length descriptor
    // (out-of-bounds reads, garbage output). Reject any non-CSR layout loudly
    // rather than silently corrupting results.
    if (sp.layout() != SparseLayout::CSR) {
        throw std::runtime_error(
            "cuda_sparse: ensure_csr_on_gpu received an unsupported sparse "
            "layout (only COO and CSR are accepted; convert CSC/BSR to CSR or "
            "COO before the CUDA sparse op)");
    }
    return sp;
}

} // anonymous namespace

// Generic (non-cuSPARSE) CSR SpMM/SpMV for the dtypes cuSPARSE cannot handle:
// integer (no cuSPARSE integer path at all) and complex (routed here for a
// single, always-compiled code path). Defined in the standalone section below
// (F034 — matches the CPU generic CSR templates).
Tensor cuda_spmm_generic(const SparseTensor& sparse, const Tensor& dense);
Tensor cuda_spmv_generic(const SparseTensor& sparse, const Tensor& vec);

Tensor cuda_spmm_kernel(const SparseTensor& sparse, const Tensor& dense, cudaStream_t stream = 0) {
    // Make the operands' CUDA device current so the device-keyed cuSPARSE handle
    // and any widen/narrow kernel launches target the correct GPU. Restored on
    // scope exit. (The dispatch path does not set the device.)
    CudaDeviceGuard dev_guard(dense.device().type == Device::Type::CUDA
                              ? dense.device().index : Device::cuda().index);
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
    // Audit L.1: F16/BF16 widen-narrow uses stream-ordered allocator (cudaMallocAsync)
    // so two concurrent streams operating on the same low-precision sparse matrix do
    // not race on temp allocation/deallocation.  Per-call (no caching), bound to
    // `stream` end-to-end.  No CPU fallback — widen kernels run GPU-side.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto sparse_f32_vals = widen_to_f32_async(sparse.values().contiguous(), stream);
        SparseTensor sparse_f32 = (sparse.layout() == SparseLayout::COO)
            ? SparseTensor::sparse_coo(sparse.indices(), sparse_f32_vals, sparse.shape())
            : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                       sparse_f32_vals, sparse.shape());
        auto dense_f32 = widen_to_f32_async(dense.contiguous(), stream);
        auto result_f32 = cuda_spmm_kernel(sparse_f32, dense_f32, stream);
        return narrow_from_f32_async(result_f32, dtype, stream);
    }
    if (dtype == DType::Int32 || dtype == DType::Int64 ||
        dtype == DType::Complex64 || dtype == DType::Complex128) {
        // cuSPARSE has no integer path and we route complex through the same
        // generic CSR kernel — matches the CPU spmm's generic dtype coverage.
        return cuda_spmm_generic(sparse, dense);
    }
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("cuda_spmm: only Float32/Float64/Float16/BFloat16/"
            "Int32/Int64/Complex64/Complex128 supported, got "
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
    CudaDeviceGuard dev_guard(vec.device().type == Device::Type::CUDA
                              ? vec.device().index : Device::cuda().index);
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
    // Audit L.1: F16/BF16 widen-narrow uses stream-ordered allocator (cudaMallocAsync)
    // so two concurrent streams operating on the same low-precision sparse matrix do
    // not race on temp allocation/deallocation.  Per-call (no caching), bound to
    // `stream` end-to-end.  No CPU fallback — widen kernels run GPU-side.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto sparse_f32_vals = widen_to_f32_async(sparse.values().contiguous(), stream);
        SparseTensor sparse_f32 = (sparse.layout() == SparseLayout::COO)
            ? SparseTensor::sparse_coo(sparse.indices(), sparse_f32_vals, sparse.shape())
            : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                       sparse_f32_vals, sparse.shape());
        auto vec_f32 = widen_to_f32_async(vec.contiguous(), stream);
        auto result_f32 = cuda_spmv_kernel(sparse_f32, vec_f32, stream);
        return narrow_from_f32_async(result_f32, dtype, stream);
    }
    if (dtype == DType::Int32 || dtype == DType::Int64 ||
        dtype == DType::Complex64 || dtype == DType::Complex128) {
        return cuda_spmv_generic(sparse, vec);
    }
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("cuda_spmv: only Float32/Float64/Float16/BFloat16/"
            "Int32/Int64/Complex64/Complex128 supported, got "
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
    CudaDeviceGuard dev_guard(a.device().type == Device::Type::CUDA
                              ? a.device().index : Device::cuda().index);
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
    // Float16/BFloat16 are widened to Float32 by sparse_ops.cpp's spgemm()
    // before this kernel is ever called (see the GPU dispatch path there), so
    // they never reach here as Float16/BFloat16. Complex64/128 are supported
    // natively by cuSPARSE's generic SpGEMM API (CUDA_C_32F/CUDA_C_64F).
    // Int32/Int64 are NOT supported by cuSPARSE's SpGEMM (its alpha/beta
    // scaling is floating-point/complex only), so those bypass cuSPARSE
    // entirely below and use the native CUDA CSR SpGEMM kernel instead —
    // matching CPU's Float32/Float64/Float16/BFloat16/Int32/Int64/Complex64/
    // Complex128 coverage (F-096).
    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Complex64 && dtype != DType::Complex128 &&
        dtype != DType::Int32 && dtype != DType::Int64) {
        throw std::runtime_error("cuda_spgemm: unsupported dtype " +
            std::string(dtype_name(dtype)) +
            " (supported: Float32, Float64, Complex64, Complex128, Int32, Int64)");
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

    // Int32/Int64: cuSPARSE's generic SpGEMM has no integer-accumulation
    // support, so use the always-compiled native CSR x CSR kernel directly
    // (same 3-pass count/prefix-sum/fill algorithm as the no-cuSPARSE
    // fallback build; it's dtype-generic and works unmodified for integer
    // types since it only relies on native +=/* on T).
    if (dtype == DType::Int32) {
        auto res = spgemm_standalone_typed<int32_t>(
            a_crow, a_col, a_vals, b_crow, b_col, b_vals, M, K, N, stream);
        return SparseTensor::sparse_csr(res[0], res[1], res[2], std::vector<int64_t>{M, N});
    }
    if (dtype == DType::Int64) {
        auto res = spgemm_standalone_typed<int64_t>(
            a_crow, a_col, a_vals, b_crow, b_col, b_vals, M, K, N, stream);
        return SparseTensor::sparse_csr(res[0], res[1], res[2], std::vector<int64_t>{M, N});
    }

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
    cuFloatComplex  alpha_c = make_cuFloatComplex(1.0f, 0.0f), beta_c = make_cuFloatComplex(0.0f, 0.0f);
    cuDoubleComplex alpha_z = make_cuDoubleComplex(1.0, 0.0),  beta_z = make_cuDoubleComplex(0.0, 0.0);
    void* alpha = nullptr;
    void* beta = nullptr;
    switch (dtype) {
        case DType::Float32:   alpha = &alpha_f; beta = &beta_f; break;
        case DType::Float64:   alpha = &alpha_d; beta = &beta_d; break;
        case DType::Complex64: alpha = &alpha_c; beta = &beta_c; break;
        default:                alpha = &alpha_z; beta = &beta_z; break; // Complex128
    }

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

// Proactive zero/missing-diagonal scan run before handing a triangular matrix
// to cusparseSpSV_analysis/solve. Whether cuSPARSE's SpSV detects a
// zero/missing diagonal depends on the algorithm/version — it may instead
// silently produce Inf/NaN with no diagnostic (F-094). CPU
// (sparse_ops.cpp's cpu_forward_substitution/cpu_backward_substitution)
// explicitly scans and throws before any division; mirror that here.
//
// Each row's diagonal check is independent (unlike the actual solve, which is
// sequential), so this runs one thread per row instead of the single-thread
// device-error-flag pattern used by sparse_trsv_seq_kernel (F-093). To match
// the row CPU would report first, the violating row nearest the solve's
// starting point wins: forward substitution (lower, unprocessed rows
// 0..N-1) reports the MIN violating row; backward substitution (upper, rows
// N-1..0) reports the MAX violating row.
// Zero test used by sparse_check_diag_kernel below. Real types compare
// directly; cuFloatComplex/cuDoubleComplex are plain float2/double2 structs
// with no operator== or single-scalar constructor, so they need a manual
// component-wise test.
template <typename T>
__device__ __forceinline__ bool sparse_diag_is_zero(const T& v) { return v == T(0); }

__device__ __forceinline__ bool sparse_diag_is_zero(const cuFloatComplex& v) {
    return v.x == 0.0f && v.y == 0.0f;
}
__device__ __forceinline__ bool sparse_diag_is_zero(const cuDoubleComplex& v) {
    return v.x == 0.0 && v.y == 0.0;
}

template <typename T>
__global__ void sparse_check_diag_kernel(
    const int64_t* __restrict__ crow,
    const int64_t* __restrict__ col,
    const T* __restrict__ vals,
    int64_t N, bool upper,
    int* __restrict__ error_flag,
    unsigned long long* __restrict__ error_row)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // Zero-initializes to the additive identity for both scalar T (0.0) and
    // aggregate complex T (component-wise {0,0}), so a row with no explicit
    // diagonal entry ends up indistinguishable from an explicit zero one —
    // matching the CPU reference and sparse_trsv_seq_kernel (F-093).
    T diag{};
    for (int64_t j = crow[row]; j < crow[row + 1]; ++j) {
        if (col[j] == row) { diag = vals[j]; break; }
    }
    if (sparse_diag_is_zero(diag)) {
        atomicExch(error_flag, 1);
        if (upper) {
            atomicMax(error_row, static_cast<unsigned long long>(row));
        } else {
            atomicMin(error_row, static_cast<unsigned long long>(row));
        }
    }
}

// Launches sparse_check_diag_kernel and throws the same "zero diagonal
// element at row N" error CPU does if any row's diagonal is missing/zero.
// crow/col/vals must already be contiguous, device-resident CSR arrays.
template <typename T>
void cuda_check_sparse_diag(const int64_t* crow, const int64_t* col,
                             const T* vals, int64_t N, bool upper,
                             cudaStream_t stream) {
    if (N == 0) return;
    CudaBuffer d_error_flag(sizeof(int));
    CudaBuffer d_error_row(sizeof(unsigned long long));
    CUDA_CHECK_SPARSE(cudaMemsetAsync(d_error_flag.as<int>(), 0, sizeof(int), stream));
    unsigned long long init_row = upper ? 0ULL : static_cast<unsigned long long>(N);
    CUDA_CHECK_SPARSE(cudaMemcpyAsync(d_error_row.as<unsigned long long>(), &init_row,
        sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = static_cast<int>((N + threads - 1) / threads);
    sparse_check_diag_kernel<T><<<blocks, threads, 0, stream>>>(
        crow, col, vals, N, upper, d_error_flag.as<int>(), d_error_row.as<unsigned long long>());
    CUDA_CHECK_SPARSE(cudaGetLastError());

    int host_error = 0;
    unsigned long long host_error_row = 0;
    CUDA_CHECK_SPARSE(cudaMemcpyAsync(&host_error, d_error_flag.as<int>(), sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE(cudaMemcpyAsync(&host_error_row, d_error_row.as<unsigned long long>(),
        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE(cudaStreamSynchronize(stream));
    if (host_error) {
        throw std::runtime_error(
            "sparse_triangular_solve: zero diagonal element at row " +
            std::to_string(host_error_row));
    }
}

// Dispatches cuda_check_sparse_diag<T> for the dtypes cuda_sparse_trsv_kernel
// / cuda_sparse_trsm_kernel support (Float32/Float64/Complex64/Complex128).
void cuda_check_sparse_diag_dispatch(const Tensor& crow, const Tensor& col,
                                      const Tensor& vals, int64_t N, bool upper,
                                      cudaStream_t stream) {
    switch (vals.dtype()) {
        case DType::Float32:
            cuda_check_sparse_diag<float>(crow.data<int64_t>(), col.data<int64_t>(),
                vals.data<float>(), N, upper, stream);
            break;
        case DType::Float64:
            cuda_check_sparse_diag<double>(crow.data<int64_t>(), col.data<int64_t>(),
                vals.data<double>(), N, upper, stream);
            break;
        case DType::Complex64:
            cuda_check_sparse_diag<cuFloatComplex>(crow.data<int64_t>(), col.data<int64_t>(),
                reinterpret_cast<const cuFloatComplex*>(vals.data_ptr()), N, upper, stream);
            break;
        case DType::Complex128:
            cuda_check_sparse_diag<cuDoubleComplex>(crow.data<int64_t>(), col.data<int64_t>(),
                reinterpret_cast<const cuDoubleComplex*>(vals.data_ptr()), N, upper, stream);
            break;
        default:
            throw std::runtime_error("cuda_check_sparse_diag: unsupported dtype " +
                std::string(dtype_name(vals.dtype())));
    }
}

Tensor cuda_sparse_trsv_kernel(const SparseTensor& L, const Tensor& b,
                                bool upper, void* stream_opaque) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_opaque);
    CudaDeviceGuard dev_guard(b.device().type == Device::Type::CUDA
                              ? b.device().index : Device::cuda().index);
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
    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Complex64 && dtype != DType::Complex128) {
        throw std::runtime_error(
            "cuda_sparse_trsv: only Float32/Float64/Complex64/Complex128 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto L_csr = ensure_csr_on_gpu(L, stream);
    const int64_t nnz = L_csr.nnz();
    auto L_crow = L_csr.crow_indices().contiguous();
    auto L_col  = L_csr.col_indices().contiguous();
    auto L_vals = L_csr.values().contiguous();

    // Proactive zero/missing-diagonal check (F-094): cuSPARSE's SpSV
    // analysis/solve below doesn't reliably detect this across
    // algorithms/versions, unlike CPU's explicit up-front scan. Throws the
    // same "zero diagonal element at row N" error CPU does instead of letting
    // cusparseSpSV silently produce Inf/NaN.
    cuda_check_sparse_diag_dispatch(L_crow, L_col, L_vals, N, upper, stream);

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
    cuFloatComplex  alpha_c = make_cuFloatComplex(1.0f, 0.0f);
    cuDoubleComplex alpha_z = make_cuDoubleComplex(1.0, 0.0);
    void* alpha = nullptr;
    switch (dtype) {
        case DType::Float32:    alpha = static_cast<void*>(&alpha_f); break;
        case DType::Float64:    alpha = static_cast<void*>(&alpha_d); break;
        case DType::Complex64:  alpha = static_cast<void*>(&alpha_c); break;
        default:                alpha = static_cast<void*>(&alpha_z); break; // Complex128
    }

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
    CudaDeviceGuard dev_guard(B.device().type == Device::Type::CUDA
                              ? B.device().index : Device::cuda().index);
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

    const DType dtype = B.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Complex64 && dtype != DType::Complex128) {
        throw std::runtime_error(
            "cuda_sparse_trsm: only Float32/Float64/Complex64/Complex128 supported, got "
            + std::string(dtype_name(dtype)));
    }
    const size_t elem_size = dtype_size(dtype);

    auto B_gpu = (B.device().type != Device::Type::CUDA)
                   ? B.to(Device::cuda()).contiguous()
                   : B.contiguous();
    auto X = zeros(std::vector<int64_t>{N, K}, dtype, Device::cuda());

    // We can't take column views of a row-major matrix cheaply (a column
    // is strided by K), so extract each column to a contiguous buffer,
    // call SpSV, and write back. Each SpSV call re-runs analysis — this
    // is O(K) suboptimal and is flagged as a follow-up: caching the
    // analysis across calls for the same L would eliminate the overhead.
    //
    // The 2D copies below operate on raw bytes (elem_size), so the same code
    // path covers Float32/Float64/Complex64/Complex128 without a per-dtype
    // branch — Complex64/128 are just 8/16-byte opaque elements here.
    auto B_t = B_gpu;  // alias
    for (int64_t k = 0; k < K; ++k) {
        // Extract B[:, k] (strided by K in the row-major buffer) into a dense
        // contiguous 1-D tensor with a single strided 2D copy instead of N tiny
        // per-element memcpys. Matches sparse_trsm_standalone.
        Tensor b_col = zeros(std::vector<int64_t>{N}, dtype, Device::cuda());
        CUDA_CHECK_SPARSE(cudaMemcpy2DAsync(
            b_col.data_ptr(), elem_size,
            static_cast<const char*>(B_t.data_ptr()) + k * elem_size, K * elem_size,
            elem_size, N,
            cudaMemcpyDeviceToDevice, stream));

        auto x_col = cuda_sparse_trsv_kernel(L, b_col, upper, stream_opaque);

        // Scatter x_col back into X[:, k] with a single strided 2D copy.
        CUDA_CHECK_SPARSE(cudaMemcpy2DAsync(
            static_cast<char*>(X.data_ptr()) + k * elem_size, K * elem_size,
            x_col.data_ptr(), elem_size,
            elem_size, N,
            cudaMemcpyDeviceToDevice, stream));
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
        // An empty COO is trivially coalesced. Mark it so callers that invoke
        // cuda_coalesce directly (e.g. cuda_coo_to_csc) and any is_coalesced()
        // checks see the same flag the CPU reference sets, rather than
        // re-running coalesce on every empty tensor.
        SparseTensor result = sparse;
        result.mark_coalesced(true);
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

    // Gather values (native per-dtype: Float32/Float64/Int32/Int64).
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
    } else if (values.dtype() == DType::Int32) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<int32_t>()),
                       thrust::device_pointer_cast(sorted_vals.data<int32_t>()));
    } else if (values.dtype() == DType::Int64) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<int64_t>()),
                       thrust::device_pointer_cast(sorted_vals.data<int64_t>()));
    } else {
        throw std::runtime_error(
            "cuda_coalesce: unsupported value dtype (supported: Float32, Float64, "
            "Int32, Int64)");
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
    } else if (values.dtype() == DType::Int32) {
        auto end = thrust::reduce_by_key(
            thrust::cuda::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<int32_t>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<int32_t>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    } else if (values.dtype() == DType::Int64) {
        auto end = thrust::reduce_by_key(
            thrust::cuda::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<int64_t>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<int64_t>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    }

    // Reconstruct unique indices from unique keys
    // key = sum(idx[d] * stride[d]) — decode back to multi-dim indices
    Tensor new_indices = zeros({sparse_dim, new_nnz}, DType::Int64, Device::cuda());
    int64_t* ni_ptr = new_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim; ++d) {
        // idx[d] = (key / stride[d]) % extent[d]. The modulus MUST be the extent
        // of dim d (sp_shape[d]), not stride[d]/stride[d+1] (== sp_shape[d+1]),
        // which is the *next* dim's extent — wrong for tall/non-square (M>K) or
        // sparse_dim>2 tensors (would wrap any index >= the next dim's extent).
        int64_t stride = h_strides[d];
        int64_t extent = sp_shape[d];
        auto ok_dptr = thrust::device_pointer_cast(ok_ptr);
        auto ni_dptr = thrust::device_pointer_cast(ni_ptr + d * new_nnz);
        thrust::transform(thrust::cuda::par,
                          ok_dptr, ok_dptr + new_nnz,
                          ni_dptr,
                          [stride, extent] __device__ (int64_t key) {
                              return (key / stride) % extent;
                          });
    }

    // Trim output values to new_nnz. Use cudaMemcpyAsync on the default stream
    // (the same stream thrust::cuda::par uses above) so the copy is
    // stream-ordered with the preceding thrust work instead of forcing a
    // blocking full-device sync the synchronous cudaMemcpy would impose.
    Tensor final_vals = zeros({new_nnz}, values.dtype(), Device::cuda());
    if (values.dtype() == DType::Float32) {
        CUDA_CHECK_SPARSE(cudaMemcpyAsync(final_vals.data<float>(), out_vals.data<float>(),
                                     new_nnz * sizeof(float), cudaMemcpyDeviceToDevice, 0));
    } else if (values.dtype() == DType::Float64) {
        CUDA_CHECK_SPARSE(cudaMemcpyAsync(final_vals.data<double>(), out_vals.data<double>(),
                                     new_nnz * sizeof(double), cudaMemcpyDeviceToDevice, 0));
    } else if (values.dtype() == DType::Int32) {
        CUDA_CHECK_SPARSE(cudaMemcpyAsync(final_vals.data<int32_t>(), out_vals.data<int32_t>(),
                                     new_nnz * sizeof(int32_t), cudaMemcpyDeviceToDevice, 0));
    } else if (values.dtype() == DType::Int64) {
        CUDA_CHECK_SPARSE(cudaMemcpyAsync(final_vals.data<int64_t>(), out_vals.data<int64_t>(),
                                     new_nnz * sizeof(int64_t), cudaMemcpyDeviceToDevice, 0));
    }

    auto coalesced = SparseTensor::sparse_coo(new_indices, final_vals,
                                    std::vector<int64_t>(sp_shape.begin(), sp_shape.end()));
    // The sort + reduce_by_key above produces sorted, duplicate-free entries,
    // so the result is coalesced. Flag it (callers like cuda_coo_to_csc invoke
    // this directly and rely on is_coalesced()).
    coalesced.mark_coalesced(true);
    return coalesced;
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
    // ccol[0] = 0, ccol[i+1] = upper_bound result for column i. Async on the
    // default stream (same stream as thrust::cuda::par) so it is stream-ordered
    // with the surrounding thrust work rather than a blocking full-device sync.
    CUDA_CHECK_SPARSE(cudaMemsetAsync(ccol_ptr, 0, sizeof(int64_t), 0));
    CUDA_CHECK_SPARSE(cudaMemcpyAsync(ccol_ptr + 1, bounds_ptr,
                                 ncols * sizeof(int64_t), cudaMemcpyDeviceToDevice, 0));

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
    } else if (values.dtype() == DType::Int32) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<int32_t>()),
                       thrust::device_pointer_cast(sorted_vals.data<int32_t>()));
    } else if (values.dtype() == DType::Int64) {
        thrust::gather(thrust::cuda::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<int64_t>()),
                       thrust::device_pointer_cast(sorted_vals.data<int64_t>()));
    } else {
        throw std::runtime_error("cuda_coo_to_csc: unsupported value dtype");
    }

    return SparseTensor::sparse_csc(ccol, sorted_rows, sorted_vals,
                                    std::vector<int64_t>{nrows, ncols});
}

// ============================================================================
// SparseAdd: sparse(M,K) + dense(M,K) -> dense(M,K)
// One thread per row; each thread iterates its CSR non-zeros and adds them
// directly into the cloned dense output.  No atomicAdd needed because CSR
// row ranges are disjoint across threads.
// ============================================================================
template <typename T>
__global__ void csr_sparse_add_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    T* __restrict__ out_ptr,
    int64_t nrows, int64_t ncols)
{
    // Grid-stride over rows so a launch whose grid is clamped (compute_grid_size
    // caps gridDim at the device limit / 2^31-1 blocks) still covers every row
    // for nrows > gridDim*blockDim. A plain one-row-per-thread + early-return
    // would silently drop the row tail at extreme nrows.
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < nrows; row += stride) {
        int64_t row_start = crow_ptr[row];
        int64_t row_end = crow_ptr[row + 1];
        for (int64_t j = row_start; j < row_end; ++j) {
            out_ptr[row * ncols + col_ptr[j]] += val_ptr[j];
        }
    }
}

Tensor cuda_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) {
    // Make the operands' CUDA device current so ensure_csr_on_gpu, the
    // zeros/clone allocations and the scatter-add launch all target the
    // correct GPU on a multi-GPU host. Matches the other cuSPARSE entry points
    // in this TU (cuda_spmm_kernel etc.). Restored on scope exit.
    CudaDeviceGuard dev_guard(dense.device().type == Device::Type::CUDA
                              ? dense.device().index : Device::cuda().index);
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("cuda_sparse_add: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (M != dense.shape()[0] || K != dense.shape()[1]) {
        throw std::runtime_error("cuda_sparse_add: shape mismatch ("
            + std::to_string(M) + "x" + std::to_string(K) + " vs "
            + std::to_string(dense.shape()[0]) + "x" + std::to_string(dense.shape()[1]) + ")");
    }

    DType dtype = dense.dtype();
    auto csr = ensure_csr_on_gpu(sparse);

    auto dense_gpu = (dense.device().type != Device::Type::CUDA)
                     ? dense.to(Device::cuda()).contiguous()
                     : dense.contiguous();

    // Clone dense as output; CSR non-zeros are added in-place.
    auto result = dense_gpu.clone();

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    // Clamp the grid to the 2^31-1 block limit; the kernel grid-strides over the
    // remaining rows, so M > gridDim*threads is still fully covered (instead of
    // the unsigned truncation silently wrapping the launch dim at extreme M).
    const int64_t blocks64_add = (M + threads - 1) / threads;
    unsigned blocks = static_cast<unsigned>(
        blocks64_add > 2147483647LL ? 2147483647LL : blocks64_add);

    if (dtype == DType::Float32) {
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            result.data<float>(), M, K);
    } else if (dtype == DType::Float64) {
        csr_sparse_add_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            result.data<double>(), M, K);
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        // Widen to Float32 for the element-wise scatter-add, then narrow back.
        // Float16/BFloat16 lack wide enough add support on older SMs and this
        // keeps the kernel simple + numerically sane.
        auto vals_f32 = vals.to(DType::Float32);
        auto result_f32 = result.to(DType::Float32);
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals_f32.data<float>(),
            result_f32.data<float>(), M, K);
        CUDA_CHECK_SPARSE(cudaGetLastError());
        result = result_f32.to(dtype);
    } else if (dtype == DType::Int32) {
        // Exact integer scatter-add (matches CPU sparse::add). Each output cell
        // receives at most one addition (CSR (row,col) pairs are unique and rows
        // are partitioned across threads), so the plain += needs no atomics.
        csr_sparse_add_kernel<int32_t><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<int32_t>(),
            result.data<int32_t>(), M, K);
    } else if (dtype == DType::Int64) {
        csr_sparse_add_kernel<int64_t><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<int64_t>(),
            result.data<int64_t>(), M, K);
    } else {
        throw std::runtime_error("cuda_sparse_add: only Float32, Float64, Float16, BFloat16, Int32, and Int64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    CUDA_CHECK_SPARSE(cudaGetLastError());

    // The scatter-add runs on the default stream (stream 0); synchronize before
    // returning so a caller that reads `result` on a different (non-default)
    // stream is correctly ordered against the in-flight add — matching the
    // trailing cudaStreamSynchronize in the other cuSPARSE entry points in this
    // TU. (The F16/BF16 path already syncs implicitly via result_f32.to(dtype).)
    CUDA_CHECK_SPARSE(cudaStreamSynchronize(0));

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
    // Grid-stride over rows (see csr_sparse_add_kernel): covers all rows even
    // when the grid is clamped for nrows > gridDim*blockDim.
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < nrows; row += stride) {
        T sum = static_cast<T>(0);
        int64_t row_start = crow_ptr[row];
        int64_t row_end = crow_ptr[row + 1];
        for (int64_t j = row_start; j < row_end; ++j) {
            sum += val_ptr[j] * x_ptr[col_ptr[j]];
        }
        y_ptr[row] = sum;
    }
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
    // Grid-stride over the (row,col) output element space so a clamped grid
    // covers every output for nrows*ncols_b > gridDim*blockDim (the total can
    // exceed 2^31 well before nrows alone does).
    const int64_t total = nrows * ncols_b;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t row = idx / ncols_b;
        int64_t col = idx % ncols_b;
        T sum = static_cast<T>(0);
        int64_t row_start = crow_ptr[row];
        int64_t row_end = crow_ptr[row + 1];
        for (int64_t j = row_start; j < row_end; ++j) {
            sum += val_ptr[j] * b_ptr[col_ptr[j] * ncols_b + col];
        }
        c_ptr[row * ncols_b + col] = sum;
    }
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
    // Clamp to the 2^31-1 block limit; the kernel grid-strides over the (row,col)
    // output space, so total > gridDim*threads is still fully covered.
    const int64_t blocks64_spmm = (total + threads - 1) / threads;
    unsigned blocks = static_cast<unsigned>(
        blocks64_spmm > 2147483647LL ? 2147483647LL : blocks64_spmm);

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
    // Clamp to the 2^31-1 block limit; the kernel grid-strides over rows, so
    // M > gridDim*threads is still fully covered.
    const int64_t blocks64_spmv = (M + threads - 1) / threads;
    unsigned blocks = static_cast<unsigned>(
        blocks64_spmv > 2147483647LL ? 2147483647LL : blocks64_spmv);

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

// ============================================================================
// SparseAdd: sparse(M,K) + dense(M,K) -> dense(M,K)
// One thread per row; each thread iterates its CSR non-zeros and adds them
// directly into the cloned dense output.  No atomicAdd needed because CSR
// row ranges are disjoint across threads.
// ============================================================================
template <typename T>
__global__ void csr_sparse_add_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    T* __restrict__ out_ptr,
    int64_t nrows, int64_t ncols)
{
    // Grid-stride over rows so a launch whose grid is clamped (compute_grid_size
    // caps gridDim at the device limit / 2^31-1 blocks) still covers every row
    // for nrows > gridDim*blockDim. A plain one-row-per-thread + early-return
    // would silently drop the row tail at extreme nrows.
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < nrows; row += stride) {
        int64_t row_start = crow_ptr[row];
        int64_t row_end = crow_ptr[row + 1];
        for (int64_t j = row_start; j < row_end; ++j) {
            out_ptr[row * ncols + col_ptr[j]] += val_ptr[j];
        }
    }
}

Tensor cuda_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) {
    // Make the operands' CUDA device current so ensure_csr_on_gpu, the
    // zeros/clone allocations and the scatter-add launch all target the
    // correct GPU on a multi-GPU host. Matches the other cuSPARSE entry points
    // in this TU (cuda_spmm_kernel etc.). Restored on scope exit.
    CudaDeviceGuard dev_guard(dense.device().type == Device::Type::CUDA
                              ? dense.device().index : Device::cuda().index);
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("cuda_sparse_add: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (M != dense.shape()[0] || K != dense.shape()[1]) {
        throw std::runtime_error("cuda_sparse_add: shape mismatch ("
            + std::to_string(M) + "x" + std::to_string(K) + " vs "
            + std::to_string(dense.shape()[0]) + "x" + std::to_string(dense.shape()[1]) + ")");
    }

    DType dtype = dense.dtype();

    // The dispatch table always passes CSR components, so we expect CSR.
    auto csr = (sparse.device().type != Device::Type::CUDA)
               ? sparse.to(Device::cuda())
               : sparse;
    if (csr.layout() != SparseLayout::CSR) {
        throw std::runtime_error("cuda_sparse_add (fallback): expected CSR layout");
    }

    auto dense_gpu = (dense.device().type != Device::Type::CUDA)
                     ? dense.to(Device::cuda()).contiguous()
                     : dense.contiguous();

    auto result = dense_gpu.clone();

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    // Clamp the grid to the 2^31-1 block limit; the kernel grid-strides over the
    // remaining rows, so M > gridDim*threads is still fully covered (instead of
    // the unsigned truncation silently wrapping the launch dim at extreme M).
    const int64_t blocks64_add = (M + threads - 1) / threads;
    unsigned blocks = static_cast<unsigned>(
        blocks64_add > 2147483647LL ? 2147483647LL : blocks64_add);

    if (dtype == DType::Float32) {
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            result.data<float>(), M, K);
    } else if (dtype == DType::Float64) {
        csr_sparse_add_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            result.data<double>(), M, K);
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto vals_f32 = vals.to(DType::Float32);
        auto result_f32 = result.to(DType::Float32);
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals_f32.data<float>(),
            result_f32.data<float>(), M, K);
        CUDA_CHECK_SPARSE_FALLBACK(cudaGetLastError());
        result = result_f32.to(dtype);
    } else if (dtype == DType::Int32) {
        // Native integer path (F102): CPU sparse::add supports Int32/Int64, so
        // add exact int instantiations here for cross-backend parity.
        csr_sparse_add_kernel<int32_t><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<int32_t>(),
            result.data<int32_t>(), M, K);
    } else if (dtype == DType::Int64) {
        csr_sparse_add_kernel<int64_t><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<int64_t>(),
            result.data<int64_t>(), M, K);
    } else {
        throw std::runtime_error("cuda_sparse_add: only Float32, Float64, Float16, BFloat16, Int32, and Int64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    CUDA_CHECK_SPARSE_FALLBACK(cudaGetLastError());

    return result;
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUSPARSE

// ============================================================================
// Standalone GPU SpGEMM and SparseTrsv/Trsm (no cuSPARSE dependency)
//
// These are always compiled and provide fallback-free GPU implementations
// when cuSPARSE is not available. When cuSPARSE IS available, these are
// still compiled but only used if the cuSPARSE kernels aren't registered.
// ============================================================================

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace cuda {

#ifndef CUDA_CHECK_SPARSE_STANDALONE
#define CUDA_CHECK_SPARSE_STANDALONE(call)                                     \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error in sparse_standalone at ") +           \
                __FILE__ + ":" + std::to_string(__LINE__) + " - " +            \
                cudaGetErrorString(err));                                       \
        }                                                                      \
    } while (0)
#endif

// ============================================================================
// SparseSoftmax / SparseLogSoftmax: row-wise (log-)softmax over CSR values.
// Always compiled (no cuSPARSE dependency). One grid-strided thread per row;
// each thread does the numerically-stable max / sum-exp / normalize over its
// own disjoint CSR value range — matching CPU sparse_(log_)softmax (F033).
// ============================================================================
template <typename T>
__global__ void csr_softmax_standalone_kernel(
    const int64_t* __restrict__ crow, const T* __restrict__ vals,
    T* __restrict__ out, int64_t M, bool log_mode)
{
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < M; i += stride) {
        int64_t start = crow[i];
        int64_t end = crow[i + 1];
        if (start == end) continue;
        T max_val = vals[start];
        for (int64_t j = start + 1; j < end; ++j)
            if (vals[j] > max_val) max_val = vals[j];
        T sum_exp = T(0);
        for (int64_t j = start; j < end; ++j)
            sum_exp += exp(vals[j] - max_val);
        if (log_mode) {
            T lse = max_val + log(sum_exp);
            for (int64_t j = start; j < end; ++j) out[j] = vals[j] - lse;
        } else {
            T inv = T(1) / sum_exp;
            for (int64_t j = start; j < end; ++j) out[j] = exp(vals[j] - max_val) * inv;
        }
    }
}

Tensor cuda_sparse_softmax_values(const Tensor& crow_in, const Tensor& vals_in,
                                  int64_t M, bool log_mode) {
    DType orig = vals_in.dtype();
    if (orig == DType::Float16 || orig == DType::BFloat16) {
        Tensor out = cuda_sparse_softmax_values(crow_in, vals_in.to(DType::Float32), M, log_mode);
        return out.to(orig);
    }
    Device dev = vals_in.device().type == Device::Type::CUDA ? vals_in.device()
                                                             : Device::cuda();
    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(dev.index);

    Tensor crow = (crow_in.device() == dev ? crow_in : crow_in.to(dev)).contiguous();
    Tensor vals = (vals_in.device() == dev ? vals_in : vals_in.to(dev)).contiguous();
    int64_t nnz = vals.numel();
    Tensor out = zeros({nnz}, vals.dtype(), dev);

    int threads = 256;
    int64_t b64 = (M + threads - 1) / threads;
    unsigned blocks = static_cast<unsigned>(b64 > 2147483647LL ? 2147483647LL : (b64 < 1 ? 1 : b64));

    if (vals.dtype() == DType::Float32) {
        csr_softmax_standalone_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), vals.data<float>(), out.data<float>(), M, log_mode);
    } else if (vals.dtype() == DType::Float64) {
        csr_softmax_standalone_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), vals.data<double>(), out.data<double>(), M, log_mode);
    } else {
        cudaSetDevice(prev_device);
        throw std::runtime_error(
            "cuda_sparse_softmax: only Float32/Float64 (+F16/BF16 widened) supported");
    }
    CUDA_CHECK_SPARSE_STANDALONE(cudaGetLastError());
    cudaSetDevice(prev_device);
    return out;
}

// ============================================================================
// Generic CSR SpMM/SpMV (F034) — integer and complex dtypes cuSPARSE cannot
// handle. Always compiled (no cuSPARSE dependency); one thread per output
// element (SpMM) / row (SpMV). Matches the CPU generic CSR templates.
// ============================================================================
__device__ inline int32_t sp_mac(int32_t acc, int32_t a, int32_t b) { return acc + a * b; }
__device__ inline int64_t sp_mac(int64_t acc, int64_t a, int64_t b) { return acc + a * b; }
__device__ inline cuFloatComplex sp_mac(cuFloatComplex acc, cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(acc, cuCmulf(a, b));
}
__device__ inline cuDoubleComplex sp_mac(cuDoubleComplex acc, cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(acc, cuCmul(a, b));
}
template <typename T> __device__ inline T sp_zero();
template <> __device__ inline int32_t sp_zero<int32_t>() { return 0; }
template <> __device__ inline int64_t sp_zero<int64_t>() { return 0; }
template <> __device__ inline cuFloatComplex sp_zero<cuFloatComplex>() { return make_cuFloatComplex(0.f, 0.f); }
template <> __device__ inline cuDoubleComplex sp_zero<cuDoubleComplex>() { return make_cuDoubleComplex(0.0, 0.0); }

template <typename T>
__global__ void csr_spmm_generic_kernel(
    const int64_t* __restrict__ crow, const int64_t* __restrict__ col,
    const T* __restrict__ vals, const T* __restrict__ B, T* __restrict__ C,
    int64_t M, int64_t N)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx >= total) return;
    int64_t i = idx / N, j = idx % N;
    T acc = sp_zero<T>();
    for (int64_t nz = crow[i]; nz < crow[i + 1]; ++nz) {
        acc = sp_mac(acc, vals[nz], B[col[nz] * N + j]);
    }
    C[idx] = acc;
}

template <typename T>
__global__ void csr_spmv_generic_kernel(
    const int64_t* __restrict__ crow, const int64_t* __restrict__ col,
    const T* __restrict__ vals, const T* __restrict__ x, T* __restrict__ y,
    int64_t M)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= M) return;
    T acc = sp_zero<T>();
    for (int64_t nz = crow[i]; nz < crow[i + 1]; ++nz) {
        acc = sp_mac(acc, vals[nz], x[col[nz]]);
    }
    y[i] = acc;
}

Tensor cuda_spmm_generic(const SparseTensor& sparse, const Tensor& dense) {
    Device dev = dense.device().type == Device::Type::CUDA ? dense.device() : Device::cuda();
    int prev = 0; cudaGetDevice(&prev); cudaSetDevice(dev.index);

    auto sp_shape = sparse.shape();
    int64_t M = sp_shape[0], N = dense.shape()[1];
    auto csr = (sparse.layout() == SparseLayout::CSR) ? sparse : sparse.to_csr();
    Tensor crow = csr.crow_indices().to(dev).contiguous();
    Tensor col  = csr.col_indices().to(dev).contiguous();
    Tensor vals = csr.values().to(dev).contiguous();
    Tensor B = dense.to(dev).contiguous();
    DType dt = dense.dtype();
    Tensor C = zeros({M, N}, dt, dev);

    int64_t total = M * N;
    int block = 256;
    unsigned blocks = static_cast<unsigned>(std::min<int64_t>((total + block - 1) / block, 2147483647LL));
    if (blocks == 0) blocks = 1;
    const int64_t* cr = crow.data<int64_t>();
    const int64_t* cl = col.data<int64_t>();
    if (dt == DType::Int32) {
        csr_spmm_generic_kernel<int32_t><<<blocks, block>>>(cr, cl, vals.data<int32_t>(), B.data<int32_t>(), C.data<int32_t>(), M, N);
    } else if (dt == DType::Int64) {
        csr_spmm_generic_kernel<int64_t><<<blocks, block>>>(cr, cl, vals.data<int64_t>(), B.data<int64_t>(), C.data<int64_t>(), M, N);
    } else if (dt == DType::Complex64) {
        csr_spmm_generic_kernel<cuFloatComplex><<<blocks, block>>>(cr, cl,
            reinterpret_cast<const cuFloatComplex*>(vals.data_ptr()),
            reinterpret_cast<const cuFloatComplex*>(B.data_ptr()),
            reinterpret_cast<cuFloatComplex*>(C.data_ptr()), M, N);
    } else {
        csr_spmm_generic_kernel<cuDoubleComplex><<<blocks, block>>>(cr, cl,
            reinterpret_cast<const cuDoubleComplex*>(vals.data_ptr()),
            reinterpret_cast<const cuDoubleComplex*>(B.data_ptr()),
            reinterpret_cast<cuDoubleComplex*>(C.data_ptr()), M, N);
    }
    CUDA_CHECK_SPARSE_STANDALONE(cudaGetLastError());
    cudaSetDevice(prev);
    return C;
}

Tensor cuda_spmv_generic(const SparseTensor& sparse, const Tensor& vec) {
    Device dev = vec.device().type == Device::Type::CUDA ? vec.device() : Device::cuda();
    int prev = 0; cudaGetDevice(&prev); cudaSetDevice(dev.index);

    auto sp_shape = sparse.shape();
    int64_t M = sp_shape[0];
    auto csr = (sparse.layout() == SparseLayout::CSR) ? sparse : sparse.to_csr();
    Tensor crow = csr.crow_indices().to(dev).contiguous();
    Tensor col  = csr.col_indices().to(dev).contiguous();
    Tensor vals = csr.values().to(dev).contiguous();
    Tensor x = vec.to(dev).contiguous();
    DType dt = vec.dtype();
    Tensor y = zeros({M}, dt, dev);

    int block = 256;
    unsigned blocks = static_cast<unsigned>(std::min<int64_t>((M + block - 1) / block, 2147483647LL));
    if (blocks == 0) blocks = 1;
    const int64_t* cr = crow.data<int64_t>();
    const int64_t* cl = col.data<int64_t>();
    if (dt == DType::Int32) {
        csr_spmv_generic_kernel<int32_t><<<blocks, block>>>(cr, cl, vals.data<int32_t>(), x.data<int32_t>(), y.data<int32_t>(), M);
    } else if (dt == DType::Int64) {
        csr_spmv_generic_kernel<int64_t><<<blocks, block>>>(cr, cl, vals.data<int64_t>(), x.data<int64_t>(), y.data<int64_t>(), M);
    } else if (dt == DType::Complex64) {
        csr_spmv_generic_kernel<cuFloatComplex><<<blocks, block>>>(cr, cl,
            reinterpret_cast<const cuFloatComplex*>(vals.data_ptr()),
            reinterpret_cast<const cuFloatComplex*>(x.data_ptr()),
            reinterpret_cast<cuFloatComplex*>(y.data_ptr()), M);
    } else {
        csr_spmv_generic_kernel<cuDoubleComplex><<<blocks, block>>>(cr, cl,
            reinterpret_cast<const cuDoubleComplex*>(vals.data_ptr()),
            reinterpret_cast<const cuDoubleComplex*>(x.data_ptr()),
            reinterpret_cast<cuDoubleComplex*>(y.data_ptr()), M);
    }
    CUDA_CHECK_SPARSE_STANDALONE(cudaGetLastError());
    cudaSetDevice(prev);
    return y;
}

// ============================================================================
// SpGEMM: 3-pass algorithm (count -> prefix sum -> fill)
// ============================================================================

// Pass 1: Count nnz per row of C = A * B (both CSR)
template <typename T>
__global__ void spgemm_count_kernel(
    const int64_t* __restrict__ a_crow,  // [M+1]
    const int64_t* __restrict__ a_col,   // [nnz_a]
    const int64_t* __restrict__ b_crow,  // [K+1]
    const int64_t* __restrict__ b_col,   // [nnz_b]
    int64_t* __restrict__ row_nnz,       // [M] output: nnz per row
    int64_t M, int64_t N)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Use a simple hash-set approach (for small row widths)
    // For each non-zero in A[row,:], iterate B cols and count unique columns
    int64_t count = 0;

    // Simple approach: for each (row, col) pair in A, add all B[col,:] columns
    // Use sorting-free counting with a boolean marker array (capped)
    // For very wide rows, fall back to counting with duplicates
    int64_t a_start = a_crow[row];
    int64_t a_end = a_crow[row + 1];

    // Count upper bound (with duplicates)
    for (int64_t ja = a_start; ja < a_end; ++ja) {
        int64_t k = a_col[ja];
        count += b_crow[k + 1] - b_crow[k];
    }

    row_nnz[row] = count;  // Upper bound; dedup happens in fill pass
}

// Pass 3: Fill C values (with deduplication via sorting)
template <typename T>
__global__ void spgemm_fill_kernel(
    const int64_t* __restrict__ a_crow,
    const int64_t* __restrict__ a_col,
    const T* __restrict__ a_vals,
    const int64_t* __restrict__ b_crow,
    const int64_t* __restrict__ b_col,
    const T* __restrict__ b_vals,
    const int64_t* __restrict__ c_crow,  // [M+1] prefix sum
    int64_t* __restrict__ c_col,         // [total_nnz] output
    T* __restrict__ c_vals,              // [total_nnz] output
    int64_t* __restrict__ c_row_nnz,     // [M] actual nnz per row (written back)
    int64_t M, int64_t N)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int64_t c_start = c_crow[row];
    int64_t c_end   = c_crow[row + 1];
    int64_t cap     = c_end - c_start;   // upper-bound nnz capacity for this row
    int64_t a_start = a_crow[row];
    int64_t a_end = a_crow[row + 1];

    if (cap <= 0) {
        c_row_nnz[row] = 0;
        return;
    }

    // Use the over-allocated output region [c_start, c_end) as an open-addressing
    // hash table keyed by column, replacing the previous O(row_nnz^2) linear
    // rescan with amortized O(1) probing. c_col was pre-filled with the sentinel
    // -1 (empty). cap is >= the number of distinct columns (it is the sum of the
    // contributing B-row lengths), so linear probing always finds a free slot.
    for (int64_t ja = a_start; ja < a_end; ++ja) {
        int64_t k = a_col[ja];
        T a_val = a_vals[ja];
        int64_t b_start = b_crow[k];
        int64_t b_end = b_crow[k + 1];

        for (int64_t jb = b_start; jb < b_end; ++jb) {
            int64_t col = b_col[jb];
            T val = a_val * b_vals[jb];

            // Hash the column into the row's slot range and linear-probe.
            uint64_t h = static_cast<uint64_t>(col) * 0x9E3779B97F4A7C15ULL;
            int64_t slot = static_cast<int64_t>(h % static_cast<uint64_t>(cap));
            for (int64_t probe = 0; probe < cap; ++probe) {
                int64_t p = c_start + slot;
                int64_t existing = c_col[p];
                if (existing == col) {
                    c_vals[p] += val;
                    break;
                }
                if (existing < 0) {  // empty sentinel
                    c_col[p] = col;
                    c_vals[p] = val;
                    break;
                }
                slot = (slot + 1) % cap;
            }
        }
    }

    // Stable-compact the populated (non-sentinel) slots to the front of the row
    // so the downstream compact kernel (which copies the first row_nnz entries)
    // sees a contiguous run.
    int64_t write_pos = c_start;
    for (int64_t p = c_start; p < c_end; ++p) {
        int64_t col = c_col[p];
        if (col >= 0) {
            if (p != write_pos) {
                c_col[write_pos] = col;
                c_vals[write_pos] = c_vals[p];
                c_col[p] = -1;
            }
            write_pos++;
        }
    }

    c_row_nnz[row] = write_pos - c_start;
}

// Compact kernel: remove gaps from over-allocated output
template <typename T>
__global__ void spgemm_compact_kernel(
    const int64_t* __restrict__ old_crow,   // [M+1] old prefix
    const int64_t* __restrict__ new_crow,   // [M+1] compacted prefix
    const int64_t* __restrict__ old_col,
    const T* __restrict__ old_vals,
    int64_t* __restrict__ new_col,
    T* __restrict__ new_vals,
    const int64_t* __restrict__ row_nnz,    // [M] actual per row
    int64_t M)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int64_t src = old_crow[row];
    int64_t dst = new_crow[row];
    int64_t count = row_nnz[row];
    for (int64_t i = 0; i < count; ++i) {
        new_col[dst + i] = old_col[src + i];
        new_vals[dst + i] = old_vals[src + i];
    }
}

template <typename T>
auto spgemm_standalone_typed(
    const Tensor& a_crow, const Tensor& a_col, const Tensor& a_vals,
    const Tensor& b_crow, const Tensor& b_col, const Tensor& b_vals,
    int64_t M, int64_t K, int64_t N, cudaStream_t stream) -> std::vector<Tensor>
{
    constexpr int BLOCK = 256;

    // Pass 1: Count nnz per row (upper bound)
    int64_t* d_row_nnz = nullptr;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_row_nnz, M * sizeof(int64_t), stream));

    int64_t count_blocks = (M + BLOCK - 1) / BLOCK;
    spgemm_count_kernel<T><<<count_blocks, BLOCK, 0, stream>>>(
        a_crow.data<int64_t>(), a_col.data<int64_t>(),
        b_crow.data<int64_t>(), b_col.data<int64_t>(),
        d_row_nnz, M, N);
    CUDA_CHECK_SPARSE_STANDALONE(cudaGetLastError());

    // Pass 2: Exclusive prefix sum on row_nnz -> c_crow (upper bound)
    int64_t* d_crow_ub = nullptr;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_crow_ub, (M + 1) * sizeof(int64_t), stream));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_row_nnz, d_crow_ub, M, stream);
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_temp, temp_bytes, stream));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_row_nnz, d_crow_ub, M, stream);
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_temp, stream));

    // Get total nnz upper bound
    int64_t last_prefix = 0, last_count = 0;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(&last_prefix, d_crow_ub + M - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(&last_count, d_row_nnz + M - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaStreamSynchronize(stream));
    int64_t total_nnz_ub = last_prefix + last_count;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(d_crow_ub + M, &total_nnz_ub, sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    if (total_nnz_ub == 0) {
        auto c_crow_t = tenzor::zeros({M + 1}, DType::Int64, a_crow.device());
        auto c_col_t = tenzor::empty({0}, DType::Int64, a_crow.device());
        auto c_vals_t = tenzor::empty({0}, a_vals.dtype(), a_crow.device());
        CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_row_nnz, stream));
        CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_crow_ub, stream));
        return {c_crow_t, c_col_t, c_vals_t};
    }

    // Allocate upper-bound output arrays
    int64_t* d_col_ub = nullptr;
    T* d_vals_ub = nullptr;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_col_ub, total_nnz_ub * sizeof(int64_t), stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_vals_ub, total_nnz_ub * sizeof(T), stream));

    // Pass 3: Fill with deduplication. Pre-fill the column buffer with the
    // empty sentinel (-1) so the in-kernel open-addressing hash table can detect
    // free slots. 0xFF bytes -> int64_t(-1).
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemsetAsync(d_col_ub, 0xFF,
        total_nnz_ub * sizeof(int64_t), stream));

    int64_t* d_actual_nnz = nullptr;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_actual_nnz, M * sizeof(int64_t), stream));

    spgemm_fill_kernel<T><<<count_blocks, BLOCK, 0, stream>>>(
        a_crow.data<int64_t>(), a_col.data<int64_t>(), a_vals.data<T>(),
        b_crow.data<int64_t>(), b_col.data<int64_t>(), b_vals.data<T>(),
        d_crow_ub, d_col_ub, d_vals_ub, d_actual_nnz, M, N);
    CUDA_CHECK_SPARSE_STANDALONE(cudaGetLastError());

    // Compact: prefix sum on actual_nnz -> new_crow
    int64_t* d_crow_final = nullptr;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_crow_final, (M + 1) * sizeof(int64_t), stream));

    temp_bytes = 0;
    d_temp = nullptr;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_actual_nnz, d_crow_final, M, stream);
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_temp, temp_bytes, stream));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_actual_nnz, d_crow_final, M, stream);
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_temp, stream));

    // Get total actual nnz
    int64_t actual_last_prefix = 0, actual_last_count = 0;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(&actual_last_prefix, d_crow_final + M - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(&actual_last_count, d_actual_nnz + M - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaStreamSynchronize(stream));
    int64_t total_nnz = actual_last_prefix + actual_last_count;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(d_crow_final + M, &total_nnz, sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Compact col/vals
    auto c_crow_t = Tensor({M + 1}, DType::Int64, a_crow.device());
    auto c_col_t = Tensor({total_nnz}, DType::Int64, a_crow.device());
    auto c_vals_t = Tensor({total_nnz}, a_vals.dtype(), a_crow.device());

    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(c_crow_t.data<int64_t>(), d_crow_final,
        (M + 1) * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));

    if (total_nnz > 0) {
        spgemm_compact_kernel<T><<<count_blocks, BLOCK, 0, stream>>>(
            d_crow_ub, d_crow_final, d_col_ub, d_vals_ub,
            c_col_t.data<int64_t>(), c_vals_t.data<T>(),
            d_actual_nnz, M);
        CUDA_CHECK_SPARSE_STANDALONE(cudaGetLastError());
    }

    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_row_nnz, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_crow_ub, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_col_ub, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_vals_ub, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_actual_nnz, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_crow_final, stream));

    return {c_crow_t, c_col_t, c_vals_t};
}

auto spgemm_standalone(std::span<const Tensor> inputs, const OpAttributes& attrs,
                       cudaStream_t stream) -> std::vector<Tensor> {
    int64_t M = attrs.get_int(AttrKey::M);
    int64_t K = attrs.get_int(AttrKey::K);
    int64_t N = attrs.get_int(AttrKey::N);

    if (inputs[2].dtype() == DType::Float32) {
        return spgemm_standalone_typed<float>(
            inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5],
            M, K, N, stream);
    } else if (inputs[2].dtype() == DType::Float64) {
        return spgemm_standalone_typed<double>(
            inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5],
            M, K, N, stream);
    } else {
        throw std::runtime_error("spgemm_standalone: only Float32/Float64 supported");
    }
}

// ============================================================================
// Sparse Triangular Solve: deterministic single-thread sequential GPU kernel.
//
// The previous version mapped one thread per row and spin-waited on
// `atomicOr(&solved[c], 0)` until dependency rows were solved by threads in
// other blocks. CUDA gives no forward-progress guarantee that a not-yet-
// scheduled block holding dependency `c` will run while this block spins on an
// SM, so for N large enough that the grid's blocks cannot all be co-resident
// the solve can hang. (The same hazard is documented for the ROCm SIMT trsv,
// which was switched to a sequential kernel for correctness.)
//
// We solve sequentially in dependency order on a single thread: forward
// substitution (row 0..N-1) for lower-triangular, backward (N-1..0) for
// upper-triangular. Each row's dependencies are already computed when it is
// processed, so no cross-block synchronization is needed and forward progress
// is guaranteed.
// ============================================================================

template <typename T>
__global__ void sparse_trsv_seq_kernel(
    const int64_t* __restrict__ crow,    // [N+1]
    const int64_t* __restrict__ col,     // [nnz]
    const T* __restrict__ vals,          // [nnz]
    const T* __restrict__ b,             // [N]
    T* __restrict__ x,                   // [N] output
    int64_t N, bool upper,
    int* __restrict__ error_flag,        // set to 1 on a missing/zero diagonal
    int64_t* __restrict__ error_row)     // offending row index (valid iff *error_flag)
{
    // Single thread drives the whole solve in dependency order.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int64_t step = 0; step < N; ++step) {
        int64_t row = upper ? (N - 1 - step) : step;

        int64_t row_start = crow[row];
        int64_t row_end = crow[row + 1];

        // x[row] = (b[row] - sum(A[row,c]*x[c] for dependent c)) / A[row,row]
        // diag starts at T(0) (matching the CPU reference, sparse_ops.cpp
        // cpu_forward_substitution/cpu_backward_substitution) so a structurally
        // missing diagonal entry (dropped by to_sparse() on an exact-zero
        // diagonal) is indistinguishable from an explicit zero entry — both must
        // error out instead of silently solving with an implicit unit diagonal.
        T rhs = b[row];
        T diag = T(0);
        for (int64_t j = row_start; j < row_end; ++j) {
            int64_t c = col[j];
            if (c == row) {
                diag = vals[j];
            } else if (upper ? (c > row) : (c < row)) {
                rhs -= vals[j] * x[c];
            }
        }
        if (diag == T(0)) {
            // Device-side error-flag pattern (mirrors indexing.cu's OOB-index
            // reporting): stay memory-safe, record the offending row, and let
            // the host wrapper check the flag after the kernel completes and
            // throw a proper C++ exception — kernels can't throw directly.
            atomicExch(error_flag, 1);
            *error_row = row;
            return;
        }
        x[row] = rhs / diag;
    }
}

auto sparse_trsv_standalone(
    const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
    const Tensor& b, int64_t N, bool upper, cudaStream_t stream) -> Tensor
{
    auto x = tenzor::zeros({N}, vals.dtype(), vals.device());

    int* d_error_flag = nullptr;
    int64_t* d_error_row = nullptr;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_error_flag, sizeof(int), stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaMallocAsync(&d_error_row, sizeof(int64_t), stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    if (vals.dtype() == DType::Float32) {
        sparse_trsv_seq_kernel<float><<<1, 1, 0, stream>>>(
            crow.data<int64_t>(), col_idx.data<int64_t>(), vals.data<float>(),
            b.data<float>(), x.data<float>(), N, upper, d_error_flag, d_error_row);
    } else if (vals.dtype() == DType::Float64) {
        sparse_trsv_seq_kernel<double><<<1, 1, 0, stream>>>(
            crow.data<int64_t>(), col_idx.data<int64_t>(), vals.data<double>(),
            b.data<double>(), x.data<double>(), N, upper, d_error_flag, d_error_row);
    } else {
        CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_error_flag, stream));
        CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_error_row, stream));
        throw std::runtime_error("sparse_trsv_standalone: only Float32/Float64 supported");
    }

    CUDA_CHECK_SPARSE_STANDALONE(cudaGetLastError());

    // Bring the error flag back synchronously — mirrors indexing.cu's OOB
    // check (cudaMemcpyAsync + cudaStreamSynchronize immediately followed by
    // a host-side throw) since a device kernel cannot itself throw.
    int host_error = 0;
    int64_t host_error_row = -1;
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(&host_error, d_error_flag, sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpyAsync(&host_error_row, d_error_row, sizeof(int64_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaStreamSynchronize(stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_error_flag, stream));
    CUDA_CHECK_SPARSE_STANDALONE(cudaFreeAsync(d_error_row, stream));

    if (host_error) {
        throw std::runtime_error(
            "sparse_triangular_solve: zero diagonal element at row " +
            std::to_string(host_error_row));
    }

    return x;
}

auto sparse_trsm_standalone(
    const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
    const Tensor& B, int64_t N, bool upper, cudaStream_t stream) -> Tensor
{
    int64_t K = B.shape()[1];
    auto X = tenzor::zeros({N, K}, vals.dtype(), vals.device());

    // Solve column-by-column
    for (int64_t k = 0; k < K; ++k) {
        auto b_col = B.slice(1, k, k + 1).squeeze(1);
        auto x_col = sparse_trsv_standalone(crow, col_idx, vals, b_col, N, upper, stream);
        // Copy x_col into X[:, k]
        if (vals.dtype() == DType::Float32) {
            CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpy2DAsync(
                X.data<float>() + k, K * sizeof(float),
                x_col.data<float>(), sizeof(float),
                sizeof(float), N,
                cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDA_CHECK_SPARSE_STANDALONE(cudaMemcpy2DAsync(
                X.data<double>() + k, K * sizeof(double),
                x_col.data<double>(), sizeof(double),
                sizeof(double), N,
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    CUDA_CHECK_SPARSE_STANDALONE(cudaStreamSynchronize(stream));
    return X;
}

} // namespace cuda
} // namespace tenzor
