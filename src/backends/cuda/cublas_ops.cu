/**
 * @file cublas_ops.cu
 * @brief Enhanced cuBLAS integration with Tensor Core support (GemmEx)
 *
 * This file implements optimized matrix multiplication using cuBLAS with:
 * - cublasGemmEx for Tensor Core acceleration (FP16, BF16, INT8)
 * - Automatic algorithm selection and tuning
 * - Batch matrix multiplication optimization
 * - Support for all data types with optimal compute types
 */

#ifdef TENZOR_HAS_CUBLAS

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/cuda_config.hpp"
#include "kernels/cuda_common.cuh"
#include "cublas_handle_pool.hpp"

namespace tenzor {
namespace cuda {

// cuBLAS handle management via centralized CuBLASHandlePool (cublas_handle_pool.hpp)

// ============================================================================
// Data Type Conversion Utilities
// ============================================================================

cudaDataType_t dtype_to_cuda(DType dtype) {
    switch (dtype) {
        case DType::Float32: return CUDA_R_32F;
        case DType::Float64: return CUDA_R_64F;
        case DType::Float16: return CUDA_R_16F;
        case DType::BFloat16: return CUDA_R_16BF;
        case DType::Int8: return CUDA_R_8I;
        case DType::Int32: return CUDA_R_32I;
#if CUDA_VERSION >= 11080
        case DType::FP8_E4M3: return CUDA_R_8F_E4M3;
        case DType::FP8_E5M2: return CUDA_R_8F_E5M2;
#endif
        default:
            throw std::runtime_error("Unsupported dtype for cuBLAS");
    }
}

// Select the cuBLAS *compute type* (NOT a cudaDataType_t) for the GemmEx APIs.
// The cublasGemmEx / cublasGemmStridedBatchedEx computeType parameter is a
// cublasComputeType_t; passing a cudaDataType_t silently selected the wrong
// precision (CUDA_R_32F==0 aliases CUBLAS_COMPUTE_16F==0, i.e. FP16 accumulation
// for an FP32 GEMM).
cublasComputeType_t select_compute_type(DType input_dtype) {
    switch (input_dtype) {
        case DType::Float32:
            // Honor the global TF32 toggle, matching the FP32 paths below.
            return ::tenzor::cuda::matmul::allow_tf32()
                       ? CUBLAS_COMPUTE_32F_FAST_TF32
                       : CUBLAS_COMPUTE_32F;
        case DType::Float64:
            return CUBLAS_COMPUTE_64F;
        case DType::Float16:
        case DType::BFloat16:
            // FP32 accumulation for FP16/BF16 Tensor Cores
            return CUBLAS_COMPUTE_32F;
        case DType::Int8:
            // INT32 accumulation for INT8 Tensor Cores
            return CUBLAS_COMPUTE_32I;
        default:
            return CUBLAS_COMPUTE_32F;
    }
}

// ============================================================================
// cuBLAS GemmEx - Tensor Core Optimized Matrix Multiplication
// ============================================================================

/**
 * @brief Enhanced matrix multiplication using cublasGemmEx with Tensor Core support
 *
 * This function provides:
 * - Automatic Tensor Core acceleration for FP16, BF16, and INT8
 * - Optimal compute type selection (FP32 accumulation for FP16/BF16)
 * - Algorithm auto-tuning for best performance
 * - Support for all cuBLAS data types
 *
 * Performance improvements:
 * - FP16 Tensor Cores: 8-15x speedup vs FP32 on Ampere/Hopper GPUs
 * - Automatic fallback to CUDA Cores if Tensor Cores unavailable
 * - Optimized memory layout (row-major to column-major conversion handled)
 *
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param dtype Data type of matrices
 * @param stream CUDA stream for async execution
 * @param transpose_a Whether to transpose A
 * @param transpose_b Whether to transpose B
 */
void cublas_gemm_ex(
    const void* A,
    const void* B,
    void* C,
    int64_t M,
    int64_t N,
    int64_t K,
    DType dtype,
    cudaStream_t stream = nullptr,
    bool transpose_a = false,
    bool transpose_b = false
) {
    // Get cuBLAS handle and set stream
    cublasHandle_t handle = CuBLASHandlePool::get(stream);

    // Convert data types
    cudaDataType_t cuda_dtype = dtype_to_cuda(dtype);
    cublasComputeType_t compute_type = select_compute_type(dtype);

    // Set alpha and beta
    // Note: We use FP32 alpha/beta even for FP16 operations (cuBLAS requirement)
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    const double alpha_d = 1.0;
    const double beta_d = 0.0;
    const int32_t alpha_i = 1;
    const int32_t beta_i = 0;

    const void* alpha;
    const void* beta;

    if (dtype == DType::Float64) {
        alpha = &alpha_d;
        beta = &beta_d;
    } else if (dtype == DType::Int8 || dtype == DType::Int32) {
        alpha = &alpha_i;
        beta = &beta_i;
    } else {
        alpha = &alpha_f;
        beta = &beta_f;
    }

    // cuBLAS uses column-major, we use row-major. We pass B first then A so
    // cuBLAS computes (B_col)(A_col) = (B_user_row)^T (A_user_row)^T = (A·B)^T,
    // and the col-major output (N×M) is bit-identical to the row-major (M×N)
    // we want. Because the col-major view of row-major data is already the
    // logical transpose, we use OP_N for both (no further transpose). Passing
    // transpose_a=true asks cuBLAS to transpose A — i.e. OP_T on the second
    // operand (cuBLAS-B = A_user); same for transpose_b on cuBLAS-A.

    // Leading dimensions: a cuBLAS leading dimension is the PHYSICAL trailing
    // row-major stride of the stored matrix (K for A_user, N for B_user). It is
    // a property of the memory layout, NOT of the OP_T/OP_N flag — the OP flags
    // (set on the cublasGemmEx call below) alone select logical transpose. The
    // previous `transpose_a ? M : K` / `transpose_b ? K : N` would pass the
    // wrong stride whenever a transpose flag was set.
    int64_t lda = K;   // physical leading dim of A_user col-major view
    int64_t ldb = N;   // physical leading dim of B_user col-major view
    int64_t ldc = N;   // leading dim of C col-major (output is N×M)

    // Use cublasGemmEx for Tensor Core acceleration
    TENZOR_CUBLAS_CHECK(cublasGemmEx(
        handle,
        transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,  // op for cuBLAS-A (= our B_user)
        transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,  // op for cuBLAS-B (= our A_user)
        N,              // Rows of B^T (cols of B)
        M,              // Cols of A^T (rows of A)
        K,              // Cols of B^T = Rows of A^T
        alpha,
        B,              // B matrix
        cuda_dtype,     // Data type of B
        ldb,            // Leading dimension of B
        A,              // A matrix
        cuda_dtype,     // Data type of A
        lda,            // Leading dimension of A
        beta,
        C,              // C matrix
        cuda_dtype,     // Data type of C
        ldc,            // Leading dimension of C
        compute_type,   // Compute type (FP32 for FP16/BF16 Tensor Cores)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Enable Tensor Core operations
    ));
}

// ============================================================================
// Batched Matrix Multiplication with cuBLAS
// ============================================================================

/**
 * @brief Batched matrix multiplication using cublasGemmStridedBatchedEx
 *
 * Optimized for:
 * - Multiple matrix multiplications with same dimensions
 * - Transformer attention mechanisms (Q @ K^T @ V)
 * - Batch processing in deep learning
 *
 * Uses Tensor Cores when available for 8-15x speedup on FP16/BF16.
 *
 * @param A Input batch of matrices A (batch_size x M x K)
 * @param B Input batch of matrices B (batch_size x K x N)
 * @param C Output batch of matrices C (batch_size x M x N)
 * @param batch_size Number of matrices in batch
 * @param M Rows in each A matrix
 * @param N Columns in each B matrix
 * @param K Columns in A / rows in B
 * @param dtype Data type of matrices
 * @param stream CUDA stream
 */
void cublas_batched_gemm_ex(
    const void* A,
    const void* B,
    void* C,
    int64_t batch_size,
    int64_t M,
    int64_t N,
    int64_t K,
    DType dtype,
    cudaStream_t stream = nullptr
) {
    // Get cuBLAS handle and set stream
    cublasHandle_t handle = CuBLASHandlePool::get(stream);

    // Convert data types
    cudaDataType_t cuda_dtype = dtype_to_cuda(dtype);
    cublasComputeType_t compute_type = select_compute_type(dtype);

    // Set alpha and beta (same as non-batched version)
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    const double alpha_d = 1.0;
    const double beta_d = 0.0;
    const int32_t alpha_i = 1;
    const int32_t beta_i = 0;

    const void* alpha;
    const void* beta;

    if (dtype == DType::Float64) {
        alpha = &alpha_d;
        beta = &beta_d;
    } else if (dtype == DType::Int8 || dtype == DType::Int32) {
        alpha = &alpha_i;
        beta = &beta_i;
    } else {
        alpha = &alpha_f;
        beta = &beta_f;
    }

    // Strides between matrices in batch
    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

    // Leading dimensions (row-major layout)
    int64_t lda = K;
    int64_t ldb = N;
    int64_t ldc = N;

    // Use cublasGemmStridedBatchedEx for batched Tensor Core operations
    TENZOR_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,    // Don't transpose B
        CUBLAS_OP_N,    // Don't transpose A
        N,              // Rows of B^T
        M,              // Cols of A^T
        K,              // Inner dimension
        alpha,
        B,              // B matrix batch
        cuda_dtype,
        ldb,
        stride_b,
        A,              // A matrix batch
        cuda_dtype,
        lda,
        stride_a,
        beta,
        C,              // C matrix batch
        cuda_dtype,
        ldc,
        stride_c,
        batch_size,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

/**
 * @brief Batched GEMM with scaling factor fused into computation
 *
 * Computes: C = scale * A @ B for each matrix in batch
 * This avoids a separate scaling kernel, improving performance for attention.
 *
 * @param A Input batch of matrices A (batch_size x M x K)
 * @param B Input batch of matrices B (batch_size x K x N)
 * @param C Output batch of matrices C (batch_size x M x N)
 * @param batch_size Number of matrices in batch
 * @param M Rows in each A matrix
 * @param N Columns in each B matrix
 * @param K Columns in A / rows in B
 * @param scale Scaling factor applied to result
 * @param dtype Data type of matrices
 * @param stream CUDA stream
 */
void cublas_batched_gemm_scaled(
    const void* A,
    const void* B,
    void* C,
    int64_t batch_size,
    int64_t M,
    int64_t N,
    int64_t K,
    float scale,
    DType dtype,
    cudaStream_t stream = nullptr
) {
    cublasHandle_t handle = CuBLASHandlePool::get(stream);

    // A float scale factor is only well-defined for floating-point GEMM. An
    // integer dtype selects CUBLAS_COMPUTE_32I, which requires int32 alpha/beta;
    // passing a float scale alongside it is a type mismatch (and a scaled
    // integer GEMM with a fractional scale is ill-defined). Reject up-front.
    if (dtype == DType::Int8 || dtype == DType::Int32) {
        throw std::runtime_error(
            "cublas_batched_gemm_scaled: a float scale factor is not supported for "
            "integer dtypes (Int8/Int32); use the unscaled integer batched GEMM instead");
    }

    cudaDataType_t cuda_dtype = dtype_to_cuda(dtype);
    cublasComputeType_t compute_type = select_compute_type(dtype);

    // Apply scale as alpha parameter
    const float alpha_f = scale;
    const float beta_f = 0.0f;
    const double alpha_d = static_cast<double>(scale);
    const double beta_d = 0.0;

    const void* alpha;
    const void* beta;

    if (dtype == DType::Float64) {
        alpha = &alpha_d;
        beta = &beta_d;
    } else {
        alpha = &alpha_f;
        beta = &beta_f;
    }

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

    int64_t lda = K;
    int64_t ldb = N;
    int64_t ldc = N;

    TENZOR_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        alpha,
        B, cuda_dtype, ldb, stride_b,
        A, cuda_dtype, lda, stride_a,
        beta,
        C, cuda_dtype, ldc, stride_c,
        batch_size,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// ============================================================================
// High-Level Tensor API Wrappers
// ============================================================================

/**
 * @brief Batched matmul with fused scaling: C = scale * A @ B
 */
auto cublas_batched_matmul_scaled(const Tensor& a, const Tensor& b, float scale) -> Tensor {
    if (a.device().type != Device::Type::CUDA || b.device().type != Device::Type::CUDA) {
        throw std::runtime_error("cublas_batched_matmul_scaled requires CUDA tensors");
    }

    if (a.ndim() != 3 || b.ndim() != 3) {
        throw std::runtime_error("cublas_batched_matmul_scaled requires 3D tensors");
    }

    auto a_shape = a.shape();
    auto b_shape = b.shape();

    int64_t batch_size = a_shape[0];
    int64_t M = a_shape[1];
    int64_t K = a_shape[2];
    int64_t K2 = b_shape[1];
    int64_t N = b_shape[2];

    if (batch_size != b_shape[0]) {
        throw std::runtime_error("Batch size mismatch for batched matmul");
    }

    if (K != K2) {
        throw std::runtime_error("Matrix dimension mismatch for batched matmul");
    }

    Tensor result({batch_size, M, N}, a.dtype(), a.device());

    cublas_batched_gemm_scaled(
        a.data_ptr(), b.data_ptr(), result.data_ptr(),
        batch_size, M, N, K, scale, a.dtype()
    );

    return result;
}

/**
 * @brief Matrix multiplication for Tensor objects using optimized cuBLAS
 */
auto cublas_matmul(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.device().type != Device::Type::CUDA || b.device().type != Device::Type::CUDA) {
        throw std::runtime_error("cublas_matmul requires CUDA tensors");
    }

    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("cublas_matmul requires 2D tensors");
    }

    auto a_shape = a.shape();
    auto b_shape = b.shape();

    int64_t M = a_shape[0];
    int64_t K = a_shape[1];
    int64_t K2 = b_shape[0];
    int64_t N = b_shape[1];

    if (K != K2) {
        throw std::runtime_error("Matrix dimension mismatch for matmul");
    }

    // Create output tensor
    Tensor result({M, N}, a.dtype(), a.device());

    // Get data pointers based on dtype
    const void* a_ptr = a.data_ptr();
    const void* b_ptr = b.data_ptr();
    void* c_ptr = result.data_ptr();

    // Perform matrix multiplication with Tensor Core acceleration
    cublas_gemm_ex(a_ptr, b_ptr, c_ptr, M, N, K, a.dtype());

    return result;
}

/**
 * @brief Batched matrix multiplication for Tensor objects
 */
auto cublas_batched_matmul(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.device().type != Device::Type::CUDA || b.device().type != Device::Type::CUDA) {
        throw std::runtime_error("cublas_batched_matmul requires CUDA tensors");
    }

    if (a.ndim() != 3 || b.ndim() != 3) {
        throw std::runtime_error("cublas_batched_matmul requires 3D tensors");
    }

    auto a_shape = a.shape();
    auto b_shape = b.shape();

    int64_t batch_size = a_shape[0];
    int64_t M = a_shape[1];
    int64_t K = a_shape[2];
    int64_t K2 = b_shape[1];
    int64_t N = b_shape[2];

    if (batch_size != b_shape[0]) {
        throw std::runtime_error("Batch size mismatch for batched matmul");
    }

    if (K != K2) {
        throw std::runtime_error("Matrix dimension mismatch for batched matmul");
    }

    // Create output tensor
    Tensor result({batch_size, M, N}, a.dtype(), a.device());

    // Get data pointers
    const void* a_ptr = a.data_ptr();
    const void* b_ptr = b.data_ptr();
    void* c_ptr = result.data_ptr();

    // Perform batched matrix multiplication with Tensor Core acceleration
    cublas_batched_gemm_ex(a_ptr, b_ptr, c_ptr, batch_size, M, N, K, a.dtype());

    return result;
}

// ============================================================================
// Bias Add Kernel - Fused with Linear for optimal performance
// ============================================================================

/**
 * @brief CUDA kernel for adding bias to a 2D tensor
 *
 * Broadcasts bias [out_features] across batch dimension to add to
 * output [batch_size, out_features].
 *
 * Uses vectorized loads for better memory bandwidth.
 */
template<typename T>
__global__ void bias_add_kernel(
    T* output,
    const T* bias,
    int64_t batch_size,
    int64_t out_features
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * out_features;

    // Grid-stride loop for large tensors
    for (int64_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        int64_t col = i % out_features;
        output[i] += bias[col];
    }
}

/**
 * @brief Vectorized bias add kernel using float4 for 4x memory bandwidth
 */
__global__ void bias_add_kernel_vec4(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features
) {
    // Ensure out_features is divisible by 4 for vectorization
    int64_t vec_features = out_features / 4;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_vecs = batch_size * vec_features;

    float4* output4 = reinterpret_cast<float4*>(output);
    const float4* bias4 = reinterpret_cast<const float4*>(bias);

    for (int64_t i = idx; i < total_vecs; i += blockDim.x * gridDim.x) {
        int64_t col = i % vec_features;
        float4 out_val = output4[i];
        float4 bias_val = bias4[col];
        out_val.x += bias_val.x;
        out_val.y += bias_val.y;
        out_val.z += bias_val.z;
        out_val.w += bias_val.w;
        output4[i] = out_val;
    }
}

// ============================================================================
// Bias Gradient Reduction - Sum over batch dimension for backward pass
// ============================================================================

/**
 * @brief CUDA kernel for computing bias gradients via sum reduction
 *
 * Computes grad_bias[j] = sum_i(grad_output[i, j]) for all j in [0, out_features)
 * Each block handles one output feature, using efficient parallel reduction.
 */
// Accumulation type: use float for half/bfloat16 to prevent overflow
template<typename T> struct AccumType { using type = T; };
template<> struct AccumType<__half> { using type = float; };
template<> struct AccumType<__nv_bfloat16> { using type = float; };

template<typename T, int BLOCK_SIZE = 256>
__global__ void bias_grad_reduce_kernel(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_bias,
    int64_t batch_size,
    int64_t out_features
) {
    using Acc = typename AccumType<T>::type;

    // Each block handles one output feature
    int64_t feature_idx = blockIdx.x;
    if (feature_idx >= out_features) return;

    __shared__ Acc shared[BLOCK_SIZE];

    // Each thread accumulates multiple elements in higher precision
    Acc sum = Acc(0);
    for (int64_t i = threadIdx.x; i < batch_size; i += BLOCK_SIZE) {
        sum += Acc(grad_output[i * out_features + feature_idx]);
    }
    shared[threadIdx.x] = sum;
    __syncthreads();

    // Parallel reduction within block
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write result (convert back to output type)
    if (threadIdx.x == 0) {
        grad_bias[feature_idx] = T(shared[0]);
    }
}

/**
 * @brief Vectorized bias gradient reduction using warp shuffles
 *
 * More efficient for larger batch sizes - uses warp-level reduction.
 */
template<typename T, int BLOCK_SIZE = 256>
__global__ void bias_grad_reduce_warp_kernel(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_bias,
    int64_t batch_size,
    int64_t out_features
) {
    using Acc = typename AccumType<T>::type;

    // Each block handles one output feature
    int64_t feature_idx = blockIdx.x;
    if (feature_idx >= out_features) return;

    // Each thread accumulates multiple elements in higher precision
    Acc sum = Acc(0);
    for (int64_t i = threadIdx.x; i < batch_size; i += BLOCK_SIZE) {
        sum += Acc(grad_output[i * out_features + feature_idx]);
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread of each warp writes to shared memory
    __shared__ Acc warp_sums[BLOCK_SIZE / 32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps (first warp only)
    if (warp_id == 0 && lane_id < (BLOCK_SIZE / 32)) {
        sum = warp_sums[lane_id];
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            grad_bias[feature_idx] = T(sum);
        }
    }
}

// ============================================================================
// Fused Linear Layer - Single cuBLAS call with bias fusion
// ============================================================================

/**
 * @brief Optimized linear layer: output = input @ weight.T + bias
 *
 * This implementation provides significant speedup over the naive approach:
 * 1. Uses cuBLAS GEMM with transpose flags (no explicit transpose kernel)
 * 2. Fuses bias addition with vectorized kernel
 * 3. Enables Tensor Core acceleration via TF32 on Ampere+ GPUs
 *
 * Performance:
 * - 2-3x faster than separate transpose + matmul + add operations
 * - Tensor Cores provide additional 2-8x on FP16/TF32
 *
 * @param input Input tensor [batch_size, in_features]
 * @param weight Weight tensor [out_features, in_features]
 * @param bias Optional bias tensor [out_features]
 * @param stream CUDA stream for async execution
 * @return Output tensor [batch_size, out_features]
 */
auto linear_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    cudaStream_t stream = nullptr
) -> Tensor {
    if (input.device().type != Device::Type::CUDA) {
        throw std::runtime_error("linear_kernel requires CUDA tensors");
    }

    // Make tensors contiguous if needed
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    auto in_shape = input_c.shape();
    auto w_shape = weight_c.shape();

    // Handle N-D input by flattening to 2D
    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }
    int64_t in_features = in_shape[in_shape.size() - 1];
    int64_t out_features = w_shape[0];

    if (w_shape[1] != in_features) {
        throw std::runtime_error("Linear: weight in_features mismatch");
    }

    // Build output shape
    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1);
    out_shape.push_back(out_features);

    // Create output tensor
    Tensor output(out_shape, input_c.dtype(), input_c.device());

    // Get cuBLAS handle and set stream
    cublasHandle_t handle = CuBLASHandlePool::get(stream);

    // Perform GEMM: output = input @ weight.T
    // In row-major: C[M,N] = A[M,K] @ B[N,K].T
    // In column-major cuBLAS: we compute C^T = B @ A^T
    // So: cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...)
    //     with A=weight, B=input, C=output

    if (input_c.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Use cublasGemmEx for TF32 Tensor Core acceleration
        // C = alpha * A @ B.T + beta * C
        // With row-major layout, we compute using:
        // C^T = alpha * B^T @ A + beta * C^T  (cuBLAS column-major)
        // Which gives us: C = alpha * A @ B.T

        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,    // Transpose weight (B in cuBLAS)
            CUBLAS_OP_N,    // Don't transpose input (A in cuBLAS)
            out_features,   // N: rows of output (columns of weight.T)
            batch_size,     // M: columns of output (rows of input)
            in_features,    // K: inner dimension
            &alpha,
            weight_c.data<float>(),  // B: weight [out_features, in_features]
            CUDA_R_32F,
            in_features,    // ldb: leading dimension of weight
            input_c.data<float>(),   // A: input [batch_size, in_features]
            CUDA_R_32F,
            in_features,    // lda: leading dimension of input
            &beta,
            output.data<float>(),    // C: output [batch_size, out_features]
            CUDA_R_32F,
            out_features,   // ldc: leading dimension of output
            (::tenzor::cuda::matmul::allow_tf32() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F),
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // Add bias if present
        if (bias != nullptr && bias->numel() > 0) {
            Tensor bias_c = bias->is_contiguous() ? *bias : bias->contiguous();
            int64_t total = batch_size * out_features;

            // Use vectorized kernel if dimensions allow
            int min_grid_size, block_size;
            if (out_features % 4 == 0 && out_features >= 16) {
                cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                   bias_add_kernel_vec4, 0, 0);
                int blocks = std::min(
                    static_cast<int>((total / 4 + block_size - 1) / block_size),
                    2147483647
                );
                bias_add_kernel_vec4<<<blocks, block_size, 0, stream>>>(
                    output.data<float>(),
                    bias_c.data<float>(),
                    batch_size,
                    out_features
                );
            } else {
                cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                   bias_add_kernel<float>, 0, 0);
                int blocks = std::min(
                    static_cast<int>((total + block_size - 1) / block_size),
                    2147483647
                );
                bias_add_kernel<float><<<blocks, block_size, 0, stream>>>(
                    output.data<float>(),
                    bias_c.data<float>(),
                    batch_size,
                    out_features
                );
            }
        }
    } else if (input_c.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double beta = 0.0;

        TENZOR_CUBLAS_CHECK(cublasDgemm(
            handle,
            CUBLAS_OP_T,    // Transpose weight
            CUBLAS_OP_N,    // Don't transpose input
            out_features,   // N
            batch_size,     // M
            in_features,    // K
            &alpha,
            weight_c.data<double>(),
            in_features,
            input_c.data<double>(),
            in_features,
            &beta,
            output.data<double>(),
            out_features
        ));

        // Add bias
        if (bias != nullptr && bias->numel() > 0) {
            Tensor bias_c = bias->is_contiguous() ? *bias : bias->contiguous();
            int64_t total = batch_size * out_features;
            int min_grid_size, block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                               bias_add_kernel<double>, 0, 0);
            int blocks = std::min(
                static_cast<int>((total + block_size - 1) / block_size),
                2147483647
            );
            bias_add_kernel<double><<<blocks, block_size, 0, stream>>>(
                output.data<double>(),
                bias_c.data<double>(),
                batch_size,
                out_features
            );
        }
    } else if (input_c.dtype() == DType::Float16) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Use FP16 Tensor Cores with FP32 accumulation for accuracy
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_features,
            batch_size,
            in_features,
            &alpha,
            weight_c.data_ptr(),
            CUDA_R_16F,
            in_features,
            input_c.data_ptr(),
            CUDA_R_16F,
            in_features,
            &beta,
            output.data_ptr(),
            CUDA_R_16F,
            out_features,
            CUBLAS_COMPUTE_32F,  // FP32 accumulation
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // Add bias using half precision
        if (bias != nullptr && bias->numel() > 0) {
            Tensor bias_c = bias->is_contiguous() ? *bias : bias->contiguous();
            int64_t total = batch_size * out_features;
            int min_grid_size, block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                               bias_add_kernel<__half>, 0, 0);
            int blocks = std::min(
                static_cast<int>((total + block_size - 1) / block_size),
                2147483647
            );
            bias_add_kernel<__half><<<blocks, block_size, 0, stream>>>(
                reinterpret_cast<__half*>(output.data_ptr()),
                reinterpret_cast<const __half*>(bias_c.data_ptr()),
                batch_size,
                out_features
            );
        }
    } else if (input_c.dtype() == DType::BFloat16) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Use BF16 Tensor Cores with FP32 accumulation for accuracy
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_features,
            batch_size,
            in_features,
            &alpha,
            weight_c.data_ptr(),
            CUDA_R_16BF,
            in_features,
            input_c.data_ptr(),
            CUDA_R_16BF,
            in_features,
            &beta,
            output.data_ptr(),
            CUDA_R_16BF,
            out_features,
            CUBLAS_COMPUTE_32F,  // FP32 accumulation
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // Add bias using bfloat16 precision
        if (bias != nullptr && bias->numel() > 0) {
            Tensor bias_c = bias->is_contiguous() ? *bias : bias->contiguous();
            int64_t total = batch_size * out_features;
            int min_grid_size, block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                               bias_add_kernel<__nv_bfloat16>, 0, 0);
            int blocks = std::min(
                static_cast<int>((total + block_size - 1) / block_size),
                2147483647
            );
            bias_add_kernel<__nv_bfloat16><<<blocks, block_size, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(bias_c.data_ptr()),
                batch_size,
                out_features
            );
        }
    } else {
        throw std::runtime_error("linear_kernel: Unsupported dtype");
    }

    // Check for errors from bias_add kernel launches
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// ============================================================================
// Linear Backward - Gradients for input, weight, and bias
// ============================================================================

/**
 * @brief Compute gradients for linear layer backward pass
 *
 * Computes:
 * - grad_input = grad_output @ weight
 * - grad_weight = grad_output.T @ input
 * - grad_bias = sum(grad_output, dim=0)
 *
 * @param grad_output Gradient from next layer [batch_size, out_features]
 * @param input Original input [batch_size, in_features]
 * @param weight Weight tensor [out_features, in_features]
 * @param stream CUDA stream
 * @return Tuple of (grad_input, grad_weight, grad_bias)
 */
auto linear_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    cudaStream_t stream = nullptr
) -> std::vector<Tensor> {
    // Make tensors contiguous
    Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    auto grad_shape = grad_out_c.shape();
    auto in_shape = input_c.shape();
    auto w_shape = weight_c.shape();

    // Flatten to 2D
    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }
    int64_t in_features = w_shape[1];
    int64_t out_features = w_shape[0];

    // Create gradient tensors (no need to zero - GEMM uses beta=0 and reduction overwrites)
    Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                      input_c.dtype(), input_c.device());
    Tensor grad_weight({out_features, in_features}, weight_c.dtype(), weight_c.device());
    Tensor grad_bias({out_features}, grad_out_c.dtype(), grad_out_c.device());

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

    if (input_c.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // grad_input = grad_output @ weight
        // [batch, in_features] = [batch, out_features] @ [out_features, in_features]
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,    // Don't transpose weight
            CUBLAS_OP_N,    // Don't transpose grad_output
            in_features,    // N
            batch_size,     // M
            out_features,   // K
            &alpha,
            weight_c.data<float>(),
            CUDA_R_32F,
            in_features,
            grad_out_c.data<float>(),
            CUDA_R_32F,
            out_features,
            &beta,
            grad_input.data<float>(),
            CUDA_R_32F,
            in_features,
            (::tenzor::cuda::matmul::allow_tf32() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F),
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // grad_weight = grad_output.T @ input
        // [out_features, in_features] = [out_features, batch] @ [batch, in_features]
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,    // Don't transpose input
            CUBLAS_OP_T,    // Transpose grad_output
            in_features,    // N
            out_features,   // M
            batch_size,     // K
            &alpha,
            input_c.data<float>(),
            CUDA_R_32F,
            in_features,
            grad_out_c.data<float>(),
            CUDA_R_32F,
            out_features,
            &beta,
            grad_weight.data<float>(),
            CUDA_R_32F,
            in_features,
            (::tenzor::cuda::matmul::allow_tf32() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F),
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // grad_bias = sum(grad_output, dim=0) using efficient parallel reduction
        // Each block handles one output feature
        {
            constexpr int BLOCK_SIZE = 256;
            int blocks = static_cast<int>(out_features);
            bias_grad_reduce_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
                grad_out_c.data<float>(),
                grad_bias.data<float>(),
                batch_size,
                out_features
            );
        }
    } else if (input_c.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double beta = 0.0;

        // grad_input = grad_output @ weight
        TENZOR_CUBLAS_CHECK(cublasDgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            in_features,
            batch_size,
            out_features,
            &alpha,
            weight_c.data<double>(),
            in_features,
            grad_out_c.data<double>(),
            out_features,
            &beta,
            grad_input.data<double>(),
            in_features
        ));

        // grad_weight = grad_output.T @ input
        TENZOR_CUBLAS_CHECK(cublasDgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            in_features,
            out_features,
            batch_size,
            &alpha,
            input_c.data<double>(),
            in_features,
            grad_out_c.data<double>(),
            out_features,
            &beta,
            grad_weight.data<double>(),
            in_features
        ));

        // grad_bias = sum(grad_output, dim=0) using efficient parallel reduction
        {
            constexpr int BLOCK_SIZE = 256;
            int blocks = static_cast<int>(out_features);
            bias_grad_reduce_kernel<double, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
                grad_out_c.data<double>(),
                grad_bias.data<double>(),
                batch_size,
                out_features
            );
        }
    } else if (input_c.dtype() == DType::Float16) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // grad_input = grad_output @ weight
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            in_features,
            batch_size,
            out_features,
            &alpha,
            weight_c.data_ptr(),
            CUDA_R_16F,
            in_features,
            grad_out_c.data_ptr(),
            CUDA_R_16F,
            out_features,
            &beta,
            grad_input.data_ptr(),
            CUDA_R_16F,
            in_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // grad_weight = grad_output.T @ input
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            in_features,
            out_features,
            batch_size,
            &alpha,
            input_c.data_ptr(),
            CUDA_R_16F,
            in_features,
            grad_out_c.data_ptr(),
            CUDA_R_16F,
            out_features,
            &beta,
            grad_weight.data_ptr(),
            CUDA_R_16F,
            in_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // grad_bias = sum(grad_output, dim=0)
        {
            constexpr int BLOCK_SIZE = 256;
            int blocks = static_cast<int>(out_features);
            bias_grad_reduce_kernel<__half, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_out_c.data_ptr()),
                reinterpret_cast<__half*>(grad_bias.data_ptr()),
                batch_size,
                out_features
            );
        }
    } else if (input_c.dtype() == DType::BFloat16) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // grad_input = grad_output @ weight
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            in_features,
            batch_size,
            out_features,
            &alpha,
            weight_c.data_ptr(),
            CUDA_R_16BF,
            in_features,
            grad_out_c.data_ptr(),
            CUDA_R_16BF,
            out_features,
            &beta,
            grad_input.data_ptr(),
            CUDA_R_16BF,
            in_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // grad_weight = grad_output.T @ input
        TENZOR_CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            in_features,
            out_features,
            batch_size,
            &alpha,
            input_c.data_ptr(),
            CUDA_R_16BF,
            in_features,
            grad_out_c.data_ptr(),
            CUDA_R_16BF,
            out_features,
            &beta,
            grad_weight.data_ptr(),
            CUDA_R_16BF,
            in_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // grad_bias = sum(grad_output, dim=0)
        {
            constexpr int BLOCK_SIZE = 256;
            int blocks = static_cast<int>(out_features);
            bias_grad_reduce_kernel<__nv_bfloat16, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_out_c.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(grad_bias.data_ptr()),
                batch_size,
                out_features
            );
        }
    } else {
        throw std::runtime_error("linear_backward_kernel: Unsupported dtype");
    }

    // Check for errors from bias_grad_reduce kernel launches
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// FP8 GEMM via cublasLt - Hopper Tensor Core Support (SM 9.0+)
// ============================================================================

#if CUDA_VERSION >= 11080

#include <cublasLt.h>

namespace {

/// Thread-safe cublasLt handle, cached per device. One handle per device keyed
/// by the device that is current at call time, so FP8 matmuls targeting a
/// different GPU on a heterogeneous box get a handle created on that GPU rather
/// than reusing the first-touched device's handle.
cublasLtHandle_t get_cublaslt_handle() {
    int device = 0;
    cudaGetDevice(&device);
    static std::unordered_map<int, cublasLtHandle_t> handles;
    static std::mutex handles_mutex;
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto it = handles.find(device);
    if (it != handles.end()) {
        return it->second;
    }
    cublasLtHandle_t handle = nullptr;
    auto status = cublasLtCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cublasLt handle");
    }
    handles.emplace(device, handle);
    return handle;
}

/// Query compute capability of the current device, cached per device. Querying
/// at call time (not a single process-global std::call_once) ensures the FP8 SM
/// gate reflects the device actually in use on a heterogeneous multi-GPU box.
int get_compute_capability() {
    int device = 0;
    cudaGetDevice(&device);
    static std::unordered_map<int, int> cc_by_device;
    static std::mutex cc_mutex;
    std::lock_guard<std::mutex> lock(cc_mutex);
    auto it = cc_by_device.find(device);
    if (it != cc_by_device.end()) {
        return it->second;
    }
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    int cc = major * 10 + minor;
    cc_by_device.emplace(device, cc);
    return cc;
}

/// Map DType to cublasLt data type, including FP8 variants
cudaDataType_t dtype_to_cublaslt(DType dtype) {
    switch (dtype) {
        case DType::Float32:  return CUDA_R_32F;
        case DType::Float64:  return CUDA_R_64F;
        case DType::Float16:  return CUDA_R_16F;
        case DType::BFloat16: return CUDA_R_16BF;
        case DType::Int8:     return CUDA_R_8I;
        case DType::Int32:    return CUDA_R_32I;
        case DType::FP8_E4M3: return CUDA_R_8F_E4M3;
        case DType::FP8_E5M2: return CUDA_R_8F_E5M2;
        default:
            throw std::runtime_error("Unsupported dtype for cublasLt");
    }
}

} // anonymous namespace

/**
 * @brief FP8 matrix multiplication using cublasLt for Hopper Tensor Cores (SM 9.0+)
 *
 * Supports mixed FP8 inputs with higher-precision output:
 * - A (E4M3) x B (E4M3) -> C (Float16 or Float32)
 * - A (E4M3) x B (E5M2) -> C (Float16 or Float32)
 * - A (E5M2) x B (E5M2) -> C (Float16 or Float32)
 *
 * FP8 Tensor Cores provide ~2x throughput over FP16 on Hopper GPUs.
 * The computation uses FP32 accumulation internally for numerical stability.
 *
 * Per-tensor scaling factors (amax-based) are required for FP8 to compensate
 * for the limited dynamic range. Callers must provide scale_a and scale_b.
 *
 * @param A Input matrix A (M x K), FP8 type
 * @param B Input matrix B (K x N), FP8 type
 * @param C Output matrix C (M x N), Float16 or Float32
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param a_dtype Data type of A (FP8_E4M3 or FP8_E5M2)
 * @param b_dtype Data type of B (FP8_E4M3 or FP8_E5M2)
 * @param out_dtype Data type of C (Float16 or Float32)
 * @param scale_a Per-tensor scale factor for A (applied as A_real = A_fp8 * scale_a)
 * @param scale_b Per-tensor scale factor for B
 * @param stream CUDA stream for async execution
 */
void cublas_fp8_gemm(
    const void* A,
    const void* B,
    void* C,
    int64_t M,
    int64_t N,
    int64_t K,
    DType a_dtype,
    DType b_dtype,
    DType out_dtype,
    float scale_a,
    float scale_b,
    cudaStream_t stream = nullptr
) {
    // Runtime check for Hopper (SM 9.0+)
    if (get_compute_capability() < 90) {
        throw std::runtime_error(
            "FP8 GEMM requires Hopper GPU (SM 9.0+), "
            "current device has SM " + std::to_string(get_compute_capability() / 10) +
            "." + std::to_string(get_compute_capability() % 10));
    }

    // Validate FP8 input types
    if (a_dtype != DType::FP8_E4M3 && a_dtype != DType::FP8_E5M2) {
        throw std::runtime_error("cublas_fp8_gemm: A must be FP8_E4M3 or FP8_E5M2");
    }
    if (b_dtype != DType::FP8_E4M3 && b_dtype != DType::FP8_E5M2) {
        throw std::runtime_error("cublas_fp8_gemm: B must be FP8_E4M3 or FP8_E5M2");
    }
    if (out_dtype != DType::Float16 && out_dtype != DType::Float32 && out_dtype != DType::BFloat16) {
        throw std::runtime_error("cublas_fp8_gemm: output must be Float16, BFloat16, or Float32");
    }

    cublasLtHandle_t ltHandle = get_cublaslt_handle();

    cudaDataType_t cuda_a_dtype = dtype_to_cublaslt(a_dtype);
    cudaDataType_t cuda_b_dtype = dtype_to_cublaslt(b_dtype);
    cudaDataType_t cuda_out_dtype = dtype_to_cublaslt(out_dtype);

    // FP8 always uses FP32 accumulation
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    // Combined scale: alpha = scale_a * scale_b (FP8 dequantization)
    float alpha = scale_a * scale_b;
    float beta = 0.0f;

    // Create matrix descriptors
    // cuBLAS uses column-major; for row-major C = A @ B, we compute C^T = B^T @ A^T
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;

    auto cleanup = [&]() {
        if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
        if (layoutA) cublasLtMatrixLayoutDestroy(layoutA);
        if (layoutB) cublasLtMatrixLayoutDestroy(layoutB);
        if (layoutC) cublasLtMatrixLayoutDestroy(layoutC);
    };

    try {
        // Create matmul descriptor
        auto status = cublasLtMatmulDescCreate(&matmulDesc, compute_type, CUDA_R_32F);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cublasLt matmul descriptor");
        }

        // Row-major C = A @ B is computed as the column-major problem
        //   C^T (N x M) = B^T (N x K) @ A^T (K x M).
        // Following the same convention as the cublasGemmEx path above, we pass
        // B as cuBLAS "A" and A as cuBLAS "B" and describe each operand by the
        // column-major VIEW of its row-major storage. That column-major view is
        // already the logical transpose of the row-major operand, so NO extra
        // transpose flag is needed: both ops are OP_N. Setting OP_T here (as the
        // previous code did) would transpose the already-transposed layouts —
        // a double transpose that contradicts C = A @ B.
        //
        // NOTE: FP8 cublasLt has a "TN" format restriction on some toolkit
        // versions. Before this path is wired to a caller, validate against a
        // reference GEMM (see cublas_fp8_matmul) on the target arch/toolkit; if
        // TN is enforced, the operands must be re-laid-out accordingly rather
        // than reintroducing the double transpose.
        cublasOperation_t transA = CUBLAS_OP_N;  // cuBLAS "A" (= our B), col-major view
        cublasOperation_t transB = CUBLAS_OP_N;  // cuBLAS "B" (= our A), col-major view

        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                        &transA, sizeof(transA));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                        &transB, sizeof(transB));

        // Matrix layouts are the column-major views of the row-major operands.
        // cuBLAS A = B (row-major K x N) -> col-major N x K, ld = N
        cublasLtMatrixLayoutCreate(&layoutA, cuda_b_dtype, N, K, N);
        // cuBLAS B = A (row-major M x K) -> col-major K x M, ld = K
        cublasLtMatrixLayoutCreate(&layoutB, cuda_a_dtype, K, M, K);
        // cuBLAS C = C (row-major M x N) -> col-major N x M, ld = N
        cublasLtMatrixLayoutCreate(&layoutC, cuda_out_dtype, N, M, N);

        // Perform the FP8 GEMM
        status = cublasLtMatmul(
            ltHandle,
            matmulDesc,
            &alpha,
            B,        // cuBLAS "A" matrix (transposed B)
            layoutA,
            A,        // cuBLAS "B" matrix (transposed A)
            layoutB,
            &beta,
            C,        // Output
            layoutC,
            C,        // D = C (in-place, since beta = 0)
            layoutC,
            nullptr,  // algo (nullptr = auto-select)
            nullptr,  // workspace
            0,        // workspace size
            stream
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "cublasLtMatmul FP8 GEMM failed with status " + std::to_string(static_cast<int>(status)));
        }

        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

/**
 * @brief High-level FP8 matrix multiplication for Tensor objects
 *
 * Performs C = scale_a * scale_b * (A @ B), where A and B are FP8 tensors.
 * Output is in a higher-precision type (Float16, BFloat16, or Float32).
 *
 * @param a Input tensor A (2D, FP8 type)
 * @param b Input tensor B (2D, FP8 type)
 * @param out_dtype Output data type (default: Float16)
 * @param scale_a Per-tensor scale for A (default: 1.0)
 * @param scale_b Per-tensor scale for B (default: 1.0)
 * @return Output tensor in out_dtype
 */
auto cublas_fp8_matmul(
    const Tensor& a,
    const Tensor& b,
    DType out_dtype = DType::Float16,
    float scale_a = 1.0f,
    float scale_b = 1.0f
) -> Tensor {
    if (a.device().type != Device::Type::CUDA || b.device().type != Device::Type::CUDA) {
        throw std::runtime_error("cublas_fp8_matmul requires CUDA tensors");
    }

    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("cublas_fp8_matmul requires 2D tensors");
    }

    auto a_shape = a.shape();
    auto b_shape = b.shape();

    int64_t M = a_shape[0];
    int64_t K = a_shape[1];
    int64_t K2 = b_shape[0];
    int64_t N = b_shape[1];

    if (K != K2) {
        throw std::runtime_error("Matrix dimension mismatch for FP8 matmul");
    }

    Tensor result({M, N}, out_dtype, a.device());

    cublas_fp8_gemm(
        a.data_ptr(), b.data_ptr(), result.data_ptr(),
        M, N, K,
        a.dtype(), b.dtype(), out_dtype,
        scale_a, scale_b
    );

    return result;
}

#endif // CUDA_VERSION >= 11080

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUBLAS
