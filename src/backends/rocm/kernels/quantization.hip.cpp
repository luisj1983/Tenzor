#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "tenzor/core/tensor.hpp"
#include <stdexcept>
#include <cstdint>

namespace tenzor {
namespace rocm {

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(error) \
            ); \
        } \
    } while(0)

// ==============================================================================
// Quantized Linear HIP Kernel (INT8 inputs, INT32 accumulation)
// ==============================================================================

__global__ void quantized_linear_kernel_hip(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float combined_scale,
    int32_t input_zp,
    int32_t weight_zp
) {
    // 2D grid: blockIdx.y = batch, blockIdx.x * blockDim.x + threadIdx.x = output feature
    int64_t b = blockIdx.y;
    int64_t o = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || o >= out_features) return;

    const int8_t* input_row = input + b * in_features;
    const int8_t* weight_row = weight + o * in_features;

    int32_t acc = 0;
    int32_t sum_x = 0;
    int32_t sum_w = 0;

    // Vectorized loading: process 16 int8 values at a time via int4 (16 bytes)
    constexpr int VEC_SIZE = 16;
    int64_t vec_steps = in_features / VEC_SIZE;

    for (int64_t v = 0; v < vec_steps; ++v) {
        int4 input_vec = reinterpret_cast<const int4*>(input_row)[v];
        int4 weight_vec = reinterpret_cast<const int4*>(weight_row)[v];

        const int8_t* input_bytes = reinterpret_cast<const int8_t*>(&input_vec);
        const int8_t* weight_bytes = reinterpret_cast<const int8_t*>(&weight_vec);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            acc += static_cast<int32_t>(input_bytes[i]) * static_cast<int32_t>(weight_bytes[i]);
            sum_x += static_cast<int32_t>(input_bytes[i]);
            sum_w += static_cast<int32_t>(weight_bytes[i]);
        }
    }

    // Remainder elements
    for (int64_t i = vec_steps * VEC_SIZE; i < in_features; ++i) {
        acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
        sum_x += static_cast<int32_t>(input_row[i]);
        sum_w += static_cast<int32_t>(weight_row[i]);
    }

    // Zero point correction:
    // Full expansion: sum((x_i - x_zp) * (w_j - w_zp))
    //   = sum(x_i * w_j) - x_zp * sum(w_j) - w_zp * sum(x_i) + x_zp * w_zp * K
    acc = acc - input_zp * sum_w - weight_zp * sum_x
          + input_zp * weight_zp * static_cast<int32_t>(in_features);

    // Dequantize to float and add bias
    float result = static_cast<float>(acc) * combined_scale;
    if (bias != nullptr) {
        result += bias[o];
    }

    output[b * out_features + o] = result;
}

/**
 * @brief Host wrapper for quantized linear (INT8 → INT32 accumulation → Float32 output).
 */
auto quantized_linear_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    float output_scale,
    int32_t output_zero_point,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t batch_size = input_shape[0];
    int64_t in_features = input_shape[1];
    int64_t out_features = weight_shape[0];

    Tensor output({batch_size, out_features}, DType::Float32, input.device());

    float combined_scale = input_scale * weight_scale / output_scale;

    const int THREADS = 256;
    dim3 blocks((out_features + THREADS - 1) / THREADS, batch_size);
    dim3 threads(THREADS);

    hipLaunchKernelGGL(quantized_linear_kernel_hip,
        blocks, threads, 0, stream,
        input.data<int8_t>(),
        weight.data<int8_t>(),
        bias ? bias->data<float>() : nullptr,
        output.data<float>(),
        batch_size, in_features, out_features,
        combined_scale, input_zero_point, weight_zero_point);

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Quantized Conv2d — im2col + rocBLAS GEMM (INT8 → INT32 accumulation)
// ==============================================================================

// Helper macros (scoped to this translation unit)
#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        throw std::runtime_error( \
            std::string("rocBLAS error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + " - status " + std::to_string(status) \
        ); \
    } \
} while(0)

/**
 * @brief im2col kernel for Int8 input: unfolds input patches into a column matrix.
 *
 * Converts NCHW input into a 2D matrix where each row is one convolution patch.
 * Output shape: (batch * out_h * out_w, in_channels * kernel_h * kernel_w)
 * Padded positions are filled with the input zero point (not 0) so that the
 * subsequent GEMM correctly accounts for zero-point subtraction.
 */
__global__ void im2col_int8_kernel(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col_buffer,
    int64_t batch,
    int64_t in_channels,
    int64_t h_in,
    int64_t w_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t h_out,
    int64_t w_out,
    int8_t input_zp
) {
    int64_t total = batch * h_out * w_out * in_channels * kernel_size * kernel_size;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode flat index to (b, oh, ow, ic, kh, kw)
    int64_t temp = idx;
    int64_t kw = temp % kernel_size; temp /= kernel_size;
    int64_t kh = temp % kernel_size; temp /= kernel_size;
    int64_t ic = temp % in_channels; temp /= in_channels;
    int64_t ow = temp % w_out; temp /= w_out;
    int64_t oh = temp % h_out; temp /= h_out;
    int64_t b = temp;

    int64_t ih = oh * stride - padding + kh * dilation;
    int64_t iw = ow * stride - padding + kw * dilation;

    // Output index in column matrix
    // Row: b * out_h * out_w + oh * out_w + ow
    // Col: ic * kernel_size * kernel_size + kh * kernel_size + kw
    int64_t col_cols = in_channels * kernel_size * kernel_size;
    int64_t out_row = b * h_out * w_out + oh * w_out + ow;
    int64_t out_col = ic * kernel_size * kernel_size + kh * kernel_size + kw;

    int8_t value = input_zp;  // Use zero point for out-of-bounds (padding)
    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
        value = input[((b * in_channels + ic) * h_in + ih) * w_in + iw];
    }
    col_buffer[out_row * col_cols + out_col] = value;
}

/**
 * @brief Dequantize Int32 GEMM output to Float32 and add bias.
 *
 * After the Int8 GEMM with Int32 accumulation, this kernel applies zero-point
 * correction, dequantization scaling, and bias addition.
 *
 * The GEMM computes: C[m][n] = sum_k (A[m][k] * B[k][n])
 * where A is im2col(input) with padding filled by input_zp, and B = weight^T.
 * So C already contains: sum_k (input_k * weight_k) with input_zp used for padding.
 *
 * The true quantized result needs:
 *   sum_k ((input_k - input_zp) * (weight_k - weight_zp))
 *   = C - input_zp * col_sum_w - weight_zp * row_sum_x + input_zp * weight_zp * K
 *
 * Since we set padding to input_zp in im2col, the GEMM result already correctly
 * handles the padding. We just need the standard zero-point correction.
 *
 * For simplicity and to avoid a separate reduction pass, we apply a per-output-channel
 * correction: precompute weight_col_sums on host and pass them in. This avoids the
 * triple-nested loop of the old kernel.
 */
// Per-row Int8 sum (rows × K) accumulated in Int32. Builds the zero-point
// correction terms (sum over each weight filter, and sum over each im2col row)
// entirely on-device — no host round-trip, no CPU fallback.
__global__ void quant_row_sum_kernel(
    const int8_t* __restrict__ mat,
    int32_t* __restrict__ row_sums,
    int64_t rows,
    int64_t K
) {
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int8_t* row_ptr = mat + r * K;
    int32_t s = 0;
    for (int64_t k = 0; k < K; ++k) {
        s += static_cast<int32_t>(row_ptr[k]);
    }
    row_sums[r] = s;
}

__global__ void dequantize_bias_kernel(
    const int32_t* __restrict__ gemm_output,
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int32_t* __restrict__ weight_col_sums,  // [out_channels] sum_k weight[oc][k]
    const int32_t* __restrict__ row_sums,         // [M] sum_k im2col_row[m][k]
    int64_t total,
    int64_t out_channels,
    int64_t spatial_size,   // h_out * w_out
    int64_t k_inner,        // in_channels * kernel_h * kernel_w
    int32_t input_zp,
    int32_t weight_zp,
    float combined_scale
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Map linear index to output channel for bias lookup
    // Output layout: (batch, out_channels, h_out, w_out)
    // GEMM output layout: (batch * h_out * w_out, out_channels) — row-major
    int64_t row = idx / out_channels;  // spatial position within batch
    int64_t oc = idx % out_channels;

    // Zero-point correction (mirrors quantized_linear_kernel_hip):
    //   sum_k (A - input_zp)(B - weight_zp)
    //     = C - input_zp*sum_k B[oc] - weight_zp*sum_k A[row] + input_zp*weight_zp*K
    // A = im2col(input) (padding filled with input_zp), B = weight[oc].
    int32_t acc = gemm_output[idx]
                  - input_zp * weight_col_sums[oc]
                  - weight_zp * row_sums[row]
                  + input_zp * weight_zp * static_cast<int32_t>(k_inner);

    float result = static_cast<float>(acc) * combined_scale;
    if (bias != nullptr) {
        result += bias[oc];
    }

    // Reorder from GEMM layout (batch*h_out*w_out, out_channels) to NCHW
    // batch_idx = row / spatial_size
    // spatial_idx = row % spatial_size
    int64_t batch_idx = row / spatial_size;
    int64_t spatial_idx = row % spatial_size;
    int64_t nchw_idx = (batch_idx * out_channels + oc) * spatial_size + spatial_idx;

    output[nchw_idx] = result;
}

/**
 * @brief Host wrapper for quantized conv2d using im2col + rocBLAS GEMM.
 *
 * Strategy:
 * 1. im2col: unfold input patches into column matrix (Int8), padding = input_zp
 * 2. rocblas_gemm_ex: Int8 x Int8 → Int32 accumulation (hardware-accelerated on MI-series GPUs)
 * 3. Dequantize kernel: Int32 → Float32 with zero-point correction and bias
 *
 * This replaces the previous naive triple-nested-loop kernel with a BLAS-based
 * approach that leverages AMD's hardware Int8 dot product instructions.
 */
auto quantized_conv2d_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    float output_scale,
    int32_t output_zero_point,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];

    auto weight_shape = weight.shape();
    int64_t out_channels = weight_shape[0];
    int64_t kernel_size = weight_shape[2];

    int64_t h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t w_out = (w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, input.device());

    float combined_scale = input_scale * weight_scale / output_scale;

    // Dimensions for im2col and GEMM
    int64_t M = batch * h_out * w_out;                          // number of patches
    int64_t K = in_channels * kernel_size * kernel_size;        // patch size
    int64_t N = out_channels;                                   // number of filters

    // Step 1: Allocate im2col buffer (Int8)
    int8_t* col_buffer = nullptr;
    HIP_CHECK(hipMalloc(&col_buffer, M * K * sizeof(int8_t)));

    // Launch im2col kernel
    int64_t im2col_total = batch * h_out * w_out * in_channels * kernel_size * kernel_size;
    const int THREADS = 256;
    int im2col_blocks = static_cast<int>((im2col_total + THREADS - 1) / THREADS);

    hipLaunchKernelGGL(im2col_int8_kernel,
        dim3(im2col_blocks), dim3(THREADS), 0, stream,
        input.data<int8_t>(),
        col_buffer,
        batch, in_channels, h_in, w_in,
        kernel_size, stride, padding, dilation,
        h_out, w_out,
        static_cast<int8_t>(input_zero_point));

    HIP_CHECK(hipGetLastError());

    // Step 1b: zero-point correction sums, computed on-device.
    //   weight_col_sums[oc] = sum_k weight[oc][k]   (rows = N = out_channels)
    //   row_sums[m]         = sum_k col_buffer[m][k] (rows = M = patches; padding=input_zp)
    int32_t* weight_col_sums = nullptr;
    int32_t* row_sums = nullptr;
    HIP_CHECK(hipMalloc(&weight_col_sums, N * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&row_sums, M * sizeof(int32_t)));

    hipLaunchKernelGGL(quant_row_sum_kernel,
        dim3(static_cast<int>((N + THREADS - 1) / THREADS)), dim3(THREADS), 0, stream,
        weight.data<int8_t>(), weight_col_sums, N, K);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(quant_row_sum_kernel,
        dim3(static_cast<int>((M + THREADS - 1) / THREADS)), dim3(THREADS), 0, stream,
        col_buffer, row_sums, M, K);
    HIP_CHECK(hipGetLastError());

    // Step 2: GEMM via rocblas_gemm_ex — Int8 × Int8 → Int32
    // col_buffer: (M, K) row-major — the im2col patches
    // weight: (N, K) row-major — each row is one filter, stored as (out_channels, in_channels*kh*kw)
    // output: (M, N) row-major
    //
    // rocBLAS uses column-major, so in column-major terms:
    //   C^T(N, M) = B^T(N, K) * A^T(K, M)
    //   C(M, N) = A(M, K) * B^T(K, N)  → rocBLAS: op(B)=T, op(A)=N
    int32_t* gemm_output = nullptr;
    HIP_CHECK(hipMalloc(&gemm_output, M * N * sizeof(int32_t)));

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    int32_t alpha_i32 = 1;
    int32_t beta_i32 = 0;

    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle,
        rocblas_operation_transpose,    // transpose weight (N,K) → (K,N) in col-major = (N,K) row-major
        rocblas_operation_none,         // col_buffer as-is
        N,                              // rows of op(B) = N
        M,                              // cols of op(A) = M
        K,                              // inner dimension
        &alpha_i32,
        weight.data<int8_t>(),          // B: weight (N, K) in row-major
        rocblas_datatype_i8_r,
        K,                              // ldb = K (row-major B has K columns)
        col_buffer,                     // A: col_buffer (M, K) in row-major
        rocblas_datatype_i8_r,
        K,                              // lda = K
        &beta_i32,
        gemm_output,                    // C: output (M, N) in row-major
        rocblas_datatype_i32_r,
        N,                              // ldc = N
        gemm_output,                    // D = C (in-place)
        rocblas_datatype_i32_r,
        N,                              // ldd = N
        rocblas_datatype_i32_r,         // compute type
        rocblas_gemm_algo_standard,
        0, 0                            // solution index, flags
    ));

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    // audit-8 GG.3: rocblas_gemm_ex is async on `stream` and reads col_buffer
    // as operand A.  Without this sync, hipFree(col_buffer) may execute (and
    // the page may be reused) before the GEMM completes, producing
    // nondeterministic INT8 garbage.  Mirrors the sibling gemm_output free.
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(col_buffer));

    // Step 3: Dequantize Int32 → Float32 and add bias
    int64_t total_output = M * N;
    int dequant_blocks = static_cast<int>((total_output + THREADS - 1) / THREADS);
    int64_t spatial_size = h_out * w_out;

    hipLaunchKernelGGL(dequantize_bias_kernel,
        dim3(dequant_blocks), dim3(THREADS), 0, stream,
        gemm_output,
        output.data<float>(),
        bias ? bias->data<float>() : nullptr,
        weight_col_sums,
        row_sums,
        total_output,
        out_channels,
        spatial_size,
        K,
        input_zero_point,
        weight_zero_point,
        combined_scale);

    HIP_CHECK(hipGetLastError());

    // Synchronize before freeing GEMM buffer (it's still in use by the kernel)
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(gemm_output));
    HIP_CHECK(hipFree(weight_col_sums));
    HIP_CHECK(hipFree(row_sums));

    return output;
}

} // namespace rocm
} // namespace tenzor
