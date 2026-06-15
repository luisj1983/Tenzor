/**
 * @file quantized_linear.cu
 * @brief CUDA kernels for quantized linear operations
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "../cuda_common.cuh"

// Include WMMA for Tensor Core support (Turing+)
#if __CUDA_ARCH__ >= 750 || !defined(__CUDA_ARCH__)
#include <mma.h>
#endif

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

/**
 * @brief CUDA kernel for quantized matrix multiplication using INT8 Tensor Cores.
 *
 * Optimized for NVIDIA GPUs with INT8 Tensor Core support (Turing+).
 */
__global__ void quantized_linear_cuda_kernel(
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
    // Thread block computes one output element
    int64_t b = blockIdx.y;
    int64_t o = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || o >= out_features) return;

    // int64 accumulator: each int8*int8 product fits in int32, but summed over
    // in_features the running total can exceed INT32_MAX for very wide layers
    // (K >~ 131072), so accumulate in int64 to avoid silent wraparound.
    int64_t acc = 0;
    // Running sums of the quantized operands, needed for the asymmetric
    // (non-zero zero-point) dequantization correction below.
    int64_t sum_i = 0;  // sum(q_i)
    int64_t sum_w = 0;  // sum(q_w)

    const int8_t* input_row = input + b * in_features;
    const int8_t* weight_row = weight + o * in_features;

    // Vectorized loading with int4 (16 bytes = 16 int8 values)
    const int VEC_SIZE = 16;
    int64_t vec_steps = in_features / VEC_SIZE;

    for (int64_t v = 0; v < vec_steps; ++v) {
        int4 input_vec = reinterpret_cast<const int4*>(input_row)[v];
        int4 weight_vec = reinterpret_cast<const int4*>(weight_row)[v];

        int8_t* input_bytes = reinterpret_cast<int8_t*>(&input_vec);
        int8_t* weight_bytes = reinterpret_cast<int8_t*>(&weight_vec);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            int32_t qi = static_cast<int32_t>(input_bytes[i]);
            int32_t qw = static_cast<int32_t>(weight_bytes[i]);
            acc += qi * qw;
            sum_i += qi;
            sum_w += qw;
        }
    }

    // Process remaining elements
    for (int64_t i = vec_steps * VEC_SIZE; i < in_features; ++i) {
        int32_t qi = static_cast<int32_t>(input_row[i]);
        int32_t qw = static_cast<int32_t>(weight_row[i]);
        acc += qi * qw;
        sum_i += qi;
        sum_w += qw;
    }

    // Asymmetric zero-point correction (computed in int64 to avoid overflow):
    //   sum(q_i*q_w) - zp_w*sum(q_i) - zp_i*sum(q_w) + N*zp_i*zp_w
    int64_t corrected = static_cast<int64_t>(acc)
                      - static_cast<int64_t>(weight_zp) * sum_i
                      - static_cast<int64_t>(input_zp) * sum_w
                      + static_cast<int64_t>(input_zp) * static_cast<int64_t>(weight_zp) * in_features;

    // Dequantize and add bias
    float result = static_cast<float>(corrected) * combined_scale;
    if (bias != nullptr) {
        result += bias[o];
    }

    output[b * out_features + o] = result;
}

/**
 * @brief Host function to launch quantized linear CUDA kernel.
 */
auto quantized_linear_cuda(
    const int8_t* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float input_scale,
    float weight_scale,
    float output_scale,
    int32_t input_zp,
    int32_t weight_zp,
    cudaStream_t stream
) -> void {
    float combined_scale = input_scale * weight_scale / output_scale;

    // Launch configuration
    const int THREADS = 256;
    dim3 blocks((out_features + THREADS - 1) / THREADS, batch_size);
    dim3 threads(THREADS);

    quantized_linear_cuda_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_features, out_features,
        combined_scale, input_zp, weight_zp
    );
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

/**
 * @brief Optimized kernel using CUDA Tensor Cores (WMMA API).
 *
 * Requires Turing+ architecture. Uses WMMA (Warp Matrix Multiply Accumulate)
 * for maximum INT8 performance.
 */
#if __CUDA_ARCH__ >= 750 || !defined(__CUDA_ARCH__)

__global__ void quantized_linear_wmma_kernel(
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
    // WMMA tile sizes for INT8 on Turing+: M=8, N=32, K=16
    const int M = 8;
    const int N = 32;
    const int K = 16;

    // Warp index
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;

    // Compute which output tile this warp handles
    int batch_tile = warp_id / ((out_features + N - 1) / N);
    int out_tile = warp_id % ((out_features + N - 1) / N);

    if (batch_tile >= (batch_size + M - 1) / M) return;

    // Declare WMMA fragments for INT8 (use signed char for INT8)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, signed char, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, signed char, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, int32_t> acc_frag;

    // Initialize accumulator to zero
    nvcuda::wmma::fill_fragment(acc_frag, 0);

    // Loop over K dimension
    for (int k_tile = 0; k_tile < (in_features + K - 1) / K; ++k_tile) {
        // Load input tile (A matrix)
        int batch_offset = batch_tile * M;
        int k_offset = k_tile * K;

        if (batch_offset < batch_size && k_offset < in_features) {
            nvcuda::wmma::load_matrix_sync(a_frag,
                                  input + batch_offset * in_features + k_offset,
                                  in_features);

            // Load weight tile (B matrix)
            int out_offset = out_tile * N;
            if (out_offset < out_features) {
                nvcuda::wmma::load_matrix_sync(b_frag,
                                      weight + out_offset * in_features + k_offset,
                                      in_features);

                // Perform matrix multiplication
                nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }
    }

    // Store result
    int batch_offset = batch_tile * M;
    int out_offset = out_tile * N;

    if (batch_offset < batch_size && out_offset < out_features) {
        // Convert accumulator to float, apply scaling and bias
        int32_t acc_values[M * N];
        nvcuda::wmma::store_matrix_sync(acc_values, acc_frag, N, nvcuda::wmma::mem_row_major);

        for (int i = 0; i < M && batch_offset + i < batch_size; ++i) {
            for (int j = 0; j < N && out_offset + j < out_features; ++j) {
                int32_t acc = acc_values[i * N + j];

                // Zero point correction
                acc -= input_zp * weight_zp * in_features;

                // Dequantize and add bias
                float result = static_cast<float>(acc) * combined_scale;
                if (bias != nullptr) {
                    result += bias[out_offset + j];
                }

                output[(batch_offset + i) * out_features + (out_offset + j)] = result;
            }
        }
    }
}

#endif // __CUDA_ARCH__ >= 750

} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor
