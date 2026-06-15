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
    int32_t weight_zp,
    bool use_vectorized
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

    // Vectorized int4 loads (16 bytes = 16 int8 values) require both row
    // pointers to be 16-byte aligned. A row pointer is `base + r*in_features`,
    // so its alignment depends on BOTH the base allocation alignment and on
    // `in_features` being a multiple of 16 (otherwise odd rows fall off the
    // 16-byte boundary and a hardware misaligned-address fault aborts the
    // kernel). The host computes this precondition once and passes it in via
    // `use_vectorized`; when it is false we take a fully scalar path that is
    // correct for every `in_features` value and every base alignment.
    int64_t i = 0;
    if (use_vectorized) {
        // Vectorized loading with int4 (16 bytes = 16 int8 values).
        const int VEC_SIZE = 16;
        int64_t vec_steps = in_features / VEC_SIZE;

        for (int64_t v = 0; v < vec_steps; ++v) {
            int4 input_vec = reinterpret_cast<const int4*>(input_row)[v];
            int4 weight_vec = reinterpret_cast<const int4*>(weight_row)[v];

            int8_t* input_bytes = reinterpret_cast<int8_t*>(&input_vec);
            int8_t* weight_bytes = reinterpret_cast<int8_t*>(&weight_vec);

            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                int32_t qi = static_cast<int32_t>(input_bytes[j]);
                int32_t qw = static_cast<int32_t>(weight_bytes[j]);
                acc += qi * qw;
                sum_i += qi;
                sum_w += qw;
            }
        }
        i = vec_steps * VEC_SIZE;
    }

    // Process remaining elements with scalar byte loads. When use_vectorized
    // is false this covers the entire row (i == 0); otherwise it handles the
    // in_features % 16 tail.
    for (; i < in_features; ++i) {
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

    // Precondition for the int4 (16-byte) vectorized load path. Each row
    // pointer is `base + r * in_features`. A 16-byte int4 load requires every
    // such pointer to be 16-byte aligned, which holds iff:
    //   (1) in_features % 16 == 0  -> r * in_features is a multiple of 16 for
    //       every row index r, so all rows share the base's alignment, AND
    //   (2) the base input/weight pointers are themselves 16-byte aligned.
    // CUDA device allocations are at least 256-byte aligned, so (2) normally
    // holds for the start of a tensor; we still check it explicitly so the
    // guard remains correct for any (e.g. offset/view) buffer. When the
    // precondition fails the kernel uses a scalar byte path that is correct
    // for all in_features values and all alignments.
    const bool use_vectorized =
        (in_features % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(weight) % 16 == 0);

    // Launch configuration
    const int THREADS = 256;
    dim3 blocks((out_features + THREADS - 1) / THREADS, batch_size);
    dim3 threads(THREADS);

    quantized_linear_cuda_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_features, out_features,
        combined_scale, input_zp, weight_zp, use_vectorized
    );
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// NOTE: An unused quantized_linear_wmma_kernel was removed here. It was dead code
// (never registered — the live kernel is quantized_linear_cuda at
// cuda_kernel_registry.cpp) and had an incomplete zero-point correction: it
// applied only acc -= input_zp*weight_zp*in_features (in int32, overflow-prone
// for large K) and omitted the -weight_zp*sum_i and -input_zp*sum_w cross terms
// that the live kernel and the CPU reference include — so it would have produced
// wrong results under non-zero zero-points if wired up. Reintroduce only with the
// full asymmetric zero-point correction (per-row sum_i / per-col sum_w computed in
// int64) and a parity test before registering.

} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor
