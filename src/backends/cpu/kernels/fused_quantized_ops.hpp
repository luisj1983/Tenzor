/**
 * @file fused_quantized_ops.hpp
 * @brief Fused quantized linear layer: INT8 x INT4 matmul + dequant + bias.
 *
 * Performs quantized matmul, rescales the int32 accumulator to float32
 * using quantization scales, adds bias, and outputs float32 result.
 * This avoids materializing the int32 intermediate.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "int4_utils.hpp"

namespace tenzor {
namespace cpu {

/**
 * @brief Fused quantized linear: output = dequant(act @ w_int4) + bias
 *
 * @param act          INT8 activations, row-major [M x K]
 * @param weights      Packed INT4 weights, row-major [K x N], 2 values per byte
 * @param bias         Float32 bias [N], may be nullptr
 * @param output       Float32 output [M x N]
 * @param M            Number of rows (batch size)
 * @param N            Number of output features
 * @param K            Number of input features
 * @param act_scale    Activation quantization scale
 * @param weight_scale Weight quantization scale
 */
inline void fused_qlinear_dequant(
    const int8_t* act,
    const uint8_t* weights,
    const float* bias,
    float* output,
    int64_t M, int64_t N, int64_t K,
    float act_scale, float weight_scale) {

    const float combined_scale = act_scale * weight_scale;
    const int64_t KN = K * N;
    if (KN <= 0 || M <= 0) {
        return;
    }

    // Pre-unpack the INT4 weight matrix into an int8 buffer [K x N] exactly
    // once, instead of doing the nibble extract + sign-extend on every (m,n,k).
    // Previously the unpack ran M times per element; now it runs a single pass,
    // and the inner dot product becomes a plain int8 multiply-accumulate the
    // compiler can auto-vectorize.
    std::vector<int8_t> w_unpacked(static_cast<size_t>(KN));
    {
        const int64_t pairs = KN / 2;
        for (int64_t p = 0; p < pairs; ++p) {
            int8_t lo, hi;
            unpack_int4(weights[p], lo, hi);
            w_unpacked[static_cast<size_t>(2 * p)]     = lo;
            w_unpacked[static_cast<size_t>(2 * p + 1)] = hi;
        }
        if (KN & 1) {
            // Trailing odd nibble lives in the low half of the last byte.
            int8_t lo, hi;
            unpack_int4(weights[pairs], lo, hi);
            w_unpacked[static_cast<size_t>(KN - 1)] = lo;
        }
    }

    const int8_t* w_data = w_unpacked.data();

    #pragma omp parallel for schedule(static) if(M * N * K > 4096)
    for (int64_t m = 0; m < M; ++m) {
        const int8_t* act_row = act + m * K;
        for (int64_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(act_row[k]) *
                       static_cast<int32_t>(w_data[k * N + n]);
            }

            float result = static_cast<float>(acc) * combined_scale;
            if (bias) {
                result += bias[n];
            }
            output[m * N + n] = result;
        }
    }
}

/**
 * @brief Fused quantized linear with ReLU activation.
 */
inline void fused_qlinear_dequant_relu(
    const int8_t* act,
    const uint8_t* weights,
    const float* bias,
    float* output,
    int64_t M, int64_t N, int64_t K,
    float act_scale, float weight_scale) {

    fused_qlinear_dequant(act, weights, bias, output, M, N, K, act_scale, weight_scale);
    for (int64_t i = 0; i < M * N; ++i) {
        if (output[i] < 0.0f) output[i] = 0.0f;
    }
}

} // namespace cpu
} // namespace tenzor
