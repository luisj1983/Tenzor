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

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t acc = 0;

            for (int64_t k = 0; k < K; ++k) {
                // Unpack INT4 weight on the fly
                int64_t linear_idx = k * N + n;
                int64_t byte_idx = linear_idx / 2;
                uint8_t packed = weights[byte_idx];
                int8_t w;
                if (linear_idx % 2 == 0) {
                    w = static_cast<int8_t>(packed & 0x0F);
                    if (w & 0x08) w |= static_cast<int8_t>(0xF0);
                } else {
                    w = static_cast<int8_t>((packed >> 4) & 0x0F);
                    if (w & 0x08) w |= static_cast<int8_t>(0xF0);
                }

                acc += static_cast<int32_t>(act[m * K + k]) * static_cast<int32_t>(w);
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
