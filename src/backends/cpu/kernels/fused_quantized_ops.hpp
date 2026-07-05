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
#include <stdexcept>
#include <vector>
#include "int4_utils.hpp"

namespace tenzor {
namespace cpu {

/**
 * @brief Fused quantized linear: output = dequant(act @ w_int4) + bias
 *
 * @param act          INT8 activations, row-major [M x K]
 * @param weights      Packed INT4 weights, out-feature-major [N x K], 2 values
 *                     per byte (channel n's K nibbles contiguous; QInt4x2 layout)
 * @param bias         Float32 bias [N], may be nullptr
 * @param output       Float32 output [M x N]
 * @param M            Number of rows (batch size)
 * @param N            Number of output features
 * @param K            Number of input features
 * @param act_scale    Activation quantization scale
 * @param weight_scale Weight quantization scale
 * @param act_zp       Activation (input) zero-point. For asymmetric activation
 *                     quantization this must be subtracted from every activation
 *                     before the dot product; the algebraically-equivalent
 *                     correction term -act_zp * sum_k(w[k,n]) is applied to the
 *                     int32 accumulator. INT4 weights are symmetric so no weight
 *                     zero-point term is needed. Defaults to 0 (symmetric).
 */
inline void fused_qlinear_dequant(
    const int8_t* act,
    const uint8_t* weights,
    const float* bias,
    float* output,
    int64_t M, int64_t N, int64_t K,
    float act_scale, float weight_scale,
    int32_t act_zp = 0) {

    const float combined_scale = act_scale * weight_scale;
    const int64_t KN = K * N;
    if (KN <= 0 || M <= 0) {
        return;
    }

    // QInt4x2 weights are packed per output row along the contraction dim K:
    // each row of K nibbles occupies (K + 1) / 2 bytes, and an odd K leaves the
    // high nibble of that row's last byte as zero padding (see the QInt4x2
    // packer in core/tensor.cpp). This kernel unpacks the weight buffer as a
    // single flat K*N nibble stream, which is only byte-aligned with that
    // per-row layout when K is even; an odd K desynchronises every row after the
    // first and silently returns wrong results. Reject it, matching
    // quantized_linear_int4_kernel.
    if (K % 2 != 0) {
        throw std::invalid_argument(
            "fused_qlinear_dequant: K (in_features) must be even for INT4 packing");
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

    // Asymmetric-activation zero-point correction:
    //   sum_k (act[k] - act_zp) * w[k,n]
    //     = sum_k act[k]*w[k,n]  -  act_zp * sum_k w[k,n]
    // so the int32 accumulator only needs the constant term act_zp*col_sum_w[n]
    // subtracted. Precompute the per-output-column weight sums once. INT4 weights
    // are symmetric (weight_zp == 0) so there is no symmetric weight-side term.
    // int64 to avoid overflow: act_zp (up to 127) * col_sum_w (up to ~8*K) can
    // exceed INT32_MAX for large K, and the dot accumulator overflows int32 at
    // K >~ 2.1M (INT4 [-8,7] × INT8 [-128,127]). Mirrors the int8 linear kernel.
    std::vector<int64_t> col_sum_w;
    if (act_zp != 0) {
        col_sum_w.assign(static_cast<size_t>(N), 0);
        for (int64_t n = 0; n < N; ++n) {
            const int8_t* w_row = w_data + n * K;
            int64_t s = 0;
            for (int64_t k = 0; k < K; ++k) {
                s += static_cast<int64_t>(w_row[k]);
            }
            col_sum_w[static_cast<size_t>(n)] = s;
        }
    }
    const int64_t* col_sum_w_ptr = col_sum_w.empty() ? nullptr : col_sum_w.data();

    #pragma omp parallel for schedule(static) if(M * N * K > 4096)
    for (int64_t m = 0; m < M; ++m) {
        const int8_t* act_row = act + m * K;
        for (int64_t n = 0; n < N; ++n) {
            int64_t acc = 0;
            // QInt4x2 weights are out-feature-major: channel n's K nibbles are
            // contiguous at w_data + n*K (w_data[n*K + k] == weight[n, k]),
            // matching quantized_linear_int4_kernel's weight_row = packed + o*(K/2).
            const int8_t* w_row = w_data + n * K;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<int64_t>(act_row[k]) *
                       static_cast<int64_t>(w_row[k]);
            }

            if (col_sum_w_ptr) {
                acc -= static_cast<int64_t>(act_zp) * col_sum_w_ptr[n];
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
    float act_scale, float weight_scale,
    int32_t act_zp = 0) {

    fused_qlinear_dequant(act, weights, bias, output, M, N, K, act_scale, weight_scale, act_zp);
    for (int64_t i = 0; i < M * N; ++i) {
        if (output[i] < 0.0f) output[i] = 0.0f;
    }
}

} // namespace cpu
} // namespace tenzor
