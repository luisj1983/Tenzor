/**
 * @file gptq.cpp
 * @brief Implementation of GPTQ quantization algorithm
 */

#include "tenzor/nn/quantization/gptq.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace nn {
namespace quantization {

GPTQQuantizer::GPTQQuantizer(GPTQConfig config)
    : config_(std::move(config)) {
    if (config_.bits != 4 && config_.bits != 8) {
        throw std::invalid_argument("GPTQ: bits must be 4 or 8, got " +
                                    std::to_string(config_.bits));
    }
    if (config_.group_size <= 0) {
        throw std::invalid_argument("GPTQ: group_size must be positive, got " +
                                    std::to_string(config_.group_size));
    }
    if (config_.damp_percent < 0.0f || config_.damp_percent > 1.0f) {
        throw std::invalid_argument("GPTQ: damp_percent must be in [0, 1], got " +
                                    std::to_string(config_.damp_percent));
    }
}

auto GPTQQuantizer::quant_range() const -> std::pair<int32_t, int32_t> {
    if (config_.sym) {
        // Symmetric: use signed range
        int32_t qmax = (1 << (config_.bits - 1)) - 1;  // 7 for 4-bit, 127 for 8-bit
        int32_t qmin = -(1 << (config_.bits - 1));       // -8 for 4-bit, -128 for 8-bit
        return {qmin, qmax};
    } else {
        // Asymmetric: use unsigned range
        int32_t qmax = (1 << config_.bits) - 1;  // 15 for 4-bit, 255 for 8-bit
        return {0, qmax};
    }
}

auto GPTQQuantizer::quantize_value(float val, float scale, float zero) const -> int32_t {
    auto [qmin, qmax] = quant_range();
    float q = val / scale + zero;
    q = std::round(q);
    q = std::clamp(q, static_cast<float>(qmin), static_cast<float>(qmax));
    return static_cast<int32_t>(q);
}

auto GPTQQuantizer::dequantize_value(int32_t qval, float scale, float zero) -> float {
    return (static_cast<float>(qval) - zero) * scale;
}

auto GPTQQuantizer::compute_hessian(const Tensor& input_activations) -> Tensor {
    // input_activations: (num_samples, in_features)
    auto shape = input_activations.shape();
    if (shape.size() != 2) {
        throw std::invalid_argument(
            "GPTQ::compute_hessian: input must be 2D (num_samples, in_features), got " +
            std::to_string(shape.size()) + "D");
    }

    int64_t num_samples = shape[0];

    // H = 2 * X^T * X / num_samples
    auto xt = input_activations.transpose(0, 1);  // (in_features, num_samples)
    auto h = tenzor::matmul(xt, input_activations);   // (in_features, in_features)
    h = h * (2.0 / static_cast<double>(num_samples));

    return h;
}

auto GPTQQuantizer::quantize_layer(const Tensor& weight, const Tensor& hessian)
    -> GPTQResult {
    // weight: (out_features, in_features)
    // hessian: (in_features, in_features)
    auto w_shape = weight.shape();
    auto h_shape = hessian.shape();

    if (w_shape.size() != 2) {
        throw std::invalid_argument(
            "GPTQ: weight must be 2D (out_features, in_features), got " +
            std::to_string(w_shape.size()) + "D");
    }
    if (h_shape.size() != 2 || h_shape[0] != h_shape[1]) {
        throw std::invalid_argument(
            "GPTQ: hessian must be square 2D, got (" +
            std::to_string(h_shape[0]) + ", " + std::to_string(h_shape[1]) + ")");
    }
    if (w_shape[1] != h_shape[0]) {
        throw std::invalid_argument(
            "GPTQ: weight in_features (" + std::to_string(w_shape[1]) +
            ") != hessian dim (" + std::to_string(h_shape[0]) + ")");
    }

    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    // This routine uses raw host-pointer loops throughout; any non-CPU input
    // must be migrated to CPU first or the host reads crash. Previously this
    // hung/crashed when weight or hessian were on a GPU backend.
    Device original_device = weight.device();
    auto weight_cpu = original_device.type == Device::Type::CPU
        ? weight : weight.to(Device::cpu());
    auto hessian_cpu = hessian.device().type == Device::Type::CPU
        ? hessian : hessian.to(Device::cpu());

    // Work in Float32 for numerical stability (all on CPU).
    auto W = weight_cpu.to(DType::Float32).clone();
    auto H = hessian_cpu.to(DType::Float32).clone();

    // Column permutation for desc_act (activation order)
    Tensor perm;
    Tensor inv_perm;
    if (config_.desc_act) {
        // Sort columns by descending diagonal of H
        auto h_diag = ops::diag(H);
        perm = ops::argsort(h_diag, 0, /*descending=*/true);

        // Apply permutation to W columns and H rows/columns
        W = ops::index_select(W, 1, perm);
        H = ops::index_select(ops::index_select(H, 0, perm), 1, perm);

        // Compute inverse permutation for unpacking later
        inv_perm = ops::argsort(perm, 0, false);
    }

    // Add dampening to Hessian diagonal
    auto h_diag = ops::diag(H);
    float damp = config_.damp_percent * ops::max(h_diag).item<float>();
    damp = std::max(damp, 1e-6f);  // Ensure non-zero dampening
    auto damp_diag = ops::eye(in_features, std::nullopt, DType::Float32,
                              H.device()) * damp;
    H = H + damp_diag;

    // Compute H_inv = inv(H). The error-compensation loop below reads H_inv
    // entries directly via hinv_ptr (H_inv[j,j] and H_inv[j, j+1:]); it does
    // not use a Cholesky factor or a precomputed diagonal, so neither is
    // materialised here. (A previous version computed cholesky(H_inv) and
    // diag(H_inv) that were never referenced — an O(n^3) waste that could also
    // spuriously throw on a borderline-PD H_inv.)
    auto H_inv = linalg::inv(H);

    // Prepare output tensors
    int64_t num_groups = (in_features + config_.group_size - 1) / config_.group_size;
    auto scales = ops::zeros({out_features, num_groups}, DType::Float32, W.device());
    auto zeros_tensor = ops::zeros({out_features, num_groups}, DType::Float32, W.device());

    // Quantized weight in int32 (will be packed later)
    auto Q = ops::zeros({out_features, in_features}, DType::Int32, W.device());

    auto [qmin, qmax] = quant_range();

    // Access raw data for element-wise operations
    auto* w_ptr = static_cast<float*>(W.data_ptr());
    auto* hinv_ptr = static_cast<const float*>(H_inv.data_ptr());
    auto* q_ptr = static_cast<int32_t*>(Q.data_ptr());
    auto* scales_ptr = static_cast<float*>(scales.data_ptr());
    auto* zeros_ptr = static_cast<float*>(zeros_tensor.data_ptr());

    // GPTQ column-wise quantization with error compensation
    // Process columns left to right in blocks of group_size
    for (int64_t col_start = 0; col_start < in_features; col_start += config_.group_size) {
        int64_t col_end = std::min(col_start + config_.group_size, in_features);
        int64_t group_idx = col_start / config_.group_size;

        // Compute per-group scale and zero_point for each output row
        for (int64_t row = 0; row < out_features; ++row) {
            // Find min/max of weights in this group for this row
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            for (int64_t j = col_start; j < col_end; ++j) {
                float val = w_ptr[row * in_features + j];
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }

            float scale;
            float zero;
            if (config_.sym) {
                // Symmetric: scale = max(|min|, |max|) / qmax
                float abs_max = std::max(std::abs(w_min), std::abs(w_max));
                abs_max = std::max(abs_max, 1e-8f);
                scale = abs_max / static_cast<float>(qmax);
                zero = 0.0f;
            } else {
                // Asymmetric: scale = (max - min) / (qmax - qmin)
                float range = w_max - w_min;
                range = std::max(range, 1e-8f);
                scale = range / static_cast<float>(qmax - qmin);
                zero = std::round(static_cast<float>(qmin) - w_min / scale);
                zero = std::clamp(zero, static_cast<float>(qmin),
                                  static_cast<float>(qmax));
            }

            scales_ptr[row * num_groups + group_idx] = scale;
            zeros_ptr[row * num_groups + group_idx] = zero;
        }

        // Quantize each column in this group and apply error compensation
        for (int64_t j = col_start; j < col_end; ++j) {
            float h_inv_jj = hinv_ptr[j * in_features + j];
            // Avoid division by zero
            if (std::abs(h_inv_jj) < 1e-10f) {
                h_inv_jj = 1e-10f;
            }

            for (int64_t row = 0; row < out_features; ++row) {
                float w_val = w_ptr[row * in_features + j];
                float scale = scales_ptr[row * num_groups + group_idx];
                float zero = zeros_ptr[row * num_groups + group_idx];

                // Quantize
                int32_t q_val = quantize_value(w_val, scale, zero);
                q_ptr[row * in_features + j] = q_val;

                // Compute quantization error
                float w_deq = dequantize_value(q_val, scale, zero);
                float error = w_val - w_deq;

                // Compensate remaining columns in the weight matrix
                // W[row, j+1:] -= error * H_inv[j, j+1:] / H_inv[j, j]
                float error_scale = error / h_inv_jj;
                for (int64_t k = j + 1; k < in_features; ++k) {
                    float h_inv_jk = hinv_ptr[j * in_features + k];
                    w_ptr[row * in_features + k] -= error_scale * h_inv_jk;
                }
            }
        }
    }

    // desc_act consistency: when desc_act is on, columns of W/H were permuted
    // by `perm`, and the per-group scales/zeros were computed indexed by the
    // *permuted* column groups. Previously Q was un-permuted back to original
    // column order here while scales/zeros were left in permuted-group order —
    // so dequant used the wrong group's scale/zero for every column whose
    // original group differs from its permuted group, corrupting the result.
    //
    // We now keep Q, scales and zeros all in permuted (activation) order so
    // they are mutually consistent: column k of Q dequantizes with group
    // k/group_size of scales/zeros. The returned `perm` lets the caller apply
    // the same permutation to input activations at inference time (and
    // inv_perm to outputs), which is the standard GPTQ desc_act protocol.
    // (inv_perm intentionally unused now that Q stays permuted.)
    (void)inv_perm;

    // Pack INT4 weights if using 4-bit quantization
    Tensor packed;
    if (config_.bits == 4) {
        // Shift to unsigned range for packing: for symmetric, shift by -qmin
        // so values are in [0, 15] for INT4
        if (config_.sym) {
            // Shift signed [-8, 7] to unsigned [0, 15] for packing
            auto shift = ops::full({out_features, in_features},
                                   static_cast<double>(-qmin), DType::Int32, Q.device());
            auto Q_unsigned = Q + shift;
            packed = pack_int4(Q_unsigned);
        } else {
            packed = pack_int4(Q);
        }
    } else {
        // 8-bit: store as Int8 directly
        packed = Q.to(DType::Int8);
    }

    // Move results back to the caller's original device.
    Tensor result_perm = config_.desc_act ? std::move(perm) : Tensor{};
    if (original_device.type != Device::Type::CPU) {
        packed = packed.to(original_device);
        scales = scales.to(original_device);
        zeros_tensor = zeros_tensor.to(original_device);
        if (result_perm.numel() > 0) result_perm = result_perm.to(original_device);
    }

    return GPTQResult{
        .packed_weight = std::move(packed),
        .in_features = in_features,
        .scales = std::move(scales),
        .zeros = std::move(zeros_tensor),
        .perm = std::move(result_perm)
    };
}

auto GPTQQuantizer::pack_int4(const Tensor& quantized_weight) const -> Tensor {
    // quantized_weight: (out_features, in_features) with values in [0, 15]
    // Pack two 4-bit values per byte: low nibble = even col, high nibble = odd col
    auto shape = quantized_weight.shape();
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    int64_t packed_cols = (cols + 1) / 2;  // Ceiling division

    auto packed = ops::zeros({rows, packed_cols}, DType::UInt8, quantized_weight.device());

    auto* src = static_cast<const int32_t*>(quantized_weight.data_ptr());
    auto* dst = static_cast<uint8_t*>(packed.data_ptr());

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; c += 2) {
            uint8_t lo = static_cast<uint8_t>(src[r * cols + c] & 0xF);
            uint8_t hi = 0;
            if (c + 1 < cols) {
                hi = static_cast<uint8_t>(src[r * cols + c + 1] & 0xF);
            }
            dst[r * packed_cols + c / 2] = lo | (hi << 4);
        }
    }

    return packed;
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
