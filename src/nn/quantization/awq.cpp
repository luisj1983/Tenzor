/**
 * @file awq.cpp
 * @brief Implementation of AWQ (Activation-Aware Weight Quantization) algorithm
 */

#include "tenzor/nn/quantization/awq.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <vector>

namespace tenzor {
namespace nn {
namespace quantization {

AWQQuantizer::AWQQuantizer(AWQConfig config)
    : config_(std::move(config)) {
    if (config_.bits != 4 && config_.bits != 8) {
        throw std::invalid_argument("AWQ: bits must be 4 or 8, got " +
                                    std::to_string(config_.bits));
    }
    if (config_.group_size <= 0) {
        throw std::invalid_argument("AWQ: group_size must be positive, got " +
                                    std::to_string(config_.group_size));
    }
    if (config_.n_grid <= 0) {
        throw std::invalid_argument("AWQ: n_grid must be positive, got " +
                                    std::to_string(config_.n_grid));
    }
    if (config_.max_alpha < 0.0f) {
        throw std::invalid_argument("AWQ: max_alpha must be non-negative, got " +
                                    std::to_string(config_.max_alpha));
    }
}

auto AWQQuantizer::quant_range() const -> std::pair<int32_t, int32_t> {
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

auto AWQQuantizer::quantize_value(float val, float scale, float zero) const -> int32_t {
    auto [qmin, qmax] = quant_range();
    float q = val / scale + zero;
    q = std::round(q);
    q = std::clamp(q, static_cast<float>(qmin), static_cast<float>(qmax));
    return static_cast<int32_t>(q);
}

auto AWQQuantizer::dequantize_value(int32_t qval, float scale, float zero) -> float {
    return (static_cast<float>(qval) - zero) * scale;
}

auto AWQQuantizer::compute_act_scales(const Tensor& input_activations) -> Tensor {
    // input_activations: (num_samples, in_features)
    auto shape = input_activations.shape();
    if (shape.size() != 2) {
        throw std::invalid_argument(
            "AWQ::compute_act_scales: input must be 2D (num_samples, in_features), got " +
            std::to_string(shape.size()) + "D");
    }

    // act_scale = mean(|X|, dim=0) -> shape (in_features,)
    auto abs_input = tenzor::abs(input_activations);
    auto act_scales = ops::mean(abs_input, 0);

    return act_scales;
}

auto AWQQuantizer::find_optimal_scales(const Tensor& weight, const Tensor& act_scales,
                                       int64_t group_start, int64_t group_end) -> Tensor {
    // weight: (out_features, in_features) - full weight matrix
    // act_scales: (in_features,) - per-channel activation scales
    // We search for the best alpha for this group of columns [group_start, group_end)

    int64_t out_features = weight.shape()[0];
    int64_t in_features = weight.shape()[1];
    int64_t group_size = group_end - group_start;

    auto* w_ptr = static_cast<const float*>(weight.data_ptr());
    auto* act_ptr = static_cast<const float*>(act_scales.data_ptr());

    auto [qmin, qmax] = quant_range();

    // Best MSE and corresponding alpha (we track one alpha for the whole group)
    float best_mse = std::numeric_limits<float>::max();
    float best_alpha = 0.0f;

    // Grid search over alpha values
    for (int grid_idx = 0; grid_idx <= config_.n_grid; ++grid_idx) {
        float alpha = config_.max_alpha * static_cast<float>(grid_idx) /
                      static_cast<float>(config_.n_grid);

        // Compute scaling factors s = act_scale^alpha for this group
        // Then simulate: scale weight columns by s, quantize, dequantize,
        // unscale by s^{-1}, measure MSE against original weight
        float total_mse = 0.0f;

        // For each output row, compute per-group scale/zero, then quantize
        for (int64_t row = 0; row < out_features; ++row) {
            // First pass: find min/max of scaled weights to compute quant params
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();

            for (int64_t j = group_start; j < group_end; ++j) {
                float act_val = act_ptr[j];
                float s = std::pow(std::max(act_val, 1e-8f), alpha);
                float w_scaled = w_ptr[row * in_features + j] * s;
                w_min = std::min(w_min, w_scaled);
                w_max = std::max(w_max, w_scaled);
            }

            // Compute quantization scale/zero for this row's group
            float scale;
            float zero;
            if (config_.sym) {
                float abs_max = std::max(std::abs(w_min), std::abs(w_max));
                abs_max = std::max(abs_max, 1e-8f);
                scale = abs_max / static_cast<float>(qmax);
                zero = 0.0f;
            } else {
                float range = w_max - w_min;
                range = std::max(range, 1e-8f);
                scale = range / static_cast<float>(qmax - qmin);
                zero = std::round(static_cast<float>(qmin) - w_min / scale);
                zero = std::clamp(zero, static_cast<float>(qmin),
                                  static_cast<float>(qmax));
            }

            // Second pass: quantize, dequantize, unscale, measure error
            for (int64_t j = group_start; j < group_end; ++j) {
                float act_val = act_ptr[j];
                float s = std::pow(std::max(act_val, 1e-8f), alpha);
                float s_inv = 1.0f / s;

                float w_orig = w_ptr[row * in_features + j];
                float w_scaled = w_orig * s;

                // Quantize and dequantize
                int32_t q_val = quantize_value(w_scaled, scale, zero);
                float w_deq = dequantize_value(q_val, scale, zero);

                // Apply inverse scaling to get reconstructed weight
                float w_recon = w_deq * s_inv;

                // Weighted MSE: weight the error by activation magnitude
                float error = w_orig - w_recon;
                // Weight error by activation magnitude for this channel
                total_mse += error * error * act_val;
            }
        }

        if (total_mse < best_mse) {
            best_mse = total_mse;
            best_alpha = alpha;
        }
    }

    // Construct the optimal scaling factors for this group
    auto optimal_scales = ops::zeros({group_size}, DType::Float32, weight.device());
    auto* s_ptr = static_cast<float*>(optimal_scales.data_ptr());

    for (int64_t j = 0; j < group_size; ++j) {
        float act_val = act_ptr[group_start + j];
        s_ptr[j] = std::pow(std::max(act_val, 1e-8f), best_alpha);
    }

    return optimal_scales;
}

auto AWQQuantizer::quantize_layer(const Tensor& weight, const Tensor& act_scales)
    -> AWQResult {
    // weight: (out_features, in_features)
    // act_scales: (in_features,)
    auto w_shape = weight.shape();
    auto a_shape = act_scales.shape();

    if (w_shape.size() != 2) {
        throw std::invalid_argument(
            "AWQ: weight must be 2D (out_features, in_features), got " +
            std::to_string(w_shape.size()) + "D");
    }
    if (a_shape.size() != 1) {
        throw std::invalid_argument(
            "AWQ: act_scales must be 1D (in_features,), got " +
            std::to_string(a_shape.size()) + "D");
    }
    if (w_shape[1] != a_shape[0]) {
        throw std::invalid_argument(
            "AWQ: weight in_features (" + std::to_string(w_shape[1]) +
            ") != act_scales length (" + std::to_string(a_shape[0]) + ")");
    }

    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    // The routine below uses raw host-pointer loops, so all intermediate
    // tensors must live on CPU even when the caller passes GPU tensors.
    // Previously this would hang/crash when weight or act_scales were on a
    // GPU backend (Variable host-ptr reads on device memory).
    Device original_device = weight.device();
    auto weight_cpu = original_device.type == Device::Type::CPU
        ? weight : weight.to(Device::cpu());
    auto act_scales_cpu = act_scales.device().type == Device::Type::CPU
        ? act_scales : act_scales.to(Device::cpu());

    // Work in Float32 for numerical stability (all on CPU).
    auto W = weight_cpu.to(DType::Float32).clone();
    auto act = act_scales_cpu.to(DType::Float32).clone();

    // Prepare output tensors on CPU; move back to original device at the end.
    int64_t num_groups = (in_features + config_.group_size - 1) / config_.group_size;
    // Standard AWQ group-wise quantization parameters:
    //   scales / zeros: (out_features, num_groups) — one (scale, zero) per output
    //     row per input-channel group. Storing one float scale per *channel* would
    //     cost more bits than the 4-bit weight it quantizes, defeating the point.
    //   act_scales: (in_features,) — the per-input-channel AWQ scaling factor s[j]
    //     applied to the weights before quantization. Exact dequant is
    //       W_recon[o,j] = (q[o,j] - zeros[o,g]) * scales[o,g] / act_scales[j].
    auto scales = ops::zeros({out_features, num_groups}, DType::Float32, Device::cpu());
    auto zeros_tensor = ops::zeros({out_features, num_groups}, DType::Float32, Device::cpu());
    auto act_scales_out = ops::zeros({in_features}, DType::Float32, Device::cpu());

    // Quantized weight in int32 (will be packed later)
    auto Q = ops::zeros({out_features, in_features}, DType::Int32, Device::cpu());

    auto [qmin, qmax] = quant_range();

    // Step 1: Find optimal scaling factors per group via grid search
    std::vector<Tensor> group_scales;
    group_scales.reserve(num_groups);

    for (int64_t col_start = 0; col_start < in_features; col_start += config_.group_size) {
        int64_t col_end = std::min(col_start + config_.group_size, in_features);
        auto s = find_optimal_scales(W, act, col_start, col_end);
        group_scales.push_back(std::move(s));
    }

    // Step 2: Apply optimal scaling, quantize, and compute per-group quant params
    auto* w_ptr = static_cast<float*>(W.data_ptr());
    auto* q_ptr = static_cast<int32_t*>(Q.data_ptr());
    auto* scales_ptr = static_cast<float*>(scales.data_ptr());
    auto* zeros_ptr = static_cast<float*>(zeros_tensor.data_ptr());
    auto* act_scales_ptr = static_cast<float*>(act_scales_out.data_ptr());

    for (int64_t col_start = 0; col_start < in_features; col_start += config_.group_size) {
        int64_t col_end = std::min(col_start + config_.group_size, in_features);
        int64_t group_idx = col_start / config_.group_size;
        int64_t group_size = col_end - col_start;

        auto* s_ptr = static_cast<const float*>(group_scales[group_idx].data_ptr());

        // Record the per-input-channel AWQ scale factors for this group so the
        // weight pre-scaling can be undone at dequant/inference time.
        for (int64_t j = 0; j < group_size; ++j) {
            act_scales_ptr[col_start + j] = s_ptr[j];
        }

        // Apply scaling to weight columns: W_scaled[:, j] = W[:, j] * s[j]
        // We modify W in-place for this group
        for (int64_t row = 0; row < out_features; ++row) {
            for (int64_t j = 0; j < group_size; ++j) {
                w_ptr[row * in_features + col_start + j] *= s_ptr[j];
            }
        }

        // Compute per-group scale and zero_point from scaled weights
        for (int64_t row = 0; row < out_features; ++row) {
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
                float abs_max = std::max(std::abs(w_min), std::abs(w_max));
                abs_max = std::max(abs_max, 1e-8f);
                scale = abs_max / static_cast<float>(qmax);
                zero = 0.0f;
            } else {
                float range = w_max - w_min;
                range = std::max(range, 1e-8f);
                scale = range / static_cast<float>(qmax - qmin);
                zero = std::round(static_cast<float>(qmin) - w_min / scale);
                zero = std::clamp(zero, static_cast<float>(qmin),
                                  static_cast<float>(qmax));
            }

            // Store the per-group quant scale and zero point. The per-channel AWQ
            // factor s[j] is recorded separately in act_scales; exact dequant is
            //   W_recon[row, j] = (q - zero) * scale / act_scales[j].
            scales_ptr[row * num_groups + group_idx] = scale;
            zeros_ptr[row * num_groups + group_idx] = zero;

            // Quantize each column in this group
            for (int64_t j = col_start; j < col_end; ++j) {
                float w_val = w_ptr[row * in_features + j];
                int32_t q_val = quantize_value(w_val, scale, zero);
                q_ptr[row * in_features + j] = q_val;
            }
        }

        // (No geometric-mean fold: the exact per-channel s[j]^{-1} is already baked
        //  into scales[row, col_start + j] above.)
    }

    // Pack INT4 weights if using 4-bit quantization
    Tensor packed;
    if (config_.bits == 4) {
        // Shift to unsigned range for packing: for symmetric, shift by -qmin
        // so values are in [0, 15] for INT4
        if (config_.sym) {
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

    // Move results back to the original input device so downstream callers
    // (e.g. inference kernels) can use the outputs without an extra copy.
    if (original_device.type != Device::Type::CPU) {
        packed = packed.to(original_device);
        scales = scales.to(original_device);
        zeros_tensor = zeros_tensor.to(original_device);
        act_scales_out = act_scales_out.to(original_device);
    }

    return AWQResult{
        .quantized_weight = std::move(packed),
        .scales = std::move(scales),
        .zeros = std::move(zeros_tensor),
        .act_scales = std::move(act_scales_out)
    };
}

auto AWQQuantizer::pack_int4(const Tensor& quantized_weight) const -> Tensor {
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
