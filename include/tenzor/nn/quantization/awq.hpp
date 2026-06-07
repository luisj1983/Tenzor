/**
 * @file awq.hpp
 * @brief AWQ (Activation-Aware Weight Quantization)
 *
 * Implements activation-aware weight quantization that finds per-channel
 * scaling factors minimizing quantization error weighted by activation
 * magnitudes. Based on:
 *   Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
 *   Compression and Acceleration", MLSys 2024.
 *
 * The algorithm searches for per-channel scaling factors s such that
 * quantizing W * diag(s) and then multiplying by diag(s)^{-1} at inference
 * preserves the activation-weighted reconstruction quality. A grid search
 * over s = act_scale^alpha selects the optimal alpha per channel group.
 */

#pragma once

#include <cstdint>
#include <tuple>
#include "../../core/tensor.hpp"
#include "quantize.hpp"

namespace tenzor {
namespace nn {
namespace quantization {

/**
 * @brief Configuration for AWQ quantization.
 */
struct AWQConfig {
    int group_size = 128;       ///< Number of columns per quantization group
    int bits = 4;               ///< Quantization bit width (4 or 8)
    bool sym = true;            ///< Use symmetric quantization (zero_point = 0)
    int n_grid = 20;            ///< Number of grid search points for alpha
    float max_alpha = 1.0f;     ///< Maximum scaling power in grid search
};

/**
 * @brief Result of AWQ quantization for a single linear layer.
 *
 * Contains the packed quantized weights along with per-group scales and
 * zero points needed for dequantization at inference time.
 */
struct AWQResult {
    Tensor quantized_weight;  ///< Packed INT4/INT8 weight tensor
    Tensor scales;            ///< Per-group dequant scales, shape (out_features, num_groups)
    Tensor zeros;             ///< Per-group zero points, shape (out_features, num_groups)
    Tensor act_scales;        ///< Per-input-channel AWQ scale factors s[j], shape
                              ///< (in_features,). Weights were pre-scaled by s[j] before
                              ///< quantization; exact dequant is
                              ///< W_recon[o,j] = (q[o,j] - zeros[o,g]) * scales[o,g] / act_scales[j].
};

/**
 * @brief AWQ quantizer for activation-aware weight quantization.
 *
 * Quantizes linear layer weights using activation magnitude statistics to
 * find optimal per-channel scaling factors. The algorithm:
 *
 * 1. Compute per-channel activation scales: act_scale = mean(|X|, dim=0)
 * 2. For each group of `group_size` weight columns:
 *    a. Grid search over alpha in [0, max_alpha] with n_grid steps
 *    b. For each alpha, compute s = act_scale^alpha
 *    c. Scale weight columns: W_scaled = W * diag(s)
 *    d. Quantize and dequantize W_scaled
 *    e. Apply inverse scaling: W_recon = W_deq * diag(s^{-1})
 *    f. Measure MSE between W_recon and original W
 *    g. Select alpha with minimum MSE per group
 * 3. Apply optimal scaling, quantize, and pack to INT4/INT8
 *
 * @code
 * // 1. Create quantizer
 * AWQConfig config;
 * config.bits = 4;
 * config.group_size = 128;
 * AWQQuantizer quantizer(config);
 *
 * // 2. Compute activation scales from calibration data
 * // input_activations: shape (num_samples, in_features)
 * Tensor act_scales = AWQQuantizer::compute_act_scales(input_activations);
 *
 * // 3. Quantize the layer
 * // weight: shape (out_features, in_features) from nn::Linear
 * auto [packed, scales, zeros] = quantizer.quantize_layer(weight, act_scales);
 * @endcode
 */
class AWQQuantizer {
public:
    /**
     * @brief Construct AWQ quantizer with given configuration.
     *
     * @param config AWQ configuration (group_size, bits, n_grid, etc.)
     */
    explicit AWQQuantizer(AWQConfig config = {});

    /**
     * @brief Compute per-channel activation scales from calibration data.
     *
     * Computes act_scale = mean(|X|, dim=0) where X has shape (N, in_features).
     * The resulting scales reflect the average activation magnitude per input
     * channel, used to weight the importance of each channel during quantization.
     *
     * @param input_activations Calibration inputs, shape (num_samples, in_features)
     * @return Per-channel activation scales, shape (in_features,)
     */
    static auto compute_act_scales(const Tensor& input_activations) -> Tensor;

    /**
     * @brief Quantize a single linear layer's weights.
     *
     * Given the weight matrix and pre-computed activation scales, performs the
     * full AWQ algorithm: grid search for optimal scaling factors, application
     * of scaling, quantization, and INT4/INT8 packing.
     *
     * @param weight Weight tensor of shape (out_features, in_features), Float32
     * @param act_scales Per-channel activation scales, shape (in_features,).
     *                   Computed from calibration inputs via compute_act_scales().
     * @return AWQResult containing packed weights, scales, and zero points
     *
     * @throws std::invalid_argument if weight/act_scales dimensions are incompatible
     */
    auto quantize_layer(const Tensor& weight, const Tensor& act_scales) -> AWQResult;

    /**
     * @brief Get the current configuration.
     */
    auto config() const -> const AWQConfig& { return config_; }

private:
    AWQConfig config_;

    /**
     * @brief Grid search for optimal scaling factor per group.
     *
     * Tries alpha values from 0 to max_alpha in n_grid steps. For each alpha,
     * computes s = act_scale^alpha, applies scaling to weight columns, quantizes,
     * dequantizes, applies inverse scaling, and measures MSE. Returns the
     * scaling factors corresponding to the best alpha.
     *
     * @param weight Full weight tensor, shape (out_features, in_features)
     * @param act_scales Per-channel activation scales, shape (in_features,)
     * @param group_start Start column index of the group (inclusive)
     * @param group_end End column index of the group (exclusive)
     * @return Optimal scaling factors for this group, shape (group_end - group_start,)
     */
    auto find_optimal_scales(const Tensor& weight, const Tensor& act_scales,
                             int64_t group_start, int64_t group_end) -> Tensor;

    /**
     * @brief Quantize a single value to the configured bit width.
     *
     * @param val Float value to quantize
     * @param scale Quantization scale
     * @param zero Zero point
     * @return Quantized integer value clamped to [qmin, qmax]
     */
    auto quantize_value(float val, float scale, float zero) const -> int32_t;

    /**
     * @brief Dequantize a single value from integer back to float.
     *
     * @param qval Quantized integer value
     * @param scale Quantization scale
     * @param zero Zero point
     * @return Dequantized float value
     */
    static auto dequantize_value(int32_t qval, float scale, float zero) -> float;

    /**
     * @brief Pack INT4 values into bytes (2 values per byte).
     *
     * Packing format: low nibble = first value, high nibble = second value.
     * For odd column counts, the last byte's high nibble is zero-padded.
     *
     * @param quantized_weight Quantized weight tensor with int32 values in [0, 15] or [-8, 7]
     * @return Packed tensor with half the columns (uint8 storage)
     */
    auto pack_int4(const Tensor& quantized_weight) const -> Tensor;

    /**
     * @brief Get the quantization range for the configured bit width.
     *
     * @return Pair of (qmin, qmax) integer bounds
     */
    auto quant_range() const -> std::pair<int32_t, int32_t>;
};

} // namespace quantization
} // namespace nn
} // namespace tenzor
