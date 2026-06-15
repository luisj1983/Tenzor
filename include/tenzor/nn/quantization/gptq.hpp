/**
 * @file gptq.hpp
 * @brief GPTQ (Generative Pre-Trained Transformer Quantization)
 *
 * Implements layer-wise weight quantization using second-order information
 * (Hessian of the reconstruction error) to find optimal INT4/INT8 weight
 * quantization with minimal accuracy loss. Based on:
 *   Frantar et al., "GPTQ: Accurate Post-Training Quantization for
 *   Generative Pre-trained Transformers", ICLR 2023.
 *
 * The algorithm processes columns of the weight matrix sequentially,
 * quantizing each column and compensating the remaining columns using
 * the inverse Hessian to minimize the overall reconstruction error.
 */

#pragma once

#include <tuple>
#include <vector>
#include "../../core/tensor.hpp"
#include "quantize.hpp"

namespace tenzor {
namespace nn {
namespace quantization {

/**
 * @brief Configuration for GPTQ quantization.
 */
struct GPTQConfig {
    int group_size = 128;       ///< Number of columns per quantization group
    float damp_percent = 0.01f; ///< Dampening percentage added to Hessian diagonal
    bool desc_act = false;      ///< Sort columns by descending Hessian diagonal (activation order)
    int bits = 4;               ///< Quantization bit width (4 or 8)
    bool sym = true;            ///< Use symmetric quantization (zero_point = 0)
};

/**
 * @brief Result of GPTQ quantization for a single linear layer.
 *
 * Contains the packed quantized weights along with per-group scales and
 * zero points needed for dequantization at inference time.
 */
struct GPTQResult {
    Tensor packed_weight;  ///< Packed INT4/INT8 weight tensor. With desc_act=true,
                           ///< columns are in PERMUTED (activation) order.
    int64_t in_features = 0; ///< Logical (unpacked) in_features. For INT4 the packed
                           ///< tensor has ceil(in_features/2) columns, so this records
                           ///< the true (possibly odd) column count for unambiguous unpack.
    Tensor scales;         ///< Per-group scale factors, shape (out_features, num_groups).
                           ///< With desc_act=true, groups index PERMUTED columns
                           ///< (consistent with packed_weight).
    Tensor zeros;          ///< Per-group zero points, shape (out_features, num_groups).
                           ///< With desc_act=true, in PERMUTED column-group order.
    Tensor perm;           ///< Column permutation (non-empty only when desc_act=true).
                           ///< REQUIRED at inference: permute input activations by
                           ///< `perm` before matmul (packed_weight/scales/zeros are
                           ///< all in this permuted order). Outputs need no reorder
                           ///< since the row (output) dimension is untouched.
};

/**
 * @brief GPTQ quantizer for layer-wise weight quantization.
 *
 * Quantizes linear layer weights using second-order (Hessian) information
 * from calibration data. The algorithm:
 *
 * 1. Accumulate the Hessian H = 2 * X^T * X / N from calibration inputs
 * 2. Add dampening: H += damp * diag(H)
 * 3. Compute Cholesky decomposition of H_inv = inv(H)
 * 4. Process columns left-to-right in groups of `group_size`:
 *    - Compute per-group scale/zero_point from the weight range
 *    - Quantize each column to INT4/INT8
 *    - Compensate remaining columns for quantization error using H_inv
 * 5. Pack quantized weights (2 values per byte for INT4)
 *
 * @code
 * // 1. Create quantizer
 * GPTQConfig config;
 * config.bits = 4;
 * config.group_size = 128;
 * GPTQQuantizer quantizer(config);
 *
 * // 2. Accumulate Hessian from calibration data
 * // input_activations: shape (num_samples, in_features)
 * Tensor H = GPTQQuantizer::compute_hessian(input_activations);
 *
 * // 3. Quantize the layer
 * // weight: shape (out_features, in_features) from nn::Linear
 * auto [packed, scales, zeros, perm] = quantizer.quantize_layer(weight, H);
 * @endcode
 */
class GPTQQuantizer {
public:
    /**
     * @brief Construct GPTQ quantizer with given configuration.
     *
     * @param config GPTQ configuration (group_size, bits, dampening, etc.)
     */
    explicit GPTQQuantizer(GPTQConfig config = {});

    /**
     * @brief Quantize a single linear layer's weights.
     *
     * Given the weight matrix and pre-computed Hessian, performs the full
     * GPTQ quantization algorithm: dampening, Cholesky of H_inv, column-wise
     * quantization with error compensation, and INT4/INT8 packing.
     *
     * @param weight Weight tensor of shape (out_features, in_features), Float32
     * @param hessian Hessian tensor of shape (in_features, in_features), Float32.
     *               Computed from calibration inputs via compute_hessian().
     * @return GPTQResult containing packed weights, scales, zeros, and permutation
     *
     * @throws std::invalid_argument if weight/hessian dimensions are incompatible
     * @throws std::runtime_error if Cholesky decomposition fails (Hessian not positive-definite)
     */
    auto quantize_layer(const Tensor& weight, const Tensor& hessian) -> GPTQResult;

    /**
     * @brief Compute Hessian matrix from calibration input activations.
     *
     * Accumulates H = 2 * X^T * X / N where X has shape (N, in_features).
     * The factor of 2 comes from the derivative of the squared reconstruction
     * error ||WX - W_q X||^2 with respect to W.
     *
     * Can be called incrementally: pass batches of activations and sum the
     * resulting Hessians, then divide by total number of batches.
     *
     * @param input_activations Calibration inputs, shape (num_samples, in_features)
     * @return Hessian matrix, shape (in_features, in_features)
     */
    static auto compute_hessian(const Tensor& input_activations) -> Tensor;

    /**
     * @brief Get the current configuration.
     */
    auto config() const -> const GPTQConfig& { return config_; }

private:
    GPTQConfig config_;

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
