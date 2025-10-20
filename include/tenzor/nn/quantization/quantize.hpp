/**
 * @file quantize.hpp
 * @brief Quantization operations and schemes for neural network inference
 *
 * Provides quantization operations for converting floating-point tensors to
 * lower-precision integer representations (INT8, UINT8) for efficient inference.
 * Supports per-tensor and per-channel quantization with symmetric and asymmetric modes.
 */

#pragma once

#include <memory>
#include <string>
#include "../../core/tensor.hpp"
#include "../../autograd/variable.hpp"

namespace tenzor {
namespace nn {
namespace quantization {

/**
 * @brief Quantization scheme enumeration.
 */
enum class QuantizationScheme {
    PerTensorSymmetric,    ///< Symmetric quantization per tensor
    PerTensorAsymmetric,   ///< Asymmetric quantization per tensor
    PerChannelSymmetric,   ///< Symmetric quantization per channel
    PerChannelAsymmetric   ///< Asymmetric quantization per channel
};

/**
 * @brief Quantized data type.
 */
enum class QuantDType {
    INT8,   ///< Signed 8-bit integer [-128, 127]
    UINT8   ///< Unsigned 8-bit integer [0, 255]
};

/**
 * @brief Quantization parameters.
 *
 * Stores scale and zero-point for quantization/dequantization:
 * quantized = round(value / scale) + zero_point
 * dequantized = (quantized - zero_point) * scale
 */
struct QuantizationParams {
    Tensor scale;       ///< Scale factor(s) for quantization
    Tensor zero_point;  ///< Zero point(s) for quantization
    QuantDType dtype;   ///< Quantized data type
    QuantizationScheme scheme;  ///< Quantization scheme
    int64_t axis{-1};   ///< Axis for per-channel quantization (default: -1 for per-tensor)

    QuantizationParams(Tensor scale_val, Tensor zero_point_val,
                      QuantDType dtype_val, QuantizationScheme scheme_val,
                      int64_t axis_val = -1)
        : scale(std::move(scale_val)), zero_point(std::move(zero_point_val)),
          dtype(dtype_val), scheme(scheme_val), axis(axis_val) {}
};

/**
 * @brief Quantized tensor representation.
 *
 * Contains quantized integer data along with quantization parameters
 * needed for dequantization.
 */
class QuantizedTensor {
public:
    QuantizedTensor(Tensor data, QuantizationParams params)
        : data_(std::move(data)), params_(std::move(params)) {}

    auto data() const -> const Tensor& { return data_; }
    auto params() const -> const QuantizationParams& { return params_; }

    /**
     * @brief Dequantize tensor back to floating point.
     *
     * @return Dequantized floating-point tensor
     */
    auto dequantize() const -> Tensor;

    /**
     * @brief Get shape of quantized tensor.
     */
    auto shape() const -> std::span<const int64_t> { return data_.shape(); }

    /**
     * @brief Get device of quantized tensor.
     */
    auto device() const -> const Device& { return data_.device(); }

private:
    Tensor data_;                    ///< Quantized integer data
    QuantizationParams params_;      ///< Quantization parameters
};

/**
 * @brief Compute quantization parameters from min/max values.
 *
 * Calculates optimal scale and zero-point for quantizing values in [min, max]
 * to the target quantized data type range.
 *
 * @param min Minimum value in the data
 * @param max Maximum value in the data
 * @param dtype Target quantized data type
 * @param scheme Quantization scheme
 * @return Quantization parameters (scale and zero_point)
 *
 * @code
 * Tensor min({1}, DType::Float32, Device::cpu());
 * Tensor max({1}, DType::Float32, Device::cpu());
 * min.fill_(-2.5f);
 * max.fill_(3.5f);
 * auto params = compute_quantization_params(min, max, QuantDType::INT8,
 *                                          QuantizationScheme::PerTensorAsymmetric);
 * @endcode
 */
auto compute_quantization_params(
    const Tensor& min,
    const Tensor& max,
    QuantDType dtype,
    QuantizationScheme scheme
) -> QuantizationParams;

/**
 * @brief Quantize floating-point tensor to integer representation.
 *
 * Converts floating-point values to quantized integers using the provided
 * quantization parameters. This is used for model inference optimization.
 *
 * @param input Floating-point tensor to quantize
 * @param params Quantization parameters (scale, zero_point, dtype)
 * @return Quantized tensor with integer data
 *
 * @code
 * Tensor weights({64, 32}, DType::Float32, Device::cpu());
 * auto params = compute_quantization_params(weights.min(), weights.max(),
 *                                          QuantDType::INT8,
 *                                          QuantizationScheme::PerTensorSymmetric);
 * QuantizedTensor q_weights = quantize_tensor(weights, params);
 * @endcode
 */
auto quantize_tensor(const Tensor& input, const QuantizationParams& params)
    -> QuantizedTensor;

/**
 * @brief Quantize tensor using per-tensor symmetric quantization.
 *
 * Convenience function for symmetric per-tensor quantization.
 * Uses scale = max(abs(min), abs(max)) / 127 and zero_point = 0.
 *
 * @param input Floating-point tensor
 * @param dtype Target quantized data type
 * @return Quantized tensor
 */
auto quantize_per_tensor_symmetric(const Tensor& input, QuantDType dtype = QuantDType::INT8)
    -> QuantizedTensor;

/**
 * @brief Quantize tensor using per-tensor asymmetric quantization.
 *
 * Convenience function for asymmetric per-tensor quantization.
 * Calculates scale and zero_point to map [min, max] to full quantized range.
 *
 * @param input Floating-point tensor
 * @param dtype Target quantized data type
 * @return Quantized tensor
 */
auto quantize_per_tensor_asymmetric(const Tensor& input, QuantDType dtype = QuantDType::INT8)
    -> QuantizedTensor;

/**
 * @brief Quantize tensor using per-channel symmetric quantization.
 *
 * Applies symmetric quantization independently along the specified channel axis.
 * Commonly used for weights where each output channel can have different ranges.
 *
 * @param input Floating-point tensor
 * @param axis Channel axis for per-channel quantization (typically 0)
 * @param dtype Target quantized data type
 * @return Quantized tensor with per-channel scales
 *
 * @code
 * Tensor conv_weights({64, 32, 3, 3}, DType::Float32, Device::cpu());
 * // Quantize each output channel (dim 0) independently
 * auto q_weights = quantize_per_channel_symmetric(conv_weights, 0);
 * @endcode
 */
auto quantize_per_channel_symmetric(const Tensor& input, int64_t axis,
                                   QuantDType dtype = QuantDType::INT8)
    -> QuantizedTensor;

/**
 * @brief Quantize tensor using per-channel asymmetric quantization.
 *
 * @param input Floating-point tensor
 * @param axis Channel axis for per-channel quantization
 * @param dtype Target quantized data type
 * @return Quantized tensor with per-channel scales and zero-points
 */
auto quantize_per_channel_asymmetric(const Tensor& input, int64_t axis,
                                    QuantDType dtype = QuantDType::INT8)
    -> QuantizedTensor;

/**
 * @brief Dequantize integer tensor back to floating point.
 *
 * Converts quantized integer values back to floating-point representation
 * using the stored quantization parameters.
 *
 * @param quantized Quantized tensor with parameters
 * @return Dequantized floating-point tensor
 *
 * @code
 * QuantizedTensor q_tensor = quantize_per_tensor_symmetric(input);
 * Tensor fp_tensor = dequantize_tensor(q_tensor);
 * // fp_tensor approximates original input with some quantization error
 * @endcode
 */
auto dequantize_tensor(const QuantizedTensor& quantized) -> Tensor;

/**
 * @brief Compute quantization error metrics.
 *
 * Measures the error introduced by quantization by comparing original
 * and dequantized tensors.
 *
 * @param original Original floating-point tensor
 * @param quantized Quantized tensor
 * @return Tuple of (mean_absolute_error, mean_squared_error, signal_to_noise_ratio_db)
 */
auto compute_quantization_error(const Tensor& original, const QuantizedTensor& quantized)
    -> std::tuple<float, float, float>;

/**
 * @brief Calibrate quantization parameters from sample data.
 *
 * Analyzes sample inputs to determine optimal quantization parameters
 * for post-training quantization (PTQ). Typically used with observer objects
 * that collect statistics during calibration.
 *
 * @param samples Vector of sample tensors for calibration
 * @param dtype Target quantized data type
 * @param scheme Quantization scheme
 * @param axis Channel axis for per-channel quantization (default: -1)
 * @return Calibrated quantization parameters
 */
auto calibrate_quantization_params(
    const std::vector<Tensor>& samples,
    QuantDType dtype,
    QuantizationScheme scheme,
    int64_t axis = -1
) -> QuantizationParams;

} // namespace quantization
} // namespace nn
} // namespace tenzor
