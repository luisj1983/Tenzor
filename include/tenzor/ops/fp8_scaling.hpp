/**
 * @file fp8_scaling.hpp
 * @brief FP8 quantization scaling utilities for training and inference.
 *
 * Provides per-tensor scaling for FP8 quantization/dequantization with
 * optional stochastic rounding for improved numerical quality during training.
 *
 * @code
 * auto [fp8_tensor, params] = quantize_to_fp8(input, DType::FP8_E4M3);
 * // Use fp8_tensor for compute...
 * Tensor output = dequantize_from_fp8(fp8_tensor, params.scale);
 * @endcode
 */

#pragma once

#include "../core/tensor.hpp"
#include "../core/dtype.hpp"
#include <optional>
#include <utility>

namespace tenzor {

/**
 * @brief Per-tensor scaling parameters for FP8 quantization.
 */
struct FP8ScalingParams {
    float scale;        ///< Scale factor: scale = amax / fp8_max
    float amax;         ///< Absolute maximum value of the original tensor
    DType fp8_dtype;    ///< Target FP8 dtype (FP8_E4M3 or FP8_E5M2)
};

/**
 * @brief Maximum representable value for an FP8 data type.
 *
 * @param fp8_dtype Must be FP8_E4M3 or FP8_E5M2
 * @return Maximum finite value
 * @throws std::invalid_argument if dtype is not an FP8 type
 */
auto fp8_max_value(DType fp8_dtype) -> float;

/**
 * @brief Compute the absolute maximum value of a tensor.
 *
 * @param t Input tensor (any floating-point dtype)
 * @return Absolute maximum value as float
 */
auto compute_amax(const Tensor& t) -> float;

/**
 * @brief Compute the FP8 quantization scale from an amax value.
 *
 * The scale maps the tensor's dynamic range to the FP8 representable range:
 *   scale = amax / fp8_max
 *
 * @param amax Absolute maximum value of the tensor
 * @param fp8_dtype Target FP8 dtype
 * @return Scale factor (>= minimum positive float to avoid division by zero)
 */
auto compute_fp8_scale(float amax, DType fp8_dtype) -> float;

/**
 * @brief Quantize a floating-point tensor to FP8 with per-tensor scaling.
 *
 * Applies: fp8_value = to_fp8(input / scale)
 *
 * If no scale is provided, one is computed from the tensor's amax.
 * Stochastic rounding adds uniform noise before rounding, reducing
 * quantization bias during training.
 *
 * @param input Input tensor (Float32, Float16, or BFloat16)
 * @param fp8_dtype Target FP8 dtype (FP8_E4M3 or FP8_E5M2)
 * @param scale Optional pre-computed scale (computed from input if nullopt)
 * @param stochastic_rounding Use stochastic rounding instead of round-to-nearest
 * @return Pair of (FP8 tensor, scaling parameters used)
 */
auto quantize_to_fp8(const Tensor& input, DType fp8_dtype,
                     std::optional<float> scale = std::nullopt,
                     bool stochastic_rounding = false) -> std::pair<Tensor, FP8ScalingParams>;

/**
 * @brief Dequantize an FP8 tensor back to Float32 using a scale factor.
 *
 * Applies: output = to_float32(fp8_tensor) * scale
 *
 * @param fp8_tensor FP8 tensor to dequantize
 * @param scale Scale factor from quantization
 * @return Float32 tensor
 */
auto dequantize_from_fp8(const Tensor& fp8_tensor, float scale) -> Tensor;

} // namespace tenzor
