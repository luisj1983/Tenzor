/**
 * @file quantize.cpp
 * @brief Implementation of quantization operations
 */

#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/core/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace nn {
namespace quantization {

namespace {

// Get quantization range for data type
auto get_quant_range(QuantDType dtype) -> std::pair<int32_t, int32_t> {
    switch (dtype) {
        case QuantDType::INT8:
            return {-128, 127};
        case QuantDType::UINT8:
            return {0, 255};
        default:
            throw std::runtime_error("Unsupported quantization dtype");
    }
}

// Compute scale for symmetric quantization
auto compute_symmetric_scale(float abs_max, QuantDType dtype) -> float {
    auto [quant_min, quant_max] = get_quant_range(dtype);
    // For INT8 symmetric quantization, use [-127, 127] to avoid asymmetry at -128
    float quant_range = (dtype == QuantDType::INT8) ? 127.0f :
                        static_cast<float>(std::max(std::abs(quant_min), std::abs(quant_max)));

    // Add epsilon to handle zero-range edge case (all values identical)
    constexpr float EPSILON = 1e-8f;
    abs_max = std::max(abs_max, EPSILON);

    return abs_max / quant_range;
}

// Compute scale and zero_point for asymmetric quantization
auto compute_asymmetric_params(float min_val, float max_val, QuantDType dtype)
    -> std::pair<float, int32_t> {
    auto [quant_min, quant_max] = get_quant_range(dtype);

    // Handle edge case where all values are identical (min == max)
    constexpr float EPSILON = 1e-8f;
    if (std::abs(max_val - min_val) < EPSILON) {
        // When all values are the same, set scale to small value and zero_point to center
        float scale = EPSILON;
        int32_t zero_point = (quant_min + quant_max) / 2;
        return {scale, zero_point};
    }

    float quant_range = static_cast<float>(quant_max - quant_min);
    float scale = (max_val - min_val) / quant_range;

    // Additional safety check for very small scales
    scale = std::max(scale, EPSILON);

    int32_t zero_point = static_cast<int32_t>(std::round(quant_min - min_val / scale));
    zero_point = std::clamp(zero_point, quant_min, quant_max);

    return {scale, zero_point};
}

} // anonymous namespace

// ============================================================================
// QuantizedTensor
// ============================================================================

auto QuantizedTensor::dequantize() const -> Tensor {
    return dequantize_tensor(*this);
}

// ============================================================================
// Quantization Parameter Computation
// ============================================================================

auto compute_quantization_params(
    const Tensor& min,
    const Tensor& max,
    QuantDType dtype,
    QuantizationScheme scheme
) -> QuantizationParams {
    // Validate inputs
    if (min.numel() != max.numel()) {
        throw std::runtime_error("Min and max tensors must have same number of elements");
    }

    // Convert to Float32 and move to CPU for data access
    Tensor min_f32 = min;
    Tensor max_f32 = max;
    if (min.dtype() != DType::Float32) {
        min_f32 = min.to(DType::Float32);
    }
    if (max.dtype() != DType::Float32) {
        max_f32 = max.to(DType::Float32);
    }
    if (min_f32.device() != Device::cpu()) {
        min_f32 = min_f32.to(Device::cpu());
    }
    if (max_f32.device() != Device::cpu()) {
        max_f32 = max_f32.to(Device::cpu());
    }

    // Create scale and zero_point on CPU for data access
    Tensor scale_cpu({min.numel()}, DType::Float32, Device::cpu());
    Tensor zero_point_cpu({min.numel()}, DType::Int32, Device::cpu());

    // Get data pointers
    const float* min_data = min_f32.data<const float>();
    const float* max_data = max_f32.data<const float>();
    float* scale_data = scale_cpu.data<float>();
    int32_t* zp_data = zero_point_cpu.data<int32_t>();

    int64_t n = min.numel();

    if (scheme == QuantizationScheme::PerTensorSymmetric ||
        scheme == QuantizationScheme::PerChannelSymmetric) {
        // Symmetric quantization
        for (int64_t i = 0; i < n; ++i) {
            float abs_max = std::max(std::abs(min_data[i]), std::abs(max_data[i]));
            scale_data[i] = compute_symmetric_scale(abs_max, dtype);
            zp_data[i] = 0;  // Symmetric uses zero_point = 0
        }
    } else {
        // Asymmetric quantization
        for (int64_t i = 0; i < n; ++i) {
            auto [s, zp] = compute_asymmetric_params(min_data[i], max_data[i], dtype);
            scale_data[i] = s;
            zp_data[i] = zp;
        }
    }

    int64_t axis = (scheme == QuantizationScheme::PerChannelSymmetric ||
                    scheme == QuantizationScheme::PerChannelAsymmetric) ? 0 : -1;

    // Keep scale and zero_point on CPU (they're small scalars, better for access)
    return QuantizationParams(scale_cpu, zero_point_cpu, dtype, scheme, axis);
}

// ============================================================================
// Quantization Operations
// ============================================================================

auto quantize_tensor(const Tensor& input, const QuantizationParams& params)
    -> QuantizedTensor {
    auto [quant_min, quant_max] = get_quant_range(params.dtype);

    // For symmetric INT8 quantization, use [-127, 127] range to maintain true symmetry
    if (params.dtype == QuantDType::INT8 &&
        (params.scheme == QuantizationScheme::PerTensorSymmetric ||
         params.scheme == QuantizationScheme::PerChannelSymmetric)) {
        quant_min = -127;
        quant_max = 127;
    }

    // Create output tensor with appropriate dtype
    DType out_dtype = (params.dtype == QuantDType::INT8) ? DType::Int8 : DType::UInt8;
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Remember original device
    auto original_device = input.device();

    // Convert input to Float32 and CPU for processing
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }
    if (input_f32.device() != Device::cpu()) {
        input_f32 = input_f32.to(Device::cpu());
    }

    // Move params to CPU for data access
    Tensor scale_cpu = params.scale;
    Tensor zp_cpu = params.zero_point;
    if (scale_cpu.device() != Device::cpu()) {
        scale_cpu = scale_cpu.to(Device::cpu());
    }
    if (zp_cpu.device() != Device::cpu()) {
        zp_cpu = zp_cpu.to(Device::cpu());
    }

    // Create output on CPU
    Tensor quantized_cpu(shape_vec, out_dtype, Device::cpu());

    const float* input_data = input_f32.data<const float>();
    const float* scale_data = scale_cpu.data<const float>();
    const int32_t* zp_data = zp_cpu.data<int32_t>();

    int64_t n = input.numel();

    if (params.axis == -1) {
        // Per-tensor quantization
        float scale = scale_data[0];
        int32_t zero_point = zp_data[0];
        float inv_scale = 1.0f / scale;

        if (params.dtype == QuantDType::INT8) {
            int8_t* out_data = quantized_cpu.data<int8_t>();
            for (int64_t i = 0; i < n; ++i) {
                int32_t quantized_val = static_cast<int32_t>(std::round(input_data[i] * inv_scale)) + zero_point;
                out_data[i] = static_cast<int8_t>(std::clamp(quantized_val, quant_min, quant_max));
            }
        } else {
            uint8_t* out_data = quantized_cpu.data<uint8_t>();
            for (int64_t i = 0; i < n; ++i) {
                int32_t quantized_val = static_cast<int32_t>(std::round(input_data[i] * inv_scale)) + zero_point;
                out_data[i] = static_cast<uint8_t>(std::clamp(quantized_val, quant_min, quant_max));
            }
        }
    } else {
        // Per-channel quantization
        auto shape = input.shape();
        int64_t num_channels = shape[params.axis];
        int64_t channel_size = n / num_channels;

        for (int64_t c = 0; c < num_channels; ++c) {
            float scale = scale_data[c];
            int32_t zero_point = zp_data[c];
            float inv_scale = 1.0f / scale;

            if (params.dtype == QuantDType::INT8) {
                int8_t* out_data = quantized_cpu.data<int8_t>();
                for (int64_t i = 0; i < channel_size; ++i) {
                    int64_t idx = c * channel_size + i;
                    int32_t quantized_val = static_cast<int32_t>(std::round(input_data[idx] * inv_scale)) + zero_point;
                    out_data[idx] = static_cast<int8_t>(std::clamp(quantized_val, quant_min, quant_max));
                }
            } else {
                uint8_t* out_data = quantized_cpu.data<uint8_t>();
                for (int64_t i = 0; i < channel_size; ++i) {
                    int64_t idx = c * channel_size + i;
                    int32_t quantized_val = static_cast<int32_t>(std::round(input_data[idx] * inv_scale)) + zero_point;
                    out_data[idx] = static_cast<uint8_t>(std::clamp(quantized_val, quant_min, quant_max));
                }
            }
        }
    }

    // Move quantized tensor back to original device
    Tensor quantized = quantized_cpu.to(original_device);

    return QuantizedTensor(quantized, params, input.dtype());
}

auto quantize_per_tensor_symmetric(const Tensor& input, QuantDType dtype)
    -> QuantizedTensor {
    // Convert to Float32 and CPU for data access
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }
    if (input_f32.device() != Device::cpu()) {
        input_f32 = input_f32.to(Device::cpu());
    }
    const float* data = input_f32.data<const float>();
    int64_t n = input.numel();

    float min_val = data[0];
    float max_val = data[0];
    for (int64_t i = 1; i < n; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    Tensor min({1}, DType::Float32, input.device());
    Tensor max({1}, DType::Float32, input.device());
    min.fill_(min_val);
    max.fill_(max_val);

    auto params = compute_quantization_params(min, max, dtype,
                                             QuantizationScheme::PerTensorSymmetric);
    return quantize_tensor(input, params);
}

auto quantize_per_tensor_asymmetric(const Tensor& input, QuantDType dtype)
    -> QuantizedTensor {
    // Convert to Float32 and CPU for data access
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }
    if (input_f32.device() != Device::cpu()) {
        input_f32 = input_f32.to(Device::cpu());
    }
    const float* data = input_f32.data<const float>();
    int64_t n = input.numel();

    float min_val = data[0];
    float max_val = data[0];
    for (int64_t i = 1; i < n; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    Tensor min({1}, DType::Float32, input.device());
    Tensor max({1}, DType::Float32, input.device());
    min.fill_(min_val);
    max.fill_(max_val);

    auto params = compute_quantization_params(min, max, dtype,
                                             QuantizationScheme::PerTensorAsymmetric);
    return quantize_tensor(input, params);
}

auto quantize_per_channel_symmetric(const Tensor& input, int64_t axis, QuantDType dtype)
    -> QuantizedTensor {
    auto shape = input.shape();
    int64_t num_channels = shape[axis];
    int64_t channel_size = input.numel() / num_channels;

    // Create min/max on CPU for data access
    Tensor min({num_channels}, DType::Float32, Device::cpu());
    Tensor max({num_channels}, DType::Float32, Device::cpu());

    // Convert input to Float32 and CPU for data access
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }
    if (input_f32.device() != Device::cpu()) {
        input_f32 = input_f32.to(Device::cpu());
    }
    const float* input_data = input_f32.data<const float>();
    float* min_data = min.data<float>();
    float* max_data = max.data<float>();

    // Compute per-channel min/max
    for (int64_t c = 0; c < num_channels; ++c) {
        float ch_min = input_data[c * channel_size];
        float ch_max = input_data[c * channel_size];

        for (int64_t i = 0; i < channel_size; ++i) {
            float val = input_data[c * channel_size + i];
            ch_min = std::min(ch_min, val);
            ch_max = std::max(ch_max, val);
        }

        min_data[c] = ch_min;
        max_data[c] = ch_max;
    }

    // Keep min/max on CPU for compute_quantization_params (params stay on CPU)
    auto params = compute_quantization_params(min, max, dtype,
                                             QuantizationScheme::PerChannelSymmetric);
    params.axis = axis;
    return quantize_tensor(input, params);
}

auto quantize_per_channel_asymmetric(const Tensor& input, int64_t axis, QuantDType dtype)
    -> QuantizedTensor {
    auto shape = input.shape();
    int64_t num_channels = shape[axis];
    int64_t channel_size = input.numel() / num_channels;

    // Create min/max on CPU for data access
    Tensor min({num_channels}, DType::Float32, Device::cpu());
    Tensor max({num_channels}, DType::Float32, Device::cpu());

    // Convert input to Float32 and CPU for data access
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }
    if (input_f32.device() != Device::cpu()) {
        input_f32 = input_f32.to(Device::cpu());
    }
    const float* input_data = input_f32.data<const float>();
    float* min_data = min.data<float>();
    float* max_data = max.data<float>();

    for (int64_t c = 0; c < num_channels; ++c) {
        float ch_min = input_data[c * channel_size];
        float ch_max = input_data[c * channel_size];

        for (int64_t i = 0; i < channel_size; ++i) {
            float val = input_data[c * channel_size + i];
            ch_min = std::min(ch_min, val);
            ch_max = std::max(ch_max, val);
        }

        min_data[c] = ch_min;
        max_data[c] = ch_max;
    }

    // Keep min/max on CPU for compute_quantization_params (params stay on CPU)
    auto params = compute_quantization_params(min, max, dtype,
                                             QuantizationScheme::PerChannelAsymmetric);
    params.axis = axis;
    return quantize_tensor(input, params);
}

// ============================================================================
// Dequantization
// ============================================================================

auto dequantize_tensor(const QuantizedTensor& quantized) -> Tensor {
    const auto& params = quantized.params();
    const auto& q_data = quantized.data();

    // Remember original device
    auto original_device = q_data.device();

    // Move to CPU for data access
    Tensor q_data_cpu = q_data;
    if (q_data_cpu.device() != Device::cpu()) {
        q_data_cpu = q_data_cpu.to(Device::cpu());
    }

    Tensor scale_cpu = params.scale;
    Tensor zp_cpu = params.zero_point;
    if (scale_cpu.device() != Device::cpu()) {
        scale_cpu = scale_cpu.to(Device::cpu());
    }
    if (zp_cpu.device() != Device::cpu()) {
        zp_cpu = zp_cpu.to(Device::cpu());
    }

    auto shape_vec = std::vector<int64_t>(q_data.shape().begin(), q_data.shape().end());
    Tensor output(shape_vec, DType::Float32, Device::cpu());
    float* out_data = output.data<float>();

    const float* scale_data = scale_cpu.data<const float>();
    const int32_t* zp_data = zp_cpu.data<int32_t>();

    int64_t n = q_data.numel();

    if (params.axis == -1) {
        // Per-tensor dequantization
        float scale = scale_data[0];
        int32_t zero_point = zp_data[0];

        if (params.dtype == QuantDType::INT8) {
            const int8_t* q_ptr = q_data_cpu.data<int8_t>();
            for (int64_t i = 0; i < n; ++i) {
                out_data[i] = (static_cast<float>(q_ptr[i]) - zero_point) * scale;
            }
        } else {
            const uint8_t* q_ptr = q_data_cpu.data<uint8_t>();
            for (int64_t i = 0; i < n; ++i) {
                out_data[i] = (static_cast<float>(q_ptr[i]) - zero_point) * scale;
            }
        }
    } else {
        // Per-channel dequantization
        auto shape = q_data.shape();
        int64_t num_channels = shape[params.axis];
        int64_t channel_size = n / num_channels;

        for (int64_t c = 0; c < num_channels; ++c) {
            float scale = scale_data[c];
            int32_t zero_point = zp_data[c];

            if (params.dtype == QuantDType::INT8) {
                const int8_t* q_ptr = q_data_cpu.data<int8_t>();
                for (int64_t i = 0; i < channel_size; ++i) {
                    int64_t idx = c * channel_size + i;
                    out_data[idx] = (static_cast<float>(q_ptr[idx]) - zero_point) * scale;
                }
            } else {
                const uint8_t* q_ptr = q_data_cpu.data<uint8_t>();
                for (int64_t i = 0; i < channel_size; ++i) {
                    int64_t idx = c * channel_size + i;
                    out_data[idx] = (static_cast<float>(q_ptr[idx]) - zero_point) * scale;
                }
            }
        }
    }

    // Move output back to original device
    output = output.to(original_device);

    // Convert to original dtype if different from Float32
    DType orig_dtype = quantized.original_dtype();
    if (orig_dtype != DType::Float32) {
        output = output.to(orig_dtype);
    }

    return output;
}

// ============================================================================
// Error Metrics
// ============================================================================

auto compute_quantization_error(const Tensor& original, const QuantizedTensor& quantized)
    -> std::tuple<float, float, float> {
    Tensor dequant = quantized.dequantize();

    // Convert to Float32 and CPU for data access
    Tensor orig_f32 = original;
    if (original.dtype() != DType::Float32) {
        orig_f32 = original.to(DType::Float32);
    }
    if (orig_f32.device() != Device::cpu()) {
        orig_f32 = orig_f32.to(Device::cpu());
    }
    Tensor deq_f32 = dequant;
    if (dequant.dtype() != DType::Float32) {
        deq_f32 = dequant.to(DType::Float32);
    }
    if (deq_f32.device() != Device::cpu()) {
        deq_f32 = deq_f32.to(Device::cpu());
    }
    const float* orig_data = orig_f32.data<const float>();
    const float* deq_data = deq_f32.data<const float>();
    int64_t n = original.numel();

    float mae = 0.0f;
    float mse = 0.0f;
    float signal_power = 0.0f;

    for (int64_t i = 0; i < n; ++i) {
        float error = orig_data[i] - deq_data[i];
        mae += std::abs(error);
        mse += error * error;
        signal_power += orig_data[i] * orig_data[i];
    }

    mae /= n;
    mse /= n;
    signal_power /= n;

    // Signal-to-noise ratio in dB
    // Handle edge cases: when signal is near zero or when quantization is perfect
    constexpr float EPSILON = 1e-10f;
    float snr_db;

    // For near-zero signals, SNR calculation is not meaningful
    // For perfect quantization (mse ≈ 0), SNR should be very high
    if (mse < EPSILON) {
        // Perfect or near-perfect quantization
        snr_db = 100.0f;  // Arbitrarily high SNR
    } else if (signal_power < EPSILON) {
        // Near-zero signal - SNR not meaningful, but return reasonable value
        snr_db = 0.0f;
    } else {
        snr_db = 10.0f * std::log10(signal_power / mse);
    }

    return {mae, mse, snr_db};
}

// ============================================================================
// Calibration
// ============================================================================

auto calibrate_quantization_params(
    const std::vector<Tensor>& samples,
    QuantDType dtype,
    QuantizationScheme scheme,
    int64_t axis
) -> QuantizationParams {
    if (samples.empty()) {
        throw std::runtime_error("Cannot calibrate with empty sample set");
    }

    constexpr float EPSILON = 1e-8f;

    // Check if per-channel quantization
    bool is_per_channel = (scheme == QuantizationScheme::PerChannelSymmetric ||
                          scheme == QuantizationScheme::PerChannelAsymmetric);

    if (is_per_channel && axis >= 0) {
        // Per-channel quantization
        auto shape = samples[0].shape();
        int64_t num_channels = shape[axis];
        int64_t channel_size = samples[0].numel() / num_channels;

        // Create min/max on CPU for data access
        Tensor min({num_channels}, DType::Float32, Device::cpu());
        Tensor max({num_channels}, DType::Float32, Device::cpu());
        float* min_data = min.data<float>();
        float* max_data = max.data<float>();

        // Initialize min/max for each channel
        for (int64_t c = 0; c < num_channels; ++c) {
            min_data[c] = std::numeric_limits<float>::max();
            max_data[c] = std::numeric_limits<float>::lowest();
        }

        // Compute per-channel min/max across all samples
        for (const auto& sample : samples) {
            // Convert to Float32 and CPU for data access
            Tensor sample_f32 = sample;
            if (sample.dtype() != DType::Float32) {
                sample_f32 = sample.to(DType::Float32);
            }
            if (sample_f32.device() != Device::cpu()) {
                sample_f32 = sample_f32.to(Device::cpu());
            }
            const float* data = sample_f32.data<const float>();
            for (int64_t c = 0; c < num_channels; ++c) {
                for (int64_t i = 0; i < channel_size; ++i) {
                    float val = data[c * channel_size + i];
                    min_data[c] = std::min(min_data[c], val);
                    max_data[c] = std::max(max_data[c], val);
                }
            }
        }

        // Handle edge case where all values in a channel are identical
        for (int64_t c = 0; c < num_channels; ++c) {
            if (std::abs(max_data[c] - min_data[c]) < EPSILON) {
                min_data[c] -= EPSILON;
                max_data[c] += EPSILON;
            }
        }

        // Keep min/max on CPU for compute_quantization_params (params stay on CPU)
        auto params = compute_quantization_params(min, max, dtype, scheme);
        params.axis = axis;
        return params;
    } else {
        // Per-tensor quantization
        float global_min = std::numeric_limits<float>::max();
        float global_max = std::numeric_limits<float>::lowest();

        bool has_data = false;
        for (const auto& sample : samples) {
            // Convert to Float32 and CPU for data access
            Tensor sample_f32 = sample;
            if (sample.dtype() != DType::Float32) {
                sample_f32 = sample.to(DType::Float32);
            }
            if (sample_f32.device() != Device::cpu()) {
                sample_f32 = sample_f32.to(Device::cpu());
            }
            const float* data = sample_f32.data<const float>();
            int64_t n = sample.numel();

            if (n == 0) continue;  // Skip empty tensors
            has_data = true;

            for (int64_t i = 0; i < n; ++i) {
                global_min = std::min(global_min, data[i]);
                global_max = std::max(global_max, data[i]);
            }
        }

        if (!has_data) {
            throw std::runtime_error("Cannot calibrate with all-empty sample tensors");
        }

        // Handle edge case where all values are identical
        if (std::abs(global_max - global_min) < EPSILON) {
            // Expand range slightly to avoid numerical issues
            global_min -= EPSILON;
            global_max += EPSILON;
        }

        Tensor min({1}, DType::Float32, samples[0].device());
        Tensor max({1}, DType::Float32, samples[0].device());
        min.fill_(global_min);
        max.fill_(global_max);

        return compute_quantization_params(min, max, dtype, scheme);
    }
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
