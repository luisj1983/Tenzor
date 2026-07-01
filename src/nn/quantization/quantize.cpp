/**
 * @file quantize.cpp
 * @brief Implementation of quantization operations
 */

#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/reduction.hpp"
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
        case QuantDType::INT4:
            return {-8, 7};
        case QuantDType::UINT4:
            return {0, 15};
        default:
            throw std::runtime_error("Unsupported quantization dtype");
    }
}

// Compute scale for symmetric quantization
auto compute_symmetric_scale(float abs_max, QuantDType dtype) -> float {
    auto [quant_min, quant_max] = get_quant_range(dtype);
    // For INT8 symmetric quantization, use [-127, 127] to avoid asymmetry at -128
    float quant_range;
    if (dtype == QuantDType::INT8) {
        quant_range = 127.0f;
    } else if (dtype == QuantDType::INT4) {
        quant_range = 7.0f;
    } else {
        quant_range = static_cast<float>(std::max(std::abs(quant_min), std::abs(quant_max)));
    }

    // Add epsilon to handle zero-range edge case (all values identical)
    constexpr float EPSILON = 1e-8f;
    abs_max = std::max(abs_max, EPSILON);

    return abs_max / quant_range;
}

// Compute scale and zero_point for asymmetric quantization
auto compute_asymmetric_params(float min_val, float max_val, QuantDType dtype)
    -> std::pair<float, int32_t> {
    auto [quant_min, quant_max] = get_quant_range(dtype);

    // Handle edge case where all values are identical (min == max). Hard-coding
    // scale=EPSILON (1e-8) destroyed the value: round(c / 1e-8) saturates to
    // quant_max and dequantizes to ~0. Instead extend the range to include 0
    // (matching PyTorch's choose_qparams) and derive a real scale/zero_point so
    // the constant dequantizes back to itself.
    constexpr float EPSILON = 1e-8f;
    if (std::abs(max_val - min_val) < EPSILON) {
        float lo = std::min(min_val, 0.0f);
        float hi = std::max(max_val, 0.0f);
        float quant_range = static_cast<float>(quant_max - quant_min);
        float scale = std::max((hi - lo) / quant_range, EPSILON);
        int32_t zero_point =
            static_cast<int32_t>(std::round(quant_min - lo / scale));
        zero_point = std::clamp(zero_point, quant_min, quant_max);
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
        // Symmetric quantization is only well-defined for signed integer
        // ranges; with an unsigned dtype zero_point=0 maps every negative
        // input to 0 (silent clamp). Reject the unsupported combination.
        if (dtype == QuantDType::UINT8 || dtype == QuantDType::UINT4) {
            throw std::runtime_error(
                "Symmetric quantization is not supported for unsigned dtypes "
                "(UINT8/UINT4); use an affine/asymmetric scheme or a signed dtype");
        }
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

    // For symmetric INT8/INT4 quantization, use symmetric range to maintain true symmetry
    if ((params.dtype == QuantDType::INT8 || params.dtype == QuantDType::INT4) &&
        (params.scheme == QuantizationScheme::PerTensorSymmetric ||
         params.scheme == QuantizationScheme::PerChannelSymmetric)) {
        if (params.dtype == QuantDType::INT8) {
            quant_min = -127;
            quant_max = 127;
        } else {
            quant_min = -7;
            quant_max = 7;
        }
    }

    // Create output tensor with appropriate dtype
    // INT4/UINT4 pack 2 elements per byte, stored in Int8/UInt8
    DType out_dtype;
    bool is_int4 = (params.dtype == QuantDType::INT4 || params.dtype == QuantDType::UINT4);
    if (params.dtype == QuantDType::INT8 || params.dtype == QuantDType::INT4) {
        out_dtype = DType::Int8;
    } else {
        out_dtype = DType::UInt8;
    }
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

    // Create output on CPU.
    // INT4/UINT4 packs 2 nibbles per byte; we use a single FLAT packing (byte
    // = flat_idx/2, nibble = flat_idx%2) with a 1-D storage of ceil(n/2) bytes.
    // This is unambiguous for any rank/odd dim AND, for per-channel, each
    // element is independently quantized with ITS OWN channel scale, so packing
    // two elements (possibly from different channels) into one byte never
    // corrupts data — fixing the old per-channel "overlapping bytes" bug.
    int64_t n = input.numel();
    std::vector<int64_t> storage_shape =
        is_int4 ? std::vector<int64_t>{(n + 1) / 2} : shape_vec;
    Tensor quantized_cpu(storage_shape, out_dtype, Device::cpu());

    const float* input_data = input_f32.data<const float>();
    const float* scale_data = scale_cpu.data<const float>();
    const int32_t* zp_data = zp_cpu.data<int32_t>();

    // Per-channel addressing: the channel of element `idx` (row-major) is
    // (idx / chan_inner) % num_channels — correct for ANY axis, not just axis 0.
    int64_t num_channels = 1;
    int64_t chan_inner = 1;
    if (params.axis != -1) {
        auto shape = input.shape();
        num_channels = shape[params.axis];
        for (int64_t d = params.axis + 1; d < static_cast<int64_t>(shape.size()); ++d) {
            chan_inner *= shape[d];
        }
    }
    auto channel_of = [&](int64_t idx) -> int64_t {
        return (params.axis == -1) ? 0 : (idx / chan_inner) % num_channels;
    };

    // Helper lambda: quantize a single float value
    auto quantize_val = [&](float val, float inv_scale, int32_t zero_point) -> int32_t {
        int32_t qval = static_cast<int32_t>(std::round(val * inv_scale)) + zero_point;
        return std::clamp(qval, quant_min, quant_max);
    };

    if (is_int4) {
        // INT4/UINT4 flat packing: low nibble = even flat index, high = odd.
        // Each element uses its own channel's scale (channel_of(idx)). The
        // packed-byte arithmetic is identical for both, but the typed accessor
        // must match the storage dtype: UINT4 storage is UInt8 (data<uint8_t>),
        // INT4 storage is Int8 (data<int8_t>). Reading UInt8 storage via
        // data<int8_t>() throws, which previously made UINT4 unusable.
        auto pack = [&](int64_t i) -> uint8_t {
            int64_t c0 = channel_of(i);
            int32_t lo =
                quantize_val(input_data[i], 1.0f / scale_data[c0], zp_data[c0]);
            int32_t hi = 0;
            if (i + 1 < n) {
                int64_t c1 = channel_of(i + 1);
                hi = quantize_val(input_data[i + 1], 1.0f / scale_data[c1], zp_data[c1]);
            }
            return static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
        };
        if (params.dtype == QuantDType::UINT4) {
            uint8_t* out_data = quantized_cpu.data<uint8_t>();
            for (int64_t i = 0; i < n; i += 2) {
                out_data[i / 2] = pack(i);
            }
        } else {  // INT4 — storage is Int8
            int8_t* out_data = quantized_cpu.data<int8_t>();
            for (int64_t i = 0; i < n; i += 2) {
                out_data[i / 2] = static_cast<int8_t>(pack(i));
            }
        }
    } else if (params.dtype == QuantDType::INT8) {
        int8_t* out_data = quantized_cpu.data<int8_t>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t c = channel_of(i);
            out_data[i] = static_cast<int8_t>(
                quantize_val(input_data[i], 1.0f / scale_data[c], zp_data[c]));
        }
    } else {  // UINT8
        uint8_t* out_data = quantized_cpu.data<uint8_t>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t c = channel_of(i);
            out_data[i] = static_cast<uint8_t>(
                quantize_val(input_data[i], 1.0f / scale_data[c], zp_data[c]));
        }
    }

    // Move quantized tensor back to original device
    Tensor quantized = quantized_cpu.to(original_device);

    return QuantizedTensor(quantized, params, input.dtype(), shape_vec);
}

auto quantize_per_tensor_symmetric(const Tensor& input, QuantDType dtype)
    -> QuantizedTensor {
    // Use dispatched min/max reductions (GPU-safe)
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }

    Tensor min = tenzor::min(input_f32);
    Tensor max = tenzor::max(input_f32);

    auto params = compute_quantization_params(min, max, dtype,
                                             QuantizationScheme::PerTensorSymmetric);
    return quantize_tensor(input, params);
}

auto quantize_per_tensor_asymmetric(const Tensor& input, QuantDType dtype)
    -> QuantizedTensor {
    // Use dispatched min/max reductions (GPU-safe)
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }

    Tensor min = tenzor::min(input_f32);
    Tensor max = tenzor::max(input_f32);

    auto params = compute_quantization_params(min, max, dtype,
                                             QuantizationScheme::PerTensorAsymmetric);
    return quantize_tensor(input, params);
}

auto quantize_per_channel_symmetric(const Tensor& input, int64_t axis, QuantDType dtype)
    -> QuantizedTensor {
    // Use dispatched reductions (GPU-safe)
    // Reshape to [num_channels, -1] to reduce all non-channel dims
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }

    auto shape = input_f32.shape();
    int64_t num_channels = shape[axis];
    int64_t rest = input_f32.numel() / num_channels;

    // Move channel dim to front if needed, then reshape to [C, rest]
    Tensor reshaped;
    if (axis == 0) {
        reshaped = input_f32.reshape({num_channels, rest});
    } else {
        reshaped = input_f32.transpose(0, axis).contiguous().reshape({num_channels, rest});
    }

    // Reduce along dim 1 to get per-channel min/max
    Tensor min = tenzor::min(reshaped, 1, /*keepdim=*/false);
    Tensor max = tenzor::max(reshaped, 1, /*keepdim=*/false);

    auto params = compute_quantization_params(min, max, dtype,
                                             QuantizationScheme::PerChannelSymmetric);
    params.axis = axis;
    return quantize_tensor(input, params);
}

auto quantize_per_channel_asymmetric(const Tensor& input, int64_t axis, QuantDType dtype)
    -> QuantizedTensor {
    // Use dispatched reductions (GPU-safe)
    // Reshape to [num_channels, -1] to reduce all non-channel dims
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }

    auto shape = input_f32.shape();
    int64_t num_channels = shape[axis];
    int64_t rest = input_f32.numel() / num_channels;

    // Move channel dim to front if needed, then reshape to [C, rest]
    Tensor reshaped;
    if (axis == 0) {
        reshaped = input_f32.reshape({num_channels, rest});
    } else {
        reshaped = input_f32.transpose(0, axis).contiguous().reshape({num_channels, rest});
    }

    // Reduce along dim 1 to get per-channel min/max
    Tensor min = tenzor::min(reshaped, 1, /*keepdim=*/false);
    Tensor max = tenzor::max(reshaped, 1, /*keepdim=*/false);

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

    bool is_int4 = (params.dtype == QuantDType::INT4 || params.dtype == QuantDType::UINT4);

    // For INT4, the stored tensor is a flat packed byte buffer; the true logical
    // shape (and odd-dim element count) is carried explicitly on the
    // QuantizedTensor since it cannot be recovered from the packed byte count.
    std::vector<int64_t> shape_vec;
    int64_t n;
    if (is_int4) {
        shape_vec = quantized.original_shape();
        n = 1;
        for (int64_t d : shape_vec) n *= d;
        if (shape_vec.empty()) n = 0;
    } else {
        shape_vec.assign(q_data.shape().begin(), q_data.shape().end());
        n = q_data.numel();
    }

    Tensor output(shape_vec, DType::Float32, Device::cpu());
    float* out_data = output.data<float>();

    const float* scale_data = scale_cpu.data<const float>();
    const int32_t* zp_data = zp_cpu.data<int32_t>();

    // Per-channel addressing (mirror of quantize_tensor): channel of element
    // `idx` is (idx / chan_inner) % num_channels for ANY axis.
    int64_t deq_num_channels = 1;
    int64_t deq_chan_inner = 1;
    if (params.axis != -1) {
        deq_num_channels = shape_vec[params.axis];
        for (int64_t d = params.axis + 1; d < static_cast<int64_t>(shape_vec.size()); ++d) {
            deq_chan_inner *= shape_vec[d];
        }
    }
    auto deq_channel_of = [&](int64_t idx) -> int64_t {
        return (params.axis == -1) ? 0 : (idx / deq_chan_inner) % deq_num_channels;
    };

    if (is_int4) {
        // INT4/UINT4 flat unpacking: low nibble = even flat index, high = odd.
        // Each element is dequantized with ITS OWN channel scale. The typed
        // accessor must match the storage dtype: UINT4 storage is UInt8, INT4
        // storage is Int8. Reading UInt8 storage via data<int8_t>() throws,
        // which previously made UINT4 dequantize unusable. Copy the packed
        // bytes to a uint8 view so the nibble math (on raw bits) is uniform.
        const bool is_signed = (params.dtype == QuantDType::INT4);
        const int64_t num_bytes = (n + 1) / 2;
        std::vector<uint8_t> packed_bytes(static_cast<size_t>(num_bytes));
        if (params.dtype == QuantDType::UINT4) {
            const uint8_t* q_ptr = q_data_cpu.data<uint8_t>();
            for (int64_t b = 0; b < num_bytes; ++b) packed_bytes[b] = q_ptr[b];
        } else {  // INT4 — storage is Int8
            const int8_t* q_ptr = q_data_cpu.data<int8_t>();
            for (int64_t b = 0; b < num_bytes; ++b) {
                packed_bytes[b] = static_cast<uint8_t>(q_ptr[b]);
            }
        }

        for (int64_t i = 0; i < n; ++i) {
            uint8_t packed = packed_bytes[static_cast<size_t>(i / 2)];
            int8_t nib = static_cast<int8_t>((i % 2 == 0) ? (packed & 0x0F)
                                                          : ((packed >> 4) & 0x0F));
            if (is_signed && (nib & 0x08)) nib |= static_cast<int8_t>(0xF0);
            int64_t c = deq_channel_of(i);
            out_data[i] = (static_cast<float>(nib) - zp_data[c]) * scale_data[c];
        }
    } else if (params.axis == -1) {
        // Per-tensor dequantization (INT8/UINT8)
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
        // Per-channel dequantization (INT8/UINT8). Use the per-element channel
        // mapping so it is correct for ANY axis (matches quantize_tensor).
        if (params.dtype == QuantDType::INT8) {
            const int8_t* q_ptr = q_data_cpu.data<int8_t>();
            for (int64_t idx = 0; idx < n; ++idx) {
                int64_t c = deq_channel_of(idx);
                out_data[idx] = (static_cast<float>(q_ptr[idx]) - zp_data[c]) * scale_data[c];
            }
        } else {
            const uint8_t* q_ptr = q_data_cpu.data<uint8_t>();
            for (int64_t idx = 0; idx < n; ++idx) {
                int64_t c = deq_channel_of(idx);
                out_data[idx] = (static_cast<float>(q_ptr[idx]) - zp_data[c]) * scale_data[c];
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
    if (n == 0) {
        throw std::invalid_argument(
            "compute_quantization_error: original tensor is empty (numel == 0)");
    }

    // Accumulate in double precision to avoid the float-accumulator-in-a-loop
    // precision loss across up to numel elements (signal_power/mse can otherwise
    // drift several percent for large tensors, distorting the reported SNR).
    double mae_acc = 0.0;
    double mse_acc = 0.0;
    double signal_power_acc = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        double error = static_cast<double>(orig_data[i]) - static_cast<double>(deq_data[i]);
        mae_acc += std::abs(error);
        mse_acc += error * error;
        signal_power_acc += static_cast<double>(orig_data[i]) * static_cast<double>(orig_data[i]);
    }

    double nd = static_cast<double>(n);
    float mae = static_cast<float>(mae_acc / nd);
    float mse = static_cast<float>(mse_acc / nd);
    float signal_power = static_cast<float>(signal_power_acc / nd);

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

    // Normalize a negative axis against the sample rank so a per-channel scheme
    // requested with axis=-1 (the last-dim convention used elsewhere) is not
    // silently downgraded to per-tensor by the axis>=0 branch below.
    if (axis < 0) axis += static_cast<int64_t>(samples[0].ndim());

    // Check if per-channel quantization
    bool is_per_channel = (scheme == QuantizationScheme::PerChannelSymmetric ||
                          scheme == QuantizationScheme::PerChannelAsymmetric);

    if (is_per_channel && axis >= 0) {
        // Per-channel quantization
        auto shape = samples[0].shape();
        int64_t num_channels = shape[axis];

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

        // Compute per-channel min/max across all samples. The naive
        // `data[c*channel_size + i]` indexing is only correct for axis==0
        // (channel-major contiguous). For axis!=0 the element at that flat
        // offset belongs to a DIFFERENT channel, so move the channel axis to
        // the front (transpose+contiguous) and reshape to [C, rest] — exactly
        // what quantize_per_channel_* and MinMaxObserver::per_channel_reduce
        // do — before grouping by channel.
        for (const auto& sample : samples) {
            // Convert to Float32 and CPU for data access
            Tensor sample_f32 = sample;
            if (sample.dtype() != DType::Float32) {
                sample_f32 = sample.to(DType::Float32);
            }
            if (sample_f32.device() != Device::cpu()) {
                sample_f32 = sample_f32.to(Device::cpu());
            }
            int64_t channel_size = sample_f32.numel() / num_channels;
            Tensor reshaped;
            if (axis == 0) {
                reshaped = sample_f32.reshape({num_channels, channel_size});
            } else {
                reshaped = sample_f32.transpose(0, axis).contiguous()
                                     .reshape({num_channels, channel_size});
            }
            const float* data = reshaped.data<const float>();
            for (int64_t c = 0; c < num_channels; ++c) {
                for (int64_t i = 0; i < channel_size; ++i) {
                    float val = data[c * channel_size + i];
                    min_data[c] = std::min(min_data[c], val);
                    max_data[c] = std::max(max_data[c], val);
                }
            }
        }

        // Degenerate (all-identical) channels are handled correctly downstream
        // by compute_asymmetric_params / compute_symmetric_scale, which extend
        // the range to include 0 so the constant round-trips. Widening min/max
        // by EPSILON here would instead bypass that path and yield a tiny,
        // value-destroying scale for large constants.

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

        // Degenerate (all-identical) inputs are handled correctly downstream by
        // compute_asymmetric_params / compute_symmetric_scale (range extended to
        // include 0 so the constant round-trips); no EPSILON widening needed.

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
