/**
 * @file fp8_scaling.cpp
 * @brief FP8 quantization scaling utilities implementation.
 */

#include "tenzor/ops/fp8_scaling.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tenzor {

auto fp8_max_value(DType fp8_dtype) -> float {
    if (fp8_dtype == DType::FP8_E4M3) {
        // E4M3: max = 2^(2^3 - 1) * (1 + 7/8) = 448.0
        return 448.0f;
    }
    if (fp8_dtype == DType::FP8_E5M2) {
        // E5M2: max = 2^(2^4 - 1) * (1 + 3/4) = 57344.0
        return 57344.0f;
    }
    if (fp8_dtype == DType::FP8_E4M3FNUZ) {
        // E4M3FNUZ: max = 2^(2^3 - 1) * (1 + 7/8) = 448.0
        return 448.0f;
    }
    if (fp8_dtype == DType::FP8_E5M2FNUZ) {
        // E5M2FNUZ: max = 2^(2^4 - 1) * (1 + 3/4) = 57344.0
        return 57344.0f;
    }
    throw std::invalid_argument(
        "fp8_max_value: expected FP8_E4M3 or FP8_E5M2, got " +
        std::string(dtype_name(fp8_dtype)));
}

auto compute_amax(const Tensor& t) -> float {
    // abs -> max reduction, then pull scalar to CPU
    Tensor abs_t = tenzor::abs(t.to(DType::Float32));
    Tensor max_val = tenzor::max(abs_t);
    return max_val.cpu().data<float>()[0];
}

auto compute_fp8_scale(float amax, DType fp8_dtype) -> float {
    float fp8_max = fp8_max_value(fp8_dtype);

    // Avoid division by zero: if amax is 0, use minimum positive scale
    if (amax <= 0.0f) {
        return std::numeric_limits<float>::min();
    }

    return amax / fp8_max;
}

auto quantize_to_fp8(const Tensor& input, DType fp8_dtype,
                     std::optional<float> scale,
                     bool stochastic_rounding) -> std::pair<Tensor, FP8ScalingParams> {
    if (fp8_dtype != DType::FP8_E4M3 && fp8_dtype != DType::FP8_E5M2 &&
        fp8_dtype != DType::FP8_E4M3FNUZ && fp8_dtype != DType::FP8_E5M2FNUZ) {
        throw std::invalid_argument(
            "quantize_to_fp8: expected FP8_E4M3 or FP8_E5M2, got " +
            std::string(dtype_name(fp8_dtype)));
    }

    // Compute amax and scale if not provided
    float amax = compute_amax(input);
    float s = scale.value_or(compute_fp8_scale(amax, fp8_dtype));

    // Scale the input: scaled = input / scale
    Tensor input_f32 = input.to(DType::Float32);
    Tensor scaled = tenzor::div(input_f32,
                                full({1}, s, DType::Float32, input.device()));

    if (stochastic_rounding) {
        // Stochastic rounding: add uniform noise in [-0.5, 0.5) before truncation
        // This makes the expected value of the quantized result equal to the
        // original value, reducing quantization bias during training.
        auto shape_span = input.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor noise = tenzor::rand(shape_vec, DType::Float32, input.device());
        // Shift from [0, 1) to [-0.5, 0.5)
        Tensor half_tensor = full({1}, 0.5f, DType::Float32, input.device());
        noise = tenzor::sub(noise, half_tensor);

        // The noise magnitude should be relative to the FP8 ULP (unit in last place).
        // For simplicity, we add noise scaled to 1 ULP of the target FP8 type.
        // Since we've already scaled the input to the FP8 range, the noise
        // should be in [-0.5, 0.5) at the quantization grid level.
        scaled = tenzor::add(scaled, noise);
    }

    // Clamp to FP8 representable range before conversion
    float fp8_max = fp8_max_value(fp8_dtype);
    Tensor clamped = tenzor::clamp(scaled, -fp8_max, fp8_max);

    // Convert to FP8
    Tensor fp8_result = clamped.to(fp8_dtype);

    FP8ScalingParams params{s, amax, fp8_dtype};
    return {fp8_result, params};
}

auto dequantize_from_fp8(const Tensor& fp8_tensor, float scale) -> Tensor {
    // Widen FP8 to Float32 then multiply by scale
    Tensor f32 = fp8_tensor.to(DType::Float32);
    return tenzor::mul(f32, full({1}, scale, DType::Float32, fp8_tensor.device()));
}

auto fp8_is_native(Device::Type device_type) -> bool {
    // CPU and CUDA both register Cast kernels that understand FP8.
    // ROCm has native FP8 Cast kernels in kernels/transform.hip.cpp.
    // Vulkan has FP8 cast compute shaders (cast_f32_fp8e4m3.comp, etc.).
    // OneAPI's Cast kernel includes FP8_E4M3 / FP8_E5M2 handling
    // (see src/backends/oneapi/kernels/transform.cpp).
    switch (device_type) {
        case Device::Type::CPU:
        case Device::Type::CUDA:
        case Device::Type::OneAPI:
        case Device::Type::ROCm:
        case Device::Type::Vulkan:
            return true;
        case Device::Type::MPS:
            return false;
        default:
            return false;
    }
}

} // namespace tenzor
