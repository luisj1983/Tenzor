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

namespace {

/// FP8 format parameters needed to compute the per-element ULP (unit in the
/// last place) used by stochastic rounding.
///   mantissa_bits : explicit mantissa bits (E4M3 -> 3, E5M2 -> 2)
///   emin          : minimum *normal* unbiased exponent
///   emax          : maximum unbiased exponent of the largest finite value
/// The ULP of a normal FP8 number with magnitude |x| and binade exponent
/// e = floor(log2(|x|)) is 2^(e - mantissa_bits). In the subnormal region the
/// spacing is constant and equals the ULP at e = emin, so clamping e to
/// [emin, emax] yields the correct grid spacing across the whole range.
struct FP8Params {
    int mantissa_bits;
    int emin;
    int emax;
};

auto fp8_grid_params(DType fp8_dtype) -> FP8Params {
    switch (fp8_dtype) {
        // E4M3: 4 exp bits, bias 7. Normal exponents [1-7, ..]; largest finite
        // value 448 = 1.75 * 2^8 -> emax = 8.
        case DType::FP8_E4M3:     return {3, -6, 8};
        // E5M2: 5 exp bits, bias 15. Largest finite 57344 = 1.75 * 2^15.
        case DType::FP8_E5M2:     return {2, -14, 15};
        // FNUZ variants use a bias one larger (no inf/nan encodings), so the
        // minimum normal exponent shifts down by one; max value is unchanged.
        case DType::FP8_E4M3FNUZ: return {3, -7, 8};
        case DType::FP8_E5M2FNUZ: return {2, -15, 15};
        default:
            throw std::invalid_argument(
                "fp8_grid_params: expected an FP8 dtype, got " +
                std::string(dtype_name(fp8_dtype)));
    }
}

} // namespace

auto compute_amax(const Tensor& t) -> float {
    // An empty tensor has no elements, so the global max reduction has no
    // identity and would throw/return garbage. amax of an empty tensor is 0
    // (no magnitude to represent), which compute_fp8_scale maps to the
    // minimum positive scale.
    if (t.numel() == 0) {
        return 0.0f;
    }
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
        // Stochastic rounding: perturb each element by uniform noise spanning one
        // FP8 ULP (unit in the last place) *at that element's magnitude*, then let
        // .to(fp8_dtype) round to nearest. Because FP8 is a floating-point grid,
        // the spacing between representable values is not constant in `scaled`
        // space (e.g. ~32 near 448 but ~0.06 near 1.0 for E4M3). Adding a fixed
        // [-0.5, 0.5) perturbation would only be unbiased on a uniform integer
        // grid; here it must be scaled per element so that
        //   E[ to_fp8(scaled + noise) ] == scaled
        // holds across the whole dynamic range.
        FP8Params fp = fp8_grid_params(fp8_dtype);

        // Binade exponent e = floor(log2(|scaled|)), clamped to the FP8 normal
        // exponent range. Below emin the subnormal spacing is constant and equals
        // the ULP at emin, so clamping is exactly right. log2(0) = -inf is folded
        // to emin by the clamp, and the max binade is capped at emax.
        Tensor abs_scaled = tenzor::abs(scaled);
        Tensor exponent = tenzor::floor(tenzor::log2(abs_scaled));
        exponent = tenzor::clamp(exponent,
                                 static_cast<double>(fp.emin),
                                 static_cast<double>(fp.emax));

        // ulp = 2^(e - mantissa_bits)
        Tensor ulp = tenzor::exp2(
            tenzor::sub(exponent, static_cast<double>(fp.mantissa_bits)));

        // noise in [-0.5, 0.5) * ulp
        auto shape_span = input.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor noise = tenzor::rand(shape_vec, DType::Float32, input.device());
        noise = tenzor::sub(noise, 0.5);          // [0, 1) -> [-0.5, 0.5)
        noise = tenzor::mul(noise, ulp);          // scale to one ULP per element

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
