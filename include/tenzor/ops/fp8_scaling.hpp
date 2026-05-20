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
 *
 * # Backend support matrix
 *
 *   Backend   | FP8 cast | FP8 matmul | Notes
 *   ----------|----------|------------|------------------------------------
 *   CPU       | native   | emulated   | Bit-level emulation, widens to F32
 *   CUDA      | native   | hardware   | Hopper (SM 9.0+) tensor cores via cublasLt
 *   ROCm      | native   | emulated   | Native FP8↔F32 HIP cast kernels
 *                                       (kernels/transform.hip.cpp). Matmul
 *                                       widens FP8 → F32 on device, runs
 *                                       rocBLAS F32 GEMM, narrows back. No
 *                                       CPU roundtrip on either path. MI300
 *                                       MFMA FP8 hardware path is a future
 *                                       perf-only enhancement.
 *   OneAPI    | native   | emulated   | No Intel GPU FP8 hardware support
 *   Vulkan    | native   | emulated   | FP8 cast compute shaders are
 *                                       registered; matmul widens to F32.
 *   MPS       | fallback | fallback   | macOS only; CPU-roundtrip story
 *                                       still applies.
 *
 * `native` means the backend has a registered Cast kernel for the FP8
 * dtype and computes quantize/dequantize on-device. `emulated` means
 * the backend widens to Float32 for compute but uses local hardware
 * for the FP32 work (also on-device — no CPU roundtrip). `fallback`
 * means tensors are moved CPU → cast → back to the device, which
 * incurs a PCIe/memory-bus round-trip per FP8 op.
 *
 * The MI300 MFMA FP8 hardware path and a portable Vulkan FP8 intrinsic
 * path are both performance enhancements; current emulation is
 * numerically equivalent because Float32 widening exactly represents
 * every FP8 value.
 */

#pragma once

#include "../core/tensor.hpp"
#include "../core/dtype.hpp"
#include <optional>
#include <utility>

namespace tenzor {

/**
 * @brief Query whether a backend has a native FP8 Cast kernel.
 *
 * When this returns `false`, calls to `quantize_to_fp8(x)` with `x`
 * on the given device still work but route through CPU for the FP8
 * conversion step. Enable `TENZOR_WARN_CPU_ROUNDTRIP=1` to get a
 * per-tuple warning the first time each GPU→CPU round-trip happens.
 *
 * @param device_type The backend device type to query.
 * @return true if the backend has a registered Cast kernel that
 *         understands FP8_E4M3 / FP8_E5M2; false otherwise.
 */
auto fp8_is_native(Device::Type device_type) -> bool;

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
