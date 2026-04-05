/**
 * @file cuda_config.hpp
 * @brief Runtime configuration for CUDA backend behavior
 *
 * Provides thread-local toggles for CUDA-specific numerical behavior:
 * - TF32 Tensor Core usage for Float32 matmul
 * - FP16 saturation warnings
 */

#pragma once

namespace tenzor::cuda::matmul {

/**
 * @brief Check if TF32 Tensor Cores are allowed for Float32 matmul.
 *
 * When enabled (default), Float32 matmul on Ampere+ GPUs uses TF32
 * Tensor Cores for ~2x speedup with ~0.1% reduced precision.
 * When disabled, full IEEE Float32 precision is used.
 *
 * Thread-local: each thread can independently control this setting.
 */
auto allow_tf32() -> bool;

/**
 * @brief Set whether TF32 Tensor Cores are allowed for Float32 matmul.
 * @param value true to enable TF32 (default), false for full FP32 precision
 */
auto set_allow_tf32(bool value) -> void;

/**
 * @brief Check if FP16 saturation warnings are enabled.
 *
 * When enabled, logs a warning if any FP16 matmul output values are
 * clamped from Inf to +/-65504. This can indicate numerical overflow
 * in the computation.
 *
 * Disabled by default (zero overhead when disabled).
 * Thread-local: each thread can independently control this setting.
 */
auto warn_fp16_saturation() -> bool;

/**
 * @brief Set whether FP16 saturation warnings are enabled.
 * @param value true to enable warnings, false to disable (default)
 */
auto set_warn_fp16_saturation(bool value) -> void;

} // namespace tenzor::cuda::matmul
