/**
 * @file cuda_config.hpp
 * @brief Runtime configuration for CUDA backend behavior
 *
 * Provides process-global toggles for CUDA-specific numerical behavior:
 * - TF32 Tensor Core usage for Float32 matmul
 * - FP16 overflow warnings
 */

#pragma once

namespace tenzor::cuda::matmul {

/**
 * @brief Check if TF32 Tensor Cores are allowed for Float32 matmul.
 *
 * DEFAULT: DISABLED (full IEEE Float32 precision). This is a deliberate
 * choice for this codebase specifically: Tenzor also ships a CPU backend and
 * targets CPU<->CUDA numerical parity, and CPU's matmul always computes
 * exact FP32. Defaulting TF32 on (as e.g. PyTorch does for backward
 * compatibility with older cuDNN/cuBLAS defaults) would silently make
 * out-of-the-box CUDA Float32 matmul -- and its gradients, since
 * MatMulBackward re-enters the same matmul path -- diverge from CPU by more
 * than rounding noise (F-108). Opt in via TENZOR_ENABLE_TF32=1 or
 * set_allow_tf32(true) for the ~2x speedup / ~0.1% reduced precision
 * tradeoff on Ampere+ GPUs.
 *
 * Process-global (not thread-local): a cuBLAS call dispatched from the
 * runtime can execute on a worker thread distinct from the one that called
 * set_allow_tf32() / set the env var, so the flag is stored in a process-wide
 * atomic and honored on every thread that issues a gemm.
 */
auto allow_tf32() -> bool;

/**
 * @brief Set whether TF32 Tensor Cores are allowed for Float32 matmul.
 * @param value true to enable TF32 (opt-in, for speed), false for full FP32
 *        precision (default)
 */
auto set_allow_tf32(bool value) -> void;

/**
 * @brief Check if FP16 overflow warnings are enabled.
 *
 * When enabled, logs a warning if any FP16 matmul output values overflowed
 * to +/-Infinity (accumulator magnitude exceeded 65504, the largest finite
 * half-precision value). This is purely informational: overflow is *not*
 * clamped/saturated — CUDA follows the same strict IEEE-754 float32->float16
 * narrowing as CPU, so genuine overflow legitimately produces signed
 * Infinity on both backends (see src/core/dtype.cpp Float16::Float16(float)
 * and src/backends/cuda/kernels/matmul.cu). This warning can indicate
 * numerical overflow in the computation.
 *
 * Disabled by default (zero overhead when disabled — no scan is performed).
 * Process-global: honored on every thread that issues a gemm.
 */
auto warn_fp16_saturation() -> bool;

/**
 * @brief Set whether FP16 overflow warnings are enabled.
 * @param value true to enable warnings, false to disable (default)
 */
auto set_warn_fp16_saturation(bool value) -> void;

} // namespace tenzor::cuda::matmul
