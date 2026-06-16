/**
 * @file parity_tolerances.hpp
 * @brief Canonical numerical tolerances for cross-backend parity tests.
 *
 * Centralizing tolerances here avoids the trap of per-test hardcoding —
 * which made it easy to silently loosen tolerances and miss regressions.
 * Each constant is grouped by operation class and dtype family.
 *
 * Usage:
 *   #include "parity_tolerances.hpp"
 *   test_operation_parity(op, ins, parity::MATMUL_RTOL, parity::MATMUL_ATOL,
 *                         "matmul");
 *
 * If you must use a looser tolerance than the canonical value, document the
 * reason inline:
 *   // Looser tol: oneDNN winograd path accumulates in fp16 internally.
 *   test_operation_parity(op, ins, 1e-3f, 1e-4f, "fused_conv2d_relu_winograd");
 */

#pragma once

namespace tenzor::testing::parity {

// ============================================================================
// Float32 tolerances — the default reference precision
// ============================================================================

// Element-wise math (add, sub, mul, div, exp, log, sin, cos, ...)
inline constexpr float MATH_RTOL = 1e-6f;
inline constexpr float MATH_ATOL = 1e-8f;

// Transcendentals (exp, log, sin, cos, tan, sinh, cosh, ...) — cross-backend
// precision differs because each backend pulls a different vendor math library.
// Looser than MATH but tighter than MATMUL.
inline constexpr float TRANSCENDENTAL_RTOL = 5e-6f;
inline constexpr float TRANSCENDENTAL_ATOL = 1e-7f;

// MatMul / BMM / Linear.
//
// All backends compute FP32 GEMM at true single precision: CPU(MKL),
// Vulkan(`float` accumulators), OneAPI(oneMKL compute_mode::standard),
// ROCm(rocBLAS), and CUDA(CUBLAS_COMPUTE_32F once TF32 is disabled). Verified:
// against a Float64 reference every backend's max abs error is ~4e-6 for a
// 32x32 GEMM — true FP32, well within 1e-4. (An earlier ~6-9e-3 divergence was
// a real bug: CUDA TF32 was left enabled because its disable flag was resolved
// at .so static-init before the test fixture's setenv ran; that is fixed by
// resolving the flag lazily on the first gemm. Do NOT loosen this to mask it.)
inline constexpr float MATMUL_RTOL = 1e-4f;
inline constexpr float MATMUL_ATOL = 1e-5f;

// Reductions (sum, mean, var) — accumulation order differs between backends
inline constexpr float REDUCTION_RTOL = 1e-5f;
inline constexpr float REDUCTION_ATOL = 1e-6f;

// Convolution (im2col + GEMM, or cuDNN/oneDNN winograd) — reduction over many MACs
inline constexpr float CONV_RTOL = 1e-3f;
inline constexpr float CONV_ATOL = 1e-5f;

// Normalization (BatchNorm, LayerNorm, RMSNorm) — division by sqrt(var + eps)
inline constexpr float NORM_RTOL = 1e-4f;
inline constexpr float NORM_ATOL = 1e-5f;

// Activations (ReLU, GELU, SiLU, ...) — pointwise, but transcendentals carry small error
inline constexpr float ACT_RTOL = 1e-6f;
inline constexpr float ACT_ATOL = 1e-7f;

// Softmax / LogSoftmax — numerically delicate due to exp + log
inline constexpr float SOFTMAX_RTOL = 1e-5f;
inline constexpr float SOFTMAX_ATOL = 1e-6f;

// Loss functions (CrossEntropy, MSE, ...) — composition of reductions and softmax
inline constexpr float LOSS_RTOL = 1e-4f;
inline constexpr float LOSS_ATOL = 1e-5f;

// Linear algebra (det, inv, solve, eigh, svd) — sensitive to conditioning
inline constexpr float LINALG_RTOL = 1e-3f;
inline constexpr float LINALG_ATOL = 1e-4f;

// FFT — accumulation over complex twiddle factors
inline constexpr float FFT_RTOL = 1e-4f;
inline constexpr float FFT_ATOL = 1e-5f;

// Backward / gradient parity (looser — accumulation paths)
inline constexpr float GRAD_RTOL = 1e-4f;
inline constexpr float GRAD_ATOL = 1e-6f;

// ============================================================================
// Float64 tolerances — tighter where possible
// ============================================================================

inline constexpr float MATH_RTOL_F64    = 1e-12f;
inline constexpr float MATH_ATOL_F64    = 1e-14f;
inline constexpr float MATMUL_RTOL_F64  = 1e-10f;
inline constexpr float MATMUL_ATOL_F64  = 1e-12f;
inline constexpr float REDUCTION_RTOL_F64 = 1e-11f;
inline constexpr float REDUCTION_ATOL_F64 = 1e-13f;

// ============================================================================
// Reduced-precision tolerances — Float16 / BFloat16 must be much looser
// ============================================================================

// Float16 (10-bit mantissa, ~3 decimal digits of precision)
inline constexpr float MATH_RTOL_F16    = 1e-2f;
inline constexpr float MATH_ATOL_F16    = 1e-2f;
inline constexpr float MATMUL_RTOL_F16  = 5e-2f;
inline constexpr float MATMUL_ATOL_F16  = 5e-2f;
inline constexpr float CONV_RTOL_F16    = 1e-1f;
inline constexpr float CONV_ATOL_F16    = 1e-1f;

// BFloat16 (7-bit mantissa, ~2 decimal digits — even looser than Float16)
inline constexpr float MATH_RTOL_BF16   = 5e-2f;
inline constexpr float MATH_ATOL_BF16   = 5e-2f;
inline constexpr float MATMUL_RTOL_BF16 = 1e-1f;
inline constexpr float MATMUL_ATOL_BF16 = 1e-1f;
inline constexpr float CONV_RTOL_BF16   = 2e-1f;
inline constexpr float CONV_ATOL_BF16   = 2e-1f;

// ============================================================================
// Stride-bug coverage tolerances (test_stride_parity.cpp)
//
// Looser than contiguous tests because reduction order differs between
// strided/contiguous paths on the same backend.
// ============================================================================

inline constexpr float STRIDE_RTOL = 1e-4f;
inline constexpr float STRIDE_ATOL = 1e-6f;

}  // namespace tenzor::testing::parity
