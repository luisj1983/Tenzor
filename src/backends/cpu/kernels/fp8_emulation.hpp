/**
 * @file fp8_emulation.hpp
 * @brief FP8 GEMM emulation via FP32 compute for non-Hopper backends.
 *
 * Provides software emulation for FP8 matrix multiplication:
 * 1. Widen FP8 inputs to FP32
 * 2. Perform FP32 GEMM (via MKL/BLAS/optimized kernels)
 * 3. Narrow FP32 output back to FP8
 *
 * This enables model portability across all backends while hardware
 * acceleration is only available on NVIDIA Hopper (SM 9.0+) via cuBLAS.
 */
#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor {
namespace cpu {

/**
 * @brief FP8 GEMM via FP32 emulation.
 *
 * Performs C = A @ B where A and B are FP8 tensors.
 * Internally widens to FP32, computes GEMM, narrows back to FP8.
 *
 * @param a Input tensor (FP8_E4M3 or FP8_E5M2)
 * @param b Input tensor (FP8_E4M3 or FP8_E5M2)
 * @return Result tensor in same FP8 dtype as input
 */
inline auto fp8_gemm_emulated(const Tensor& a, const Tensor& b) -> Tensor {
    DType orig_dtype = a.dtype();

    // Widen FP8 inputs to FP32 for compute
    Tensor a_f32 = a.to(DType::Float32);
    Tensor b_f32 = b.to(DType::Float32);

    // Perform FP32 GEMM using optimized backend (MKL/BLAS)
    Tensor result_f32 = tenzor::matmul(a_f32, b_f32);

    // Narrow back to original FP8 dtype
    return result_f32.to(orig_dtype);
}

} // namespace cpu
} // namespace tenzor
