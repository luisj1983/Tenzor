#pragma once
// W.7: CUDA NaN-bit-pattern helpers, mirror of rocm_nan_helpers.hip.h.
//
// `__float2half(NaN)` and `__half2float(NaN_F16)` are documented as RTNE
// but compiler fast-math, --use_fast_math, and a small set of toolkit
// versions have been observed to silently canonicalize / drop NaN
// payloads. Kernels in the CUDA backend that round-trip half values
// through float32 add/exp (e.g. indexing / vision atomics) rely on the
// NaN bit pattern propagating exactly, the same way the ROCm helpers do.
//
// safe_f2half / safe_half2f / safe_f2bf16 / safe_bf162f short-circuit
// NaN inputs to a canonical qNaN, otherwise route through the standard
// intrinsic. Drop-in replacements at every CUDA half↔float conversion
// site that participates in NaN-correct arithmetic.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tenzor {
namespace cuda {

// Float32: exp == 0xFF, mantissa != 0.
__device__ __host__ inline bool is_nan_bits(float x) {
    union { float f; uint32_t u; } pun;
    pun.f = x;
    uint32_t exp  = (pun.u >> 23) & 0xFFu;
    uint32_t mant =  pun.u        & 0x7FFFFFu;
    return (exp == 0xFFu) && (mant != 0u);
}

// Float64: exp == 0x7FF, mantissa != 0.
__device__ __host__ inline bool is_nan_bits(double x) {
    union { double d; uint64_t u; } pun;
    pun.d = x;
    uint64_t exp  = (pun.u >> 52) & 0x7FFull;
    uint64_t mant =  pun.u        & 0xFFFFFFFFFFFFFull;
    return (exp == 0x7FFull) && (mant != 0ull);
}

// Float16: exp == 0x1F, mantissa != 0.
__device__ __host__ inline bool is_nan_bits(__half x) {
    uint16_t bits = *reinterpret_cast<uint16_t*>(&x);
    uint16_t exp  = (bits >> 10) & 0x1Fu;
    uint16_t mant =  bits        & 0x3FFu;
    return (exp == 0x1Fu) && (mant != 0u);
}

// BFloat16: exp == 0xFF, mantissa != 0.
__device__ __host__ inline bool is_nan_bits(__nv_bfloat16 x) {
    uint16_t bits = *reinterpret_cast<uint16_t*>(&x);
    uint16_t exp  = (bits >> 7) & 0xFFu;
    uint16_t mant =  bits       & 0x7Fu;
    return (exp == 0xFFu) && (mant != 0u);
}

// ============================================================================
// NaN-preserving conversions.
// ============================================================================

// Float → Float16. Canonical qNaN on NaN input, RTNE otherwise.
__device__ __host__ inline __half safe_f2half(float x) {
    if (is_nan_bits(x)) {
        uint16_t nan_bits = 0x7E00u;   // canonical Float16 qNaN
        __half h;
        *reinterpret_cast<uint16_t*>(&h) = nan_bits;
        return h;
    }
#ifdef __CUDA_ARCH__
    return __float2half_rn(x);
#else
    return __float2half(x);
#endif
}

// Float16 → Float. Canonical qNaN on NaN input.
__device__ __host__ inline float safe_half2f(__half x) {
    if (is_nan_bits(x)) {
        union { float f; uint32_t u; } pun;
        pun.u = 0x7FC00000u;            // canonical Float32 qNaN
        return pun.f;
    }
    return __half2float(x);
}

// Float → BFloat16. Canonical qNaN on NaN input, RTNE otherwise.
__device__ __host__ inline __nv_bfloat16 safe_f2bf16(float x) {
    if (is_nan_bits(x)) {
        uint16_t nan_bits = 0x7FC0u;
        __nv_bfloat16 b;
        *reinterpret_cast<uint16_t*>(&b) = nan_bits;
        return b;
    }
#ifdef __CUDA_ARCH__
    return __float2bfloat16_rn(x);
#else
    return __float2bfloat16(x);
#endif
}

// BFloat16 → Float. Canonical qNaN on NaN input.
__device__ __host__ inline float safe_bf162f(__nv_bfloat16 x) {
    if (is_nan_bits(x)) {
        union { float f; uint32_t u; } pun;
        pun.u = 0x7FC00000u;
        return pun.f;
    }
    return __bfloat162float(x);
}

}  // namespace cuda
}  // namespace tenzor
