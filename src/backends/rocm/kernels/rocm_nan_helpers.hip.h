#pragma once
// ROCm NaN-bit-pattern helpers (F7/F8 followup).
//
// HIP's `isnan()` intrinsic has been observed to silently return the wrong
// answer under HCC/Clang fast-math optimisation modes (and on certain GFX
// architectures, e.g. gfx900/gfx906 with --offload-arch quirks). Reductions
// (nansum/nanmean/etc.) that rely on it can then either:
//   - admit NaN payloads into the sum (turning the whole reduction into NaN);
//   - or count NaN samples toward the non-NaN denominator (drifting the mean).
//
// The fix is to inspect IEEE-754 bits directly via `memcpy`. The bit
// patterns are stable across host/device compiles. This header centralises
// the helpers so the same code can be used by every ROCm reduction kernel
// (and by the math kernels in F8) without re-deriving the masks each time.

#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace tenzor {
namespace rocm {

// Float32: exp == 0xFF, mantissa != 0 → NaN.
// (Use a union for the reinterpret cast: `std::memcpy` is a __host__-only
// function in HIP and triggers "reference to __host__ function memcpy in
// __host__ __device__ function" on device compiles. The union pattern is
// device-safe and produces identical codegen.)
__device__ __host__ inline bool is_nan_bits(float x) {
    union { float f; uint32_t u; } pun;
    pun.f = x;
    uint32_t exp  = (pun.u >> 23) & 0xFFu;
    uint32_t mant =  pun.u        & 0x7FFFFFu;
    return (exp == 0xFFu) && (mant != 0u);
}

// Float64: exp == 0x7FF, mantissa != 0 → NaN.
__device__ __host__ inline bool is_nan_bits(double x) {
    union { double d; uint64_t u; } pun;
    pun.d = x;
    uint64_t exp  = (pun.u >> 52) & 0x7FFull;
    uint64_t mant =  pun.u        & 0xFFFFFFFFFFFFFull;
    return (exp == 0x7FFull) && (mant != 0ull);
}

// Float16: exp == 0x1F, mantissa != 0 → NaN.
__device__ __host__ inline bool is_nan_bits(__half x) {
    union { __half h; uint16_t u; } pun;
    pun.h = x;
    uint16_t bits = pun.u;
    uint16_t exp  = (bits >> 10) & 0x1Fu;
    uint16_t mant =  bits        & 0x3FFu;
    return (exp == 0x1Fu) && (mant != 0u);
}

// BFloat16: exp == 0xFF, mantissa != 0 → NaN (same as F32 layout, narrower).
// `hip_bfloat16` is the canonical name in the HIP headers; `__hip_bfloat16`
// exists only on newer toolchains.
__device__ __host__ inline bool is_nan_bits(hip_bfloat16 x) {
    union { hip_bfloat16 b; uint16_t u; } pun;
    pun.b = x;
    uint16_t bits = pun.u;
    uint16_t exp  = (bits >> 7) & 0xFFu;
    uint16_t mant =  bits       & 0x7Fu;
    return (exp == 0xFFu) && (mant != 0u);
}

// ============================================================================
// E.2: NaN-preserving half / bfloat16 conversions.
//
// HIP's __float2half / __half2float / __float2bfloat16 / __bfloat162float
// intrinsics have been observed to canonicalize or strip NaN payloads
// under fast-math (and on a few driver versions silently turn NaN into
// ±inf). The kernels in this backend rely on NaN propagating exactly for
// gradcheck and for the nansum/nanmean family of reductions.
//
// safe_f2h / safe_h2f / safe_f2bf / safe_bf2f fast-path finite values
// through the vendor intrinsic and emit a canonical qNaN bit pattern
// whenever the input is NaN. They're drop-in replacements for the raw
// intrinsics in NaN-correct arithmetic.
// ============================================================================

__device__ __host__ inline __half safe_f2h(float x) {
    if (is_nan_bits(x)) {
        union { __half h; uint16_t u; } pun;
        pun.u = 0x7E00u;               // canonical Float16 qNaN
        return pun.h;
    }
    return __float2half(x);
}

__device__ __host__ inline float safe_h2f(__half x) {
    if (is_nan_bits(x)) {
        union { float f; uint32_t u; } pun;
        pun.u = 0x7FC00000u;            // canonical Float32 qNaN
        return pun.f;
    }
    return __half2float(x);
}

__device__ __host__ inline hip_bfloat16 safe_f2bf(float x) {
    if (is_nan_bits(x)) {
        union { hip_bfloat16 b; uint16_t u; } pun;
        pun.u = 0x7FC0u;                // canonical bf16 qNaN
        return pun.b;
    }
    // S.10 / R.11: round-to-nearest-even on float32 → bfloat16 instead of
    // the truncating `hip_bfloat16(float)` ctor. We inline the bit-twiddle
    // here (rather than #including bfloat16_helpers.hpp) because this header
    // already has __host__/__device__ visibility and the helper file is
    // device-only.
    union { float f; uint32_t u; } pun;
    pun.f = x;
    uint32_t bits = pun.u;
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    bits += rounding_bias;
    uint16_t out_bits = static_cast<uint16_t>(bits >> 16);
    union { hip_bfloat16 b; uint16_t u; } out;
    out.u = out_bits;
    return out.b;
}

__device__ __host__ inline float safe_bf2f(hip_bfloat16 x) {
    if (is_nan_bits(x)) {
        union { float f; uint32_t u; } pun;
        pun.u = 0x7FC00000u;
        return pun.f;
    }
    return static_cast<float>(x);
}

}  // namespace rocm
}  // namespace tenzor
