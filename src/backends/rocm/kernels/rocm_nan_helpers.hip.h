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
    uint16_t bits = *reinterpret_cast<uint16_t*>(&x);
    uint16_t exp  = (bits >> 10) & 0x1Fu;
    uint16_t mant =  bits        & 0x3FFu;
    return (exp == 0x1Fu) && (mant != 0u);
}

// BFloat16: exp == 0xFF, mantissa != 0 → NaN (same as F32 layout, narrower).
// `hip_bfloat16` is the canonical name in the HIP headers; `__hip_bfloat16`
// exists only on newer toolchains.
__device__ __host__ inline bool is_nan_bits(hip_bfloat16 x) {
    uint16_t bits = *reinterpret_cast<uint16_t*>(&x);
    uint16_t exp  = (bits >> 7) & 0xFFu;
    uint16_t mant =  bits       & 0x7Fu;
    return (exp == 0xFFu) && (mant != 0u);
}

}  // namespace rocm
}  // namespace tenzor
