// S.10: shared BFloat16 round-to-nearest-even (RNE) helper for ROCm HIP kernels.
//
// Background
// ----------
// `hip_bfloat16(float)` is value-truncating: it drops the lower 16 bits of
// the IEEE-754 float32 representation with no rounding. For atomic
// accumulation (atomicCAS-retry loops in scatter/index_add/embedding_bag),
// the truncation bias compounds on every CAS retry — each iteration drops
// the partial-sum's low mantissa bits and reloads. Over a backward of a
// 1M-element BF16 embedding the systematic bias can reach several ULP.
//
// PyTorch's BFloat16 path uses round-to-nearest-even on float→bf16
// conversion (see c10/util/BFloat16-inl.h `round_to_nearest_even` /
// `__float2bfloat16`). We mirror that here for any ROCm kernel that
// writes a BFloat16 value derived from a Float32 accumulator.
//
// Usage
// -----
//   #include "bfloat16_helpers.hpp"
//   ...
//   out[idx] = tenzor::rocm::f32_to_bf16_rne(some_float_accumulator);
//
// Per the R.11 finding, the truncating `hip_bfloat16(float)` ctor is to
// be replaced at every site whose float source has more than 8 fractional
// significant bits — which is essentially every accumulation, every
// elementwise unary that goes through float32 internally, and every
// `value` cast at scatter / index_add / fill sites.
//
// This header is shared because the helper is identical across
// indexing.hip.cpp, math.hip.cpp, transform.hip.cpp, pooling.hip.cpp etc.;
// inlining it via __device__ __forceinline__ leaves no call overhead.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace tenzor {
namespace rocm {

// Round-to-nearest-even float32 → bfloat16 conversion.
//
// IEEE-754 layout:
//   float32:  1 sign | 8 exponent | 23 mantissa
//   bfloat16: 1 sign | 8 exponent | 7 mantissa  (upper 16 bits of float32)
//
// RNE rule: take the upper 16 bits, then add a rounding offset equal to
// 0x7fff + lsb_of_kept_mantissa. This implements banker's rounding (ties
// round to the value whose kept mantissa LSB is 0).
//
// NaN handling: we detect NaN via the exponent==0xff && mantissa!=0 pattern
// and preserve the NaN payload (top mantissa bit set) so isnan(out) stays
// true downstream.
//
// Bit punning: we use a union (typed) instead of memcpy/reinterpret_cast
// because std::memcpy is __host__-only on the HIP toolchain; the union
// pun is type-safe under HIP's flat address space and compiles down to a
// register-level copy with no narrowing.
__device__ __host__ __forceinline__ hip_bfloat16 f32_to_bf16_rne(float f) {
    union { float f; uint32_t u; } in_pun;
    in_pun.f = f;
    uint32_t bits = in_pun.u;

    // Preserve NaN: bfloat16 NaN is exponent=0xff, mantissa!=0. Force the
    // top mantissa bit so the truncated value is still NaN.
    const uint32_t exp_mask = 0x7f800000u;
    const uint32_t mantissa_mask = 0x007fffffu;
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0u) {
        // Quiet-NaN: set the high mantissa bit of bf16 (bit 22 of f32).
        bits |= 0x00400000u;
        uint16_t nan_bits = static_cast<uint16_t>(bits >> 16);
        union { uint16_t u; hip_bfloat16 bf; } out_pun;
        out_pun.u = nan_bits;
        return out_pun.bf;
    }

    // RNE rounding: lsb-of-kept-mantissa + 0x7fff bias.
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    bits += rounding_bias;

    uint16_t out_bits = static_cast<uint16_t>(bits >> 16);
    union { uint16_t u; hip_bfloat16 bf; } out_pun;
    out_pun.u = out_bits;
    return out_pun.bf;
}

} // namespace rocm
} // namespace tenzor
