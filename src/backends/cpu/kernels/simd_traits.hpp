/**
 * @file simd_traits.hpp
 * @brief SIMD dispatch traits for binary pointwise operations.
 *
 * Maps (Op, scalar_t) → best SIMD implementation at compile time.
 * Includes op tag types (AddOp, SubOp, MulOp, DivOp) and SimdTrait
 * specializations for Float32, Float64, Float16, BFloat16, and integer types.
 */

#pragma once

#include "tenzor/core/dtype.hpp"
#include <cstddef>
#include <cstdint>
#include <complex>

// Forward declarations for SIMD functions used by traits
namespace tenzor::cpu::detail {
    // These are defined in simd_fast_math.hpp / int_simd.hpp / float16_simd.hpp
    void add_avx512_f32(const float*, const float*, float*, size_t);
    void add_avx2_f32(const float*, const float*, float*, size_t);
    void add_scalar(const float*, const float*, float*, size_t);
    void add_avx512_f64(const double*, const double*, double*, size_t);
    void add_avx2_f64(const double*, const double*, double*, size_t);
    void add_scalar(const double*, const double*, double*, size_t);
    void add_avx512_i32(const int32_t*, const int32_t*, int32_t*, size_t);
    void add_avx512_i64(const int64_t*, const int64_t*, int64_t*, size_t);

    void sub_avx512_f32(const float*, const float*, float*, size_t);
    void sub_avx2_f32(const float*, const float*, float*, size_t);
    void sub_scalar(const float*, const float*, float*, size_t);
    void sub_avx512_f64(const double*, const double*, double*, size_t);
    void sub_avx2_f64(const double*, const double*, double*, size_t);
    void sub_scalar(const double*, const double*, double*, size_t);
    void sub_avx512_i32(const int32_t*, const int32_t*, int32_t*, size_t);
    void sub_avx512_i64(const int64_t*, const int64_t*, int64_t*, size_t);

    void mul_avx512_f32(const float*, const float*, float*, size_t);
    void mul_avx2_f32(const float*, const float*, float*, size_t);
    void mul_scalar(const float*, const float*, float*, size_t);
    void mul_avx512_f64(const double*, const double*, double*, size_t);
    void mul_avx2_f64(const double*, const double*, double*, size_t);
    void mul_scalar(const double*, const double*, double*, size_t);
    void mul_avx512_i32(const int32_t*, const int32_t*, int32_t*, size_t);
    void mul_avx512_i64(const int64_t*, const int64_t*, int64_t*, size_t);

    void div_avx512_f32(const float*, const float*, float*, size_t);
    void div_avx2_f32(const float*, const float*, float*, size_t);
    void div_scalar(const float*, const float*, float*, size_t);
    void div_avx512_f64(const double*, const double*, double*, size_t);
    void div_avx2_f64(const double*, const double*, double*, size_t);
    void div_scalar(const double*, const double*, double*, size_t);
}

namespace tenzor::cpu::int_simd {
    void add_i32(const int32_t*, const int32_t*, int32_t*, size_t);
    void add_i16(const int16_t*, const int16_t*, int16_t*, size_t);
    void add_i8(const int8_t*, const int8_t*, int8_t*, size_t);
    void add_u8(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void sub_i32(const int32_t*, const int32_t*, int32_t*, size_t);
    void sub_i16(const int16_t*, const int16_t*, int16_t*, size_t);
    void sub_i8(const int8_t*, const int8_t*, int8_t*, size_t);
    void sub_u8(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void mul_i32(const int32_t*, const int32_t*, int32_t*, size_t);
    void mul_i16(const int16_t*, const int16_t*, int16_t*, size_t);
    void mul_i8(const int8_t*, const int8_t*, int8_t*, size_t);
    void mul_u8(const uint8_t*, const uint8_t*, uint8_t*, size_t);
}

namespace tenzor::cpu::float16_simd {
    void add_f16(const uint16_t*, const uint16_t*, uint16_t*, size_t);
    void sub_f16(const uint16_t*, const uint16_t*, uint16_t*, size_t);
    void mul_f16(const uint16_t*, const uint16_t*, uint16_t*, size_t);
    void div_f16(const uint16_t*, const uint16_t*, uint16_t*, size_t);
}

namespace tenzor::cpu {

// ============================================================================
// Op tag types — used for SIMD trait specialization
// ============================================================================
struct AddOp {
    template<typename T>
    static T scalar(T a, T b) { return a + b; }
};

// Float16/BFloat16 need explicit float conversion (no native operators)
template<> inline Float16 AddOp::scalar<Float16>(Float16 a, Float16 b) {
    return Float16(static_cast<float>(a) + static_cast<float>(b));
}
template<> inline BFloat16 AddOp::scalar<BFloat16>(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) + static_cast<float>(b));
}
template<> inline bool AddOp::scalar<bool>(bool a, bool b) {
    return (static_cast<int>(a) + static_cast<int>(b)) != 0;
}

struct SubOp {
    template<typename T>
    static T scalar(T a, T b) { return a - b; }
};

template<> inline Float16 SubOp::scalar<Float16>(Float16 a, Float16 b) {
    return Float16(static_cast<float>(a) - static_cast<float>(b));
}
template<> inline BFloat16 SubOp::scalar<BFloat16>(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) - static_cast<float>(b));
}
template<> inline bool SubOp::scalar<bool>(bool a, bool b) {
    return (static_cast<int>(a) - static_cast<int>(b)) != 0;
}

struct MulOp {
    template<typename T>
    static T scalar(T a, T b) { return a * b; }
};

template<> inline Float16 MulOp::scalar<Float16>(Float16 a, Float16 b) {
    return Float16(static_cast<float>(a) * static_cast<float>(b));
}
template<> inline BFloat16 MulOp::scalar<BFloat16>(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) * static_cast<float>(b));
}
template<> inline bool MulOp::scalar<bool>(bool a, bool b) {
    return a && b;
}

struct DivOp {
    template<typename T>
    static T scalar(T a, T b) { return a / b; }
};

template<> inline Float16 DivOp::scalar<Float16>(Float16 a, Float16 b) {
    return Float16(static_cast<float>(a) / static_cast<float>(b));
}
template<> inline BFloat16 DivOp::scalar<BFloat16>(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) / static_cast<float>(b));
}

// ============================================================================
// SIMD dispatch traits — maps (Op, scalar_t) to best SIMD implementation
// ============================================================================

/// Default: no SIMD, use scalar loop
template<typename Op, typename T>
struct SimdTrait {
    static void apply(const T* a, const T* b, T* c, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            c[i] = Op::template scalar<T>(a[i], b[i]);
        }
    }
};

/// BFloat16 needs float conversion for all ops (no native BF16 arithmetic)
template<typename Op>
struct SimdTrait<Op, BFloat16> {
    static void apply(const BFloat16* a, const BFloat16* b, BFloat16* c, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            c[i] = BFloat16(Op::template scalar<float>(
                static_cast<float>(a[i]), static_cast<float>(b[i])));
        }
    }
};

// --- AddOp specializations ---

template<> struct SimdTrait<AddOp, float> {
    static void apply(const float* a, const float* b, float* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::add_avx512_f32(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::add_avx2_f32(a, b, c, n);
#else
        detail::add_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<AddOp, double> {
    static void apply(const double* a, const double* b, double* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::add_avx512_f64(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::add_avx2_f64(a, b, c, n);
#else
        detail::add_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<AddOp, int32_t> {
    static void apply(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::add_avx512_i32(a, b, c, n);
#else
        int_simd::add_i32(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<AddOp, int64_t> {
    static void apply(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::add_avx512_i64(a, b, c, n);
#else
        for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
    }
};

template<> struct SimdTrait<AddOp, int16_t> {
    static void apply(const int16_t* a, const int16_t* b, int16_t* c, size_t n) {
        int_simd::add_i16(a, b, c, n);
    }
};

template<> struct SimdTrait<AddOp, int8_t> {
    static void apply(const int8_t* a, const int8_t* b, int8_t* c, size_t n) {
        int_simd::add_i8(a, b, c, n);
    }
};

template<> struct SimdTrait<AddOp, uint8_t> {
    static void apply(const uint8_t* a, const uint8_t* b, uint8_t* c, size_t n) {
        int_simd::add_u8(a, b, c, n);
    }
};

template<> struct SimdTrait<AddOp, Float16> {
    static void apply(const Float16* a, const Float16* b, Float16* c, size_t n) {
        float16_simd::add_f16(
            reinterpret_cast<const uint16_t*>(a),
            reinterpret_cast<const uint16_t*>(b),
            reinterpret_cast<uint16_t*>(c), n);
    }
};

// --- SubOp specializations ---

template<> struct SimdTrait<SubOp, float> {
    static void apply(const float* a, const float* b, float* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::sub_avx512_f32(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::sub_avx2_f32(a, b, c, n);
#else
        detail::sub_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<SubOp, double> {
    static void apply(const double* a, const double* b, double* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::sub_avx512_f64(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::sub_avx2_f64(a, b, c, n);
#else
        detail::sub_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<SubOp, Float16> {
    static void apply(const Float16* a, const Float16* b, Float16* c, size_t n) {
        float16_simd::sub_f16(
            reinterpret_cast<const uint16_t*>(a),
            reinterpret_cast<const uint16_t*>(b),
            reinterpret_cast<uint16_t*>(c), n);
    }
};

// --- MulOp specializations ---

template<> struct SimdTrait<MulOp, float> {
    static void apply(const float* a, const float* b, float* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::mul_avx512_f32(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::mul_avx2_f32(a, b, c, n);
#else
        detail::mul_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<MulOp, double> {
    static void apply(const double* a, const double* b, double* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::mul_avx512_f64(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::mul_avx2_f64(a, b, c, n);
#else
        detail::mul_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<MulOp, Float16> {
    static void apply(const Float16* a, const Float16* b, Float16* c, size_t n) {
        float16_simd::mul_f16(
            reinterpret_cast<const uint16_t*>(a),
            reinterpret_cast<const uint16_t*>(b),
            reinterpret_cast<uint16_t*>(c), n);
    }
};

// --- DivOp specializations ---

template<> struct SimdTrait<DivOp, float> {
    static void apply(const float* a, const float* b, float* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::div_avx512_f32(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::div_avx2_f32(a, b, c, n);
#else
        detail::div_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<DivOp, double> {
    static void apply(const double* a, const double* b, double* c, size_t n) {
#ifdef TENZOR_HAS_AVX512
        detail::div_avx512_f64(a, b, c, n);
#elif defined(TENZOR_HAS_AVX2)
        detail::div_avx2_f64(a, b, c, n);
#else
        detail::div_scalar(a, b, c, n);
#endif
    }
};

template<> struct SimdTrait<DivOp, Float16> {
    static void apply(const Float16* a, const Float16* b, Float16* c, size_t n) {
        float16_simd::div_f16(
            reinterpret_cast<const uint16_t*>(a),
            reinterpret_cast<const uint16_t*>(b),
            reinterpret_cast<uint16_t*>(c), n);
    }
};

