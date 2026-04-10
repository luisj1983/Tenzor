/**
 * @file dtype.hpp
 * @brief Data type system for Tenzor tensors
 *
 * Provides type definitions, traits, and utilities for tensor data types.
 * Supports various numeric types including floating-point, integers, and complex numbers.
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <complex>

namespace tenzor {

// Forward declarations for half-precision and FP8 types
struct Float16;
struct BFloat16;
struct FP8_E4M3;
struct FP8_E5M2;

/**
 * @brief Enumeration of supported tensor data types.
 *
 * Defines all numeric types that can be stored in tensors, including
 * floating-point, integer, boolean, and complex number types.
 */
enum class DType : uint8_t {
    Float32,    ///< 32-bit floating point (float)
    Float64,    ///< 64-bit floating point (double)
    Float16,    ///< 16-bit floating point (half precision)
    BFloat16,   ///< Brain floating point (16-bit, Google format)
    Int8,       ///< 8-bit signed integer
    Int16,      ///< 16-bit signed integer
    Int32,      ///< 32-bit signed integer
    Int64,      ///< 64-bit signed integer
    UInt8,      ///< 8-bit unsigned integer
    UInt16,     ///< 16-bit unsigned integer
    UInt32,     ///< 32-bit unsigned integer
    UInt64,     ///< 64-bit unsigned integer
    Bool,       ///< Boolean type
    Complex64,  ///< 64-bit complex (two float32)
    Complex128, ///< 128-bit complex (two float64)
    FP8_E4M3,   ///< 8-bit float (4 exponent, 3 mantissa) - Hopper Tensor Cores
    FP8_E5M2,   ///< 8-bit float (5 exponent, 2 mantissa) - Hopper Tensor Cores
    QInt8,      ///< Quantized 8-bit signed integer (with scale/zero_point)
    QUInt8,     ///< Quantized 8-bit unsigned integer (with scale/zero_point)
    QInt4x2     ///< Quantized 4-bit packed (2 values per byte, with scale/zero_point)
};

/**
 * @brief Concept for valid scalar types.
 *
 * Requires type to be arithmetic (integers, floats), complex numbers,
 * or half-precision types (Float16, BFloat16).
 * Used to constrain template parameters to valid tensor element types.
 *
 * @tparam T Type to check
 */
template<typename T>
concept ScalarType = std::is_arithmetic_v<T> ||
                     std::is_same_v<T, std::complex<float>> ||
                     std::is_same_v<T, std::complex<double>> ||
                     std::is_same_v<T, Float16> ||
                     std::is_same_v<T, BFloat16> ||
                     std::is_same_v<T, FP8_E4M3> ||
                     std::is_same_v<T, FP8_E5M2>;

/**
 * @brief Concept for integral types.
 *
 * Requires type to be an integer (signed or unsigned).
 *
 * @tparam T Type to check
 */
template<typename T>
concept IntegralType = std::is_integral_v<T>;

/**
 * @brief Concept for floating-point types.
 *
 * Requires type to be float or double.
 *
 * @tparam T Type to check
 */
template<typename T>
concept FloatingType = std::is_floating_point_v<T>;

/**
 * @brief Type traits for DType enumeration.
 *
 * Maps DType enumeration values to their corresponding C++ types.
 * Specialized for each supported data type.
 *
 * @tparam dt DType enumeration value
 */
template<DType dt>
struct dtype_traits;

/// @brief Specialization for Float32
template<> struct dtype_traits<DType::Float32> { using type = float; };
/// @brief Specialization for Float64
template<> struct dtype_traits<DType::Float64> { using type = double; };
/// @brief Specialization for Int8
template<> struct dtype_traits<DType::Int8> { using type = int8_t; };
/// @brief Specialization for Int16
template<> struct dtype_traits<DType::Int16> { using type = int16_t; };
/// @brief Specialization for Int32
template<> struct dtype_traits<DType::Int32> { using type = int32_t; };
/// @brief Specialization for Int64
template<> struct dtype_traits<DType::Int64> { using type = int64_t; };
/// @brief Specialization for UInt8
template<> struct dtype_traits<DType::UInt8> { using type = uint8_t; };
/// @brief Specialization for UInt16
template<> struct dtype_traits<DType::UInt16> { using type = uint16_t; };
/// @brief Specialization for UInt32
template<> struct dtype_traits<DType::UInt32> { using type = uint32_t; };
/// @brief Specialization for UInt64
template<> struct dtype_traits<DType::UInt64> { using type = uint64_t; };
/// @brief Specialization for Bool
template<> struct dtype_traits<DType::Bool> { using type = bool; };
/// @brief Specialization for Complex64
template<> struct dtype_traits<DType::Complex64> { using type = std::complex<float>; };
/// @brief Specialization for Complex128
template<> struct dtype_traits<DType::Complex128> { using type = std::complex<double>; };

/**
 * @brief Type alias for extracting C++ type from DType.
 *
 * @tparam dt DType enumeration value
 *
 * @code
 * using T = dtype_t<DType::Float32>;  // T is float
 * @endcode
 */
template<DType dt>
using dtype_t = typename dtype_traits<dt>::type;

/**
 * @brief Type trait to map C++ types to DType values (reverse of dtype_traits).
 *
 * @tparam T C++ type
 *
 * @code
 * constexpr DType dt = type_to_dtype<float>::value;  // dt is DType::Float32
 * @endcode
 */
template<typename T>
struct type_to_dtype;

template<> struct type_to_dtype<float> { static constexpr DType value = DType::Float32; };
template<> struct type_to_dtype<double> { static constexpr DType value = DType::Float64; };
template<> struct type_to_dtype<int8_t> { static constexpr DType value = DType::Int8; };
template<> struct type_to_dtype<int16_t> { static constexpr DType value = DType::Int16; };
template<> struct type_to_dtype<int32_t> { static constexpr DType value = DType::Int32; };
template<> struct type_to_dtype<int64_t> { static constexpr DType value = DType::Int64; };
template<> struct type_to_dtype<uint8_t> { static constexpr DType value = DType::UInt8; };
template<> struct type_to_dtype<uint16_t> { static constexpr DType value = DType::UInt16; };
template<> struct type_to_dtype<uint32_t> { static constexpr DType value = DType::UInt32; };
template<> struct type_to_dtype<uint64_t> { static constexpr DType value = DType::UInt64; };
template<> struct type_to_dtype<bool> { static constexpr DType value = DType::Bool; };
template<> struct type_to_dtype<std::complex<float>> { static constexpr DType value = DType::Complex64; };
template<> struct type_to_dtype<std::complex<double>> { static constexpr DType value = DType::Complex128; };

/// @brief Helper variable template for type_to_dtype
template<typename T>
inline constexpr DType type_to_dtype_v = type_to_dtype<T>::value;

// ============================================================================
// Half-Precision Types
// ============================================================================

/**
 * @brief IEEE 754 16-bit floating point (half precision)
 *
 * Memory layout: 1 sign bit, 5 exponent bits, 10 mantissa bits
 * Range: approximately ±65,504
 * Precision: ~3 decimal digits
 */
struct Float16 {
    uint16_t bits{0};

    Float16() = default;
    explicit Float16(float f);
    explicit Float16(uint16_t b) : bits(b) {}

    explicit operator float() const;

    auto operator==(const Float16& other) const -> bool { return bits == other.bits; }
    auto operator!=(const Float16& other) const -> bool { return bits != other.bits; }
};

/**
 * @brief Brain Float16 (BFloat16) - Google's 16-bit floating point
 *
 * Memory layout: 1 sign bit, 8 exponent bits, 7 mantissa bits
 * Same range as Float32, but lower precision
 * Range: approximately ±3.4×10^38
 * Precision: ~2 decimal digits
 *
 * Advantages over Float16:
 * - Same dynamic range as Float32 (same exponent bits)
 * - Direct truncation from Float32 (simple conversion)
 * - Better for deep learning (wider range, less overflow/underflow)
 */
struct BFloat16 {
    uint16_t bits{0};

    BFloat16() = default;
    explicit BFloat16(float f);
    explicit BFloat16(uint16_t b) : bits(b) {}

    explicit operator float() const;

    auto operator==(const BFloat16& other) const -> bool { return bits == other.bits; }
    auto operator!=(const BFloat16& other) const -> bool { return bits != other.bits; }
};

/// @brief Specialization for Float16
template<> struct dtype_traits<DType::Float16> { using type = Float16; };
/// @brief Specialization for BFloat16
template<> struct dtype_traits<DType::BFloat16> { using type = BFloat16; };

/// @brief Reverse mapping for Float16
template<> struct type_to_dtype<Float16> { static constexpr DType value = DType::Float16; };
/// @brief Reverse mapping for BFloat16
template<> struct type_to_dtype<BFloat16> { static constexpr DType value = DType::BFloat16; };

/**
 * @brief FP8 E4M3 format (1 sign, 4 exponent, 3 mantissa bits)
 *
 * Used for forward pass activations and weights on Hopper+ GPUs.
 * Range: approximately +/-448, Precision: ~1.5 decimal digits
 * Supported on NVIDIA H100 (SM 9.0+) via Tensor Cores.
 */
struct FP8_E4M3 {
    uint8_t bits{0};

    FP8_E4M3() = default;
    explicit FP8_E4M3(uint8_t b) : bits(b) {}
    explicit FP8_E4M3(float f);
    explicit operator float() const;

    auto operator==(const FP8_E4M3& other) const -> bool { return bits == other.bits; }
    auto operator!=(const FP8_E4M3& other) const -> bool { return bits != other.bits; }
};

/**
 * @brief FP8 E5M2 format (1 sign, 5 exponent, 2 mantissa bits)
 *
 * Used primarily for gradient storage on Hopper+ GPUs.
 * Range: approximately +/-57344, Precision: ~1 decimal digit
 * Wider range than E4M3 but lower precision, ideal for gradients.
 * Supported on NVIDIA H100 (SM 9.0+) via Tensor Cores.
 */
struct FP8_E5M2 {
    uint8_t bits{0};

    FP8_E5M2() = default;
    explicit FP8_E5M2(uint8_t b) : bits(b) {}
    explicit FP8_E5M2(float f);
    explicit operator float() const;

    auto operator==(const FP8_E5M2& other) const -> bool { return bits == other.bits; }
    auto operator!=(const FP8_E5M2& other) const -> bool { return bits != other.bits; }
};

/// @brief Specialization for FP8_E4M3
template<> struct dtype_traits<DType::FP8_E4M3> { using type = FP8_E4M3; };
/// @brief Specialization for FP8_E5M2
template<> struct dtype_traits<DType::FP8_E5M2> { using type = FP8_E5M2; };

/// @brief Reverse mapping for FP8_E4M3
template<> struct type_to_dtype<FP8_E4M3> { static constexpr DType value = DType::FP8_E4M3; };
/// @brief Reverse mapping for FP8_E5M2
template<> struct type_to_dtype<FP8_E5M2> { static constexpr DType value = DType::FP8_E5M2; };

/**
 * @brief Get the size in bytes of a data type.
 *
 * @param dtype Data type enumeration
 * @return Size in bytes (1, 2, 4, 8, or 16)
 *
 * @code
 * size_t size = dtype_size(DType::Float32);  // Returns 4
 * @endcode
 */
constexpr auto dtype_size(DType dtype) -> size_t {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int8: return 1;
        case DType::Int16: return 2;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
        case DType::UInt8: return 1;
        case DType::UInt16: return 2;
        case DType::UInt32: return 4;
        case DType::UInt64: return 8;
        case DType::Bool: return 1;
        case DType::Complex64: return 8;
        case DType::Complex128: return 16;
        case DType::FP8_E4M3: return 1;
        case DType::FP8_E5M2: return 1;
        case DType::QInt8: return 1;
        case DType::QUInt8: return 1;
        case DType::QInt4x2: return 1;  // 2 values per byte, but storage is per-byte
    }
    return 0;
}

/**
 * @brief Get the string name of a data type.
 *
 * @param dtype Data type enumeration
 * @return String representation of the type (e.g., "float32", "int64")
 *
 * @code
 * auto name = dtype_name(DType::Float32);  // Returns "float32"
 * @endcode
 */
constexpr auto dtype_name(DType dtype) -> std::string_view {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int8: return "int8";
        case DType::Int16: return "int16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt8: return "uint8";
        case DType::UInt16: return "uint16";
        case DType::UInt32: return "uint32";
        case DType::UInt64: return "uint64";
        case DType::Bool: return "bool";
        case DType::Complex64: return "complex64";
        case DType::Complex128: return "complex128";
        case DType::FP8_E4M3: return "fp8_e4m3";
        case DType::FP8_E5M2: return "fp8_e5m2";
        case DType::QInt8: return "qint8";
        case DType::QUInt8: return "quint8";
        case DType::QInt4x2: return "qint4x2";
    }
    return "unknown";
}

/**
 * @brief Check if a dtype is a quantized type (QInt8, QUInt8, QInt4x2).
 *
 * @param dtype Data type to check
 * @return true if dtype is a quantized type
 */
constexpr auto is_quantized(DType dtype) -> bool {
    return dtype == DType::QInt8 || dtype == DType::QUInt8 || dtype == DType::QInt4x2;
}

/**
 * @brief Check if a dtype is a floating-point type (including Float16, BFloat16, FP8).
 */
constexpr auto is_floating_type(DType dtype) -> bool {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::Float16 || dtype == DType::BFloat16 ||
           dtype == DType::FP8_E4M3 || dtype == DType::FP8_E5M2;
}

/**
 * @brief Check if a dtype is a signed or unsigned integer (not Bool, not quantized).
 */
constexpr auto is_integer_type(DType dtype) -> bool {
    return dtype == DType::Int8  || dtype == DType::Int16  ||
           dtype == DType::Int32 || dtype == DType::Int64  ||
           dtype == DType::UInt8 || dtype == DType::UInt16 ||
           dtype == DType::UInt32 || dtype == DType::UInt64;
}

/**
 * @brief Check if a dtype is a complex type.
 */
constexpr auto is_complex_type(DType dtype) -> bool {
    return dtype == DType::Complex64 || dtype == DType::Complex128;
}

namespace detail {

// Priority ordering for type promotion (higher = wider type).
// Used internally by promote_types().
constexpr auto dtype_priority(DType dt) -> int {
    switch (dt) {
        case DType::Bool:       return 0;
        case DType::UInt8:      return 1;
        case DType::Int8:       return 2;
        case DType::UInt16:     return 3;
        case DType::Int16:      return 4;
        case DType::UInt32:     return 5;
        case DType::Int32:      return 6;
        case DType::UInt64:     return 7;
        case DType::Int64:      return 8;
        case DType::FP8_E4M3:   return 8;   // FP8 types promote to Float32
        case DType::FP8_E5M2:   return 8;   // (same priority as narrow floats)
        case DType::Float16:    return 9;
        case DType::BFloat16:   return 10;
        case DType::Float32:    return 11;
        case DType::Float64:    return 12;
        case DType::Complex64:  return 13;
        case DType::Complex128: return 14;
        default:                return -1;
    }
}

} // namespace detail

/**
 * @brief Determine the promoted dtype for two dtypes.
 *
 * Follows NumPy/PyTorch promotion rules (torch.result_type semantics):
 *
 * 1. Complex dominates: Float + Complex -> Complex (widens if the float is wider).
 * 2. Float wins over integer — but Float16/BFloat16 + any integer -> Float32
 *    because those float types cannot exactly represent typical integers.
 * 3. Within category, higher precision wins.
 * 4. Bool promotes to the other type.
 * 5. Quantized + Float -> at-least-Float32 (implicit dequantization).
 * 6. Quantized + Integer -> Float32 (dequantization needed).
 * 7. Mixed FP8 (E4M3 + E5M2) -> E5M2 (wider range).
 *
 * This function is `constexpr` and lives in the core header so any layer
 * (core, ops, autograd, nn, backends) can use it without pulling in `ops/`.
 *
 * @param a First dtype
 * @param b Second dtype
 * @return The promoted (common) dtype
 */
constexpr auto promote_types(DType a, DType b) -> DType {
    // Same type: no promotion needed.
    if (a == b) return a;

    // Bool promotes to anything (except quantized, which we handle below).
    if (a == DType::Bool && !is_quantized(b)) return b;
    if (b == DType::Bool && !is_quantized(a)) return a;

    // Complex wins over everything.
    if (is_complex_type(a) && is_complex_type(b)) {
        return detail::dtype_priority(a) >= detail::dtype_priority(b) ? a : b;
    }
    if (is_complex_type(a)) {
        if (b == DType::Float64) return DType::Complex128;
        return a;
    }
    if (is_complex_type(b)) {
        if (a == DType::Float64) return DType::Complex128;
        return b;
    }

    // Quantized type promotion:
    // - Quantized + Float -> Float32 (or wider if the float type is wider)
    // - Quantized + Quantized -> QInt8 (signed wins over unsigned)
    // - Quantized + Integer -> Float32 (dequantization needed)
    if (is_quantized(a) || is_quantized(b)) {
        if (is_floating_type(a)) {
            return (detail::dtype_priority(a) >= detail::dtype_priority(DType::Float32))
                       ? a
                       : DType::Float32;
        }
        if (is_floating_type(b)) {
            return (detail::dtype_priority(b) >= detail::dtype_priority(DType::Float32))
                       ? b
                       : DType::Float32;
        }
        if (is_complex_type(a)) return a;
        if (is_complex_type(b)) return b;
        if (is_quantized(a) && is_quantized(b)) {
            if (a == b) return a;
            return DType::QInt8;  // signed type wins
        }
        return DType::Float32;  // quantized + integer -> dequant needed
    }

    // FP8 type promotion:
    // - FP8 + same FP8 -> same FP8 (handled by a == b above)
    // - FP8_E4M3 + FP8_E5M2 -> FP8_E5M2 (wider dynamic range)
    // - FP8 + non-FP8 -> Float32 (or wider if the other type is wider)
    if (a == DType::FP8_E4M3 || a == DType::FP8_E5M2) {
        if (b == DType::Float64) return DType::Float64;
        if (b == DType::FP8_E4M3 || b == DType::FP8_E5M2) {
            return DType::FP8_E5M2;  // mixed FP8 -> wider range
        }
        if (detail::dtype_priority(b) >= detail::dtype_priority(DType::Float32)) return b;
        return DType::Float32;
    }
    if (b == DType::FP8_E4M3 || b == DType::FP8_E5M2) {
        if (a == DType::Float64) return DType::Float64;
        if (detail::dtype_priority(a) >= detail::dtype_priority(DType::Float32)) return a;
        return DType::Float32;
    }

    // Float wins over integer.
    // Float16/BFloat16 can only represent integers up to 2048 exactly
    // (mantissa is 10/7 bits respectively), so any integer type mixed with
    // Float16/BFloat16 promotes to Float32 to avoid silent precision loss.
    // This matches NumPy/PyTorch semantics (torch.result_type).
    if (is_floating_type(a) && is_integer_type(b)) {
        if (a == DType::Float16 || a == DType::BFloat16) return DType::Float32;
        return a;
    }
    if (is_floating_type(b) && is_integer_type(a)) {
        if (b == DType::Float16 || b == DType::BFloat16) return DType::Float32;
        return b;
    }

    // Both floating: promote to wider.
    if (is_floating_type(a) && is_floating_type(b)) {
        return detail::dtype_priority(a) >= detail::dtype_priority(b) ? a : b;
    }

    // Both integer: promote to wider.
    if (is_integer_type(a) && is_integer_type(b)) {
        return detail::dtype_priority(a) >= detail::dtype_priority(b) ? a : b;
    }

    // Fallback: take the higher priority type.
    return detail::dtype_priority(a) >= detail::dtype_priority(b) ? a : b;
}

} // namespace tenzor
