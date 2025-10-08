#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <complex>

namespace tenzor {

// Data type enumeration
enum class DType : uint8_t {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
    Complex64,
    Complex128
};

// Type traits
template<typename T>
concept ScalarType = std::is_arithmetic_v<T> ||
                     std::is_same_v<T, std::complex<float>> ||
                     std::is_same_v<T, std::complex<double>>;

template<typename T>
concept IntegralType = std::is_integral_v<T>;

template<typename T>
concept FloatingType = std::is_floating_point_v<T>;

// DType traits
template<DType dt>
struct dtype_traits;

template<> struct dtype_traits<DType::Float32> { using type = float; };
template<> struct dtype_traits<DType::Float64> { using type = double; };
template<> struct dtype_traits<DType::Int32> { using type = int32_t; };
template<> struct dtype_traits<DType::Int64> { using type = int64_t; };
template<> struct dtype_traits<DType::UInt8> { using type = uint8_t; };
template<> struct dtype_traits<DType::Bool> { using type = bool; };
template<> struct dtype_traits<DType::Complex64> { using type = std::complex<float>; };
template<> struct dtype_traits<DType::Complex128> { using type = std::complex<double>; };

template<DType dt>
using dtype_t = typename dtype_traits<dt>::type;

// Helper functions
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
    }
    return 0;
}

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
    }
    return "unknown";
}

} // namespace tenzor
