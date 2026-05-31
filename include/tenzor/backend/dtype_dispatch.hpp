#pragma once

/**
 * @file dtype_dispatch.hpp
 * @brief Backend-agnostic dtype dispatch macros
 *
 * Provides AT_DISPATCH-style macros that eliminate repetitive dtype switch
 * statements across all backends (CPU, CUDA, OneAPI, ROCm). Each macro
 * defines a `scalar_t` type alias within a lambda and invokes it for the
 * matching DType.
 *
 * Header-only for dlopen compatibility — dynamically loaded backend .so
 * files can include this without linking against the core library.
 *
 * Usage:
 * @code
 * TENZOR_DISPATCH_ALL_TYPES(tensor.dtype(), "my_op", [&]() {
 *     auto* data = tensor.data<scalar_t>();
 *     // ... use scalar_t as the element type
 * });
 * @endcode
 */

#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <string>

namespace tenzor {

// ============================================================================
// TENZOR_DISPATCH_ALL_TYPES — All numeric types including half-precision
// Float32, Float64, Float16, BFloat16, Int8, UInt8, Int16, Int32, Int64
// ============================================================================
#define TENZOR_DISPATCH_ALL_TYPES(DTYPE, NAME, ...)                            \
    [&] {                                                                       \
        switch (DTYPE) {                                                        \
            case DType::Float32: {                                              \
                using scalar_t = float;                                         \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Float64: {                                              \
                using scalar_t = double;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Float16: {                                              \
                using scalar_t = Float16;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::BFloat16: {                                             \
                using scalar_t = BFloat16;                                      \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int32: {                                                \
                using scalar_t = int32_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int64: {                                                \
                using scalar_t = int64_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int16: {                                                \
                using scalar_t = int16_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int8: {                                                 \
                using scalar_t = int8_t;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::UInt8: {                                                \
                using scalar_t = uint8_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    std::string(NAME) + ": unsupported dtype");                 \
        }                                                                       \
    }()

// ============================================================================
// TENZOR_DISPATCH_FLOATING_TYPES — Float32, Float64 only
// ============================================================================
#define TENZOR_DISPATCH_FLOATING_TYPES(DTYPE, NAME, ...)                        \
    [&] {                                                                       \
        switch (DTYPE) {                                                        \
            case DType::Float32: {                                              \
                using scalar_t = float;                                         \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Float64: {                                              \
                using scalar_t = double;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    std::string(NAME) + ": unsupported dtype (expected floating point)"); \
        }                                                                       \
    }()

// ============================================================================
// TENZOR_DISPATCH_FLOAT_AND_HALF — Float32, Float64, Float16, BFloat16
// ============================================================================
#define TENZOR_DISPATCH_FLOAT_AND_HALF(DTYPE, NAME, ...)                        \
    [&] {                                                                       \
        switch (DTYPE) {                                                        \
            case DType::Float32: {                                              \
                using scalar_t = float;                                         \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Float64: {                                              \
                using scalar_t = double;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Float16: {                                              \
                using scalar_t = Float16;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::BFloat16: {                                             \
                using scalar_t = BFloat16;                                      \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    std::string(NAME) + ": unsupported dtype (expected floating point)"); \
        }                                                                       \
    }()

// ============================================================================
// TENZOR_DISPATCH_INTEGER_TYPES — Int8, UInt8, Int16, Int32, Int64
// ============================================================================
#define TENZOR_DISPATCH_INTEGER_TYPES(DTYPE, NAME, ...)                         \
    [&] {                                                                       \
        switch (DTYPE) {                                                        \
            case DType::Int8: {                                                 \
                using scalar_t = int8_t;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::UInt8: {                                                \
                using scalar_t = uint8_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int16: {                                                \
                using scalar_t = int16_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int32: {                                                \
                using scalar_t = int32_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int64: {                                                \
                using scalar_t = int64_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    std::string(NAME) + ": unsupported dtype (expected integer)"); \
        }                                                                       \
    }()

// ============================================================================
// TENZOR_DISPATCH_ALL_TYPES_AND_COMPLEX — All types + Bool + Complex64/128
// ============================================================================
#define TENZOR_DISPATCH_ALL_TYPES_AND_COMPLEX(DTYPE, NAME, ...)                \
    [&] {                                                                       \
        switch (DTYPE) {                                                        \
            case DType::Float32: {                                              \
                using scalar_t = float;                                         \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Float64: {                                              \
                using scalar_t = double;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Float16: {                                              \
                using scalar_t = Float16;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::BFloat16: {                                             \
                using scalar_t = BFloat16;                                      \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int32: {                                                \
                using scalar_t = int32_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int64: {                                                \
                using scalar_t = int64_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int16: {                                                \
                using scalar_t = int16_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Int8: {                                                 \
                using scalar_t = int8_t;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::UInt8: {                                                \
                using scalar_t = uint8_t;                                       \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Bool: {                                                 \
                using scalar_t = bool;                                          \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Complex64: {                                            \
                using scalar_t = std::complex<float>;                           \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::Complex128: {                                           \
                using scalar_t = std::complex<double>;                          \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    std::string(NAME) + ": unsupported dtype");                 \
        }                                                                       \
    }()

} // namespace tenzor
