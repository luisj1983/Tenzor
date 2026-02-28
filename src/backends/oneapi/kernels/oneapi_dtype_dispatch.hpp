#pragma once

/**
 * @file oneapi_dtype_dispatch.hpp
 * @brief SYCL-specific dtype dispatch macros for OneAPI kernels
 *
 * Maps DType to the correct SYCL-compatible C++ types:
 * - Float32  -> float
 * - Float64  -> double
 * - Float16  -> sycl::half
 * - BFloat16 -> uint16_t  (with bf16_to_f32/f32_to_bf16 helpers)
 * - Int8     -> int8_t
 * - Int16    -> int16_t
 * - Int32    -> int32_t
 * - Int64    -> int64_t
 * - UInt8    -> uint8_t
 * - Bool     -> bool
 *
 * Usage:
 * @code
 * ONEAPI_DISPATCH_ALL_TYPES(tensor.dtype(), "my_op", [&]() {
 *     auto* data = get_data_ptr<scalar_t>(tensor);
 *     queue.parallel_for<MyKernel<scalar_t>>(range, [=](id<1> idx) {
 *         // ... use scalar_t
 *     }).wait();
 * });
 * @endcode
 */

#include "tenzor/core/dtype.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace tenzor {
namespace oneapi {

// ============================================================================
// ONEAPI_DISPATCH_FLOATING_TYPES — Float32, Float64 only
// ============================================================================
#define ONEAPI_DISPATCH_FLOATING_TYPES(DTYPE, NAME, ...)                        \
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
// ONEAPI_DISPATCH_FLOAT_AND_HALF — Float32, Float64, Float16(sycl::half), BFloat16(uint16_t)
// ============================================================================
#define ONEAPI_DISPATCH_FLOAT_AND_HALF(DTYPE, NAME, ...)                        \
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
                using scalar_t = sycl::half;                                    \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::BFloat16: {                                             \
                using scalar_t = uint16_t;                                      \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    std::string(NAME) + ": unsupported dtype (expected floating point)"); \
        }                                                                       \
    }()

// ============================================================================
// ONEAPI_DISPATCH_ALL_TYPES — All numeric types with SYCL mappings
// ============================================================================
#define ONEAPI_DISPATCH_ALL_TYPES(DTYPE, NAME, ...)                             \
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
                using scalar_t = sycl::half;                                    \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::BFloat16: {                                             \
                using scalar_t = uint16_t;                                      \
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
// ONEAPI_DISPATCH_ALL_TYPES_AND_BOOL — All types including Bool
// ============================================================================
#define ONEAPI_DISPATCH_ALL_TYPES_AND_BOOL(DTYPE, NAME, ...)                    \
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
                using scalar_t = sycl::half;                                    \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case DType::BFloat16: {                                             \
                using scalar_t = uint16_t;                                      \
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
            default:                                                            \
                throw std::runtime_error(                                       \
                    std::string(NAME) + ": unsupported dtype");                 \
        }                                                                       \
    }()

// ============================================================================
// ONEAPI_DISPATCH_INTEGER_TYPES — Int8, UInt8, Int16, Int32, Int64
// ============================================================================
#define ONEAPI_DISPATCH_INTEGER_TYPES(DTYPE, NAME, ...)                         \
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

} // namespace oneapi
} // namespace tenzor
