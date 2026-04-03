/**
 * @file simd_elementwise.hpp
 * @brief Templated SIMD-accelerated elementwise operation helpers
 *
 * Provides a unified interface for applying scalar operations with
 * SIMD acceleration for float/double and scalar fallback for half types.
 * This eliminates repetitive dtype-specific SIMD boilerplate in kernels.
 *
 * Usage:
 * @code
 * TENZOR_DISPATCH_FLOAT_AND_HALF(tensor.dtype(), "relu", [&]() {
 *     elementwise_unary<scalar_t>(
 *         input.data<scalar_t>(), output.data<scalar_t>(), n,
 *         [](scalar_t x) { return std::max(scalar_t(0), x); });
 * });
 * @endcode
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include "tenzor/core/dtype.hpp"

#ifdef __AVX512F__
#define TENZOR_HAS_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define TENZOR_HAS_AVX2 1
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace tenzor::cpu {

/// Threshold for OpenMP parallelization of elementwise ops
inline constexpr size_t ELEMENTWISE_OMP_THRESHOLD = 65536;

/**
 * @brief Apply a unary scalar operation elementwise with OpenMP parallelization.
 *
 * For float and double, this uses OpenMP for large tensors.
 * For Float16/BFloat16, converts to float, applies op, converts back.
 *
 * @tparam T Element type (float, double, Float16, BFloat16)
 * @tparam Op Unary operation callable: T(T)
 */
template<typename T, typename Op>
void elementwise_unary(const T* input, T* output, size_t n, Op op) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(input[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(input[i]);
            }
        }
    } else {
        // Half types: convert through float
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(input[i]);
            output[i] = static_cast<T>(op(val));
        }
    }
}

/**
 * @brief Apply a binary scalar operation elementwise with OpenMP parallelization.
 *
 * @tparam T Element type
 * @tparam Op Binary operation callable: T(T, T)
 */
template<typename T, typename Op>
void elementwise_binary(const T* a, const T* b, T* output, size_t n, Op op) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(a[i], b[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(a[i], b[i]);
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            float va = static_cast<float>(a[i]);
            float vb = static_cast<float>(b[i]);
            output[i] = static_cast<T>(op(va, vb));
        }
    }
}

} // namespace tenzor::cpu
