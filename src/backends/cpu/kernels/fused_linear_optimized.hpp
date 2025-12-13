/**
 * @file fused_linear_optimized.hpp
 * @brief Optimized fused linear layer operations
 *
 * Key features:
 * - MatMul + Bias + Activation fused into single pass
 * - Uses optimized GEMM micro-kernels
 * - SIMD-accelerated bias addition and activation
 * - Support for ReLU, GELU, Swish, Sigmoid activations
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "gemm_optimized.hpp"
#include "simd_fast_math.hpp"
#include "buffer_pool.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_LINEAR_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_LINEAR_AVX2
    #endif
    #if defined(__FMA__)
        #define TENZOR_LINEAR_FMA
    #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {
namespace fused_linear {

// ============================================================================
// Activation Types
// ============================================================================

enum class Activation {
    None,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,      // SiLU
    LeakyReLU
};

// ============================================================================
// SIMD Bias + Activation Kernels
// ============================================================================

#ifdef TENZOR_LINEAR_AVX2

/**
 * @brief Add bias + ReLU (AVX2)
 */
__attribute__((target("avx2")))
inline void bias_relu_avx2(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features
) {
    __m256 zero = _mm256_setzero_ps();

    #pragma omp parallel for if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * out_features;

        int64_t j = 0;
        for (; j + 8 <= out_features; j += 8) {
            __m256 v = _mm256_loadu_ps(out_row + j);
            __m256 b_vec = _mm256_loadu_ps(bias + j);
            v = _mm256_add_ps(v, b_vec);
            v = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(out_row + j, v);
        }
        for (; j < out_features; ++j) {
            out_row[j] = std::max(0.0f, out_row[j] + bias[j]);
        }
    }
}

/**
 * @brief Add bias + GELU (AVX2)
 */
__attribute__((target("avx2,fma")))
inline void bias_gelu_avx2(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features
) {
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sqrt_2_over_pi = _mm256_set1_ps(0.7978845608f);
    __m256 coeff = _mm256_set1_ps(0.044715f);

    #pragma omp parallel for if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * out_features;

        int64_t j = 0;
        for (; j + 8 <= out_features; j += 8) {
            __m256 v = _mm256_loadu_ps(out_row + j);
            __m256 b_vec = _mm256_loadu_ps(bias + j);
            __m256 x = _mm256_add_ps(v, b_vec);

            // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);

#ifdef TENZOR_LINEAR_FMA
            __m256 inner = _mm256_fmadd_ps(coeff, x3, x);
#else
            __m256 inner = _mm256_add_ps(x, _mm256_mul_ps(coeff, x3));
#endif
            inner = _mm256_mul_ps(sqrt_2_over_pi, inner);

            // tanh approximation using fast_math
            __m256 th = fast_math::tanh_avx2(inner);

            __m256 result = _mm256_mul_ps(half, x);
            result = _mm256_mul_ps(result, _mm256_add_ps(one, th));

            _mm256_storeu_ps(out_row + j, result);
        }

        // Scalar remainder
        for (; j < out_features; ++j) {
            float x = out_row[j] + bias[j];
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);
            out_row[j] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }
}

/**
 * @brief Add bias + Sigmoid (AVX2)
 */
__attribute__((target("avx2,fma")))
inline void bias_sigmoid_avx2(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features
) {
    #pragma omp parallel for if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * out_features;

        int64_t j = 0;
        for (; j + 8 <= out_features; j += 8) {
            __m256 v = _mm256_loadu_ps(out_row + j);
            __m256 b_vec = _mm256_loadu_ps(bias + j);
            __m256 x = _mm256_add_ps(v, b_vec);

            // Sigmoid using fast_math
            __m256 result = fast_math::sigmoid_avx2(x);
            _mm256_storeu_ps(out_row + j, result);
        }

        for (; j < out_features; ++j) {
            float x = out_row[j] + bias[j];
            out_row[j] = 1.0f / (1.0f + std::exp(-x));
        }
    }
}

/**
 * @brief Add bias + Tanh (AVX2)
 */
__attribute__((target("avx2,fma")))
inline void bias_tanh_avx2(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features
) {
    #pragma omp parallel for if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * out_features;

        int64_t j = 0;
        for (; j + 8 <= out_features; j += 8) {
            __m256 v = _mm256_loadu_ps(out_row + j);
            __m256 b_vec = _mm256_loadu_ps(bias + j);
            __m256 x = _mm256_add_ps(v, b_vec);

            __m256 result = fast_math::tanh_avx2(x);
            _mm256_storeu_ps(out_row + j, result);
        }

        for (; j < out_features; ++j) {
            float x = out_row[j] + bias[j];
            out_row[j] = std::tanh(x);
        }
    }
}

/**
 * @brief Add bias + Swish/SiLU (AVX2)
 * Swish(x) = x * sigmoid(x)
 */
__attribute__((target("avx2,fma")))
inline void bias_swish_avx2(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features
) {
    #pragma omp parallel for if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * out_features;

        int64_t j = 0;
        for (; j + 8 <= out_features; j += 8) {
            __m256 v = _mm256_loadu_ps(out_row + j);
            __m256 b_vec = _mm256_loadu_ps(bias + j);
            __m256 x = _mm256_add_ps(v, b_vec);

            __m256 sig = fast_math::sigmoid_avx2(x);
            __m256 result = _mm256_mul_ps(x, sig);
            _mm256_storeu_ps(out_row + j, result);
        }

        for (; j < out_features; ++j) {
            float x = out_row[j] + bias[j];
            float sig = 1.0f / (1.0f + std::exp(-x));
            out_row[j] = x * sig;
        }
    }
}

/**
 * @brief Add bias + Leaky ReLU (AVX2)
 */
__attribute__((target("avx2")))
inline void bias_leaky_relu_avx2(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features,
    float negative_slope = 0.01f
) {
    __m256 zero = _mm256_setzero_ps();
    __m256 slope = _mm256_set1_ps(negative_slope);

    #pragma omp parallel for if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * out_features;

        int64_t j = 0;
        for (; j + 8 <= out_features; j += 8) {
            __m256 v = _mm256_loadu_ps(out_row + j);
            __m256 b_vec = _mm256_loadu_ps(bias + j);
            __m256 x = _mm256_add_ps(v, b_vec);

            // LeakyReLU: max(x, slope * x)
            __m256 scaled = _mm256_mul_ps(x, slope);
            __m256 result = _mm256_max_ps(x, scaled);

            _mm256_storeu_ps(out_row + j, result);
        }

        for (; j < out_features; ++j) {
            float x = out_row[j] + bias[j];
            out_row[j] = x > 0.0f ? x : negative_slope * x;
        }
    }
}

/**
 * @brief Add bias only (no activation) - AVX2
 */
__attribute__((target("avx2")))
inline void bias_only_avx2(
    float* output,
    const float* bias,
    int64_t batch_size,
    int64_t out_features
) {
    #pragma omp parallel for if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * out_features;

        int64_t j = 0;
        for (; j + 8 <= out_features; j += 8) {
            __m256 v = _mm256_loadu_ps(out_row + j);
            __m256 b_vec = _mm256_loadu_ps(bias + j);
            v = _mm256_add_ps(v, b_vec);
            _mm256_storeu_ps(out_row + j, v);
        }
        for (; j < out_features; ++j) {
            out_row[j] += bias[j];
        }
    }
}

#endif // TENZOR_LINEAR_AVX2

// ============================================================================
// Scalar Fallback Functions
// ============================================================================

namespace scalar {

inline void bias_relu(float* output, const float* bias, int64_t batch_size, int64_t out_features) {
    #pragma omp parallel for collapse(2) if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < out_features; ++j) {
            float val = output[b * out_features + j] + bias[j];
            output[b * out_features + j] = std::max(0.0f, val);
        }
    }
}

inline void bias_gelu(float* output, const float* bias, int64_t batch_size, int64_t out_features) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    #pragma omp parallel for collapse(2) if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < out_features; ++j) {
            float x = output[b * out_features + j] + bias[j];
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x3);
            output[b * out_features + j] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }
}

inline void bias_sigmoid(float* output, const float* bias, int64_t batch_size, int64_t out_features) {
    #pragma omp parallel for collapse(2) if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < out_features; ++j) {
            float x = output[b * out_features + j] + bias[j];
            output[b * out_features + j] = 1.0f / (1.0f + std::exp(-x));
        }
    }
}

inline void bias_tanh(float* output, const float* bias, int64_t batch_size, int64_t out_features) {
    #pragma omp parallel for collapse(2) if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < out_features; ++j) {
            float x = output[b * out_features + j] + bias[j];
            output[b * out_features + j] = std::tanh(x);
        }
    }
}

inline void bias_swish(float* output, const float* bias, int64_t batch_size, int64_t out_features) {
    #pragma omp parallel for collapse(2) if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < out_features; ++j) {
            float x = output[b * out_features + j] + bias[j];
            float sig = 1.0f / (1.0f + std::exp(-x));
            output[b * out_features + j] = x * sig;
        }
    }
}

inline void bias_leaky_relu(float* output, const float* bias, int64_t batch_size, int64_t out_features, float negative_slope = 0.01f) {
    #pragma omp parallel for collapse(2) if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < out_features; ++j) {
            float x = output[b * out_features + j] + bias[j];
            output[b * out_features + j] = x > 0.0f ? x : negative_slope * x;
        }
    }
}

inline void bias_only(float* output, const float* bias, int64_t batch_size, int64_t out_features) {
    #pragma omp parallel for collapse(2) if(batch_size * out_features > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < out_features; ++j) {
            output[b * out_features + j] += bias[j];
        }
    }
}

} // namespace scalar

// ============================================================================
// Main Fused Linear Function
// ============================================================================

/**
 * @brief Fused Linear layer: output = activation(input @ weight^T + bias)
 *
 * @param input Input tensor (batch_size x in_features)
 * @param weight Weight tensor (out_features x in_features)
 * @param bias Bias tensor (out_features) or nullptr
 * @param output Output tensor (batch_size x out_features)
 * @param batch_size Number of samples
 * @param in_features Input feature dimension
 * @param out_features Output feature dimension
 * @param activation Activation function to apply
 * @param negative_slope For LeakyReLU only
 */
inline void linear_bias_activation(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    Activation activation = Activation::None,
    float negative_slope = 0.01f
) {
    // Step 1: Matrix multiplication using optimized GEMM
    // input: (batch_size, in_features)
    // weight: (out_features, in_features) - will be transposed
    // output: (batch_size, out_features)

    std::memset(output, 0, batch_size * out_features * sizeof(float));

    gemm::gemm_transB_optimized(
        input, weight, output,
        batch_size, out_features, in_features
    );

    // Step 2: Bias + Activation (fused)
    if (bias == nullptr) {
        // Apply activation only
        switch (activation) {
            case Activation::None:
                break;
            case Activation::ReLU: {
                int64_t n = batch_size * out_features;
                #pragma omp parallel for if(n > 10000)
                for (int64_t i = 0; i < n; ++i) {
                    output[i] = std::max(0.0f, output[i]);
                }
                break;
            }
            // Add other activations as needed
            default:
                break;
        }
        return;
    }

    // Apply bias + activation
#ifdef TENZOR_LINEAR_AVX2
    switch (activation) {
        case Activation::None:
            bias_only_avx2(output, bias, batch_size, out_features);
            break;
        case Activation::ReLU:
            bias_relu_avx2(output, bias, batch_size, out_features);
            break;
        case Activation::GELU:
            bias_gelu_avx2(output, bias, batch_size, out_features);
            break;
        case Activation::Sigmoid:
            bias_sigmoid_avx2(output, bias, batch_size, out_features);
            break;
        case Activation::Tanh:
            bias_tanh_avx2(output, bias, batch_size, out_features);
            break;
        case Activation::Swish:
            bias_swish_avx2(output, bias, batch_size, out_features);
            break;
        case Activation::LeakyReLU:
            bias_leaky_relu_avx2(output, bias, batch_size, out_features, negative_slope);
            break;
    }
#else
    switch (activation) {
        case Activation::None:
            scalar::bias_only(output, bias, batch_size, out_features);
            break;
        case Activation::ReLU:
            scalar::bias_relu(output, bias, batch_size, out_features);
            break;
        case Activation::GELU:
            scalar::bias_gelu(output, bias, batch_size, out_features);
            break;
        case Activation::Sigmoid:
            scalar::bias_sigmoid(output, bias, batch_size, out_features);
            break;
        case Activation::Tanh:
            scalar::bias_tanh(output, bias, batch_size, out_features);
            break;
        case Activation::Swish:
            scalar::bias_swish(output, bias, batch_size, out_features);
            break;
        case Activation::LeakyReLU:
            scalar::bias_leaky_relu(output, bias, batch_size, out_features, negative_slope);
            break;
    }
#endif
}

/**
 * @brief Convenience wrapper for Linear + ReLU
 */
inline void linear_relu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features
) {
    linear_bias_activation(input, weight, bias, output, batch_size, in_features, out_features, Activation::ReLU);
}

/**
 * @brief Convenience wrapper for Linear + GELU
 */
inline void linear_gelu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features
) {
    linear_bias_activation(input, weight, bias, output, batch_size, in_features, out_features, Activation::GELU);
}

} // namespace fused_linear
} // namespace cpu
} // namespace tenzor
