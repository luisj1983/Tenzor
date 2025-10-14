#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define TENZOR_HAS_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define TENZOR_HAS_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define TENZOR_HAS_SSE2 1
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// ReLU Activation
// ============================================================================

// Forward: max(0, x)
auto relu_kernel(const Tensor& input) -> Tensor {
    auto output = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_HAS_AVX512
        size_t i = 0;
        const size_t simd_width = 16;
        __m512 zero = _mm512_setzero_ps();

        for (; i + simd_width <= n; i += simd_width) {
            __m512 x = _mm512_loadu_ps(in_data + i);
            __m512 result = _mm512_max_ps(x, zero);
            _mm512_storeu_ps(out_data + i, result);
        }

        for (; i < n; ++i) {
            out_data[i] = std::max(0.0f, in_data[i]);
        }
#elif defined(TENZOR_HAS_AVX2)
        size_t i = 0;
        const size_t simd_width = 8;
        __m256 zero = _mm256_setzero_ps();

        for (; i + simd_width <= n; i += simd_width) {
            __m256 x = _mm256_loadu_ps(in_data + i);
            __m256 result = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(out_data + i, result);
        }

        for (; i < n; ++i) {
            out_data[i] = std::max(0.0f, in_data[i]);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(0.0f, in_data[i]);
        }
#endif
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

#ifdef TENZOR_HAS_AVX512
        size_t i = 0;
        const size_t simd_width = 8;
        __m512d zero = _mm512_setzero_pd();

        for (; i + simd_width <= n; i += simd_width) {
            __m512d x = _mm512_loadu_pd(in_data + i);
            __m512d result = _mm512_max_pd(x, zero);
            _mm512_storeu_pd(out_data + i, result);
        }

        for (; i < n; ++i) {
            out_data[i] = std::max(0.0, in_data[i]);
        }
#elif defined(TENZOR_HAS_AVX2)
        size_t i = 0;
        const size_t simd_width = 4;
        __m256d zero = _mm256_setzero_pd();

        for (; i + simd_width <= n; i += simd_width) {
            __m256d x = _mm256_loadu_pd(in_data + i);
            __m256d result = _mm256_max_pd(x, zero);
            _mm256_storeu_pd(out_data + i, result);
        }

        for (; i < n; ++i) {
            out_data[i] = std::max(0.0, in_data[i]);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(0.0, in_data[i]);
        }
#endif
    } else {
        throw std::runtime_error("ReLU only supports Float32 and Float64");
    }

    return output;
}

// Backward: grad_out * (x > 0)
auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor {
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_HAS_AVX512
        size_t i = 0;
        const size_t simd_width = 16;
        __m512 zero = _mm512_setzero_ps();

        for (; i + simd_width <= n; i += simd_width) {
            __m512 x = _mm512_loadu_ps(in_data + i);
            __m512 grad_out = _mm512_loadu_ps(grad_out_data + i);
            __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
            __m512 grad_in = _mm512_mask_blend_ps(mask, zero, grad_out);
            _mm512_storeu_ps(grad_in_data + i, grad_in);
        }

        for (; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : 0.0f);
        }
#elif defined(TENZOR_HAS_AVX2)
        size_t i = 0;
        const size_t simd_width = 8;
        __m256 zero = _mm256_setzero_ps();

        for (; i + simd_width <= n; i += simd_width) {
            __m256 x = _mm256_loadu_ps(in_data + i);
            __m256 grad_out = _mm256_loadu_ps(grad_out_data + i);
            __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 grad_in = _mm256_and_ps(mask, grad_out);
            _mm256_storeu_ps(grad_in_data + i, grad_in);
        }

        for (; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : 0.0f);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : 0.0f);
        }
#endif
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : 0.0);
        }
    } else {
        throw std::runtime_error("ReLU backward only supports Float32 and Float64");
    }

    return grad_input;
}

// ============================================================================
// Sigmoid Activation
// ============================================================================

// Forward: 1 / (1 + exp(-x))
// Use numerically stable version: if x >= 0: 1/(1+exp(-x)), else: exp(x)/(1+exp(x))
auto sigmoid_kernel(const Tensor& input) -> Tensor {
    auto output = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        // Numerically stable sigmoid
        for (size_t i = 0; i < n; ++i) {
            if (in_data[i] >= 0.0f) {
                out_data[i] = 1.0f / (1.0f + std::exp(-in_data[i]));
            } else {
                float exp_x = std::exp(in_data[i]);
                out_data[i] = exp_x / (1.0f + exp_x);
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            if (in_data[i] >= 0.0) {
                out_data[i] = 1.0 / (1.0 + std::exp(-in_data[i]));
            } else {
                double exp_x = std::exp(in_data[i]);
                out_data[i] = exp_x / (1.0 + exp_x);
            }
        }
    } else {
        throw std::runtime_error("Sigmoid only supports Float32 and Float64");
    }

    return output;
}

// Backward: grad_out * sigmoid(x) * (1 - sigmoid(x))
auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor {
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            float sigmoid_x = 1.0f / (1.0f + std::exp(-in_data[i]));
            grad_in_data[i] = grad_out_data[i] * sigmoid_x * (1.0f - sigmoid_x);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            double sigmoid_x = 1.0 / (1.0 + std::exp(-in_data[i]));
            grad_in_data[i] = grad_out_data[i] * sigmoid_x * (1.0 - sigmoid_x);
        }
    } else {
        throw std::runtime_error("Sigmoid backward only supports Float32 and Float64");
    }

    return grad_input;
}

// ============================================================================
// Tanh Activation
// ============================================================================

// Forward: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
auto tanh_kernel(const Tensor& input) -> Tensor {
    auto output = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::tanh(in_data[i]);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::tanh(in_data[i]);
        }
    } else {
        throw std::runtime_error("Tanh only supports Float32 and Float64");
    }

    return output;
}

// Backward: grad_out * (1 - tanh(x)^2)
auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor {
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            float tanh_x = std::tanh(in_data[i]);
            grad_in_data[i] = grad_out_data[i] * (1.0f - tanh_x * tanh_x);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            double tanh_x = std::tanh(in_data[i]);
            grad_in_data[i] = grad_out_data[i] * (1.0 - tanh_x * tanh_x);
        }
    } else {
        throw std::runtime_error("Tanh backward only supports Float32 and Float64");
    }

    return grad_input;
}

// ============================================================================
// GELU Activation
// ============================================================================

// Forward: 0.5 * x * (1 + erf(x / sqrt(2)))
// GELU (Gaussian Error Linear Unit)
auto gelu_kernel(const Tensor& input) -> Tensor {
    auto output = zeros_like(input);
    constexpr float sqrt_2 = 1.41421356237f;

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = 0.5f * x * (1.0f + std::erf(x / sqrt_2));
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();
        constexpr double sqrt_2_d = 1.41421356237;

        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            out_data[i] = 0.5 * x * (1.0 + std::erf(x / sqrt_2_d));
        }
    } else {
        throw std::runtime_error("GELU only supports Float32 and Float64");
    }

    return output;
}

// Backward: grad_out * (0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2))
auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor {
    auto grad_input = zeros_like(input);
    constexpr float sqrt_2 = 1.41421356237f;
    constexpr float sqrt_2_pi = 2.50662827463f;  // sqrt(2*pi)

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float cdf = 0.5f * (1.0f + std::erf(x / sqrt_2));
            float pdf = (1.0f / sqrt_2_pi) * std::exp(-0.5f * x * x);
            grad_in_data[i] = grad_out_data[i] * (cdf + x * pdf);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();
        constexpr double sqrt_2_d = 1.41421356237;
        constexpr double sqrt_2_pi_d = 2.50662827463;

        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            double cdf = 0.5 * (1.0 + std::erf(x / sqrt_2_d));
            double pdf = (1.0 / sqrt_2_pi_d) * std::exp(-0.5 * x * x);
            grad_in_data[i] = grad_out_data[i] * (cdf + x * pdf);
        }
    } else {
        throw std::runtime_error("GELU backward only supports Float32 and Float64");
    }

    return grad_input;
}

// ============================================================================
// Leaky ReLU Activation
// ============================================================================

// Forward: x if x > 0 else alpha * x
auto leaky_relu_kernel(const Tensor& input, float alpha) -> Tensor {
    auto output = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_HAS_AVX512
        size_t i = 0;
        const size_t simd_width = 16;
        __m512 alpha_vec = _mm512_set1_ps(alpha);
        __m512 zero = _mm512_setzero_ps();

        for (; i + simd_width <= n; i += simd_width) {
            __m512 x = _mm512_loadu_ps(in_data + i);
            __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
            __m512 negative_part = _mm512_mul_ps(x, alpha_vec);
            __m512 result = _mm512_mask_blend_ps(mask, negative_part, x);
            _mm512_storeu_ps(out_data + i, result);
        }

        for (; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : alpha * in_data[i];
        }
#elif defined(TENZOR_HAS_AVX2)
        size_t i = 0;
        const size_t simd_width = 8;
        __m256 alpha_vec = _mm256_set1_ps(alpha);
        __m256 zero = _mm256_setzero_ps();

        for (; i + simd_width <= n; i += simd_width) {
            __m256 x = _mm256_loadu_ps(in_data + i);
            __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 negative_part = _mm256_mul_ps(x, alpha_vec);
            __m256 result = _mm256_blendv_ps(negative_part, x, mask);
            _mm256_storeu_ps(out_data + i, result);
        }

        for (; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : alpha * in_data[i];
        }
#else
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : alpha * in_data[i];
        }
#endif
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();
        double alpha_d = static_cast<double>(alpha);

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0 ? in_data[i] : alpha_d * in_data[i];
        }
    } else {
        throw std::runtime_error("Leaky ReLU only supports Float32 and Float64");
    }

    return output;
}

// Backward: grad_out * (1 if x > 0 else alpha)
auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha) -> Tensor {
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : alpha);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();
        double alpha_d = static_cast<double>(alpha);

        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : alpha_d);
        }
    } else {
        throw std::runtime_error("Leaky ReLU backward only supports Float32 and Float64");
    }

    return grad_input;
}

// ============================================================================
// Softmax Activation
// ============================================================================

// Helper: Compute max along dimension
static auto compute_max_along_dim(const float* data, const std::vector<int64_t>& shape, int64_t dim) -> std::vector<float> {
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }

    int64_t dim_size = shape[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
        inner_size *= shape[i];
    }

    std::vector<float> max_vals(outer_size * inner_size, -std::numeric_limits<float>::infinity());

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            for (int64_t j = 0; j < dim_size; ++j) {
                int64_t idx = (i * dim_size + j) * inner_size + k;
                int64_t max_idx = i * inner_size + k;
                max_vals[max_idx] = std::max(max_vals[max_idx], data[idx]);
            }
        }
    }

    return max_vals;
}

// Forward: exp(x_i - max) / sum(exp(x_j - max))
auto softmax_kernel(const Tensor& input, int64_t dim) -> Tensor {
    auto output = zeros_like(input);

    // Handle negative dimension
    if (dim < 0) {
        dim += input.ndim();
    }

    if (dim < 0 || dim >= input.ndim()) {
        throw std::runtime_error("Softmax dimension out of range");
    }

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();

        auto shape_span = input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        // Compute max values for numerical stability
        auto max_vals = compute_max_along_dim(in_data, shape, dim);

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Compute exp(x - max) and sum
        std::vector<float> sum_exp(outer_size * inner_size, 0.0f);

        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float max_val = max_vals[max_idx];

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    float exp_val = std::exp(in_data[idx] - max_val);
                    out_data[idx] = exp_val;
                    sum_exp[max_idx] += exp_val;
                }
            }
        }

        // Normalize
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float sum = sum_exp[max_idx];

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] /= sum;
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();

        auto shape_span = input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Simple implementation for double
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                // Find max
                double max_val = -std::numeric_limits<double>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, in_data[idx]);
                }

                // Compute exp and sum
                double sum = 0.0;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    double exp_val = std::exp(in_data[idx] - max_val);
                    out_data[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] /= sum;
                }
            }
        }
    } else {
        throw std::runtime_error("Softmax only supports Float32 and Float64");
    }

    return output;
}

// Backward: Jacobian-vector product
// grad_input[i] = softmax[i] * (grad_output[i] - sum(grad_output * softmax))
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor {
    auto grad_input = zeros_like(output);

    // Handle negative dimension
    if (dim < 0) {
        dim += output.ndim();
    }

    if (output.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* out_data = output.data<float>();
        float* grad_in_data = grad_input.data<float>();

        auto shape_span = output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Compute sum(grad_output * softmax) for each position
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum = 0.0f;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum += grad_out_data[idx] * out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = out_data[idx] * (grad_out_data[idx] - sum);
                }
            }
        }
    } else if (output.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* out_data = output.data<double>();
        double* grad_in_data = grad_input.data<double>();

        auto shape_span = output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                double sum = 0.0;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum += grad_out_data[idx] * out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = out_data[idx] * (grad_out_data[idx] - sum);
                }
            }
        }
    } else {
        throw std::runtime_error("Softmax backward only supports Float32 and Float64");
    }

    return grad_input;
}

// ============================================================================
// LogSoftmax Activation
// ============================================================================

// Forward: log(softmax(x, dim)) = x - max(x) - log(sum(exp(x - max(x))))
// More numerically stable than computing softmax then taking log
auto log_softmax_kernel(const Tensor& input, int64_t dim) -> Tensor {
    auto output = zeros_like(input);

    // Handle negative dimension
    if (dim < 0) {
        dim += input.ndim();
    }

    if (dim < 0 || dim >= input.ndim()) {
        throw std::runtime_error("LogSoftmax dimension out of range");
    }

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();

        auto shape_span = input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        // Compute max values for numerical stability
        auto max_vals = compute_max_along_dim(in_data, shape, dim);

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Compute log(sum(exp(x - max)))
        std::vector<float> log_sum_exp(outer_size * inner_size, 0.0f);

        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float max_val = max_vals[max_idx];
                float sum_exp = 0.0f;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_exp += std::exp(in_data[idx] - max_val);
                }

                log_sum_exp[max_idx] = std::log(sum_exp);
            }
        }

        // Compute log_softmax = x - max - log_sum_exp
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float max_val = max_vals[max_idx];
                float lse = log_sum_exp[max_idx];

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = in_data[idx] - max_val - lse;
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();

        auto shape_span = input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                // Find max
                double max_val = -std::numeric_limits<double>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, in_data[idx]);
                }

                // Compute log(sum(exp(x - max)))
                double sum_exp = 0.0;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_exp += std::exp(in_data[idx] - max_val);
                }
                double log_sum_exp = std::log(sum_exp);

                // Compute log_softmax
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = in_data[idx] - max_val - log_sum_exp;
                }
            }
        }
    } else {
        throw std::runtime_error("LogSoftmax only supports Float32 and Float64");
    }

    return output;
}

// Backward: grad_input = grad_output - exp(log_softmax) * sum(grad_output)
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor {
    auto grad_input = zeros_like(output);

    // Handle negative dimension
    if (dim < 0) {
        dim += output.ndim();
    }

    if (output.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* out_data = output.data<float>();
        float* grad_in_data = grad_input.data<float>();

        auto shape_span = output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum_grad = 0.0f;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_grad += grad_out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = grad_out_data[idx] - std::exp(out_data[idx]) * sum_grad;
                }
            }
        }
    } else if (output.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* out_data = output.data<double>();
        double* grad_in_data = grad_input.data<double>();

        auto shape_span = output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                double sum_grad = 0.0;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_grad += grad_out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = grad_out_data[idx] - std::exp(out_data[idx]) * sum_grad;
                }
            }
        }
    } else {
        throw std::runtime_error("LogSoftmax backward only supports Float32 and Float64");
    }

    return grad_input;
}

} // namespace cpu
} // namespace tenzor
