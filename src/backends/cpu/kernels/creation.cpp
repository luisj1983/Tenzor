#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/utils/error.hpp"
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>

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
// Zeros Kernel - Create tensor filled with zeros
// ============================================================================

auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    size_t n = static_cast<size_t>(result.numel());

    if (dtype == DType::Float32) {
        float* data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
        // Process 16 floats at a time with AVX-512
        size_t simd_end = (n / 16) * 16;
        __m512 zero_vec = _mm512_setzero_ps();
        for (size_t i = 0; i < simd_end; i += 16) {
            _mm512_storeu_ps(&data[i], zero_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 0.0f;
        }
#elif defined(TENZOR_HAS_AVX2)
        // Process 8 floats at a time with AVX2
        size_t simd_end = (n / 8) * 8;
        __m256 zero_vec = _mm256_setzero_ps();
        for (size_t i = 0; i < simd_end; i += 8) {
            _mm256_storeu_ps(&data[i], zero_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 0.0f;
        }
#else
        // Scalar fallback - use std::fill for better optimization
        std::fill_n(data, n, 0.0f);
#endif

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();

#ifdef TENZOR_HAS_AVX512
        // Process 8 doubles at a time with AVX-512
        size_t simd_end = (n / 8) * 8;
        __m512d zero_vec = _mm512_setzero_pd();
        for (size_t i = 0; i < simd_end; i += 8) {
            _mm512_storeu_pd(&data[i], zero_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 0.0;
        }
#elif defined(TENZOR_HAS_AVX2)
        // Process 4 doubles at a time with AVX2
        size_t simd_end = (n / 4) * 4;
        __m256d zero_vec = _mm256_setzero_pd();
        for (size_t i = 0; i < simd_end; i += 4) {
            _mm256_storeu_pd(&data[i], zero_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 0.0;
        }
#else
        std::fill_n(data, n, 0.0);
#endif

    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        std::fill_n(data, n, 0);

    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        std::fill_n(data, n, 0);

    } else {
        throw std::runtime_error("zeros operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Ones Kernel - Create tensor filled with ones
// ============================================================================

auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    size_t n = static_cast<size_t>(result.numel());

    if (dtype == DType::Float32) {
        float* data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
        // Process 16 floats at a time with AVX-512
        size_t simd_end = (n / 16) * 16;
        __m512 one_vec = _mm512_set1_ps(1.0f);
        for (size_t i = 0; i < simd_end; i += 16) {
            _mm512_storeu_ps(&data[i], one_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 1.0f;
        }
#elif defined(TENZOR_HAS_AVX2)
        // Process 8 floats at a time with AVX2
        size_t simd_end = (n / 8) * 8;
        __m256 one_vec = _mm256_set1_ps(1.0f);
        for (size_t i = 0; i < simd_end; i += 8) {
            _mm256_storeu_ps(&data[i], one_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 1.0f;
        }
#else
        std::fill_n(data, n, 1.0f);
#endif

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();

#ifdef TENZOR_HAS_AVX512
        // Process 8 doubles at a time with AVX-512
        size_t simd_end = (n / 8) * 8;
        __m512d one_vec = _mm512_set1_pd(1.0);
        for (size_t i = 0; i < simd_end; i += 8) {
            _mm512_storeu_pd(&data[i], one_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 1.0;
        }
#elif defined(TENZOR_HAS_AVX2)
        // Process 4 doubles at a time with AVX2
        size_t simd_end = (n / 4) * 4;
        __m256d one_vec = _mm256_set1_pd(1.0);
        for (size_t i = 0; i < simd_end; i += 4) {
            _mm256_storeu_pd(&data[i], one_vec);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = 1.0;
        }
#else
        std::fill_n(data, n, 1.0);
#endif

    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        std::fill_n(data, n, 1);

    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        std::fill_n(data, n, 1);

    } else {
        throw std::runtime_error("ones operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Random Number Generator - Thread-local for thread safety
// ============================================================================

namespace detail {
    // Thread-local random number generator for thread safety
    thread_local std::mt19937 rng(std::random_device{}());
}

// ============================================================================
// Rand Kernel - Create tensor with uniform random values in [0, 1)
// ============================================================================

auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    size_t n = static_cast<size_t>(result.numel());

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < n; ++i) {
            data[i] = dist(detail::rng);
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < n; ++i) {
            data[i] = dist(detail::rng);
        }

    } else {
        throw std::runtime_error("rand operation only supports Float32 and Float64 dtypes");
    }

    return result;
}

// ============================================================================
// Randn Kernel - Create tensor with standard normal distribution N(0, 1)
// ============================================================================

auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    size_t n = static_cast<size_t>(result.numel());

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < n; ++i) {
            data[i] = dist(detail::rng);
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        std::normal_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < n; ++i) {
            data[i] = dist(detail::rng);
        }

    } else {
        throw std::runtime_error("randn operation only supports Float32 and Float64 dtypes");
    }

    return result;
}

// ============================================================================
// Full Kernel - Create tensor filled with a specific value
// ============================================================================

auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    size_t n = static_cast<size_t>(result.numel());

    if (dtype == DType::Float32) {
        float* data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
        size_t simd_end = (n / 16) * 16;
        __m512 val_vec = _mm512_set1_ps(value);
        for (size_t i = 0; i < simd_end; i += 16) {
            _mm512_storeu_ps(&data[i], val_vec);
        }
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = value;
        }
#elif defined(TENZOR_HAS_AVX2)
        size_t simd_end = (n / 8) * 8;
        __m256 val_vec = _mm256_set1_ps(value);
        for (size_t i = 0; i < simd_end; i += 8) {
            _mm256_storeu_ps(&data[i], val_vec);
        }
        for (size_t i = simd_end; i < n; ++i) {
            data[i] = value;
        }
#else
        std::fill_n(data, n, value);
#endif

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        std::fill_n(data, n, static_cast<double>(value));

    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        std::fill_n(data, n, static_cast<int32_t>(value));

    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        std::fill_n(data, n, static_cast<int64_t>(value));

    } else {
        throw std::runtime_error("full operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Arange Kernel - Create tensor with evenly spaced values
// ============================================================================

auto arange_kernel(float start, float end, float step, DType dtype, const Device& device) -> Tensor {
    if (step == 0.0f) {
        throw std::runtime_error("arange: step must be non-zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        // Return empty tensor
        return Tensor({0}, dtype, device);
    }

    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    Tensor result({numel}, dtype, device);

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = start + static_cast<float>(i) * step;
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<double>(start) + static_cast<double>(i) * static_cast<double>(step);
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int32_t>(start + static_cast<float>(i) * step);
        }
    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int64_t>(start + static_cast<float>(i) * step);
        }
    } else {
        throw std::runtime_error("arange operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Linspace Kernel - Create tensor with linearly spaced values
// ============================================================================

auto linspace_kernel(float start, float end, int64_t steps, DType dtype, const Device& device) -> Tensor {
    if (steps < 0) {
        throw std::runtime_error("linspace: number of steps must be non-negative");
    }
    if (steps == 0) {
        return Tensor({0}, dtype, device);
    }

    Tensor result({steps}, dtype, device);

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        if (steps == 1) {
            data[0] = start;
        } else {
            float step = (end - start) / static_cast<float>(steps - 1);
            for (int64_t i = 0; i < steps; ++i) {
                data[i] = start + static_cast<float>(i) * step;
            }
            data[steps - 1] = end;  // Exact endpoint
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        if (steps == 1) {
            data[0] = static_cast<double>(start);
        } else {
            double step = (static_cast<double>(end) - static_cast<double>(start)) / static_cast<double>(steps - 1);
            for (int64_t i = 0; i < steps; ++i) {
                data[i] = static_cast<double>(start) + static_cast<double>(i) * step;
            }
            data[steps - 1] = static_cast<double>(end);  // Exact endpoint
        }
    } else {
        throw std::runtime_error("linspace operation: only Float32 and Float64 supported");
    }

    return result;
}

// ============================================================================
// Eye Kernel - Create identity matrix
// ============================================================================

auto eye_kernel(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor {
    if (m < 0) m = n;  // Square matrix by default

    Tensor result({n, m}, dtype, device);

    // Zero-initialize first
    size_t total = static_cast<size_t>(n * m);

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        std::fill_n(data, total, 0.0f);
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1.0f;
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        std::fill_n(data, total, 0.0);
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1.0;
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        std::fill_n(data, total, 0);
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1;
        }
    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        std::fill_n(data, total, static_cast<int64_t>(0));
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1;
        }
    } else {
        throw std::runtime_error("eye operation: unsupported dtype");
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
