/**
 * @file simd_dispatch.cpp
 * @brief Implementation of function pointer dispatch system
 */

#include "simd_dispatch.hpp"
#include "simd_fast_math.hpp"
#include <mutex>

namespace tenzor {
namespace cpu {
namespace dispatch {

// Global dispatch table
SIMDDispatch g_dispatch = {
    nullptr, nullptr, nullptr, nullptr,  // add, sub, mul, div
    nullptr, nullptr, nullptr, nullptr,  // sqrt, exp, log, fma
    nullptr, nullptr, nullptr, nullptr,  // relu, sigmoid, tanh, gelu
    false  // initialized
};

// Initialization mutex for thread safety
static std::mutex init_mutex;

// ============================================================================
// Scalar implementations (fallback)
// ============================================================================

namespace scalar_impl {

void add(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
}

void sub(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] - b[i];
}

void mul(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
}

void div(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] / b[i];
}

void sqrt(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::sqrt(a[i]);
}

void exp(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::exp(a[i]);
}

void log(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::log(a[i]);
}

void fma(const float* a, const float* b, const float* c, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i] + c[i];
}

void relu(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::max(0.0f, a[i]);
}

void sigmoid(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = 1.0f / (1.0f + std::exp(-a[i]));
}

void tanh(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::tanh(a[i]);
}

void gelu(const float* a, float* out, size_t size) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    for (size_t i = 0; i < size; ++i) {
        float x = a[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

} // namespace scalar_impl

// ============================================================================
// AVX2 wrappers
// ============================================================================

#if defined(__AVX2__)

namespace avx2_impl {

void add(const float* a, const float* b, float* out, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] + b[i];
}

void sub(const float* a, const float* b, float* out, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] - b[i];
}

void mul(const float* a, const float* b, float* out, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] * b[i];
}

void div(const float* a, const float* b, float* out, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] / b[i];
}

void sqrt(const float* a, float* out, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_sqrt_ps(va));
    }
    for (; i < size; ++i) out[i] = std::sqrt(a[i]);
}

void exp(const float* a, float* out, size_t size) {
    fast_math::exp_batch_avx2(a, out, size);
}

void log(const float* a, float* out, size_t size) {
    fast_math::log_batch_avx2(a, out, size);
}

void fma(const float* a, const float* b, const float* c, float* out, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(va, vb, vc));
    }
    for (; i < size; ++i) out[i] = a[i] * b[i] + c[i];
}

void relu(const float* a, float* out, size_t size) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(zero, va));
    }
    for (; i < size; ++i) out[i] = std::max(0.0f, a[i]);
}

void sigmoid(const float* a, float* out, size_t size) {
    fast_math::sigmoid_batch_avx2(a, out, size);
}

void tanh(const float* a, float* out, size_t size) {
    fast_math::tanh_batch_avx2(a, out, size);
}

void gelu(const float* a, float* out, size_t size) {
    fast_math::gelu_batch_avx2(a, out, size);
}

} // namespace avx2_impl

#endif // __AVX2__

// ============================================================================
// Initialization
// AVX-512 implementations come from separate TUs (math_simd_avx512.cpp,
// activations_simd_avx512.cpp) via the avx512:: namespace declared in simd.hpp.
// ============================================================================

void init_dispatch() {
    std::lock_guard<std::mutex> lock(init_mutex);

    // Check if already initialized
    if (g_dispatch.initialized) {
        return;
    }

    const auto& cpu = CPUInfo::get();

    if (cpu.has_avx512()) {
        // Use AVX-512 implementations (from separate TU)
        g_dispatch.add = avx512::add;
        g_dispatch.sub = avx512::sub;
        g_dispatch.mul = avx512::mul;
        g_dispatch.div = avx512::div;
        g_dispatch.sqrt = avx512::sqrt;
        g_dispatch.exp = avx512::exp;
        g_dispatch.log = avx512::log;
        g_dispatch.fma = avx512::fma;
        g_dispatch.relu = avx512::relu;
        g_dispatch.sigmoid = avx512::sigmoid;
        g_dispatch.tanh = avx512::tanh;
        g_dispatch.gelu = avx512::gelu;
        g_dispatch.initialized = true;
        return;
    }

#if defined(__AVX2__)
    if (cpu.has_avx2()) {
        // Use AVX2 implementations
        g_dispatch.add = avx2_impl::add;
        g_dispatch.sub = avx2_impl::sub;
        g_dispatch.mul = avx2_impl::mul;
        g_dispatch.div = avx2_impl::div;
        g_dispatch.sqrt = avx2_impl::sqrt;
        g_dispatch.exp = avx2_impl::exp;
        g_dispatch.log = avx2_impl::log;
        g_dispatch.fma = avx2_impl::fma;
        g_dispatch.relu = avx2_impl::relu;
        g_dispatch.sigmoid = avx2_impl::sigmoid;
        g_dispatch.tanh = avx2_impl::tanh;
        g_dispatch.gelu = avx2_impl::gelu;
        g_dispatch.initialized = true;
        return;
    }
#endif

    // Fall back to scalar implementations
    g_dispatch.add = scalar_impl::add;
    g_dispatch.sub = scalar_impl::sub;
    g_dispatch.mul = scalar_impl::mul;
    g_dispatch.div = scalar_impl::div;
    g_dispatch.sqrt = scalar_impl::sqrt;
    g_dispatch.exp = scalar_impl::exp;
    g_dispatch.log = scalar_impl::log;
    g_dispatch.fma = scalar_impl::fma;
    g_dispatch.relu = scalar_impl::relu;
    g_dispatch.sigmoid = scalar_impl::sigmoid;
    g_dispatch.tanh = scalar_impl::tanh;
    g_dispatch.gelu = scalar_impl::gelu;
    g_dispatch.initialized = true;
}

// ============================================================================
// Static initialization (optional auto-init)
// ============================================================================

namespace {

// Auto-initialize on library load
struct AutoInit {
    AutoInit() {
        init_dispatch();
    }
};

static AutoInit auto_init;

} // anonymous namespace

} // namespace dispatch
} // namespace cpu
} // namespace tenzor
