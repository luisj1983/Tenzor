/**
 * @file simd_dispatch.cpp
 * @brief Implementation of function pointer dispatch system
 *
 * Populates g_dispatch at startup based on runtime CPU feature detection.
 * Respects TENZOR_FORCE_SIMD_LEVEL env var for testing/override:
 *   "avx512"  — force AVX-512 (only if hardware actually supports it)
 *   "avx2"    — force AVX2
 *   "sse2"    — force SSE2 (scalar-width paths; uses scalar fallback here)
 *   "scalar"  — force scalar fallback
 */

#include "simd_dispatch.hpp"
#include "simd_fast_math.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace tenzor {
namespace cpu {
namespace dispatch {

// Global dispatch table — all nullptrs until init_dispatch() runs.
SIMDDispatch g_dispatch = {
    nullptr, nullptr, nullptr, nullptr,  // add, sub, mul, div (f32)
    nullptr, nullptr, nullptr, nullptr,  // sqrt, exp, log, fma (f32)
    nullptr, nullptr,                    // neg, abs_f32
    nullptr, nullptr, nullptr, nullptr,  // relu, sigmoid, tanh, gelu
    nullptr, nullptr, nullptr, nullptr,  // add_f64, sub_f64, mul_f64, div_f64
    nullptr, nullptr, nullptr,           // sqrt_f64, neg_f64, abs_f64
    false,                               // initialized
    "none"                               // simd_level
};

// Initialization mutex for thread safety
static std::mutex init_mutex;

// ============================================================================
// Scalar implementations — float32
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

void neg(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = -a[i];
}

void abs_f32(const float* a, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::abs(a[i]);
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

// --- float64 ---

void add_f64(const double* a, const double* b, double* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
}

void sub_f64(const double* a, const double* b, double* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] - b[i];
}

void mul_f64(const double* a, const double* b, double* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
}

void div_f64(const double* a, const double* b, double* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = a[i] / b[i];
}

void sqrt_f64(const double* a, double* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::sqrt(a[i]);
}

void neg_f64(const double* a, double* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = -a[i];
}

void abs_f64(const double* a, double* out, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = std::abs(a[i]);
}

} // namespace scalar_impl

// ============================================================================
// AVX2 wrappers — compiled with -mavx2 per-file in portable mode
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

void neg(const float* a, float* out, size_t size) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(zero, va));
    }
    for (; i < size; ++i) out[i] = -a[i];
}

void abs_f32(const float* a, float* out, size_t size) {
    size_t i = 0;
    // Clear sign bit via AND with 0x7FFFFFFF mask
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_and_ps(va, sign_mask));
    }
    for (; i < size; ++i) out[i] = std::abs(a[i]);
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

// --- float64 ---

void add_f64(const double* a, const double* b, double* out, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] + b[i];
}

void sub_f64(const double* a, const double* b, double* out, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] - b[i];
}

void mul_f64(const double* a, const double* b, double* out, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] * b[i];
}

void div_f64(const double* a, const double* b, double* out, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_div_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] / b[i];
}

void sqrt_f64(const double* a, double* out, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_sqrt_pd(va));
    }
    for (; i < size; ++i) out[i] = std::sqrt(a[i]);
}

void neg_f64(const double* a, double* out, size_t size) {
    size_t i = 0;
    __m256d zero = _mm256_setzero_pd();
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(zero, va));
    }
    for (; i < size; ++i) out[i] = -a[i];
}

void abs_f64(const double* a, double* out, size_t size) {
    size_t i = 0;
    // Clear sign bit: AND with 0x7FFFFFFFFFFFFFFF mask (all bits except sign)
    __m256d sign_mask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(static_cast<int64_t>(0x7FFFFFFFFFFFFFFFll)));
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_and_pd(va, sign_mask));
    }
    for (; i < size; ++i) out[i] = std::abs(a[i]);
}

} // namespace avx2_impl

#endif // __AVX2__

} // namespace dispatch
} // namespace cpu
} // namespace tenzor

// ============================================================================
// AVX-512 F64 forward declarations — defined in simd_kernels_avx512_f64.cpp
// which is compiled with -mavx512f per-file flags.
// These live in tenzor::cpu::avx512_f64_impl (no dispatch sub-namespace).
// ============================================================================

namespace tenzor {
namespace cpu {
namespace avx512_f64_impl {
void add_f64(const double*, const double*, double*, size_t);
void sub_f64(const double*, const double*, double*, size_t);
void mul_f64(const double*, const double*, double*, size_t);
void div_f64(const double*, const double*, double*, size_t);
void sqrt_f64(const double*, double*, size_t);
void neg_f64(const double*, double*, size_t);
void abs_f64(const double*, double*, size_t);
void neg_f32(const float*, float*, size_t);
void abs_f32(const float*, float*, size_t);
} // namespace avx512_f64_impl
} // namespace cpu
} // namespace tenzor

namespace tenzor {
namespace cpu {
namespace dispatch {

// ============================================================================
// Internal: populate g_dispatch for a given ISA level
// Called with init_mutex held.
// ============================================================================

static void populate_avx512() {
    g_dispatch.add   = avx512::add;
    g_dispatch.sub   = avx512::sub;
    g_dispatch.mul   = avx512::mul;
    g_dispatch.div   = avx512::div;
    g_dispatch.sqrt  = avx512::sqrt;
    g_dispatch.exp   = avx512::exp;
    g_dispatch.log   = avx512::log;
    g_dispatch.fma   = avx512::fma;
    g_dispatch.neg   = tenzor::cpu::avx512_f64_impl::neg_f32;
    g_dispatch.abs_f32 = tenzor::cpu::avx512_f64_impl::abs_f32;
    g_dispatch.relu  = avx512::relu;
    g_dispatch.sigmoid = avx512::sigmoid;
    g_dispatch.tanh  = avx512::tanh;
    g_dispatch.gelu  = avx512::gelu;
    g_dispatch.add_f64  = tenzor::cpu::avx512_f64_impl::add_f64;
    g_dispatch.sub_f64  = tenzor::cpu::avx512_f64_impl::sub_f64;
    g_dispatch.mul_f64  = tenzor::cpu::avx512_f64_impl::mul_f64;
    g_dispatch.div_f64  = tenzor::cpu::avx512_f64_impl::div_f64;
    g_dispatch.sqrt_f64 = tenzor::cpu::avx512_f64_impl::sqrt_f64;
    g_dispatch.neg_f64  = tenzor::cpu::avx512_f64_impl::neg_f64;
    g_dispatch.abs_f64  = tenzor::cpu::avx512_f64_impl::abs_f64;
    g_dispatch.simd_level = "avx512";
}

#if defined(__AVX2__)
static void populate_avx2() {
    g_dispatch.add   = avx2_impl::add;
    g_dispatch.sub   = avx2_impl::sub;
    g_dispatch.mul   = avx2_impl::mul;
    g_dispatch.div   = avx2_impl::div;
    g_dispatch.sqrt  = avx2_impl::sqrt;
    g_dispatch.exp   = avx2_impl::exp;
    g_dispatch.log   = avx2_impl::log;
    g_dispatch.fma   = avx2_impl::fma;
    g_dispatch.neg   = avx2_impl::neg;
    g_dispatch.abs_f32 = avx2_impl::abs_f32;
    g_dispatch.relu  = avx2_impl::relu;
    g_dispatch.sigmoid = avx2_impl::sigmoid;
    g_dispatch.tanh  = avx2_impl::tanh;
    g_dispatch.gelu  = avx2_impl::gelu;
    g_dispatch.add_f64  = avx2_impl::add_f64;
    g_dispatch.sub_f64  = avx2_impl::sub_f64;
    g_dispatch.mul_f64  = avx2_impl::mul_f64;
    g_dispatch.div_f64  = avx2_impl::div_f64;
    g_dispatch.sqrt_f64 = avx2_impl::sqrt_f64;
    g_dispatch.neg_f64  = avx2_impl::neg_f64;
    g_dispatch.abs_f64  = avx2_impl::abs_f64;
    g_dispatch.simd_level = "avx2";
}
#endif

static void populate_scalar() {
    g_dispatch.add   = scalar_impl::add;
    g_dispatch.sub   = scalar_impl::sub;
    g_dispatch.mul   = scalar_impl::mul;
    g_dispatch.div   = scalar_impl::div;
    g_dispatch.sqrt  = scalar_impl::sqrt;
    g_dispatch.exp   = scalar_impl::exp;
    g_dispatch.log   = scalar_impl::log;
    g_dispatch.fma   = scalar_impl::fma;
    g_dispatch.neg   = scalar_impl::neg;
    g_dispatch.abs_f32 = scalar_impl::abs_f32;
    g_dispatch.relu  = scalar_impl::relu;
    g_dispatch.sigmoid = scalar_impl::sigmoid;
    g_dispatch.tanh  = scalar_impl::tanh;
    g_dispatch.gelu  = scalar_impl::gelu;
    g_dispatch.add_f64  = scalar_impl::add_f64;
    g_dispatch.sub_f64  = scalar_impl::sub_f64;
    g_dispatch.mul_f64  = scalar_impl::mul_f64;
    g_dispatch.div_f64  = scalar_impl::div_f64;
    g_dispatch.sqrt_f64 = scalar_impl::sqrt_f64;
    g_dispatch.neg_f64  = scalar_impl::neg_f64;
    g_dispatch.abs_f64  = scalar_impl::abs_f64;
    g_dispatch.simd_level = "scalar";
}

// ============================================================================
// Internal: determine requested ISA level, respecting env override
// Returns one of: "avx512", "avx2", "sse2", "scalar"
// ============================================================================

static const char* resolve_simd_level() {
    const char* env = std::getenv("TENZOR_FORCE_SIMD_LEVEL");
    if (env) {
        if (std::strcmp(env, "scalar") == 0) return "scalar";
        if (std::strcmp(env, "sse2")   == 0) return "sse2";   // treated as scalar here
        if (std::strcmp(env, "avx2")   == 0) return "avx2";
        if (std::strcmp(env, "avx512") == 0) return "avx512";
    }
    // Auto-detect
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) return "avx512";
    if (cpu.has_avx2())   return "avx2";
    return "scalar";
}

// ============================================================================
// Internal: do the actual init (must be called with init_mutex held)
// ============================================================================

static void do_init() {
    const char* level = resolve_simd_level();

    if (std::strcmp(level, "avx512") == 0 && CPUInfo::get().has_avx512()) {
        populate_avx512();
        return;
    }

#if defined(__AVX2__)
    if ((std::strcmp(level, "avx2") == 0 || std::strcmp(level, "avx512") == 0)
            && CPUInfo::get().has_avx2()) {
        populate_avx2();
        return;
    }
#endif

    // "sse2", "scalar", or no SIMD available
    populate_scalar();
}

// ============================================================================
// Public API
// ============================================================================

void init_dispatch() {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (g_dispatch.initialized) {
        return;
    }
    do_init();
    g_dispatch.initialized = true;
}

void reinit_dispatch() {
    std::lock_guard<std::mutex> lock(init_mutex);
    // Force re-initialisation even if already done (for testing).
    g_dispatch.initialized = false;
    do_init();
    g_dispatch.initialized = true;
}

std::string get_simd_level() {
    if (!g_dispatch.initialized) {
        init_dispatch();
    }
    return std::string(g_dispatch.simd_level);
}

// ============================================================================
// Static initialization (auto-init on library load)
// ============================================================================

namespace {

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
