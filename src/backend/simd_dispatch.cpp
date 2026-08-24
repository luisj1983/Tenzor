/**
 * @file simd_dispatch.cpp
 * @brief Implementation of runtime SIMD dispatch system
 */

#include "tenzor/backend/simd_dispatch.hpp"
#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>

// Platform-specific includes for SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
    #define TENZOR_X86
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define TENZOR_ARM
    #if defined(__linux__)
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
    #endif
#endif

namespace tenzor {
namespace backend {

// ============================================================================
// CPU Feature Detection
// ============================================================================

namespace {

struct CPUFeatures {
    bool avx512 = false;
    bool avx2 = false;
    bool sse42 = false;
    bool neon = false;
    bool detected = false;
};

std::atomic<CPUFeatures*> g_cpu_features{nullptr};
std::mutex g_cpu_features_mutex;

#ifdef TENZOR_X86
// Read XCR0 (Extended Control Register 0) via XGETBV. XCR0 indicates which
// extended CPU state components the OS has enabled for save/restore across
// context switches. Executing AVX2 (YMM) / AVX-512 (ZMM, opmask) instructions
// when the OS has NOT enabled the corresponding state raises #UD, even if the
// CPUID feature bit is set, so this must be checked before selecting those
// kernels. Mirrors runtime_simd.cpp.
inline uint64_t simd_xgetbv(uint32_t xcr_index) {
#if defined(_MSC_VER)
    return _xgetbv(xcr_index);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr_index));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#else
    (void)xcr_index;
    return 0;
#endif
}
#endif // TENZOR_X86

void detect_cpu_features() {
    std::lock_guard<std::mutex> lock(g_cpu_features_mutex);

    if (g_cpu_features.load()) {
        return; // Already detected
    }

    CPUFeatures* features = new CPUFeatures();

#ifdef TENZOR_X86
    // Use CPUID to detect x86/x64 features
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

#if defined(_MSC_VER)
    // MSVC has no <cpuid.h>; <intrin.h>'s __cpuid/__cpuidex write into an
    // int[4] (eax,ebx,ecx,edx) instead of taking eax/ebx/ecx/edx by reference.
    int cpu_info[4] = {0, 0, 0, 0};
    __cpuid(cpu_info, 0);
    unsigned int max_leaf = static_cast<unsigned int>(cpu_info[0]);

    if (max_leaf >= 1) {
        __cpuid(cpu_info, 1);
        eax = static_cast<unsigned int>(cpu_info[0]);
        ebx = static_cast<unsigned int>(cpu_info[1]);
        ecx = static_cast<unsigned int>(cpu_info[2]);
        edx = static_cast<unsigned int>(cpu_info[3]);
        (void)eax; (void)edx;
#else
    // Check if CPUID is supported
    if (__get_cpuid_max(0, nullptr) >= 1) {
        // Get feature flags
        __cpuid(1, eax, ebx, ecx, edx);
#endif

        // SSE4.2: ECX bit 20
        features->sse42 = (ecx & (1 << 20)) != 0;

        // OS extended-state support: OSXSAVE (ECX bit 27) means XGETBV is
        // usable and the OS manages extended state. Without it, AVX/AVX-512
        // kernels would #UD regardless of the CPUID feature bits below.
        const bool osxsave = (ecx & (1 << 27)) != 0;
        bool os_avx = false;     // OS saves YMM (AVX) state
        bool os_avx512 = false;  // OS saves ZMM/opmask (AVX-512) state
        if (osxsave) {
            const uint64_t xcr0 = simd_xgetbv(0);
            // Bits 1 (SSE/XMM) and 2 (AVX/YMM) must both be set for AVX2.
            os_avx = ((xcr0 & 0x6) == 0x6);
            // Bits 5,6,7 (opmask, ZMM upper, ZMM16-31) for AVX-512.
            os_avx512 = os_avx && ((xcr0 & 0xE0) == 0xE0);
        }

        // Check extended features (leaf 7)
#if defined(_MSC_VER)
        if (max_leaf >= 7) {
            __cpuidex(cpu_info, 7, 0);
            ebx = static_cast<unsigned int>(cpu_info[1]);
#else
        if (__get_cpuid_max(0, nullptr) >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

            // AVX2: EBX bit 5 — gated on OS saving YMM state.
            features->avx2 = ((ebx & (1 << 5)) != 0) && os_avx;

            // AVX-512F: EBX bit 16 — gated on OS saving ZMM/opmask state.
            features->avx512 = ((ebx & (1 << 16)) != 0) && os_avx512;
        }
    }
#endif

#ifdef TENZOR_ARM
    // ARM NEON detection
    #if defined(__aarch64__)
        // NEON is mandatory in ARMv8-A (AArch64)
        features->neon = true;
    #elif defined(__ARM_NEON)
        // Compile-time NEON support
        features->neon = true;
    #elif defined(__linux__)
        // Runtime detection on Linux
        unsigned long hwcaps = getauxval(AT_HWCAP);
        features->neon = (hwcaps & HWCAP_NEON) != 0;
    #endif
#endif

    features->detected = true;
    g_cpu_features.store(features);
}

const CPUFeatures& get_cpu_features_impl() {
    if (!g_cpu_features.load()) {
        detect_cpu_features();
    }
    return *g_cpu_features.load();
}

} // anonymous namespace

bool cpu_supports_avx512() {
    return get_cpu_features_impl().avx512;
}

bool cpu_supports_avx2() {
    return get_cpu_features_impl().avx2;
}

bool cpu_supports_sse42() {
    return get_cpu_features_impl().sse42;
}

bool cpu_supports_neon() {
    return get_cpu_features_impl().neon;
}

const char* get_cpu_features() {
    // Build the feature string exactly once into a function-local static
    // std::string (thread-safe initialization under C++11 magic statics). The
    // detected feature set is deterministic, so the value never changes after
    // the first call; returning its c_str() avoids the previous shared mutable
    // `static char buffer[256]` that concurrent callers raced on (torn string
    // / dangling-on-overwrite return). The string lives for the program's
    // lifetime, so the returned pointer stays valid.
    static const std::string features_str = [] {
        const auto& features = get_cpu_features_impl();
        std::string result;
        auto append = [&](const char* name) {
            if (!result.empty()) result += ", ";
            result += name;
        };

        if (features.avx512) append("AVX-512");
        if (features.avx2) append("AVX2");
        if (features.sse42) append("SSE4.2");
        if (features.neon) append("NEON");

        if (result.empty()) {
            result = "Scalar (no SIMD)";
        }
        return result;
    }();

    return features_str.c_str();
}

// ============================================================================
// Scalar Kernel Implementations (Fallback)
// ============================================================================

namespace kernels {

void add_scalar(float* dst, const float* src1, const float* src2, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

void mul_scalar(float* dst, const float* src1, const float* src2, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

void matmul_scalar(float* dst, const float* src1, const float* src2, size_t size) {
    // G.5: actual square-matrix matmul (NxN where N = sqrt(size)).
    // The previous implementation silently did element-wise multiplication
    // instead of matmul, which produced wrong results for any caller using
    // `get_optimal_matmul_kernel()`. Production code paths use MKL/BLAS
    // through the CPU backend's separate matmul kernels; this dispatcher
    // is exercised by tests + small in-kernel reuses, so it must compute
    // the real thing.
    //
    // Convention: src1 is row-major (M x K), src2 is row-major (K x N),
    // dst is row-major (M x N). Caller guarantees size == N*N (square).
    const size_t N = static_cast<size_t>(
        std::lround(std::sqrt(static_cast<double>(size))));
    if (N * N != size) {
        // Non-square — fall back to element-wise so we don't read OOB.
        // Caller error; real matmul must hit cblas_sgemm.
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src1[i] * src2[i];
        }
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < N; ++k) {
                acc += src1[i * N + k] * src2[k * N + j];
            }
            dst[i * N + j] = acc;
        }
    }
}

void relu_scalar(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused for unary operation
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src1[i] > 0.0f ? src1[i] : 0.0f;
    }
}

void sigmoid_scalar(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused for unary operation
    for (size_t i = 0; i < size; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src1[i]));
    }
}

void tanh_scalar(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused for unary operation
    for (size_t i = 0; i < size; ++i) {
        dst[i] = std::tanh(src1[i]);
    }
}

float reduce_sum_scalar(const float* src, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += src[i];
    }
    return sum;
}

float reduce_max_scalar(const float* src, size_t size) {
    if (size == 0) return 0.0f;
    float max_val = src[0];
    for (size_t i = 1; i < size; ++i) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

// ============================================================================
// x86/x64 SIMD Implementations
// ============================================================================

// MSVC has no equivalent of GCC/Clang's __attribute__((target(...)))
// function-multiversioning, and compiles a whole translation unit against a
// single /arch: flag rather than per-function ISA targets — so these
// SSE4.2/AVX2/AVX-512 variants (which rely on that attribute to coexist in
// one TU) aren't compiled under MSVC. initialize_kernel_table() below always
// selects the scalar kernels in that case; CPU feature detection above still
// works and is used by the public cpu_supports_*() queries.
#if defined(TENZOR_X86) && !defined(_MSC_VER)

// SSE4.2 Implementations (4 floats per operation)
__attribute__((target("sse4.2")))
void add_sse42(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 4);

    for (; i < simd_end; i += 4) {
        __m128 a = _mm_loadu_ps(src1 + i);
        __m128 b = _mm_loadu_ps(src2 + i);
        __m128 result = _mm_add_ps(a, b);
        _mm_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

__attribute__((target("sse4.2")))
void mul_sse42(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 4);

    for (; i < simd_end; i += 4) {
        __m128 a = _mm_loadu_ps(src1 + i);
        __m128 b = _mm_loadu_ps(src2 + i);
        __m128 result = _mm_mul_ps(a, b);
        _mm_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

__attribute__((target("sse4.2")))
void matmul_sse42(float* dst, const float* src1, const float* src2, size_t size) {
    // G.5: delegate to the corrected scalar matmul. The SIMD speedup for
    // square matmul on modest sizes is best achieved through the CPU
    // backend's MKL-backed kernels (cblas_sgemm); the dispatcher's role
    // here is correctness, not peak FLOPS.
    matmul_scalar(dst, src1, src2, size);
}

__attribute__((target("sse4.2")))
void relu_sse42(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    size_t i = 0;
    const size_t simd_end = size - (size % 4);
    const __m128 zero = _mm_setzero_ps();

    for (; i < simd_end; i += 4) {
        __m128 a = _mm_loadu_ps(src1 + i);
        __m128 result = _mm_max_ps(a, zero); // max(a, 0)
        _mm_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] > 0.0f ? src1[i] : 0.0f;
    }
}

__attribute__((target("sse4.2")))
void sigmoid_sse42(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // SSE doesn't have native exp, fall back to scalar for simplicity
    // A production implementation would use vectorized approximations
    sigmoid_scalar(dst, src1, src2, size);
}

__attribute__((target("sse4.2")))
void tanh_sse42(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // SSE doesn't have native tanh, fall back to scalar
    // A production implementation would use vectorized approximations
    tanh_scalar(dst, src1, src2, size);
}

__attribute__((target("sse4.2")))
float reduce_sum_sse42(const float* src, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 4);
    __m128 sum_vec = _mm_setzero_ps();

    for (; i < simd_end; i += 4) {
        __m128 a = _mm_loadu_ps(src + i);
        sum_vec = _mm_add_ps(sum_vec, a);
    }

    // Horizontal sum: sum all 4 elements
    __m128 shuf = _mm_movehdup_ps(sum_vec);        // [1,1,3,3]
    __m128 sums = _mm_add_ps(sum_vec, shuf);       // [0+1, 1+1, 2+3, 3+3]
    shuf = _mm_movehl_ps(shuf, sums);              // [2+3, 3+3, ?, ?]
    sums = _mm_add_ss(sums, shuf);                 // [0+1+2+3, ...]
    float sum = _mm_cvtss_f32(sums);

    // Handle remainder
    for (; i < size; ++i) {
        sum += src[i];
    }
    return sum;
}

__attribute__((target("sse4.2")))
float reduce_max_sse42(const float* src, size_t size) {
    if (size == 0) return 0.0f;

    size_t i = 0;
    const size_t simd_end = size - (size % 4);
    __m128 max_vec = _mm_set1_ps(-INFINITY);

    for (; i < simd_end; i += 4) {
        __m128 a = _mm_loadu_ps(src + i);
        max_vec = _mm_max_ps(max_vec, a);
    }

    // Horizontal max: find max of 4 elements
    __m128 shuf = _mm_movehdup_ps(max_vec);
    __m128 maxs = _mm_max_ps(max_vec, shuf);
    shuf = _mm_movehl_ps(shuf, maxs);
    maxs = _mm_max_ss(maxs, shuf);
    float max_val = _mm_cvtss_f32(maxs);

    // Handle remainder
    for (; i < size; ++i) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

// AVX2 Implementations (8 floats per operation)
__attribute__((target("avx2")))
void add_avx2(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 8);

    for (; i < simd_end; i += 8) {
        __m256 a = _mm256_loadu_ps(src1 + i);
        __m256 b = _mm256_loadu_ps(src2 + i);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

__attribute__((target("avx2")))
void mul_avx2(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 8);

    for (; i < simd_end; i += 8) {
        __m256 a = _mm256_loadu_ps(src1 + i);
        __m256 b = _mm256_loadu_ps(src2 + i);
        __m256 result = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

__attribute__((target("avx2")))
void matmul_avx2(float* dst, const float* src1, const float* src2, size_t size) {
    // G.5: delegate to the corrected scalar matmul (see matmul_sse42).
    matmul_scalar(dst, src1, src2, size);
}

__attribute__((target("avx2")))
void relu_avx2(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    size_t i = 0;
    const size_t simd_end = size - (size % 8);
    const __m256 zero = _mm256_setzero_ps();

    for (; i < simd_end; i += 8) {
        __m256 a = _mm256_loadu_ps(src1 + i);
        __m256 result = _mm256_max_ps(a, zero);
        _mm256_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] > 0.0f ? src1[i] : 0.0f;
    }
}

__attribute__((target("avx2")))
void sigmoid_avx2(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // AVX2 doesn't have native exp, fall back to scalar
    // Production implementations use polynomial approximations
    sigmoid_scalar(dst, src1, src2, size);
}

__attribute__((target("avx2")))
void tanh_avx2(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // AVX2 doesn't have native tanh, fall back to scalar
    tanh_scalar(dst, src1, src2, size);
}

__attribute__((target("avx2")))
float reduce_sum_avx2(const float* src, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 8);
    __m256 sum_vec = _mm256_setzero_ps();

    for (; i < simd_end; i += 8) {
        __m256 a = _mm256_loadu_ps(src + i);
        sum_vec = _mm256_add_ps(sum_vec, a);
    }

    // Horizontal sum of 8 floats
    // sum_vec = [a0, a1, a2, a3, a4, a5, a6, a7]
    __m128 lo = _mm256_castps256_ps128(sum_vec);        // [a0, a1, a2, a3]
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);      // [a4, a5, a6, a7]
    __m128 sum128 = _mm_add_ps(lo, hi);                 // [a0+a4, a1+a5, a2+a6, a3+a7]

    // Now reduce 4 elements
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum = _mm_cvtss_f32(sums);

    // Handle remainder
    for (; i < size; ++i) {
        sum += src[i];
    }
    return sum;
}

__attribute__((target("avx2")))
float reduce_max_avx2(const float* src, size_t size) {
    if (size == 0) return 0.0f;

    size_t i = 0;
    const size_t simd_end = size - (size % 8);
    __m256 max_vec = _mm256_set1_ps(-INFINITY);

    for (; i < simd_end; i += 8) {
        __m256 a = _mm256_loadu_ps(src + i);
        max_vec = _mm256_max_ps(max_vec, a);
    }

    // Horizontal max of 8 floats
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 max128 = _mm_max_ps(lo, hi);

    __m128 shuf = _mm_movehdup_ps(max128);
    __m128 maxs = _mm_max_ps(max128, shuf);
    shuf = _mm_movehl_ps(shuf, maxs);
    maxs = _mm_max_ss(maxs, shuf);
    float max_val = _mm_cvtss_f32(maxs);

    // Handle remainder
    for (; i < size; ++i) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

// AVX-512 Implementations (16 floats per operation)
__attribute__((target("avx512f")))
void add_avx512(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 16);

    for (; i < simd_end; i += 16) {
        __m512 a = _mm512_loadu_ps(src1 + i);
        __m512 b = _mm512_loadu_ps(src2 + i);
        __m512 result = _mm512_add_ps(a, b);
        _mm512_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

__attribute__((target("avx512f")))
void mul_avx512(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 16);

    for (; i < simd_end; i += 16) {
        __m512 a = _mm512_loadu_ps(src1 + i);
        __m512 b = _mm512_loadu_ps(src2 + i);
        __m512 result = _mm512_mul_ps(a, b);
        _mm512_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

__attribute__((target("avx512f")))
void matmul_avx512(float* dst, const float* src1, const float* src2, size_t size) {
    // G.5: delegate to the corrected scalar matmul (see matmul_sse42).
    matmul_scalar(dst, src1, src2, size);
}

__attribute__((target("avx512f")))
void relu_avx512(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    size_t i = 0;
    const size_t simd_end = size - (size % 16);
    const __m512 zero = _mm512_setzero_ps();

    for (; i < simd_end; i += 16) {
        __m512 a = _mm512_loadu_ps(src1 + i);
        __m512 result = _mm512_max_ps(a, zero);
        _mm512_storeu_ps(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] > 0.0f ? src1[i] : 0.0f;
    }
}

__attribute__((target("avx512f")))
void sigmoid_avx512(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // AVX-512 doesn't have native exp, fall back to scalar
    // Production implementations use polynomial approximations or SVML
    sigmoid_scalar(dst, src1, src2, size);
}

__attribute__((target("avx512f")))
void tanh_avx512(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // AVX-512 doesn't have native tanh, fall back to scalar
    tanh_scalar(dst, src1, src2, size);
}

__attribute__((target("avx512f")))
float reduce_sum_avx512(const float* src, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 16);
    __m512 sum_vec = _mm512_setzero_ps();

    for (; i < simd_end; i += 16) {
        __m512 a = _mm512_loadu_ps(src + i);
        sum_vec = _mm512_add_ps(sum_vec, a);
    }

    // Horizontal sum using AVX-512 reduce instruction
    float sum = _mm512_reduce_add_ps(sum_vec);

    // Handle remainder
    for (; i < size; ++i) {
        sum += src[i];
    }
    return sum;
}

__attribute__((target("avx512f")))
float reduce_max_avx512(const float* src, size_t size) {
    if (size == 0) return 0.0f;

    size_t i = 0;
    const size_t simd_end = size - (size % 16);
    __m512 max_vec = _mm512_set1_ps(-INFINITY);

    for (; i < simd_end; i += 16) {
        __m512 a = _mm512_loadu_ps(src + i);
        max_vec = _mm512_max_ps(max_vec, a);
    }

    // Horizontal max using AVX-512 reduce instruction
    float max_val = _mm512_reduce_max_ps(max_vec);

    // Handle remainder
    for (; i < size; ++i) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

#endif // TENZOR_X86 && !_MSC_VER

// ============================================================================
// ARM NEON Implementations
// ============================================================================

#ifdef TENZOR_ARM

void add_neon(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 4);

    for (; i < simd_end; i += 4) {
        float32x4_t a = vld1q_f32(src1 + i);
        float32x4_t b = vld1q_f32(src2 + i);
        float32x4_t result = vaddq_f32(a, b);
        vst1q_f32(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

void mul_neon(float* dst, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 4);

    for (; i < simd_end; i += 4) {
        float32x4_t a = vld1q_f32(src1 + i);
        float32x4_t b = vld1q_f32(src2 + i);
        float32x4_t result = vmulq_f32(a, b);
        vst1q_f32(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

void matmul_neon(float* dst, const float* src1, const float* src2, size_t size) {
    // G.5: delegate to the corrected scalar matmul (see matmul_sse42).
    matmul_scalar(dst, src1, src2, size);
}

void relu_neon(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    size_t i = 0;
    const size_t simd_end = size - (size % 4);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    for (; i < simd_end; i += 4) {
        float32x4_t a = vld1q_f32(src1 + i);
        float32x4_t result = vmaxq_f32(a, zero);
        vst1q_f32(dst + i, result);
    }

    // Handle remainder
    for (; i < size; ++i) {
        dst[i] = src1[i] > 0.0f ? src1[i] : 0.0f;
    }
}

void sigmoid_neon(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // NEON doesn't have native exp, fall back to scalar
    // Production implementations use polynomial approximations
    sigmoid_scalar(dst, src1, src2, size);
}

void tanh_neon(float* dst, const float* src1, const float* src2, size_t size) {
    (void)src2; // Unused
    // NEON doesn't have native tanh, fall back to scalar
    tanh_scalar(dst, src1, src2, size);
}

float reduce_sum_neon(const float* src, size_t size) {
    size_t i = 0;
    const size_t simd_end = size - (size % 4);
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    for (; i < simd_end; i += 4) {
        float32x4_t a = vld1q_f32(src + i);
        sum_vec = vaddq_f32(sum_vec, a);
    }

    // Horizontal sum of 4 floats
    float32x2_t sum_lo = vget_low_f32(sum_vec);
    float32x2_t sum_hi = vget_high_f32(sum_vec);
    float32x2_t sum_pair = vadd_f32(sum_lo, sum_hi);
    float sum = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Handle remainder
    for (; i < size; ++i) {
        sum += src[i];
    }
    return sum;
}

float reduce_max_neon(const float* src, size_t size) {
    if (size == 0) return 0.0f;

    size_t i = 0;
    const size_t simd_end = size - (size % 4);
    float32x4_t max_vec = vdupq_n_f32(-INFINITY);

    for (; i < simd_end; i += 4) {
        float32x4_t a = vld1q_f32(src + i);
        max_vec = vmaxq_f32(max_vec, a);
    }

    // Horizontal max of 4 floats
    float32x2_t max_lo = vget_low_f32(max_vec);
    float32x2_t max_hi = vget_high_f32(max_vec);
    float32x2_t max_pair = vmax_f32(max_lo, max_hi);
    float max_val = vget_lane_f32(vpmax_f32(max_pair, max_pair), 0);

    // Handle remainder
    for (; i < size; ++i) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

#endif // TENZOR_ARM

} // namespace kernels

// ============================================================================
// Kernel Selection
// ============================================================================

namespace {

struct KernelTable {
    KernelFunc add = nullptr;
    KernelFunc mul = nullptr;
    KernelFunc matmul = nullptr;
    KernelFunc relu = nullptr;
    KernelFunc sigmoid = nullptr;
    KernelFunc tanh = nullptr;
    ReductionFunc reduce_sum = nullptr;
    ReductionFunc reduce_max = nullptr;
    bool initialized = false;
};

std::atomic<KernelTable*> g_kernel_table{nullptr};
std::mutex g_kernel_table_mutex;

void initialize_kernel_table() {
    std::lock_guard<std::mutex> lock(g_kernel_table_mutex);

    if (g_kernel_table.load()) {
        return; // Already initialized
    }

    KernelTable* table = new KernelTable();
    const auto& features = get_cpu_features_impl();

    // Select best available implementation for each operation

#if defined(TENZOR_X86) && !defined(_MSC_VER)
    if (features.avx512) {
        table->add = kernels::add_avx512;
        table->mul = kernels::mul_avx512;
        table->matmul = kernels::matmul_avx512;
        table->relu = kernels::relu_avx512;
        table->sigmoid = kernels::sigmoid_avx512;
        table->tanh = kernels::tanh_avx512;
        table->reduce_sum = kernels::reduce_sum_avx512;
        table->reduce_max = kernels::reduce_max_avx512;
    } else if (features.avx2) {
        table->add = kernels::add_avx2;
        table->mul = kernels::mul_avx2;
        table->matmul = kernels::matmul_avx2;
        table->relu = kernels::relu_avx2;
        table->sigmoid = kernels::sigmoid_avx2;
        table->tanh = kernels::tanh_avx2;
        table->reduce_sum = kernels::reduce_sum_avx2;
        table->reduce_max = kernels::reduce_max_avx2;
    } else if (features.sse42) {
        table->add = kernels::add_sse42;
        table->mul = kernels::mul_sse42;
        table->matmul = kernels::matmul_sse42;
        table->relu = kernels::relu_sse42;
        table->sigmoid = kernels::sigmoid_sse42;
        table->tanh = kernels::tanh_sse42;
        table->reduce_sum = kernels::reduce_sum_sse42;
        table->reduce_max = kernels::reduce_max_sse42;
    } else {
        table->add = kernels::add_scalar;
        table->mul = kernels::mul_scalar;
        table->matmul = kernels::matmul_scalar;
        table->relu = kernels::relu_scalar;
        table->sigmoid = kernels::sigmoid_scalar;
        table->tanh = kernels::tanh_scalar;
        table->reduce_sum = kernels::reduce_sum_scalar;
        table->reduce_max = kernels::reduce_max_scalar;
    }
#elif defined(TENZOR_ARM)
    if (features.neon) {
        table->add = kernels::add_neon;
        table->mul = kernels::mul_neon;
        table->matmul = kernels::matmul_neon;
        table->relu = kernels::relu_neon;
        table->sigmoid = kernels::sigmoid_neon;
        table->tanh = kernels::tanh_neon;
        table->reduce_sum = kernels::reduce_sum_neon;
        table->reduce_max = kernels::reduce_max_neon;
    } else {
        table->add = kernels::add_scalar;
        table->mul = kernels::mul_scalar;
        table->matmul = kernels::matmul_scalar;
        table->relu = kernels::relu_scalar;
        table->sigmoid = kernels::sigmoid_scalar;
        table->tanh = kernels::tanh_scalar;
        table->reduce_sum = kernels::reduce_sum_scalar;
        table->reduce_max = kernels::reduce_max_scalar;
    }
#else
    // Fallback for unknown architectures
    table->add = kernels::add_scalar;
    table->mul = kernels::mul_scalar;
    table->matmul = kernels::matmul_scalar;
    table->relu = kernels::relu_scalar;
    table->sigmoid = kernels::sigmoid_scalar;
    table->tanh = kernels::tanh_scalar;
    table->reduce_sum = kernels::reduce_sum_scalar;
    table->reduce_max = kernels::reduce_max_scalar;
#endif

    table->initialized = true;
    g_kernel_table.store(table);
}

const KernelTable& get_kernel_table() {
    if (!g_kernel_table.load()) {
        initialize_kernel_table();
    }
    return *g_kernel_table.load();
}

} // anonymous namespace

void initialize_simd_dispatch() {
    // Trigger detection and initialization
    get_kernel_table();
}

KernelFunc get_optimal_add_kernel() {
    return get_kernel_table().add;
}

KernelFunc get_optimal_mul_kernel() {
    return get_kernel_table().mul;
}

KernelFunc get_optimal_matmul_kernel() {
    return get_kernel_table().matmul;
}

KernelFunc get_optimal_relu_kernel() {
    return get_kernel_table().relu;
}

KernelFunc get_optimal_sigmoid_kernel() {
    return get_kernel_table().sigmoid;
}

KernelFunc get_optimal_tanh_kernel() {
    return get_kernel_table().tanh;
}

ReductionFunc get_optimal_reduce_sum_kernel() {
    return get_kernel_table().reduce_sum;
}

ReductionFunc get_optimal_reduce_max_kernel() {
    return get_kernel_table().reduce_max;
}

} // namespace backend
} // namespace tenzor
