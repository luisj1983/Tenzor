// =============================================================================
// tests/backends/test_neon_kernels.cpp — ARM NEON kernel correctness tests
//
// Guarded by __ARM_NEON so the binary compiles (with a trivial test) on x86.
// =============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

// Trivial test so the binary is never empty on non-ARM platforms
TEST(NeonKernels, PlatformPlaceholder) {
#ifdef __ARM_NEON
    SUCCEED() << "ARM NEON available";
#else
    GTEST_SKIP() << "ARM NEON not available on this platform";
#endif
}

#ifdef __ARM_NEON

#include "src/backends/cpu/kernels/simd_arm.hpp"
#include "src/backends/cpu/kernels/simd_arm_quantized.hpp"

using namespace tenzor::cpu::neon;

// ---------------------------------------------------------------------------
// sigmoid_f32
// ---------------------------------------------------------------------------
TEST(NeonKernels, SigmoidF32) {
    const int64_t n = 131; // not a multiple of 4
    std::vector<float> src(n), dst(n);
    for (int64_t i = 0; i < n; ++i)
        src[i] = -6.0f + 12.0f * static_cast<float>(i) / static_cast<float>(n - 1);

    sigmoid_f32(src.data(), dst.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-src[i]));
        EXPECT_NEAR(dst[i], expected, 1e-5f)
            << "mismatch at index " << i << " (x=" << src[i] << ")";
    }
}

// ---------------------------------------------------------------------------
// tanh_f32
// ---------------------------------------------------------------------------
TEST(NeonKernels, TanhF32) {
    const int64_t n = 131;
    std::vector<float> src(n), dst(n);
    for (int64_t i = 0; i < n; ++i)
        src[i] = -4.0f + 8.0f * static_cast<float>(i) / static_cast<float>(n - 1);

    tanh_f32(src.data(), dst.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        float expected = std::tanh(src[i]);
        EXPECT_NEAR(dst[i], expected, 1e-5f)
            << "mismatch at index " << i << " (x=" << src[i] << ")";
    }
}

// ---------------------------------------------------------------------------
// gelu_f32
// ---------------------------------------------------------------------------
TEST(NeonKernels, GeluF32) {
    const int64_t n = 131;
    std::vector<float> src(n), dst(n);
    for (int64_t i = 0; i < n; ++i)
        src[i] = -4.0f + 8.0f * static_cast<float>(i) / static_cast<float>(n - 1);

    gelu_f32(src.data(), dst.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        float x = src[i];
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        float expected = 0.5f * x * (1.0f + std::tanh(inner));
        EXPECT_NEAR(dst[i], expected, 1e-5f)
            << "mismatch at index " << i << " (x=" << x << ")";
    }
}

// ---------------------------------------------------------------------------
// gemv_f32
// ---------------------------------------------------------------------------
TEST(NeonKernels, GemvF32) {
    const int64_t M = 17, N = 33; // not multiples of 4
    std::vector<float> A(M * N), x(N), y(M), y_ref(M, 0.0f);

    // Fill with simple deterministic data
    for (int64_t i = 0; i < M * N; ++i) A[i] = 0.01f * static_cast<float>(i % 97 - 48);
    for (int64_t i = 0; i < N; ++i)     x[i] = 0.1f * static_cast<float>(i % 13 - 6);

    // Reference
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < N; ++k) {
            y_ref[m] += A[m * N + k] * x[k];
        }
    }

    gemv_f32(A.data(), x.data(), y.data(), M, N);

    for (int64_t m = 0; m < M; ++m) {
        EXPECT_NEAR(y[m], y_ref[m], 1e-5f)
            << "mismatch at row " << m;
    }
}

// ---------------------------------------------------------------------------
// gemm_4x4_f32
// ---------------------------------------------------------------------------
TEST(NeonKernels, Gemm4x4F32) {
    const int64_t M = 9, N = 11, K = 7; // not multiples of 4
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), C_ref(M * N, 0.0f);

    for (int64_t i = 0; i < M * K; ++i) A[i] = 0.01f * static_cast<float>(i % 53 - 26);
    for (int64_t i = 0; i < K * N; ++i) B[i] = 0.01f * static_cast<float>(i % 41 - 20);

    // Reference: C_ref += A * B
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k)
                C_ref[m * N + n] += A[m * K + k] * B[k * N + n];

    gemm_4x4_f32(A.data(), B.data(), C.data(), M, N, K);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f)
            << "mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// qdot_s8
// ---------------------------------------------------------------------------
TEST(NeonKernels, QdotS8) {
    const int64_t n = 67; // not a multiple of 16
    std::vector<int8_t> a(n), b(n);
    for (int64_t i = 0; i < n; ++i) {
        a[i] = static_cast<int8_t>((i * 7 + 3) % 251 - 125);
        b[i] = static_cast<int8_t>((i * 13 + 5) % 251 - 125);
    }

    // Reference
    int32_t ref = 0;
    for (int64_t i = 0; i < n; ++i) {
        ref += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    int32_t result = qdot_s8(a.data(), b.data(), n);
    EXPECT_EQ(result, ref);
}

// ---------------------------------------------------------------------------
// qgemv_s8
// ---------------------------------------------------------------------------
TEST(NeonKernels, QgemvS8) {
    const int64_t M = 5, K = 35;
    std::vector<int8_t> A(M * K), x(K);
    std::vector<int32_t> y(M), y_ref(M, 0);

    for (int64_t i = 0; i < M * K; ++i)
        A[i] = static_cast<int8_t>((i * 11 + 7) % 251 - 125);
    for (int64_t i = 0; i < K; ++i)
        x[i] = static_cast<int8_t>((i * 3 + 1) % 251 - 125);

    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k)
            y_ref[m] += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(x[k]);

    qgemv_s8(A.data(), x.data(), y.data(), M, K);

    for (int64_t m = 0; m < M; ++m) {
        EXPECT_EQ(y[m], y_ref[m]) << "mismatch at row " << m;
    }
}

#endif // __ARM_NEON
