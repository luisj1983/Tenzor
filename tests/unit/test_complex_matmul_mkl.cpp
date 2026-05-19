/**
 * @file test_complex_matmul_mkl.cpp
 * @brief Audit P0 #5: verify complex matmul uses cblas_cgemm / cblas_zgemm
 *        and does NOT fall through to the scalar triple-loop.
 *
 * math.cpp guarded both cblas_cgemm (Complex64) and cblas_zgemm (Complex128)
 * behind #ifdef TENZOR_HAS_MKL, but CMake defines TENZOR_USE_MKL.  The
 * mismatch meant every complex matmul silently used the O(N^3) scalar path —
 * ~100x slower than the MKL BLAS path for large matrices.
 *
 * Fix: rename both guards to #ifdef TENZOR_USE_MKL.
 *
 * Tests:
 *   Correctness1x1_{C64,C128} — (1+2i)(3+4i) = -5+10i
 *   Complex64Correctness2x2Identity — K-loop exercise with identity matmul
 *   ScalarFallbackNotUsed_PerfSmoke_{C64,C128} — 256×256 must finish in <100 ms
 *     (scalar fallback takes 300–800 ms on a modern CPU; MKL takes <10 ms).
 *     These tests always run — the CPU backend links against MKL in this build.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <complex>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

// Initialize backends once for the whole binary.
class ComplexMatmulMKLEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new ComplexMatmulMKLEnv);

// ============================================================================
// Correctness: 1×1 complex matmul
//   (1+2i)(3+4i) = 3+4i+6i+8i² = (3-8) + (4+6)i = -5+10i
// ============================================================================

TEST(ComplexMatmulMKL, Complex64Correctness1x1) {
    // Build 1×1 Complex64 tensors via data pointer (established project pattern).
    auto a = tz::Tensor({int64_t(1), int64_t(1)}, tz::DType::Complex64, tz::Device::cpu());
    auto b = tz::Tensor({int64_t(1), int64_t(1)}, tz::DType::Complex64, tz::Device::cpu());
    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    auto c = tz::matmul(a, b);
    auto cc = c.to(tz::Device::cpu());
    auto val = cc.data<std::complex<float>>()[0];

    EXPECT_NEAR(val.real(), -5.0f, 1e-5f);
    EXPECT_NEAR(val.imag(), 10.0f, 1e-5f);
}

TEST(ComplexMatmulMKL, Complex128Correctness1x1) {
    auto a = tz::Tensor({int64_t(1), int64_t(1)}, tz::DType::Complex128, tz::Device::cpu());
    auto b = tz::Tensor({int64_t(1), int64_t(1)}, tz::DType::Complex128, tz::Device::cpu());
    a.data<std::complex<double>>()[0] = {1.0, 2.0};
    b.data<std::complex<double>>()[0] = {3.0, 4.0};

    auto c = tz::matmul(a, b);
    auto cc = c.to(tz::Device::cpu());
    auto val = cc.data<std::complex<double>>()[0];

    EXPECT_NEAR(val.real(), -5.0, 1e-12);
    EXPECT_NEAR(val.imag(), 10.0, 1e-12);
}

// ============================================================================
// 2×2 correctness — exercises the K loop (not just K=1).
//   B = identity  →  C = A
// ============================================================================

TEST(ComplexMatmulMKL, Complex64Correctness2x2Identity) {
    auto a = tz::Tensor({int64_t(2), int64_t(2)}, tz::DType::Complex64, tz::Device::cpu());
    auto b = tz::Tensor({int64_t(2), int64_t(2)}, tz::DType::Complex64, tz::Device::cpu());
    auto* ap = a.data<std::complex<float>>();
    auto* bp = b.data<std::complex<float>>();
    ap[0] = {1.0f, 1.0f}; ap[1] = {2.0f, 0.0f};
    ap[2] = {0.0f, 0.0f}; ap[3] = {1.0f, 0.0f};
    // B = identity
    bp[0] = {1.0f, 0.0f}; bp[1] = {0.0f, 0.0f};
    bp[2] = {0.0f, 0.0f}; bp[3] = {1.0f, 0.0f};

    auto c = tz::matmul(a, b);
    auto cc = c.to(tz::Device::cpu());
    auto* cp = cc.data<std::complex<float>>();

    EXPECT_NEAR(cp[0].real(), ap[0].real(), 1e-5f);
    EXPECT_NEAR(cp[0].imag(), ap[0].imag(), 1e-5f);
    EXPECT_NEAR(cp[1].real(), ap[1].real(), 1e-5f);
    EXPECT_NEAR(cp[3].real(), ap[3].real(), 1e-5f);
}

// ============================================================================
// Performance smoke tests
//   256×256 Complex64 matmul: scalar O(N³) path ~300–800 ms on a modern CPU.
//   MKL cblas_cgemm: ~2–15 ms.  Threshold is 100 ms — very generous.
//
//   256×256 Complex128 (double): scalar ~600–1500 ms. cblas_zgemm: ~5–25 ms.
//   Threshold is 200 ms.
//
//   These tests always run because the CPU backend in this project links MKL.
//   If the scalar path is accidentally used, the timing assertion exposes it.
// ============================================================================

TEST(ComplexMatmulMKL, ScalarFallbackNotUsed_PerfSmoke_C64) {
    const int N = 256;
    auto a = tz::randn({N, N}, tz::DType::Complex64, tz::Device::cpu());
    auto b = tz::randn({N, N}, tz::DType::Complex64, tz::Device::cpu());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto c = tz::matmul(a, b);
    (void)c;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_LT(ms, 100.0)
        << "256×256 Complex64 matmul took " << ms << " ms — likely scalar fallback.\n"
        << "Audit P0 #5: TENZOR_HAS_MKL guard in math.cpp must be TENZOR_USE_MKL.";
}

TEST(ComplexMatmulMKL, ScalarFallbackNotUsed_PerfSmoke_C128) {
    const int N = 256;
    auto a = tz::randn({N, N}, tz::DType::Complex128, tz::Device::cpu());
    auto b = tz::randn({N, N}, tz::DType::Complex128, tz::Device::cpu());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto c = tz::matmul(a, b);
    (void)c;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_LT(ms, 200.0)
        << "256×256 Complex128 matmul took " << ms << " ms — likely scalar fallback.\n"
        << "Audit P0 #5: TENZOR_HAS_MKL guard in math.cpp must be TENZOR_USE_MKL.";
}
