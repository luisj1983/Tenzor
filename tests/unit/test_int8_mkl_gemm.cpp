/**
 * @file test_int8_mkl_gemm.cpp
 * @brief Tests for MKL cblas_gemm_s8u8s32-based Int8 matmul
 *
 * Task 6.3: Wire cblas_gemm_s8u8s32 for S8×S8 matmul with int32 accumulator
 * and saturation back to int8. Verify correctness against scalar reference
 * and check perf for 1024^3.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cmath>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor { void initialize(); }
namespace tz = ::tenzor;

class Int8MklGemmEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new Int8MklGemmEnv);

// Scalar reference: matmul with int32 accumulator + saturation
static std::vector<int8_t> scalar_int8_matmul(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    int M, int K, int N) {
    std::vector<int8_t> C(M * N, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[i * K + k])
                     * static_cast<int32_t>(B[k * N + j]);
            }
            if (acc > 127) acc = 127;
            else if (acc < -128) acc = -128;
            C[i * N + j] = static_cast<int8_t>(acc);
        }
    }
    return C;
}

TEST(Int8MklGemm, SmallMatmulCorrectness) {
    // 4x4 @ 4x4, all ones → each output = 4, which fits in int8
    int M = 4, K = 4, N = 4;
    auto a = tz::full({M, K}, 1.0, tz::DType::Int8);
    auto b = tz::full({K, N}, 1.0, tz::DType::Int8);
    auto c = tz::matmul(a, b);
    auto p = c.cpu().data<int8_t>();
    for (int i = 0; i < M * N; ++i) EXPECT_EQ(p[i], static_cast<int8_t>(K));
}

TEST(Int8MklGemm, MixedSignCorrectness) {
    // 4x4 @ 4x4, A=2, B=-3 → each output = 4*2*(-3) = -24
    int M = 4, K = 4, N = 4;
    auto a = tz::full({M, K}, 2.0, tz::DType::Int8);
    auto b = tz::full({K, N}, -3.0, tz::DType::Int8);
    auto c = tz::matmul(a, b);
    auto p = c.cpu().data<int8_t>();
    for (int i = 0; i < M * N; ++i) EXPECT_EQ(p[i], static_cast<int8_t>(-24));
}

TEST(Int8MklGemm, NegativeAPositiveBCorrectness) {
    // A=-1, B=5 → each output = K * (-1) * 5 = -4*5 = -20
    int M = 4, K = 4, N = 4;
    auto a = tz::full({M, K}, -1.0, tz::DType::Int8);
    auto b = tz::full({K, N},  5.0, tz::DType::Int8);
    auto c = tz::matmul(a, b);
    auto p = c.cpu().data<int8_t>();
    for (int i = 0; i < M * N; ++i) EXPECT_EQ(p[i], static_cast<int8_t>(-20));
}

TEST(Int8MklGemm, SaturationToMax127) {
    // 4x32: A=4, B=4 → each = 32*16 = 512 → saturates to 127
    int M = 4, K = 32, N = 4;
    auto a = tz::full({M, K}, 4.0, tz::DType::Int8);
    auto b = tz::full({K, N}, 4.0, tz::DType::Int8);
    auto c = tz::matmul(a, b);
    auto p = c.cpu().data<int8_t>();
    for (int i = 0; i < M * N; ++i) EXPECT_EQ(p[i], static_cast<int8_t>(127));
}

TEST(Int8MklGemm, SaturationToMin128) {
    // 4x32: A=4, B=-4 → each = 32*(-16) = -512 → saturates to -128
    int M = 4, K = 32, N = 4;
    auto a = tz::full({M, K},  4.0, tz::DType::Int8);
    auto b = tz::full({K, N}, -4.0, tz::DType::Int8);
    auto c = tz::matmul(a, b);
    auto p = c.cpu().data<int8_t>();
    for (int i = 0; i < M * N; ++i) EXPECT_EQ(p[i], static_cast<int8_t>(-128));
}

TEST(Int8MklGemm, LargerMatmulVsScalarRef) {
    // 32x64 @ 64x48, values in [-3, 3], verify against scalar reference
    int M = 32, K = 64, N = 48;
    std::vector<int8_t> A_data(M * K), B_data(K * N);
    for (int i = 0; i < M * K; ++i) A_data[i] = static_cast<int8_t>((i % 7) - 3);
    for (int i = 0; i < K * N; ++i) B_data[i] = static_cast<int8_t>((i % 5) - 2);

    // Build tensors from data
    auto a = tz::zeros({M, K}, tz::DType::Int8);
    auto b = tz::zeros({K, N}, tz::DType::Int8);
    std::memcpy(a.data<int8_t>(), A_data.data(), M * K);
    std::memcpy(b.data<int8_t>(), B_data.data(), K * N);

    auto c = tz::matmul(a, b);
    auto p = c.cpu().data<int8_t>();

    auto ref = scalar_int8_matmul(A_data, B_data, M, K, N);
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(p[i], ref[i]) << "Mismatch at index " << i;
    }
}

TEST(Int8MklGemm, PerfLarge1024) {
    // 1024x1024 @ 1024x1024 — scalar would take multi-seconds; MKL <50ms
    const int N = 1024;
    auto a = tz::full({N, N}, 1.0, tz::DType::Int8);
    auto b = tz::full({N, N}, 1.0, tz::DType::Int8);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto c = tz::matmul(a, b);
    (void)c.cpu();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(ms, 500.0) << "Int8 matmul 1024^3 too slow: " << ms << " ms";
}
