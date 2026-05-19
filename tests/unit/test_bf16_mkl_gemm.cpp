/**
 * @file test_bf16_mkl_gemm.cpp
 * @brief Task 6.4: BF16 matmul via MKL gemm_bf16bf16f32
 *
 * Tests correctness of the BF16×BF16→BF16 matmul kernel that routes
 * through MKL's gemm_bf16bf16f32 (F32 accumulation, then narrow back
 * to BF16).
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

namespace tenzor { void initialize(); }

struct BF16GemmEnv : ::testing::Environment {
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const bf16_env =
    ::testing::AddGlobalTestEnvironment(new BF16GemmEnv);

// Helper: make a BF16 tensor from float values
static tenzor::Tensor make_bf16(const std::vector<float>& vals,
                                  std::vector<int64_t> shape) {
    auto t = tenzor::ops::zeros(shape, tenzor::DType::Float32, tenzor::Device::cpu());
    float* ptr = t.data<float>();
    for (size_t i = 0; i < vals.size(); ++i) ptr[i] = vals[i];
    return t.to(tenzor::DType::BFloat16);
}

// ===========================================================================
// Correctness tests
// ===========================================================================

TEST(BF16MklGemm, SmallMatmulCorrectness) {
    // 2×3 × 3×2 = 2×2
    // C[0][0] = 1*7+2*9+3*11 = 58   C[0][1] = 1*8+2*10+3*12 = 64
    // C[1][0] = 4*7+5*9+6*11 = 139  C[1][1] = 4*8+5*10+6*12 = 154
    auto A = make_bf16({1,2,3,4,5,6}, {2,3});
    auto B = make_bf16({7,8,9,10,11,12}, {3,2});
    auto C = tenzor::matmul(A, B);

    ASSERT_EQ(C.dtype(), tenzor::DType::BFloat16);

    auto C_f32 = C.to(tenzor::DType::Float32);
    const float* p = C_f32.data<float>();

    // BF16 has ~0.4% relative error; allow 1 unit
    EXPECT_NEAR(p[0], 58.0f,  1.5f);
    EXPECT_NEAR(p[1], 64.0f,  1.5f);
    EXPECT_NEAR(p[2], 139.0f, 2.5f);
    EXPECT_NEAR(p[3], 154.0f, 2.5f);
}

TEST(BF16MklGemm, IdentityMatrix) {
    const int N = 8;
    std::vector<float> a_vals(N * N, 0.0f);
    std::vector<float> i_vals(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        a_vals[i * N + i] = static_cast<float>(i + 1);
        i_vals[i * N + i] = 1.0f;
    }
    auto A = make_bf16(a_vals, {N, N});
    auto I = make_bf16(i_vals, {N, N});
    auto C = tenzor::matmul(A, I);
    auto C_f32 = C.to(tenzor::DType::Float32);
    const float* p = C_f32.data<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(p[i * N + i], static_cast<float>(i + 1), 0.5f)
            << "diagonal[" << i << "]";
        for (int j = 0; j < N; ++j) {
            if (j != i) {
                EXPECT_NEAR(p[i * N + j], 0.0f, 0.5f)
                    << "off-diagonal[" << i << "," << j << "]";
            }
        }
    }
}

TEST(BF16MklGemm, MediumMatmulVsReference) {
    const int M = 32, K = 64, N = 32;
    std::vector<float> A_f(M * K), B_f(K * N);
    for (int i = 0; i < M * K; ++i) A_f[i] = 0.1f * ((i % 10) - 5);
    for (int i = 0; i < K * N; ++i) B_f[i] = 0.1f * ((i % 7)  - 3);

    auto A = make_bf16(A_f, {M, K});
    auto B = make_bf16(B_f, {K, N});
    auto C = tenzor::matmul(A, B);
    auto C_f32 = C.to(tenzor::DType::Float32);
    const float* p = C_f32.data<float>();

    // Reference in F32
    std::vector<float> ref(M * N, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                ref[i * N + j] += A_f[i * K + k] * B_f[k * N + j];

    // Allow 5% relative tolerance for BF16 accumulation error
    for (int i = 0; i < M * N; ++i) {
        float tol = std::max(std::abs(ref[i]) * 0.05f, 0.05f);
        EXPECT_NEAR(p[i], ref[i], tol) << "element[" << i << "]";
        if (HasFatalFailure()) FAIL();
    }
}

TEST(BF16MklGemm, ZeroMatrix) {
    auto A = make_bf16({0,0,0,0}, {2,2});
    auto B = make_bf16({1,2,3,4}, {2,2});
    auto C = tenzor::matmul(A, B);
    auto C_f32 = C.to(tenzor::DType::Float32);
    const float* p = C_f32.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(p[i], 0.0f, 0.01f) << "element " << i;
    }
}

// ===========================================================================
// Performance test
// ===========================================================================

TEST(BF16MklGemm, Perf1024) {
    const int M = 1024, K = 1024, N = 1024;
    std::vector<float> A_f(M * K, 0.01f), B_f(K * N, 0.01f);
    auto A = make_bf16(A_f, {M, K});
    auto B = make_bf16(B_f, {K, N});

    tenzor::matmul(A, B);  // warm-up

    auto t0 = std::chrono::high_resolution_clock::now();
    auto C = tenzor::matmul(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)C;
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_LT(ms, 2000.0) << "BF16 1024^3 matmul took " << ms << " ms";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
