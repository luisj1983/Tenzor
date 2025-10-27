/**
 * @file test_simd_dispatch.cpp
 * @brief Comprehensive tests for SIMD dispatch system
 */

#include <gtest/gtest.h>
#include "tenzor/backend/simd_dispatch.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

using namespace tenzor::backend;

// Test fixture for SIMD dispatch tests
class SIMDDispatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize SIMD dispatch system
        initialize_simd_dispatch();

        // Setup random number generator for reproducible tests
        rng.seed(42);
        dist = std::uniform_real_distribution<float>(-10.0f, 10.0f);
    }

    // Helper to generate random test data
    std::vector<float> generate_random_data(size_t size) {
        std::vector<float> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(rng);
        }
        return data;
    }

    // Helper to check if two floats are approximately equal
    bool approx_equal(float a, float b, float epsilon = 1e-5f) {
        return std::abs(a - b) < epsilon;
    }

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
};

// ============================================================================
// CPU Feature Detection Tests
// ============================================================================

TEST_F(SIMDDispatchTest, CPUFeatureDetection) {
    // Test that feature detection doesn't crash
    bool has_avx512 = cpu_supports_avx512();
    bool has_avx2 = cpu_supports_avx2();
    bool has_sse42 = cpu_supports_sse42();
    bool has_neon = cpu_supports_neon();

    // Get feature string
    const char* features = get_cpu_features();
    ASSERT_NE(features, nullptr);
    ASSERT_GT(std::strlen(features), 0);

    std::cout << "Detected CPU features: " << features << std::endl;

    // On x86/x64, at least one of SSE4.2, AVX2, or AVX-512 should be available
    // On ARM, NEON might be available
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // Most modern x86 CPUs have at least SSE4.2
    std::cout << "  AVX-512: " << (has_avx512 ? "yes" : "no") << std::endl;
    std::cout << "  AVX2: " << (has_avx2 ? "yes" : "no") << std::endl;
    std::cout << "  SSE4.2: " << (has_sse42 ? "yes" : "no") << std::endl;
    EXPECT_FALSE(has_neon); // Should be false on x86
#elif defined(__ARM_NEON) || defined(__aarch64__)
    std::cout << "  NEON: " << (has_neon ? "yes" : "no") << std::endl;
    EXPECT_FALSE(has_avx512); // Should be false on ARM
    EXPECT_FALSE(has_avx2);
    EXPECT_FALSE(has_sse42);
#endif
}

// ============================================================================
// Element-wise Addition Tests
// ============================================================================

TEST_F(SIMDDispatchTest, AdditionSmallArray) {
    const size_t size = 8;
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
    std::vector<float> result(size);
    std::vector<float> expected = {1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f, 15.5f};

    auto add_kernel = get_optimal_add_kernel();
    ASSERT_NE(add_kernel, nullptr);

    add_kernel(result.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SIMDDispatchTest, AdditionLargeArray) {
    const size_t size = 1024;
    auto a = generate_random_data(size);
    auto b = generate_random_data(size);
    std::vector<float> result(size);

    auto add_kernel = get_optimal_add_kernel();
    add_kernel(result.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SIMDDispatchTest, AdditionUnalignedSize) {
    // Test with size not multiple of SIMD width
    const size_t size = 1000; // Not divisible by 4, 8, or 16
    auto a = generate_random_data(size);
    auto b = generate_random_data(size);
    std::vector<float> result(size);

    auto add_kernel = get_optimal_add_kernel();
    add_kernel(result.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]) << "Mismatch at index " << i;
    }
}

// ============================================================================
// Element-wise Multiplication Tests
// ============================================================================

TEST_F(SIMDDispatchTest, MultiplicationSmallArray) {
    const size_t size = 8;
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> result(size);
    std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f};

    auto mul_kernel = get_optimal_mul_kernel();
    ASSERT_NE(mul_kernel, nullptr);

    mul_kernel(result.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SIMDDispatchTest, MultiplicationLargeArray) {
    const size_t size = 1024;
    auto a = generate_random_data(size);
    auto b = generate_random_data(size);
    std::vector<float> result(size);

    auto mul_kernel = get_optimal_mul_kernel();
    mul_kernel(result.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] * b[i]) << "Mismatch at index " << i;
    }
}

// ============================================================================
// ReLU Activation Tests
// ============================================================================

TEST_F(SIMDDispatchTest, ReLUBasic) {
    const size_t size = 16;
    std::vector<float> input = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f,
                                 4.0f, -0.5f, 0.5f, -10.0f, 10.0f, -100.0f, 100.0f, 0.0f};
    std::vector<float> result(size);
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 0.0f, 0.5f, 0.0f, 10.0f, 0.0f, 100.0f, 0.0f};

    auto relu_kernel = get_optimal_relu_kernel();
    ASSERT_NE(relu_kernel, nullptr);

    relu_kernel(result.data(), input.data(), nullptr, size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SIMDDispatchTest, ReLULargeArray) {
    const size_t size = 1024;
    auto input = generate_random_data(size);
    std::vector<float> result(size);

    auto relu_kernel = get_optimal_relu_kernel();
    relu_kernel(result.data(), input.data(), nullptr, size);

    for (size_t i = 0; i < size; ++i) {
        float expected = input[i] > 0.0f ? input[i] : 0.0f;
        EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
    }
}

// ============================================================================
// Sigmoid Activation Tests
// ============================================================================

TEST_F(SIMDDispatchTest, SigmoidBasic) {
    const size_t size = 8;
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -5.0f, 5.0f, 0.5f};
    std::vector<float> result(size);

    auto sigmoid_kernel = get_optimal_sigmoid_kernel();
    ASSERT_NE(sigmoid_kernel, nullptr);

    sigmoid_kernel(result.data(), input.data(), nullptr, size);

    for (size_t i = 0; i < size; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-input[i]));
        EXPECT_TRUE(approx_equal(result[i], expected, 1e-5f))
            << "Mismatch at index " << i << ": got " << result[i]
            << ", expected " << expected;
    }
}

TEST_F(SIMDDispatchTest, SigmoidBounds) {
    const size_t size = 4;
    std::vector<float> input = {-100.0f, 100.0f, 0.0f, -0.0f};
    std::vector<float> result(size);

    auto sigmoid_kernel = get_optimal_sigmoid_kernel();
    sigmoid_kernel(result.data(), input.data(), nullptr, size);

    // sigmoid(-100) should be very close to 0
    EXPECT_LT(result[0], 0.01f);
    // sigmoid(100) should be very close to 1
    EXPECT_GT(result[1], 0.99f);
    // sigmoid(0) should be exactly 0.5
    EXPECT_FLOAT_EQ(result[2], 0.5f);
    EXPECT_FLOAT_EQ(result[3], 0.5f);
}

// ============================================================================
// Tanh Activation Tests
// ============================================================================

TEST_F(SIMDDispatchTest, TanhBasic) {
    const size_t size = 8;
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -5.0f, 5.0f, 0.5f};
    std::vector<float> result(size);

    auto tanh_kernel = get_optimal_tanh_kernel();
    ASSERT_NE(tanh_kernel, nullptr);

    tanh_kernel(result.data(), input.data(), nullptr, size);

    for (size_t i = 0; i < size; ++i) {
        float expected = std::tanh(input[i]);
        EXPECT_TRUE(approx_equal(result[i], expected, 1e-5f))
            << "Mismatch at index " << i << ": got " << result[i]
            << ", expected " << expected;
    }
}

TEST_F(SIMDDispatchTest, TanhBounds) {
    const size_t size = 4;
    std::vector<float> input = {-100.0f, 100.0f, 0.0f, -0.0f};
    std::vector<float> result(size);

    auto tanh_kernel = get_optimal_tanh_kernel();
    tanh_kernel(result.data(), input.data(), nullptr, size);

    // tanh(-100) should be very close to -1
    EXPECT_LT(result[0], -0.99f);
    // tanh(100) should be very close to 1
    EXPECT_GT(result[1], 0.99f);
    // tanh(0) should be exactly 0
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], 0.0f);
}

// ============================================================================
// Reduction Sum Tests
// ============================================================================

TEST_F(SIMDDispatchTest, ReduceSumBasic) {
    const size_t size = 8;
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float expected = 36.0f; // 1+2+3+4+5+6+7+8

    auto sum_kernel = get_optimal_reduce_sum_kernel();
    ASSERT_NE(sum_kernel, nullptr);

    float result = sum_kernel(input.data(), size);
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(SIMDDispatchTest, ReduceSumLargeArray) {
    const size_t size = 1024;
    auto input = generate_random_data(size);

    // Compute expected sum
    float expected = 0.0f;
    for (float val : input) {
        expected += val;
    }

    auto sum_kernel = get_optimal_reduce_sum_kernel();
    float result = sum_kernel(input.data(), size);

    EXPECT_TRUE(approx_equal(result, expected, 1e-3f))
        << "Sum mismatch: got " << result << ", expected " << expected;
}

TEST_F(SIMDDispatchTest, ReduceSumUnalignedSize) {
    const size_t size = 1000; // Not divisible by 4, 8, or 16
    auto input = generate_random_data(size);

    float expected = 0.0f;
    for (float val : input) {
        expected += val;
    }

    auto sum_kernel = get_optimal_reduce_sum_kernel();
    float result = sum_kernel(input.data(), size);

    EXPECT_TRUE(approx_equal(result, expected, 1e-3f));
}

TEST_F(SIMDDispatchTest, ReduceSumEmpty) {
    auto sum_kernel = get_optimal_reduce_sum_kernel();
    float result = sum_kernel(nullptr, 0);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

// ============================================================================
// Reduction Max Tests
// ============================================================================

TEST_F(SIMDDispatchTest, ReduceMaxBasic) {
    const size_t size = 8;
    std::vector<float> input = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 8.0f, 4.0f, 6.0f};
    float expected = 9.0f;

    auto max_kernel = get_optimal_reduce_max_kernel();
    ASSERT_NE(max_kernel, nullptr);

    float result = max_kernel(input.data(), size);
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(SIMDDispatchTest, ReduceMaxNegativeValues) {
    const size_t size = 8;
    std::vector<float> input = {-10.0f, -5.0f, -20.0f, -1.0f, -100.0f, -3.0f, -7.0f, -2.0f};
    float expected = -1.0f;

    auto max_kernel = get_optimal_reduce_max_kernel();
    float result = max_kernel(input.data(), size);
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(SIMDDispatchTest, ReduceMaxLargeArray) {
    const size_t size = 1024;
    auto input = generate_random_data(size);

    float expected = *std::max_element(input.begin(), input.end());

    auto max_kernel = get_optimal_reduce_max_kernel();
    float result = max_kernel(input.data(), size);

    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(SIMDDispatchTest, ReduceMaxUnalignedSize) {
    const size_t size = 1000;
    auto input = generate_random_data(size);

    float expected = *std::max_element(input.begin(), input.end());

    auto max_kernel = get_optimal_reduce_max_kernel();
    float result = max_kernel(input.data(), size);

    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(SIMDDispatchTest, ReduceMaxEmpty) {
    auto max_kernel = get_optimal_reduce_max_kernel();
    float result = max_kernel(nullptr, 0);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

// ============================================================================
// Cross-Implementation Consistency Tests
// ============================================================================

TEST_F(SIMDDispatchTest, AllImplementationsConsistent) {
    const size_t size = 100;
    auto a = generate_random_data(size);
    auto b = generate_random_data(size);

    // Test that all available implementations produce the same results
    std::vector<float> result_scalar(size);
    std::vector<float> result_simd(size);

    // Test addition
    kernels::add_scalar(result_scalar.data(), a.data(), b.data(), size);
    auto add_kernel = get_optimal_add_kernel();
    add_kernel(result_simd.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result_scalar[i], result_simd[i])
            << "Addition mismatch at index " << i;
    }

    // Test multiplication
    kernels::mul_scalar(result_scalar.data(), a.data(), b.data(), size);
    auto mul_kernel = get_optimal_mul_kernel();
    mul_kernel(result_simd.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result_scalar[i], result_simd[i])
            << "Multiplication mismatch at index " << i;
    }

    // Test ReLU
    kernels::relu_scalar(result_scalar.data(), a.data(), nullptr, size);
    auto relu_kernel = get_optimal_relu_kernel();
    relu_kernel(result_simd.data(), a.data(), nullptr, size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result_scalar[i], result_simd[i])
            << "ReLU mismatch at index " << i;
    }

    // Test reductions
    float sum_scalar = kernels::reduce_sum_scalar(a.data(), size);
    float sum_simd = get_optimal_reduce_sum_kernel()(a.data(), size);
    EXPECT_TRUE(approx_equal(sum_scalar, sum_simd, 1e-3f))
        << "Sum reduction mismatch";

    float max_scalar = kernels::reduce_max_scalar(a.data(), size);
    float max_simd = get_optimal_reduce_max_kernel()(a.data(), size);
    EXPECT_FLOAT_EQ(max_scalar, max_simd) << "Max reduction mismatch";
}

// ============================================================================
// Performance-oriented Tests (verify optimizations work)
// ============================================================================

TEST_F(SIMDDispatchTest, KernelSelectionNotNull) {
    // Verify that all kernel getters return valid function pointers
    EXPECT_NE(get_optimal_add_kernel(), nullptr);
    EXPECT_NE(get_optimal_mul_kernel(), nullptr);
    EXPECT_NE(get_optimal_matmul_kernel(), nullptr);
    EXPECT_NE(get_optimal_relu_kernel(), nullptr);
    EXPECT_NE(get_optimal_sigmoid_kernel(), nullptr);
    EXPECT_NE(get_optimal_tanh_kernel(), nullptr);
    EXPECT_NE(get_optimal_reduce_sum_kernel(), nullptr);
    EXPECT_NE(get_optimal_reduce_max_kernel(), nullptr);
}

TEST_F(SIMDDispatchTest, InitializationIdempotent) {
    // Test that multiple initializations are safe
    initialize_simd_dispatch();
    initialize_simd_dispatch();
    initialize_simd_dispatch();

    // Kernels should still work
    const size_t size = 16;
    auto a = generate_random_data(size);
    auto b = generate_random_data(size);
    std::vector<float> result(size);

    auto add_kernel = get_optimal_add_kernel();
    add_kernel(result.data(), a.data(), b.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]);
    }
}
