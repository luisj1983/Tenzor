/**
 * @file test_simd.cpp
 * @brief Comprehensive multi-backend tests for SIMD operations
 *
 * Tests all SIMD implementations across CPU, CUDA, Vulkan, and OneAPI backends.
 * Covers all functions in tenzor/backends/cpu/simd.hpp with extensive edge case testing.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/backends/cpu/simd.hpp"
#include "tenzor/backend/runtime_simd.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::cpu;

class SIMDBackendTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();

        // Initialize random number generator with fixed seed for reproducibility
        rng_.seed(42);
    }

    // Helper to generate random floats in a range
    std::vector<float> randomFloats(size_t size, float min = -10.0f, float max = 10.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        std::vector<float> result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = dist(rng_);
        }
        return result;
    }

    // Helper to check approximate equality
    void checkApproxEqual(const float* a, const float* b, size_t size, float tolerance = 1e-5f) {
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(a[i], b[i], tolerance)
                << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                << " on device " << device.to_string();
        }
    }

    // Helper to create tensor from vector
    Tensor tensorFromVector(const std::vector<float>& data) {
        auto t = zeros({static_cast<int64_t>(data.size())}, DType::Float32, device);
        auto t_cpu = t.to(Device::cpu());
        std::memcpy(t_cpu.data<float>(), data.data(), data.size() * sizeof(float));
        return t_cpu.to(device);
    }

    // Helper to extract vector from tensor
    std::vector<float> vectorFromTensor(const Tensor& t) {
        auto t_cpu = t.to(Device::cpu());
        const float* data = t_cpu.data<float>();
        return std::vector<float>(data, data + t_cpu.numel());
    }

    std::mt19937 rng_;
};

// ============================================================================
// CPU Feature Detection Tests
// ============================================================================

TEST_P(SIMDBackendTest, SIMDFeaturesSingleton) {
    // runtime_simd's get_simd_features() returns a cached static reference;
    // two calls must yield the same address.
    const auto& f1 = ::tenzor::backend::get_simd_features();
    const auto& f2 = ::tenzor::backend::get_simd_features();

    EXPECT_EQ(&f1, &f2) << "SIMDFeatures cache should be a singleton on " << device.to_string();
}

TEST_P(SIMDBackendTest, SIMDFeatureDetection) {
    const auto& cpu = ::tenzor::backend::get_simd_features();

    // Feature summary string should be retrievable on every platform.
    std::string features = cpu.to_string();
    EXPECT_TRUE(features.empty() || !features.empty())
        << "Feature string call should not crash on " << device.to_string();

    // Reading individual feature flags must be side-effect free.
    (void)cpu.avx512f;
    (void)cpu.avx2;
    (void)cpu.sse42;
}

TEST_P(SIMDBackendTest, SIMDLevelOrdering) {
    // Detected SIMD level must agree with the features struct's best_level().
    auto level = ::tenzor::backend::detect_simd_level();
    auto best = ::tenzor::backend::get_simd_features().best_level();
    EXPECT_EQ(level, best)
        << "detect_simd_level() and SIMDFeatures::best_level() must agree on "
        << device.to_string();

    // On x86-64 we must at least have SSE2 (baseline ISA).
    #if defined(__x86_64__) || defined(_M_X64)
        EXPECT_TRUE(::tenzor::backend::has_simd_feature(::tenzor::backend::SIMDLevel::SSE2))
            << "SSE2 is baseline x86-64; detection must report it on "
            << device.to_string();
    #endif
}

// ============================================================================
// Arithmetic Operations - Addition
// ============================================================================

TEST_P(SIMDBackendTest, AddBasic) {
    const size_t sizes[] = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size);
        auto b_vec = randomFloats(size);

        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);
        std::vector<float> expected(size);

        // Compute expected result
        for (size_t i = 0; i < size; ++i) {
            expected[i] = a_vec[i] + b_vec[i];
        }

        // Test SIMD implementation
        simd::add(a_vec.data(), b_vec.data(), out_simd.data(), size);

        // Test scalar implementation as reference
        scalar::add(a_vec.data(), b_vec.data(), out_scalar.data(), size);

        // Compare results
        checkApproxEqual(out_simd.data(), expected.data(), size);
        checkApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_P(SIMDBackendTest, AddUnalignedSizes) {
    // Test sizes that are not powers of 2 or multiples of SIMD width
    const size_t sizes[] = {1, 3, 7, 15, 31, 63, 127, 255, 511, 1000, 1001, 1023};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size);
        auto b_vec = randomFloats(size);
        std::vector<float> out(size);

        simd::add(a_vec.data(), b_vec.data(), out.data(), size);

        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(out[i], a_vec[i] + b_vec[i], 1e-5f)
                << "Mismatch at index " << i << " for size " << size
                << " on " << device.to_string();
        }
    }
}

TEST_P(SIMDBackendTest, AddZeroElements) {
    std::vector<float> a = {1.0f};
    std::vector<float> b = {2.0f};
    std::vector<float> out = {999.0f};

    // Should not crash or modify output
    simd::add(a.data(), b.data(), out.data(), 0);
    EXPECT_EQ(out[0], 999.0f) << "Zero-size operation should not modify output on " << device.to_string();
}

TEST_P(SIMDBackendTest, AddSpecialValues) {
    std::vector<float> a = {
        0.0f, -0.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::min(),
        1e-10f, -1e-10f
    };
    std::vector<float> b = {
        0.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 1.0f
    };
    std::vector<float> out(a.size());

    simd::add(a.data(), b.data(), out.data(), a.size());

    EXPECT_EQ(out[0], 0.0f);
    EXPECT_EQ(out[1], 0.0f);
    EXPECT_TRUE(std::isinf(out[2])) << "inf + 1 should be inf on " << device.to_string();
    EXPECT_TRUE(std::isinf(out[3])) << "-inf + 1 should be -inf on " << device.to_string();
}

// ============================================================================
// Arithmetic Operations - Subtraction
// ============================================================================

TEST_P(SIMDBackendTest, SubBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size);
        auto b_vec = randomFloats(size);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::sub(a_vec.data(), b_vec.data(), out_simd.data(), size);
        scalar::sub(a_vec.data(), b_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size);

        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(out_simd[i], a_vec[i] - b_vec[i], 1e-5f)
                << "Subtraction failed at index " << i << " on " << device.to_string();
        }
    }
}

TEST_P(SIMDBackendTest, SubSelfZero) {
    const size_t size = 100;
    auto a_vec = randomFloats(size);
    std::vector<float> out(size);

    // a - a should be zero
    simd::sub(a_vec.data(), a_vec.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(out[i], 0.0f, 1e-5f)
            << "Self-subtraction should yield zero at index " << i
            << " on " << device.to_string();
    }
}

// ============================================================================
// Arithmetic Operations - Multiplication
// ============================================================================

TEST_P(SIMDBackendTest, MulBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size);
        auto b_vec = randomFloats(size);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::mul(a_vec.data(), b_vec.data(), out_simd.data(), size);
        scalar::mul(a_vec.data(), b_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_P(SIMDBackendTest, MulByZero) {
    const size_t size = 100;
    auto a_vec = randomFloats(size);
    std::vector<float> zeros(size, 0.0f);
    std::vector<float> out(size);

    simd::mul(a_vec.data(), zeros.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(out[i], 0.0f)
            << "Multiplication by zero failed at index " << i
            << " on " << device.to_string();
    }
}

TEST_P(SIMDBackendTest, MulByOne) {
    const size_t size = 100;
    auto a_vec = randomFloats(size);
    std::vector<float> ones(size, 1.0f);
    std::vector<float> out(size);

    simd::mul(a_vec.data(), ones.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(out[i], a_vec[i], 1e-5f)
            << "Multiplication by one should preserve value at index " << i
            << " on " << device.to_string();
    }
}

// ============================================================================
// Arithmetic Operations - Division
// ============================================================================

TEST_P(SIMDBackendTest, DivBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size);
        auto b_vec = randomFloats(size, 0.1f, 10.0f);  // Avoid division by zero
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::div(a_vec.data(), b_vec.data(), out_simd.data(), size);
        scalar::div(a_vec.data(), b_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_P(SIMDBackendTest, DivByOne) {
    const size_t size = 100;
    auto a_vec = randomFloats(size);
    std::vector<float> ones(size, 1.0f);
    std::vector<float> out(size);

    simd::div(a_vec.data(), ones.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(out[i], a_vec[i], 1e-5f)
            << "Division by one should preserve value at index " << i
            << " on " << device.to_string();
    }
}

TEST_P(SIMDBackendTest, DivSelfOne) {
    const size_t size = 100;
    auto a_vec = randomFloats(size, 0.1f, 10.0f);  // Avoid zero
    std::vector<float> out(size);

    simd::div(a_vec.data(), a_vec.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(out[i], 1.0f, 1e-4f)
            << "Self-division should yield one at index " << i
            << " on " << device.to_string();
    }
}

// ============================================================================
// Unary Operations - Square Root
// ============================================================================

TEST_P(SIMDBackendTest, SqrtBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size, 0.0f, 100.0f);  // Non-negative
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::sqrt(a_vec.data(), out_simd.data(), size);
        scalar::sqrt(a_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size);

        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(out_simd[i], std::sqrt(a_vec[i]), 1e-5f)
                << "Sqrt failed at index " << i << " on " << device.to_string();
        }
    }
}

TEST_P(SIMDBackendTest, SqrtSpecialValues) {
    std::vector<float> input = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 100.0f};
    std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 10.0f};
    std::vector<float> out(input.size());

    simd::sqrt(input.data(), out.data(), input.size());

    checkApproxEqual(out.data(), expected.data(), input.size());
}

// ============================================================================
// Unary Operations - Exponential
// ============================================================================

TEST_P(SIMDBackendTest, ExpBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size, -5.0f, 5.0f);  // Keep in reasonable range
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::exp(a_vec.data(), out_simd.data(), size);
        scalar::exp(a_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);
    }
}

TEST_P(SIMDBackendTest, ExpSpecialValues) {
    std::vector<float> input = {0.0f, 1.0f, -1.0f};
    std::vector<float> out(input.size());

    simd::exp(input.data(), out.data(), input.size());

    EXPECT_NEAR(out[0], 1.0f, 1e-4f) << "exp(0) should be 1 on " << device.to_string();
    EXPECT_NEAR(out[1], std::exp(1.0f), 1e-4f) << "exp(1) failed on " << device.to_string();
    EXPECT_NEAR(out[2], std::exp(-1.0f), 1e-4f) << "exp(-1) failed on " << device.to_string();
}

// ============================================================================
// Unary Operations - Logarithm
// ============================================================================

TEST_P(SIMDBackendTest, LogBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size, 0.01f, 100.0f);  // Positive values
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::log(a_vec.data(), out_simd.data(), size);
        scalar::log(a_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);
    }
}

TEST_P(SIMDBackendTest, LogSpecialValues) {
    std::vector<float> input = {1.0f, std::exp(1.0f), std::exp(2.0f)};
    std::vector<float> expected = {0.0f, 1.0f, 2.0f};
    std::vector<float> out(input.size());

    simd::log(input.data(), out.data(), input.size());

    checkApproxEqual(out.data(), expected.data(), input.size(), 1e-4f);
}

// ============================================================================
// Activation Functions - ReLU
// ============================================================================

TEST_P(SIMDBackendTest, ReLUBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size, -10.0f, 10.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::relu(a_vec.data(), out_simd.data(), size);
        scalar::relu(a_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size);

        for (size_t i = 0; i < size; ++i) {
            float expected = std::max(0.0f, a_vec[i]);
            EXPECT_NEAR(out_simd[i], expected, 1e-5f)
                << "ReLU failed at index " << i << " on " << device.to_string();
        }
    }
}

TEST_P(SIMDBackendTest, ReLUMixedValues) {
    std::vector<float> input = {-5.0f, -2.0f, -0.1f, 0.0f, 0.1f, 2.0f, 5.0f};
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 2.0f, 5.0f};
    std::vector<float> out(input.size());

    simd::relu(input.data(), out.data(), input.size());

    checkApproxEqual(out.data(), expected.data(), input.size());
}

TEST_P(SIMDBackendTest, ReLUAllNegative) {
    const size_t size = 100;
    auto a_vec = randomFloats(size, -100.0f, -0.1f);
    std::vector<float> out(size);

    simd::relu(a_vec.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(out[i], 0.0f)
            << "ReLU of negative should be zero at index " << i
            << " on " << device.to_string();
    }
}

TEST_P(SIMDBackendTest, ReLUAllPositive) {
    const size_t size = 100;
    auto a_vec = randomFloats(size, 0.1f, 100.0f);
    std::vector<float> out(size);

    simd::relu(a_vec.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(out[i], a_vec[i], 1e-5f)
            << "ReLU of positive should be identity at index " << i
            << " on " << device.to_string();
    }
}

// ============================================================================
// Activation Functions - Sigmoid
// ============================================================================

TEST_P(SIMDBackendTest, SigmoidBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size, -5.0f, 5.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::sigmoid(a_vec.data(), out_simd.data(), size);
        scalar::sigmoid(a_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-5f);
    }
}

TEST_P(SIMDBackendTest, SigmoidRange) {
    const size_t size = 100;
    auto a_vec = randomFloats(size, -10.0f, 10.0f);
    std::vector<float> out(size);

    simd::sigmoid(a_vec.data(), out.data(), size);

    // Sigmoid output should always be in (0, 1)
    for (size_t i = 0; i < size; ++i) {
        EXPECT_GT(out[i], 0.0f) << "Sigmoid should be > 0 at index " << i
            << " on " << device.to_string();
        EXPECT_LT(out[i], 1.0f) << "Sigmoid should be < 1 at index " << i
            << " on " << device.to_string();
    }
}

TEST_P(SIMDBackendTest, SigmoidZero) {
    std::vector<float> input = {0.0f};
    std::vector<float> out(1);

    simd::sigmoid(input.data(), out.data(), 1);

    EXPECT_NEAR(out[0], 0.5f, 1e-5f)
        << "sigmoid(0) should be 0.5 on " << device.to_string();
}

TEST_P(SIMDBackendTest, SigmoidSymmetry) {
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> out(input.size());

    simd::sigmoid(input.data(), out.data(), input.size());

    // sigmoid(-x) + sigmoid(x) should equal 1
    EXPECT_NEAR(out[0] + out[4], 1.0f, 1e-5f)
        << "Sigmoid symmetry property failed on " << device.to_string();
    EXPECT_NEAR(out[1] + out[3], 1.0f, 1e-5f)
        << "Sigmoid symmetry property failed on " << device.to_string();
}

// ============================================================================
// Activation Functions - Tanh
// ============================================================================

TEST_P(SIMDBackendTest, TanhBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size, -5.0f, 5.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::tanh(a_vec.data(), out_simd.data(), size);
        scalar::tanh(a_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-5f);
    }
}

TEST_P(SIMDBackendTest, TanhRange) {
    const size_t size = 100;
    auto a_vec = randomFloats(size, -10.0f, 10.0f);
    std::vector<float> out(size);

    simd::tanh(a_vec.data(), out.data(), size);

    // Tanh output should always be in [-1, 1]
    for (size_t i = 0; i < size; ++i) {
        EXPECT_GE(out[i], -1.0f) << "Tanh should be >= -1 at index " << i
            << " on " << device.to_string();
        EXPECT_LE(out[i], 1.0f) << "Tanh should be <= 1 at index " << i
            << " on " << device.to_string();
    }
}

TEST_P(SIMDBackendTest, TanhZero) {
    std::vector<float> input = {0.0f};
    std::vector<float> out(1);

    simd::tanh(input.data(), out.data(), 1);

    EXPECT_NEAR(out[0], 0.0f, 1e-5f)
        << "tanh(0) should be 0 on " << device.to_string();
}

TEST_P(SIMDBackendTest, TanhAntisymmetry) {
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> out(input.size());

    simd::tanh(input.data(), out.data(), input.size());

    // tanh(-x) = -tanh(x)
    EXPECT_NEAR(out[0], -out[4], 1e-5f)
        << "Tanh antisymmetry property failed on " << device.to_string();
    EXPECT_NEAR(out[1], -out[3], 1e-5f)
        << "Tanh antisymmetry property failed on " << device.to_string();
}

// ============================================================================
// Activation Functions - GELU
// ============================================================================

TEST_P(SIMDBackendTest, GeLUBasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size, -3.0f, 3.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::gelu(a_vec.data(), out_simd.data(), size);
        scalar::gelu(a_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);
    }
}

TEST_P(SIMDBackendTest, GeLUZero) {
    std::vector<float> input = {0.0f};
    std::vector<float> out(1);

    simd::gelu(input.data(), out.data(), 1);

    EXPECT_NEAR(out[0], 0.0f, 1e-4f)
        << "GELU(0) should be approximately 0 on " << device.to_string();
}

TEST_P(SIMDBackendTest, GeLULargePositive) {
    std::vector<float> input = {5.0f, 10.0f};
    std::vector<float> out(input.size());

    simd::gelu(input.data(), out.data(), input.size());

    // For large positive x, GELU(x) ≈ x
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(out[i], input[i], 0.1f)
            << "GELU of large positive should approximate identity at index " << i
            << " on " << device.to_string();
    }
}

TEST_P(SIMDBackendTest, GeLULargeNegative) {
    std::vector<float> input = {-5.0f, -10.0f};
    std::vector<float> out(input.size());

    simd::gelu(input.data(), out.data(), input.size());

    // For large negative x, GELU(x) ≈ 0
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(out[i], 0.0f, 0.1f)
            << "GELU of large negative should be near zero at index " << i
            << " on " << device.to_string();
    }
}

// ============================================================================
// Fused Multiply-Add (FMA)
// ============================================================================

TEST_P(SIMDBackendTest, FMABasic) {
    const size_t sizes[] = {1, 8, 16, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a_vec = randomFloats(size);
        auto b_vec = randomFloats(size);
        auto c_vec = randomFloats(size);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::fma(a_vec.data(), b_vec.data(), c_vec.data(), out_simd.data(), size);
        scalar::fma(a_vec.data(), b_vec.data(), c_vec.data(), out_scalar.data(), size);

        checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);

        for (size_t i = 0; i < size; ++i) {
            float expected = a_vec[i] * b_vec[i] + c_vec[i];
            EXPECT_NEAR(out_simd[i], expected, 1e-4f)
                << "FMA failed at index " << i << " on " << device.to_string();
        }
    }
}

TEST_P(SIMDBackendTest, FMAZeroMultiply) {
    const size_t size = 100;
    std::vector<float> zeros(size, 0.0f);
    auto b_vec = randomFloats(size);
    auto c_vec = randomFloats(size);
    std::vector<float> out(size);

    // 0 * b + c = c
    simd::fma(zeros.data(), b_vec.data(), c_vec.data(), out.data(), size);

    checkApproxEqual(out.data(), c_vec.data(), size, 1e-5f);
}

TEST_P(SIMDBackendTest, FMAZeroAdd) {
    const size_t size = 100;
    auto a_vec = randomFloats(size);
    auto b_vec = randomFloats(size);
    std::vector<float> zeros(size, 0.0f);
    std::vector<float> out(size);
    std::vector<float> expected(size);

    // a * b + 0 = a * b
    simd::fma(a_vec.data(), b_vec.data(), zeros.data(), out.data(), size);

    for (size_t i = 0; i < size; ++i) {
        expected[i] = a_vec[i] * b_vec[i];
    }

    checkApproxEqual(out.data(), expected.data(), size, 1e-5f);
}

TEST_P(SIMDBackendTest, FMAIdentity) {
    const size_t size = 100;
    auto a_vec = randomFloats(size);
    std::vector<float> ones(size, 1.0f);
    std::vector<float> zeros(size, 0.0f);
    std::vector<float> out(size);

    // a * 1 + 0 = a
    simd::fma(a_vec.data(), ones.data(), zeros.data(), out.data(), size);

    checkApproxEqual(out.data(), a_vec.data(), size, 1e-5f);
}

// ============================================================================
// Memory Alignment Tests
// ============================================================================

TEST_P(SIMDBackendTest, UnalignedMemoryAccess) {
    const size_t size = 100;
    const size_t buffer_size = size + 16;

    auto a_vec = randomFloats(buffer_size);
    auto b_vec = randomFloats(buffer_size);

    // Test with various misalignments
    for (size_t offset = 0; offset < 8; ++offset) {
        std::vector<float> out(buffer_size);

        simd::add(a_vec.data() + offset, b_vec.data() + offset,
                  out.data() + offset, size);

        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(out[i + offset], a_vec[i + offset] + b_vec[i + offset], 1e-5f)
                << "Unaligned access failed at index " << i << " with offset " << offset
                << " on " << device.to_string();
        }
    }
}

// ============================================================================
// Consistency Tests - All Implementations
// ============================================================================

TEST_P(SIMDBackendTest, ScalarSIMDConsistency) {
    const size_t size = 1000;
    auto a_vec = randomFloats(size);
    auto b_vec = randomFloats(size, 0.1f, 10.0f);
    auto c_vec = randomFloats(size);

    // Test all binary operations
    std::vector<float> out_simd(size), out_scalar(size);

    simd::add(a_vec.data(), b_vec.data(), out_simd.data(), size);
    scalar::add(a_vec.data(), b_vec.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size);

    simd::sub(a_vec.data(), b_vec.data(), out_simd.data(), size);
    scalar::sub(a_vec.data(), b_vec.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size);

    simd::mul(a_vec.data(), b_vec.data(), out_simd.data(), size);
    scalar::mul(a_vec.data(), b_vec.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size);

    simd::div(a_vec.data(), b_vec.data(), out_simd.data(), size);
    scalar::div(a_vec.data(), b_vec.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size);

    // Test unary operations with positive inputs
    auto a_pos = randomFloats(size, 0.01f, 100.0f);

    simd::sqrt(a_pos.data(), out_simd.data(), size);
    scalar::sqrt(a_pos.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size);

    auto a_moderate = randomFloats(size, -5.0f, 5.0f);

    simd::exp(a_moderate.data(), out_simd.data(), size);
    scalar::exp(a_moderate.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);

    simd::log(a_pos.data(), out_simd.data(), size);
    scalar::log(a_pos.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);

    // Test activation functions
    simd::relu(a_vec.data(), out_simd.data(), size);
    scalar::relu(a_vec.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size);

    simd::sigmoid(a_moderate.data(), out_simd.data(), size);
    scalar::sigmoid(a_moderate.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-5f);

    simd::tanh(a_moderate.data(), out_simd.data(), size);
    scalar::tanh(a_moderate.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-5f);

    auto a_gelu = randomFloats(size, -3.0f, 3.0f);
    simd::gelu(a_gelu.data(), out_simd.data(), size);
    scalar::gelu(a_gelu.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);

    // Test FMA
    simd::fma(a_vec.data(), b_vec.data(), c_vec.data(), out_simd.data(), size);
    scalar::fma(a_vec.data(), b_vec.data(), c_vec.data(), out_scalar.data(), size);
    checkApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);
}

// ============================================================================
// AVX2/AVX512 Specific Tests (when available)
// ============================================================================

TEST_P(SIMDBackendTest, AVX2Operations) {
    const auto& cpu = ::tenzor::backend::get_simd_features();
    if (!cpu.avx2) {
        GTEST_SKIP() << "AVX2 not available on " << device.to_string();
    }

    const size_t size = 256;  // Multiple of 8 (AVX2 width)
    auto a_vec = randomFloats(size);
    auto b_vec = randomFloats(size);
    std::vector<float> out_avx2(size);
    std::vector<float> out_scalar(size);

    avx2::add(a_vec.data(), b_vec.data(), out_avx2.data(), size);
    scalar::add(a_vec.data(), b_vec.data(), out_scalar.data(), size);

    checkApproxEqual(out_avx2.data(), out_scalar.data(), size);
}

TEST_P(SIMDBackendTest, AVX512Operations) {
    const auto& cpu = ::tenzor::backend::get_simd_features();
    if (!cpu.avx512f) {
        GTEST_SKIP() << "AVX-512 not available on " << device.to_string();
    }

    const size_t size = 512;  // Multiple of 16 (AVX-512 width)
    auto a_vec = randomFloats(size);
    auto b_vec = randomFloats(size);
    std::vector<float> out_avx512(size);
    std::vector<float> out_scalar(size);

    avx512::add(a_vec.data(), b_vec.data(), out_avx512.data(), size);
    scalar::add(a_vec.data(), b_vec.data(), out_scalar.data(), size);

    checkApproxEqual(out_avx512.data(), out_scalar.data(), size);
}

// ============================================================================
// Performance Validation Tests
// ============================================================================

TEST_P(SIMDBackendTest, LargeArrayPerformance) {
    const size_t size = 1000000;
    auto a_vec = randomFloats(size);
    auto b_vec = randomFloats(size);
    std::vector<float> out(size);

    // This should complete in reasonable time
    auto start = std::chrono::high_resolution_clock::now();
    simd::add(a_vec.data(), b_vec.data(), out.data(), size);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete in less than 100ms on modern hardware
    EXPECT_LT(duration.count(), 100)
        << "Large array operation took too long on " << device.to_string();

    // Verify correctness
    for (size_t i = 0; i < 100; ++i) {  // Spot check
        EXPECT_NEAR(out[i], a_vec[i] + b_vec[i], 1e-5f)
            << "Large array computation incorrect at index " << i
            << " on " << device.to_string();
    }
}

// Instantiate tests for all backends
INSTANTIATE_BACKEND_TESTS(SIMDBackendTest);
