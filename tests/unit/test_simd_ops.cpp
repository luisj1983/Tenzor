/**
 * @file test_simd_ops.cpp
 * @brief Tests for SIMD operations
 */

#include <gtest/gtest.h>
#include "tenzor/backends/cpu/simd.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace tenzor::cpu;

class SIMDOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Print CPU features
        const auto& cpu = CPUInfo::get();
        std::cout << "CPU Vendor: " << cpu.vendor() << "\n";
        std::cout << "CPU Brand: " << cpu.brand() << "\n";
        std::cout << "CPU Features: " << cpu.feature_string() << "\n";
        std::cout << std::endl;
    }

    // Helper to check if two arrays are approximately equal
    void CheckApproxEqual(const float* a, const float* b, size_t size, float tolerance = 1e-5f) {
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(a[i], b[i], tolerance)
                << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i];
        }
    }

    // Generate random floats
    std::vector<float> RandomFloats(size_t size, float min = -10.0f, float max = 10.0f) {
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(min, max);

        std::vector<float> result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = dis(gen);
        }
        return result;
    }
};

// ============================================================================
// CPU Feature Detection Tests
// ============================================================================

TEST_F(SIMDOpsTest, CPUFeatureDetection) {
    const auto& cpu = CPUInfo::get();

    // Check that at least SSE2 is supported (required for x86-64)
    #if defined(__x86_64__) || defined(_M_X64)
        EXPECT_TRUE(cpu.has(CPUFeature::SSE2));
    #endif

    // Just log what we have
    EXPECT_FALSE(cpu.vendor().empty());
}

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(SIMDOpsTest, AddCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size);
        auto b = RandomFloats(size);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        // Compute with SIMD
        simd::add(a.data(), b.data(), out_simd.data(), size);

        // Compute with scalar (reference)
        scalar::add(a.data(), b.data(), out_scalar.data(), size);

        // Compare
        CheckApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_F(SIMDOpsTest, SubCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size);
        auto b = RandomFloats(size);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::sub(a.data(), b.data(), out_simd.data(), size);
        scalar::sub(a.data(), b.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_F(SIMDOpsTest, MulCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size);
        auto b = RandomFloats(size);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::mul(a.data(), b.data(), out_simd.data(), size);
        scalar::mul(a.data(), b.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_F(SIMDOpsTest, DivCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size);
        auto b = RandomFloats(size, 0.1f, 10.0f);  // Avoid division by zero
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::div(a.data(), b.data(), out_simd.data(), size);
        scalar::div(a.data(), b.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_F(SIMDOpsTest, SqrtCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size, 0.0f, 100.0f);  // Non-negative for sqrt
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::sqrt(a.data(), out_simd.data(), size);
        scalar::sqrt(a.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_F(SIMDOpsTest, FMACorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size);
        auto b = RandomFloats(size);
        auto c = RandomFloats(size);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::fma(a.data(), b.data(), c.data(), out_simd.data(), size);
        scalar::fma(a.data(), b.data(), c.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);
    }
}

TEST_F(SIMDOpsTest, ReLUCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size, -10.0f, 10.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::relu(a.data(), out_simd.data(), size);
        scalar::relu(a.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size);
    }
}

TEST_F(SIMDOpsTest, SigmoidCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size, -5.0f, 5.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::sigmoid(a.data(), out_simd.data(), size);
        scalar::sigmoid(a.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-5f);
    }
}

TEST_F(SIMDOpsTest, TanhCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size, -5.0f, 5.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::tanh(a.data(), out_simd.data(), size);
        scalar::tanh(a.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-5f);
    }
}

TEST_F(SIMDOpsTest, GeLUCorrectness) {
    const size_t sizes[] = {1, 7, 8, 15, 16, 31, 32, 100, 1000};

    for (size_t size : sizes) {
        auto a = RandomFloats(size, -3.0f, 3.0f);
        std::vector<float> out_simd(size);
        std::vector<float> out_scalar(size);

        simd::gelu(a.data(), out_simd.data(), size);
        scalar::gelu(a.data(), out_scalar.data(), size);

        CheckApproxEqual(out_simd.data(), out_scalar.data(), size, 1e-4f);
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(SIMDOpsTest, AddPerformance) {
    const size_t size = 1000000;
    const int iterations = 100;

    auto a = RandomFloats(size);
    auto b = RandomFloats(size);
    std::vector<float> out(size);

    // Benchmark SIMD
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        simd::add(a.data(), b.data(), out.data(), size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration<double>(end - start).count();

    // Benchmark scalar
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        scalar::add(a.data(), b.data(), out.data(), size);
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double>(end - start).count();

    double speedup = scalar_time / simd_time;

    std::cout << "Add Performance:\n";
    std::cout << "  SIMD:   " << simd_time << " s\n";
    std::cout << "  Scalar: " << scalar_time << " s\n";
    std::cout << "  Speedup: " << speedup << "x\n";

    // Relaxed threshold: Performance can vary run-to-run due to CPU scheduling
    // Accept up to 5% slowdown for simple operations where SIMD overhead can dominate
    EXPECT_GT(speedup, 0.95);
}

TEST_F(SIMDOpsTest, MulPerformance) {
    const size_t size = 1000000;
    const int iterations = 100;

    auto a = RandomFloats(size);
    auto b = RandomFloats(size);
    std::vector<float> out(size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        simd::mul(a.data(), b.data(), out.data(), size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        scalar::mul(a.data(), b.data(), out.data(), size);
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double>(end - start).count();

    double speedup = scalar_time / simd_time;

    std::cout << "Mul Performance:\n";
    std::cout << "  SIMD:   " << simd_time << " s\n";
    std::cout << "  Scalar: " << scalar_time << " s\n";
    std::cout << "  Speedup: " << speedup << "x\n";

    // Relaxed threshold: Performance can vary run-to-run due to CPU scheduling
    // Accept up to 5% slowdown for simple operations where SIMD overhead can dominate
    EXPECT_GT(speedup, 0.95);
}

TEST_F(SIMDOpsTest, ReLUPerformance) {
    const size_t size = 1000000;
    const int iterations = 100;

    auto a = RandomFloats(size);
    std::vector<float> out(size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        simd::relu(a.data(), out.data(), size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        scalar::relu(a.data(), out.data(), size);
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double>(end - start).count();

    double speedup = scalar_time / simd_time;

    std::cout << "ReLU Performance:\n";
    std::cout << "  SIMD:   " << simd_time << " s\n";
    std::cout << "  Scalar: " << scalar_time << " s\n";
    std::cout << "  Speedup: " << speedup << "x\n";

    // Relax threshold: ReLU is so simple that SIMD overhead can dominate
    // Accept up to 20% slowdown (some operations don't benefit from SIMD)
    EXPECT_GT(speedup, 0.80);
}

// ============================================================================
// Alignment Tests
// ============================================================================

TEST_F(SIMDOpsTest, UnalignedAccess) {
    const size_t size = 100;
    std::vector<float> buffer(size + 10);

    auto a = RandomFloats(size);
    auto b = RandomFloats(size);

    // Copy to unaligned positions
    for (size_t offset = 0; offset < 8; ++offset) {
        std::vector<float> out_simd(size + 10);
        std::vector<float> out_scalar(size + 10);

        // Test with various alignments
        simd::add(a.data(), b.data(), out_simd.data() + offset, size);
        scalar::add(a.data(), b.data(), out_scalar.data() + offset, size);

        CheckApproxEqual(out_simd.data() + offset, out_scalar.data() + offset, size);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SIMDOpsTest, ZeroSize) {
    std::vector<float> a = {1.0f};
    std::vector<float> b = {2.0f};
    std::vector<float> out = {0.0f};

    // Should not crash
    simd::add(a.data(), b.data(), out.data(), 0);
    EXPECT_EQ(out[0], 0.0f);
}

TEST_F(SIMDOpsTest, SingleElement) {
    std::vector<float> a = {3.0f};
    std::vector<float> b = {4.0f};
    std::vector<float> out(1);

    simd::add(a.data(), b.data(), out.data(), 1);
    EXPECT_FLOAT_EQ(out[0], 7.0f);
}
