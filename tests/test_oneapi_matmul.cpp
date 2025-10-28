#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <iostream>

using namespace tenzor;

/**
 * @file test_oneapi_matmul.cpp
 * @brief Standalone test for OneAPI MatMul operation
 *
 * This test verifies that the MatMul operation works correctly with oneMKL
 * integration and tests various matrix sizes.
 */

// Helper to check if OneAPI is available
bool is_oneapi_available() {
    try {
        tenzor::initialize();
        auto device = Device::oneapi(0);
        auto test_tensor = ones({2, 2}, DType::Float32, device);
        return true;
    } catch (...) {
        return false;
    }
}

class OneAPIMatMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!is_oneapi_available()) {
            GTEST_SKIP() << "OneAPI backend not available";
        }
        device_ = Device::oneapi(0);
        tenzor::initialize();
    }

    Device device_;
};

// Test square matrix multiplication
TEST_F(OneAPIMatMulTest, SquareMatrixMultiplication) {
    // Create 3x3 matrices
    auto a = randn({3, 3}, DType::Float32, device_);
    auto b = randn({3, 3}, DType::Float32, device_);

    // Compute matmul on OneAPI
    auto c_oneapi = matmul(a, b);

    // Copy to CPU and compute reference
    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto c_cpu = matmul(a_cpu, b_cpu);

    // Copy OneAPI result to CPU for comparison
    auto c_oneapi_cpu = c_oneapi.to(Device::cpu());

    // Compare results
    auto shape_oneapi = c_oneapi_cpu.shape();
    auto shape_cpu = c_cpu.shape();
    ASSERT_EQ(shape_oneapi.size(), shape_cpu.size());
    for (size_t i = 0; i < shape_cpu.size(); ++i) {
        ASSERT_EQ(shape_oneapi[i], shape_cpu[i]);
    }

    const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
    const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());

    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(c_oneapi_data[i], c_cpu_data[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    std::cout << "Square 3x3 MatMul: PASSED" << std::endl;
}

// Test rectangular matrix multiplication
TEST_F(OneAPIMatMulTest, RectangularMatrixMultiplication) {
    // Create matrices: A (4x5), B (5x3), C (4x3)
    auto a = randn({4, 5}, DType::Float32, device_);
    auto b = randn({5, 3}, DType::Float32, device_);

    auto c_oneapi = matmul(a, b);

    // Reference on CPU
    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto c_cpu = matmul(a_cpu, b_cpu);

    auto c_oneapi_cpu = c_oneapi.to(Device::cpu());

    auto shape_oneapi = c_oneapi_cpu.shape(); auto shape_cpu = c_cpu.shape(); ASSERT_EQ(shape_oneapi.size(), shape_cpu.size());
    EXPECT_EQ(c_oneapi_cpu.shape()[0], 4);
    EXPECT_EQ(c_oneapi_cpu.shape()[1], 3);

    const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
    const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());

    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(c_oneapi_data[i], c_cpu_data[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    std::cout << "Rectangular 4x5 @ 5x3 MatMul: PASSED" << std::endl;
}

// Test larger matrix multiplication
TEST_F(OneAPIMatMulTest, LargeMatrixMultiplication) {
    // Create matrices: A (64x128), B (128x32), C (64x32)
    auto a = randn({64, 128}, DType::Float32, device_);
    auto b = randn({128, 32}, DType::Float32, device_);

    auto c_oneapi = matmul(a, b);

    // Reference on CPU
    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto c_cpu = matmul(a_cpu, b_cpu);

    auto c_oneapi_cpu = c_oneapi.to(Device::cpu());

    auto shape_oneapi = c_oneapi_cpu.shape(); auto shape_cpu = c_cpu.shape(); ASSERT_EQ(shape_oneapi.size(), shape_cpu.size());
    EXPECT_EQ(c_oneapi_cpu.shape()[0], 64);
    EXPECT_EQ(c_oneapi_cpu.shape()[1], 32);

    const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
    const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());

    // Use slightly larger tolerance for accumulation errors
    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        float rel_error = std::abs(c_oneapi_data[i] - c_cpu_data[i]) /
                         (std::abs(c_cpu_data[i]) + 1e-8f);
        EXPECT_LT(rel_error, 1e-3f)
            << "Mismatch at index " << i
            << ": OneAPI=" << c_oneapi_data[i]
            << ", CPU=" << c_cpu_data[i];
    }

    std::cout << "Large 64x128 @ 128x32 MatMul: PASSED" << std::endl;
}

// Test vector-matrix multiplication
TEST_F(OneAPIMatMulTest, VectorMatrixMultiplication) {
    // Create vector (1x4) and matrix (4x3)
    auto a = randn({1, 4}, DType::Float32, device_);
    auto b = randn({4, 3}, DType::Float32, device_);

    auto c_oneapi = matmul(a, b);

    // Reference on CPU
    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto c_cpu = matmul(a_cpu, b_cpu);

    auto c_oneapi_cpu = c_oneapi.to(Device::cpu());

    auto shape_oneapi = c_oneapi_cpu.shape(); auto shape_cpu = c_cpu.shape(); ASSERT_EQ(shape_oneapi.size(), shape_cpu.size());
    EXPECT_EQ(c_oneapi_cpu.shape()[0], 1);
    EXPECT_EQ(c_oneapi_cpu.shape()[1], 3);

    const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
    const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());

    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(c_oneapi_data[i], c_cpu_data[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    std::cout << "Vector 1x4 @ Matrix 4x3 MatMul: PASSED" << std::endl;
}

// Test matrix-vector multiplication
TEST_F(OneAPIMatMulTest, MatrixVectorMultiplication) {
    // Create matrix (3x4) and vector (4x1)
    auto a = randn({3, 4}, DType::Float32, device_);
    auto b = randn({4, 1}, DType::Float32, device_);

    auto c_oneapi = matmul(a, b);

    // Reference on CPU
    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto c_cpu = matmul(a_cpu, b_cpu);

    auto c_oneapi_cpu = c_oneapi.to(Device::cpu());

    auto shape_oneapi = c_oneapi_cpu.shape(); auto shape_cpu = c_cpu.shape(); ASSERT_EQ(shape_oneapi.size(), shape_cpu.size());
    EXPECT_EQ(c_oneapi_cpu.shape()[0], 3);
    EXPECT_EQ(c_oneapi_cpu.shape()[1], 1);

    const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
    const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());

    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(c_oneapi_data[i], c_cpu_data[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    std::cout << "Matrix 3x4 @ Vector 4x1 MatMul: PASSED" << std::endl;
}

// Test Double precision
TEST_F(OneAPIMatMulTest, Float64MatrixMultiplication) {
    // Create 3x3 double matrices
    auto a = randn({3, 3}, DType::Float64, device_);
    auto b = randn({3, 3}, DType::Float64, device_);

    auto c_oneapi = matmul(a, b);

    // Reference on CPU
    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto c_cpu = matmul(a_cpu, b_cpu);

    auto c_oneapi_cpu = c_oneapi.to(Device::cpu());

    auto shape_oneapi = c_oneapi_cpu.shape(); auto shape_cpu = c_cpu.shape(); ASSERT_EQ(shape_oneapi.size(), shape_cpu.size());

    const double* c_oneapi_data = static_cast<const double*>(c_oneapi_cpu.data_ptr());
    const double* c_cpu_data = static_cast<const double*>(c_cpu.data_ptr());

    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(c_oneapi_data[i], c_cpu_data[i], 1e-10)
            << "Mismatch at index " << i;
    }

    std::cout << "Float64 3x3 MatMul: PASSED" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
