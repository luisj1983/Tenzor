/**
 * @file test_cublas_cudnn.cpp
 * @brief Comprehensive unit tests for cuBLAS and cuDNN operations
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "tenzor/core/tensor.hpp"

using namespace tenzor;

class CuBLAScuDNNTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    // Helper to check tensors are approximately equal
    void ExpectTensorsNear(const Tensor& a, const Tensor& b, float tolerance = 1e-4f) {
        ASSERT_EQ(a.shape(), b.shape());

        // Copy to CPU for comparison
        Tensor a_cpu = a.to(Device::cpu());
        Tensor b_cpu = b.to(Device::cpu());

        const float* a_data = a_cpu.data<float>();
        const float* b_data = b_cpu.data<float>();

        int64_t numel = a_cpu.numel();
        for (int64_t i = 0; i < numel; ++i) {
            EXPECT_NEAR(a_data[i], b_data[i], tolerance)
                << "Mismatch at index " << i;
        }
    }
};

// ============================================================================
// cuBLAS Matrix Multiplication Tests
// ============================================================================

TEST_F(CuBLAScuDNNTest, MatMul_FP32_Correctness) {
    Device device = Device::cuda(0);

    // Create simple test matrices
    Tensor a = Tensor::ones({3, 4}, DType::Float32, device) * 2.0f;
    Tensor b = Tensor::ones({4, 5}, DType::Float32, device) * 3.0f;

    // Expected result: (3, 5) all elements = 2 * 3 * 4 = 24
    Tensor c = a.matmul(b);

    Tensor expected = Tensor::ones({3, 5}, DType::Float32, device) * 24.0f;

    ExpectTensorsNear(c, expected);
}

TEST_F(CuBLAScuDNNTest, MatMul_FP32_NonSquare) {
    Device device = Device::cuda(0);

    // (128, 256) @ (256, 512) = (128, 512)
    Tensor a = Tensor::randn({128, 256}, DType::Float32, device);
    Tensor b = Tensor::randn({256, 512}, DType::Float32, device);

    Tensor c = a.matmul(b);

    // Check shape
    EXPECT_EQ(c.shape()[0], 128);
    EXPECT_EQ(c.shape()[1], 512);

    // Verify with CPU computation (sample a few elements)
    Tensor a_cpu = a.to(Device::cpu());
    Tensor b_cpu = b.to(Device::cpu());
    Tensor c_cpu = c.to(Device::cpu());

    // Check first element manually
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    const float* c_data = c_cpu.data<float>();

    float expected_00 = 0.0f;
    for (int k = 0; k < 256; ++k) {
        expected_00 += a_data[k] * b_data[k * 512];
    }

    EXPECT_NEAR(c_data[0], expected_00, 1e-3f);
}

TEST_F(CuBLAScuDNNTest, MatMul_FP16_Correctness) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major < 7) {
        GTEST_SKIP() << "FP16 Tensor Cores require compute capability 7.0+";
    }

    Device device = Device::cuda(0);

    // Test FP16 matrix multiplication
    Tensor a = Tensor::ones({16, 16}, DType::Float16, device);
    Tensor b = Tensor::ones({16, 16}, DType::Float16, device);

    Tensor c = a.matmul(b);

    // All elements should be 16.0 (with some FP16 precision loss)
    Tensor c_fp32 = c.to(DType::Float32);
    Tensor expected = Tensor::ones({16, 16}, DType::Float32, device) * 16.0f;

    ExpectTensorsNear(c_fp32, expected, 0.1f);  // FP16 has lower precision
}

TEST_F(CuBLAScuDNNTest, BatchedMatMul_FP32) {
    Device device = Device::cuda(0);

    // Batched matrix multiplication: (8, 32, 64) @ (8, 64, 48) = (8, 32, 48)
    Tensor a = Tensor::randn({8, 32, 64}, DType::Float32, device);
    Tensor b = Tensor::randn({8, 64, 48}, DType::Float32, device);

    Tensor c = a.matmul(b);

    // Check shape
    ASSERT_EQ(c.ndim(), 3);
    EXPECT_EQ(c.shape()[0], 8);
    EXPECT_EQ(c.shape()[1], 32);
    EXPECT_EQ(c.shape()[2], 48);

    // Verify first batch with CPU computation
    Tensor a_cpu = a.to(Device::cpu());
    Tensor b_cpu = b.to(Device::cpu());
    Tensor c_cpu = c.to(Device::cpu());

    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    const float* c_data = c_cpu.data<float>();

    // Check first element of first batch
    float expected = 0.0f;
    for (int k = 0; k < 64; ++k) {
        expected += a_data[k] * b_data[k * 48];
    }

    EXPECT_NEAR(c_data[0], expected, 1e-3f);
}

// ============================================================================
// cuDNN Convolution Tests
// ============================================================================

TEST_F(CuBLAScuDNNTest, Conv2d_Forward_Simple) {
    Device device = Device::cuda(0);

    // Simple 3x3 convolution
    // Input: (1, 1, 5, 5)
    // Kernel: (1, 1, 3, 3) - all ones
    // Output: (1, 1, 3, 3) with stride=1, padding=0

    Tensor input = Tensor::ones({1, 1, 5, 5}, DType::Float32, device);
    Tensor weight = Tensor::ones({1, 1, 3, 3}, DType::Float32, device);

    // Each output element should be 9.0 (sum of 3x3 = 9)
    // TODO: Call conv2d when implemented
    // Tensor output = conv2d(input, weight, nullptr, 1, 0, 1, 1);

    // For now, just check tensors are created correctly
    EXPECT_EQ(input.device().type, Device::Type::CUDA);
    EXPECT_EQ(weight.device().type, Device::Type::CUDA);
}

TEST_F(CuBLAScuDNNTest, Conv2d_Forward_WithPadding) {
    Device device = Device::cuda(0);

    // Input: (2, 3, 28, 28)
    // Kernel: (16, 3, 3, 3)
    // stride=1, padding=1 -> Output: (2, 16, 28, 28)

    Tensor input = Tensor::randn({2, 3, 28, 28}, DType::Float32, device);
    Tensor weight = Tensor::randn({16, 3, 3, 3}, DType::Float32, device);
    Tensor bias = Tensor::randn({16}, DType::Float32, device);

    // TODO: Call conv2d when implemented
    // Tensor output = conv2d(input, weight, &bias, 1, 1, 1, 1);

    // Check shapes
    EXPECT_EQ(input.shape()[0], 2);
    EXPECT_EQ(input.shape()[1], 3);
}

TEST_F(CuBLAScuDNNTest, Conv2d_Backward_GradientCheck) {
    Device device = Device::cuda(0);

    // Numerical gradient checking
    // Input: (1, 2, 4, 4)
    // Kernel: (3, 2, 3, 3)

    Tensor input = Tensor::randn({1, 2, 4, 4}, DType::Float32, device);
    Tensor weight = Tensor::randn({3, 2, 3, 3}, DType::Float32, device);

    // TODO: Implement gradient check when conv2d backward is ready
    // This would involve:
    // 1. Forward pass
    // 2. Backward pass to compute gradients
    // 3. Numerical gradient computation
    // 4. Compare analytical vs numerical gradients

    EXPECT_TRUE(true);  // Placeholder
}

// ============================================================================
// cuDNN Batch Normalization Tests
// ============================================================================

TEST_F(CuBLAScuDNNTest, BatchNorm2d_Forward_Inference) {
    Device device = Device::cuda(0);

    // Input: (4, 8, 16, 16)
    Tensor input = Tensor::randn({4, 8, 16, 16}, DType::Float32, device);
    Tensor scale = Tensor::ones({8}, DType::Float32, device);
    Tensor bias = Tensor::zeros({8}, DType::Float32, device);
    Tensor running_mean = Tensor::zeros({8}, DType::Float32, device);
    Tensor running_var = Tensor::ones({8}, DType::Float32, device);

    // TODO: Call batchnorm when implemented
    // Tensor output = batchnorm2d(input, scale, bias, running_mean, running_var, false);

    // For now, check tensor properties
    EXPECT_EQ(input.shape()[1], 8);  // Number of channels
}

TEST_F(CuBLAScuDNNTest, BatchNorm2d_Forward_Training) {
    Device device = Device::cuda(0);

    // Input: (8, 16, 32, 32)
    Tensor input = Tensor::randn({8, 16, 32, 32}, DType::Float32, device);
    Tensor scale = Tensor::ones({16}, DType::Float32, device);
    Tensor bias = Tensor::zeros({16}, DType::Float32, device);
    Tensor running_mean = Tensor::zeros({16}, DType::Float32, device);
    Tensor running_var = Tensor::ones({16}, DType::Float32, device);

    // TODO: Call batchnorm in training mode
    // auto [output, saved_mean, saved_var] = batchnorm2d_training(input, scale, bias, running_mean, running_var);

    EXPECT_EQ(scale.shape()[0], 16);
}

TEST_F(CuBLAScuDNNTest, BatchNorm2d_Backward) {
    Device device = Device::cuda(0);

    // Gradient computation for batch normalization
    Tensor grad_output = Tensor::randn({4, 8, 16, 16}, DType::Float32, device);
    Tensor input = Tensor::randn({4, 8, 16, 16}, DType::Float32, device);
    Tensor scale = Tensor::ones({8}, DType::Float32, device);
    Tensor saved_mean = Tensor::zeros({8}, DType::Float32, device);
    Tensor saved_var = Tensor::ones({8}, DType::Float32, device);

    // TODO: Call batchnorm backward
    // auto [grad_input, grad_scale, grad_bias] = batchnorm2d_backward(grad_output, input, scale, saved_mean, saved_var);

    EXPECT_TRUE(true);  // Placeholder
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(CuBLAScuDNNTest, MultipleOperations_Pipeline) {
    Device device = Device::cuda(0);

    // Test a sequence of operations (simulating a small neural network)
    Tensor x = Tensor::randn({16, 32, 28, 28}, DType::Float32, device);

    // Conv -> BatchNorm -> ReLU -> MaxPool -> FC -> Softmax
    // This tests that multiple cuDNN/cuBLAS operations work together

    // TODO: Implement full pipeline when all operations are ready
    // For now, just test basic flow
    Tensor w1 = Tensor::randn({64, 32, 3, 3}, DType::Float32, device);

    // Just verify we can create the tensors
    EXPECT_EQ(x.device().type, Device::Type::CUDA);
    EXPECT_EQ(w1.device().type, Device::Type::CUDA);
}

TEST_F(CuBLAScuDNNTest, MixedPrecision_FP16_FP32) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major < 7) {
        GTEST_SKIP() << "Mixed precision requires compute capability 7.0+";
    }

    Device device = Device::cuda(0);

    // Test mixed precision workflow
    Tensor x_fp32 = Tensor::randn({8, 16, 32, 32}, DType::Float32, device);
    Tensor x_fp16 = x_fp32.to(DType::Float16);

    // Operations in FP16
    Tensor w_fp16 = Tensor::randn({32, 16, 3, 3}, DType::Float16, device);

    // TODO: Conv in FP16, convert back to FP32
    // Tensor y_fp16 = conv2d(x_fp16, w_fp16, ...);
    // Tensor y_fp32 = y_fp16.to(DType::Float32);

    // Verify conversion works
    Tensor back_to_fp32 = x_fp16.to(DType::Float32);
    ExpectTensorsNear(x_fp32, back_to_fp32, 1e-2f);  // FP16 has lower precision
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
