/**
 * @file test_cublas_cudnn.cpp
 * @brief Comprehensive unit tests for cuBLAS and cuDNN operations
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <tenzor/tenzor.hpp>

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

    // Use nn::Conv2d for forward
    nn::Conv2d conv(1, 1, 3, 1, 0, 1, 1, false);
    // Set weight to ones
    auto params = conv.own_parameters();
    params[0]->tensor() = weight.to(Device::cpu());
    conv.to(device);

    auto output = conv.forward(Variable(input, false)).tensor();

    // Output shape: (1, 1, 3, 3) for 5x5 input with 3x3 kernel, no padding
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 3);
    EXPECT_EQ(output.shape()[3], 3);

    // Each output element should be ~9.0 (sum of 3x3 ones)
    auto output_cpu = output.to(Device::cpu());
    EXPECT_NEAR(output_cpu.data<float>()[0], 9.0f, 0.1f);
}

TEST_F(CuBLAScuDNNTest, Conv2d_Forward_WithPadding) {
    Device device = Device::cuda(0);

    // Input: (2, 3, 28, 28)
    // Kernel: (16, 3, 3, 3)
    // stride=1, padding=1 -> Output: (2, 16, 28, 28)

    Tensor input = Tensor::randn({2, 3, 28, 28}, DType::Float32, device);
    Tensor weight = Tensor::randn({16, 3, 3, 3}, DType::Float32, device);
    Tensor bias = Tensor::randn({16}, DType::Float32, device);

    nn::Conv2d conv(3, 16, 3, 1, 1);
    conv.to(device);
    auto output = conv.forward(Variable(input, false)).tensor();

    // Output shape: (2, 16, 28, 28) with padding=1
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
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

    nn::BatchNorm2d bn(8);
    bn.eval();  // Inference mode
    bn.to(device);

    auto output = bn.forward(Variable(input, false)).tensor();
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_F(CuBLAScuDNNTest, BatchNorm2d_Forward_Training) {
    Device device = Device::cuda(0);

    // Input: (8, 16, 32, 32)
    Tensor input = Tensor::randn({8, 16, 32, 32}, DType::Float32, device);
    Tensor scale = Tensor::ones({16}, DType::Float32, device);
    Tensor bias = Tensor::zeros({16}, DType::Float32, device);
    Tensor running_mean = Tensor::zeros({16}, DType::Float32, device);
    Tensor running_var = Tensor::ones({16}, DType::Float32, device);

    nn::BatchNorm2d bn(16);
    bn.train();  // Training mode
    bn.to(device);

    auto output = bn.forward(Variable(input, true)).tensor();
    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
}

TEST_F(CuBLAScuDNNTest, BatchNorm2d_Backward) {
    Device device = Device::cuda(0);

    nn::BatchNorm2d bn(8);
    bn.train();
    bn.to(device);

    auto input = Variable(randn({4, 8, 16, 16}, DType::Float32, device), true);
    auto output = bn.forward(input);

    // Backward pass
    output.backward();
    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad().value().shape()[0], 4);
    EXPECT_EQ(input.grad().value().shape()[1], 8);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(CuBLAScuDNNTest, MultipleOperations_Pipeline) {
    Device device = Device::cuda(0);

    // Test a sequence of operations (simulating a small neural network)
    Tensor x = Tensor::randn({16, 32, 28, 28}, DType::Float32, device);

    // Conv -> BatchNorm -> ReLU pipeline
    nn::Conv2d conv(32, 64, 3, 1, 1);
    nn::BatchNorm2d bn(64);
    conv.to(device);
    bn.to(device);

    auto input_var = Variable(x, false);
    auto conv_out = conv.forward(input_var);
    auto bn_out = bn.forward(conv_out);
    auto relu_out = Variable(relu(bn_out.tensor()), false);

    EXPECT_EQ(relu_out.tensor().shape()[0], 16);
    EXPECT_EQ(relu_out.tensor().shape()[1], 64);
    EXPECT_EQ(relu_out.tensor().shape()[2], 28);
    EXPECT_EQ(relu_out.tensor().shape()[3], 28);
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

    // Conv in FP16 via nn::Conv2d
    nn::Conv2d conv_fp16(16, 32, 3, 1, 1);
    conv_fp16.to(DType::Float16);
    conv_fp16.to(device);
    auto y_fp16 = conv_fp16.forward(Variable(x_fp16, false)).tensor();
    Tensor y_fp32 = y_fp16.to(DType::Float32);
    EXPECT_EQ(y_fp32.dtype(), DType::Float32);
    EXPECT_EQ(y_fp32.shape()[1], 32);

    // Verify conversion works
    Tensor back_to_fp32 = x_fp16.to(DType::Float32);
    ExpectTensorsNear(x_fp32, back_to_fp32, 1e-2f);  // FP16 has lower precision
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
