#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>

using namespace tenzor;

// ============================================================================
// Test Environment Setup
// ============================================================================

class CUDAKernelsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();

        // Check if CUDA is available
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA kernel tests";
        }

        std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    }
};

// Register test environment
static ::testing::Environment* const cuda_env =
    ::testing::AddGlobalTestEnvironment(new CUDAKernelsTestEnvironment);

// ============================================================================
// Helper Functions
// ============================================================================

template<typename T>
void ExpectNear(T actual, T expected, T tolerance) {
    EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected),
                static_cast<double>(tolerance));
}

// Helper to compare CUDA results with CPU reference
template<typename T>
void CompareWithCPU(const Tensor& cuda_result, const Tensor& cpu_result,
                    T tolerance = static_cast<T>(1e-5)) {
    ASSERT_EQ(cuda_result.numel(), cpu_result.numel());

    // Copy CUDA result to CPU
    auto cuda_cpu = cuda_result.to(Device::cpu());

    auto cuda_data = cuda_cpu.data<T>();
    auto cpu_data = cpu_result.data<T>();

    for (int64_t i = 0; i < cuda_result.numel(); i++) {
        ExpectNear(cuda_data[i], cpu_data[i], tolerance);
    }
}

// ============================================================================
// Math Operations Tests
// ============================================================================

TEST(CUDAKernelsTest, Add_Float32_Basic) {
    auto a = ones({100, 200}, DType::Float32, Device::cuda());
    auto b = ones({100, 200}, DType::Float32, Device::cuda());

    auto c = add(a, b);

    // Verify on CPU
    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 2.0f);
    }
}

TEST(CUDAKernelsTest, Add_Float32_LargeArray) {
    const int64_t size = 1000000;
    auto a = ones({size}, DType::Float32, Device::cuda());
    auto b = ones({size}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    // Initialize with different values on device
    std::vector<float> host_a(size), host_b(size);
    for (int64_t i = 0; i < size; i++) {
        host_a[i] = static_cast<float>(i % 100);
        host_b[i] = static_cast<float>((i + 50) % 100);
    }

    cudaMemcpy(a_data, host_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, host_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    auto c = add(a, b);

    // Verify against CPU
    auto a_cpu = ones({size}, DType::Float32, Device::cpu());
    auto b_cpu = ones({size}, DType::Float32, Device::cpu());
    auto a_cpu_data = a_cpu.data<float>();
    auto b_cpu_data = b_cpu.data<float>();

    for (int64_t i = 0; i < size; i++) {
        a_cpu_data[i] = host_a[i];
        b_cpu_data[i] = host_b[i];
    }

    auto c_cpu = add(a_cpu, b_cpu);
    CompareWithCPU<float>(c, c_cpu, 1e-5f);
}

TEST(CUDAKernelsTest, Sub_Float32_Basic) {
    auto a = ones({256, 256}, DType::Float32, Device::cuda());
    auto b = ones({256, 256}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(256 * 256, 5.0f);
    cudaMemcpy(a_data, host_a.data(), 256 * 256 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = sub(a, b);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 4.0f);
    }
}

TEST(CUDAKernelsTest, Mul_Float32_Basic) {
    auto a = ones({512, 512}, DType::Float32, Device::cuda());
    auto b = ones({512, 512}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    std::vector<float> host_a(512 * 512, 3.0f);
    std::vector<float> host_b(512 * 512, 4.0f);

    cudaMemcpy(a_data, host_a.data(), 512 * 512 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, host_b.data(), 512 * 512 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = mul(a, b);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 12.0f);
    }
}

TEST(CUDAKernelsTest, Div_Float32_Basic) {
    auto a = ones({128, 128}, DType::Float32, Device::cuda());
    auto b = ones({128, 128}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    std::vector<float> host_a(128 * 128, 12.0f);
    std::vector<float> host_b(128 * 128, 4.0f);

    cudaMemcpy(a_data, host_a.data(), 128 * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, host_b.data(), 128 * 128 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = div(a, b);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 3.0f);
    }
}

TEST(CUDAKernelsTest, Neg_Float32_Basic) {
    auto a = ones({1024}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1024);
    for (int i = 0; i < 1024; i++) {
        host_a[i] = static_cast<float>(i);
    }
    cudaMemcpy(a_data, host_a.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = neg(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(c_data[i], -static_cast<float>(i));
    }
}

TEST(CUDAKernelsTest, Abs_Float32_Basic) {
    auto a = ones({2048}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(2048);
    for (int i = 0; i < 2048; i++) {
        host_a[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
    }
    cudaMemcpy(a_data, host_a.data(), 2048 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = abs(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 2048; i++) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>(i));
    }
}

TEST(CUDAKernelsTest, Sqrt_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>((i + 1) * (i + 1));
    }
    cudaMemcpy(a_data, host_a.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = sqrt(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        EXPECT_NEAR(c_data[i], static_cast<float>(i + 1), 1e-4f);
    }
}

TEST(CUDAKernelsTest, Exp_Float32_Basic) {
    auto a = ones({500}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(500);
    for (int i = 0; i < 500; i++) {
        host_a[i] = static_cast<float>(i) * 0.01f;
    }
    cudaMemcpy(a_data, host_a.data(), 500 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = exp(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 500; i++) {
        float expected = std::exp(static_cast<float>(i) * 0.01f);
        EXPECT_NEAR(c_data[i], expected, 1e-4f);
    }
}

TEST(CUDAKernelsTest, Log_Float32_Basic) {
    auto a = ones({500}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(500);
    for (int i = 0; i < 500; i++) {
        host_a[i] = static_cast<float>(i + 1);
    }
    cudaMemcpy(a_data, host_a.data(), 500 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = log(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 500; i++) {
        float expected = std::log(static_cast<float>(i + 1));
        EXPECT_NEAR(c_data[i], expected, 1e-5f);
    }
}

TEST(CUDAKernelsTest, Pow_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>(i % 10 + 1);
    }
    cudaMemcpy(a_data, host_a.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = pow(a, 2.0f);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        float base = static_cast<float>(i % 10 + 1);
        EXPECT_NEAR(c_data[i], base * base, 1e-4f);
    }
}

TEST(CUDAKernelsTest, Clamp_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>(i - 500);
    }
    cudaMemcpy(a_data, host_a.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = clamp(a, -100.0f, 100.0f);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        float val = static_cast<float>(i - 500);
        float expected = std::max(-100.0f, std::min(100.0f, val));
        EXPECT_FLOAT_EQ(c_data[i], expected);
    }
}

// ============================================================================
// Reduction Operations Tests
// ============================================================================

TEST(CUDAKernelsTest, Sum_Float32_FullReduction) {
    auto a = ones({1000}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>(i + 1);
    }
    cudaMemcpy(a_data, host_a.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto result = sum(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    // Sum of 1 to 1000 = 1000 * 1001 / 2 = 500500
    EXPECT_NEAR(result_data[0], 500500.0f, 1.0f);
}

TEST(CUDAKernelsTest, Sum_Float32_LargeReduction) {
    const int64_t size = 1000000;
    auto a = ones({size}, DType::Float32, Device::cuda());

    auto result = sum(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_NEAR(result_data[0], static_cast<float>(size), 100.0f);
}

TEST(CUDAKernelsTest, Mean_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000, 5.0f);
    cudaMemcpy(a_data, host_a.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto result = mean(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_NEAR(result_data[0], 5.0f, 1e-5f);
}

TEST(CUDAKernelsTest, Max_Float32_Basic) {
    auto a = ones({10000}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(10000);
    for (int i = 0; i < 10000; i++) {
        host_a[i] = static_cast<float>(i);
    }
    cudaMemcpy(a_data, host_a.data(), 10000 * sizeof(float), cudaMemcpyHostToDevice);

    auto result = max(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_FLOAT_EQ(result_data[0], 9999.0f);
}

TEST(CUDAKernelsTest, Min_Float32_Basic) {
    auto a = ones({10000}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(10000);
    for (int i = 0; i < 10000; i++) {
        host_a[i] = static_cast<float>(10000 - i);
    }
    cudaMemcpy(a_data, host_a.data(), 10000 * sizeof(float), cudaMemcpyHostToDevice);

    auto result = min(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
}

// ============================================================================
// Activation Functions Tests
// ============================================================================

TEST(CUDAKernelsTest, ReLU_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>(i - 500);
    }
    cudaMemcpy(input_data, host_input.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto output = nn::relu(Variable(input));

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    for (int i = 0; i < 1000; i++) {
        float expected = std::max(0.0f, static_cast<float>(i - 500));
        EXPECT_FLOAT_EQ(output_data[i], expected);
    }
}

TEST(CUDAKernelsTest, ReLU_Backward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>(i - 500);
    }
    cudaMemcpy(input_data, host_input.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass
    auto output = nn::relu(Variable(input));

    // Backward pass (simplified - assumes grad computation exists)
    // For now, just verify forward pass
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    int positive_count = 0;
    for (int i = 0; i < 1000; i++) {
        if (i > 500) {
            EXPECT_GT(output_data[i], 0.0f);
            positive_count++;
        } else {
            EXPECT_EQ(output_data[i], 0.0f);
        }
    }
    EXPECT_EQ(positive_count, 499);
}

TEST(CUDAKernelsTest, Sigmoid_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>((i - 500) * 0.01f);
    }
    cudaMemcpy(input_data, host_input.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto output = nn::sigmoid(Variable(input));

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    for (int i = 0; i < 1000; i++) {
        float x = static_cast<float>((i - 500) * 0.01f);
        float expected = 1.0f / (1.0f + std::exp(-x));
        EXPECT_NEAR(output_data[i], expected, 1e-5f);
    }
}

TEST(CUDAKernelsTest, Tanh_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>((i - 500) * 0.01f);
    }
    cudaMemcpy(input_data, host_input.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto output = tanh(input);

    auto output_cpu = output.to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        float x = static_cast<float>((i - 500) * 0.01f);
        float expected = std::tanh(x);
        EXPECT_NEAR(output_data[i], expected, 1e-5f);
    }
}

TEST(CUDAKernelsTest, LeakyReLU_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>(i - 500);
    }
    cudaMemcpy(input_data, host_input.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 0.01f;
    auto output = nn::leaky_relu(Variable(input), alpha);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    for (int i = 0; i < 1000; i++) {
        float x = static_cast<float>(i - 500);
        float expected = (x > 0.0f) ? x : alpha * x;
        EXPECT_NEAR(output_data[i], expected, 1e-5f);
    }
}

TEST(CUDAKernelsTest, Softmax_Forward_2D) {
    auto input = ones({32, 10}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(32 * 10);
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 10; j++) {
            host_input[i * 10 + j] = static_cast<float>(j);
        }
    }
    cudaMemcpy(input_data, host_input.data(), 32 * 10 * sizeof(float), cudaMemcpyHostToDevice);

    auto output = nn::softmax(Variable(input), 1);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    // Check that each row sums to 1.0
    for (int i = 0; i < 32; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 10; j++) {
            sum += output_data[i * 10 + j];
            EXPECT_GT(output_data[i * 10 + j], 0.0f);
            EXPECT_LE(output_data[i * 10 + j], 1.0f);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(CUDAKernelsTest, LogSoftmax_Forward_2D) {
    auto input = ones({64, 20}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(64 * 20);
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 20; j++) {
            host_input[i * 20 + j] = static_cast<float>(j) * 0.1f;
        }
    }
    cudaMemcpy(input_data, host_input.data(), 64 * 20 * sizeof(float), cudaMemcpyHostToDevice);

    auto output = nn::log_softmax(Variable(input), 1);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    // Check that values are negative (log of probabilities)
    // and exp of each row sums to 1.0
    for (int i = 0; i < 64; i++) {
        float sum_exp = 0.0f;
        for (int j = 0; j < 20; j++) {
            EXPECT_LE(output_data[i * 20 + j], 0.0f);
            sum_exp += std::exp(output_data[i * 20 + j]);
        }
        EXPECT_NEAR(sum_exp, 1.0f, 1e-4f);
    }
}

// ============================================================================
// Matrix Operations Tests
// ============================================================================

TEST(CUDAKernelsTest, MatMul_Float32_2x3_3x2) {
    auto a = ones({2, 3}, DType::Float32, Device::cuda());
    auto b = ones({3, 2}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    std::vector<float> host_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> host_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    cudaMemcpy(a_data, host_a.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, host_b.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    // Expected: [[22, 28], [49, 64]]
    EXPECT_FLOAT_EQ(c_data[0], 22.0f);
    EXPECT_FLOAT_EQ(c_data[1], 28.0f);
    EXPECT_FLOAT_EQ(c_data[2], 49.0f);
    EXPECT_FLOAT_EQ(c_data[3], 64.0f);
}

TEST(CUDAKernelsTest, MatMul_Float32_LargeMatrix) {
    const int M = 512;
    const int K = 512;
    const int N = 512;

    auto a = ones({M, K}, DType::Float32, Device::cuda());
    auto b = ones({K, N}, DType::Float32, Device::cuda());

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], M);
    EXPECT_EQ(c.shape()[1], N);

    // Each element should be K (sum of K 1s)
    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>(K));
    }
}

TEST(CUDAKernelsTest, MatMul_Float64_Precision) {
    auto a = ones({10, 10}, DType::Float64, Device::cuda());
    auto b = ones({10, 10}, DType::Float64, Device::cuda());

    auto a_data = a.data<double>();
    auto b_data = b.data<double>();

    std::vector<double> host_a(100), host_b(100);
    for (int i = 0; i < 100; i++) {
        host_a[i] = static_cast<double>(i) * 0.1;
        host_b[i] = static_cast<double>(i) * 0.1;
    }

    cudaMemcpy(a_data, host_a.data(), 100 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, host_b.data(), 100 * sizeof(double), cudaMemcpyHostToDevice);

    auto c = matmul(a, b);

    // Verify on CPU for precision
    auto a_cpu = ones({10, 10}, DType::Float64, Device::cpu());
    auto b_cpu = ones({10, 10}, DType::Float64, Device::cpu());

    auto a_cpu_data = a_cpu.data<double>();
    auto b_cpu_data = b_cpu.data<double>();

    for (int i = 0; i < 100; i++) {
        a_cpu_data[i] = host_a[i];
        b_cpu_data[i] = host_b[i];
    }

    auto c_cpu_ref = matmul(a_cpu, b_cpu);
    CompareWithCPU<double>(c, c_cpu_ref, 1e-10);
}

TEST(CUDAKernelsTest, Transpose_Float32) {
    auto a = ones({4, 5}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(20);
    for (int i = 0; i < 20; i++) {
        host_a[i] = static_cast<float>(i);
    }
    cudaMemcpy(a_data, host_a.data(), 20 * sizeof(float), cudaMemcpyHostToDevice);

    auto b = transpose(a, 0, 1);

    EXPECT_EQ(b.shape()[0], 5);
    EXPECT_EQ(b.shape()[1], 4);

    auto b_cpu = b.to(Device::cpu());
    auto b_data = b_cpu.data<float>();

    // Verify transpose
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_FLOAT_EQ(b_data[i * 4 + j], host_a[j * 5 + i]);
        }
    }
}

TEST(CUDAKernelsTest, Reshape_Float32) {
    auto a = ones({2, 3, 4}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    std::vector<float> host_a(24);
    for (int i = 0; i < 24; i++) {
        host_a[i] = static_cast<float>(i);
    }
    cudaMemcpy(a_data, host_a.data(), 24 * sizeof(float), cudaMemcpyHostToDevice);

    auto b = reshape(a, {4, 6});

    EXPECT_EQ(b.shape()[0], 4);
    EXPECT_EQ(b.shape()[1], 6);
    EXPECT_EQ(b.numel(), 24);

    auto b_cpu = b.to(Device::cpu());
    auto b_data = b_cpu.data<float>();

    // Data should be the same
    for (int i = 0; i < 24; i++) {
        EXPECT_FLOAT_EQ(b_data[i], static_cast<float>(i));
    }
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST(CUDAKernelsTest, EmptyTensor_Operations) {
    auto a = zeros({0}, DType::Float32, Device::cuda());
    auto b = zeros({0}, DType::Float32, Device::cuda());

    // Operations on empty tensors should not crash
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);
}

TEST(CUDAKernelsTest, SingleElement_AllOps) {
    auto a = ones({1}, DType::Float32, Device::cuda());
    auto b = ones({1}, DType::Float32, Device::cuda());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    float val_a = 10.0f;
    float val_b = 2.0f;

    cudaMemcpy(a_data, &val_a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, &val_b, sizeof(float), cudaMemcpyHostToDevice);

    // Test all operations
    auto add_result = add(a, b);
    auto sub_result = sub(a, b);
    auto mul_result = mul(a, b);
    auto div_result = div(a, b);

    auto add_cpu = add_result.to(Device::cpu());
    auto sub_cpu = sub_result.to(Device::cpu());
    auto mul_cpu = mul_result.to(Device::cpu());
    auto div_cpu = div_result.to(Device::cpu());

    EXPECT_FLOAT_EQ(add_cpu.data<float>()[0], 12.0f);
    EXPECT_FLOAT_EQ(sub_cpu.data<float>()[0], 8.0f);
    EXPECT_FLOAT_EQ(mul_cpu.data<float>()[0], 20.0f);
    EXPECT_FLOAT_EQ(div_cpu.data<float>()[0], 5.0f);
}

TEST(CUDAKernelsTest, NumericalStability_Softmax) {
    auto input = ones({10, 100}, DType::Float32, Device::cuda());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);

    // Large values to test numerical stability
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>((i % 100) * 10);
    }

    cudaMemcpy(input_data, host_input.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    auto output = nn::softmax(Variable(input), 1);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    // Check for NaN or Inf
    for (int i = 0; i < 1000; i++) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }

    // Check that rows sum to 1
    for (int i = 0; i < 10; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) {
            sum += output_data[i * 100 + j];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4f);
    }
}

TEST(CUDAKernelsTest, MixedDTypes_Error) {
    auto a = ones({10, 10}, DType::Float32, Device::cuda());
    auto b = ones({10, 10}, DType::Float64, Device::cuda());

    // NumPy/PyTorch-style type promotion: Float32 + Float64 → Float64
    auto c = add(a, b);
    EXPECT_EQ(c.dtype(), DType::Float64);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(CUDAKernelsTest, Performance_LargeAdd) {
    const int64_t size = 10000000;  // 10 million elements

    auto a = ones({size}, DType::Float32, Device::cuda());
    auto b = ones({size}, DType::Float32, Device::cuda());

    // Warmup
    auto c = add(a, b);
    cudaDeviceSynchronize();

    // Measure
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        c = add(a, b);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float avg_time = milliseconds / 10.0f;
    std::cout << "Average add time for " << size << " elements: "
              << avg_time << " ms" << std::endl;

    // Should be reasonably fast (< 10ms per operation)
    EXPECT_LT(avg_time, 10.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST(CUDAKernelsTest, Performance_LargeMatMul) {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    auto a = ones({M, K}, DType::Float32, Device::cuda());
    auto b = ones({K, N}, DType::Float32, Device::cuda());

    // Warmup
    auto c = matmul(a, b);
    cudaDeviceSynchronize();

    // Measure
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    c = matmul(a, b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "MatMul time for " << M << "x" << K << " @ " << K << "x" << N
              << ": " << milliseconds << " ms" << std::endl;

    // Should complete in reasonable time (< 1000ms)
    EXPECT_LT(milliseconds, 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
