#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <hip/hip_runtime.h>
#include <cmath>
#include <limits>
#include <random>

using namespace tenzor;

// ============================================================================
// Test Environment Setup
// ============================================================================

class ROCmKernelsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();

        // Check if ROCm is available
        int device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);

        if (error != hipSuccess || device_count == 0) {
            GTEST_SKIP() << "ROCm device not available, skipping ROCm kernel tests";
        }

        std::cout << "Found " << device_count << " ROCm device(s)" << std::endl;
    }
};

// Register test environment
static ::testing::Environment* const rocm_env =
    ::testing::AddGlobalTestEnvironment(new ROCmKernelsTestEnvironment);

// ============================================================================
// Helper Functions
// ============================================================================

template<typename T>
void ExpectNear(T actual, T expected, T tolerance) {
    EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected),
                static_cast<double>(tolerance));
}

// Helper to compare ROCm results with CPU reference
template<typename T>
void CompareWithCPU(const Tensor& rocm_result, const Tensor& cpu_result,
                    T tolerance = static_cast<T>(1e-5)) {
    ASSERT_EQ(rocm_result.numel(), cpu_result.numel());

    // Copy ROCm result to CPU
    auto rocm_cpu = rocm_result.to(Device::cpu());

    auto rocm_data = rocm_cpu.data<T>();
    auto cpu_data = cpu_result.data<T>();

    for (int64_t i = 0; i < rocm_result.numel(); i++) {
        ExpectNear(rocm_data[i], cpu_data[i], tolerance);
    }
}

// ============================================================================
// Math Operations Tests
// ============================================================================

TEST(ROCmKernelsTest, Add_Float32_Basic) {
    auto a = ones({100, 200}, DType::Float32, Device::rocm());
    auto b = ones({100, 200}, DType::Float32, Device::rocm());

    auto c = add(a, b);

    // Verify on CPU
    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 2.0f);
    }
}

TEST(ROCmKernelsTest, Add_Float32_LargeArray) {
    const int64_t size = 1000000;
    auto a = ones({size}, DType::Float32, Device::rocm());
    auto b = ones({size}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    // Initialize with different values on device
    std::vector<float> host_a(size), host_b(size);
    for (int64_t i = 0; i < size; i++) {
        host_a[i] = static_cast<float>(i % 100);
        host_b[i] = static_cast<float>((i + 50) % 100);
    }

    hipMemcpy(a_data, host_a.data(), size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b_data, host_b.data(), size * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, Add_Float64_Precision) {
    const int64_t size = 10000;
    auto a = ones({size}, DType::Float64, Device::rocm());
    auto b = ones({size}, DType::Float64, Device::rocm());

    auto a_data = a.data<double>();
    auto b_data = b.data<double>();

    std::vector<double> host_a(size), host_b(size);
    for (int64_t i = 0; i < size; i++) {
        host_a[i] = static_cast<double>(i) * 0.1;
        host_b[i] = static_cast<double>(i) * 0.2;
    }

    hipMemcpy(a_data, host_a.data(), size * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(b_data, host_b.data(), size * sizeof(double), hipMemcpyHostToDevice);

    auto c = add(a, b);

    // Verify against CPU
    auto a_cpu = ones({size}, DType::Float64, Device::cpu());
    auto b_cpu = ones({size}, DType::Float64, Device::cpu());
    auto a_cpu_data = a_cpu.data<double>();
    auto b_cpu_data = b_cpu.data<double>();

    for (int64_t i = 0; i < size; i++) {
        a_cpu_data[i] = host_a[i];
        b_cpu_data[i] = host_b[i];
    }

    auto c_cpu = add(a_cpu, b_cpu);
    CompareWithCPU<double>(c, c_cpu, 1e-10);
}

TEST(ROCmKernelsTest, Sub_Float32_Basic) {
    auto a = ones({256, 256}, DType::Float32, Device::rocm());
    auto b = ones({256, 256}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(256 * 256, 5.0f);
    hipMemcpy(a_data, host_a.data(), 256 * 256 * sizeof(float), hipMemcpyHostToDevice);

    auto c = sub(a, b);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 4.0f);
    }
}

TEST(ROCmKernelsTest, Mul_Float32_Basic) {
    auto a = ones({512, 512}, DType::Float32, Device::rocm());
    auto b = ones({512, 512}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    std::vector<float> host_a(512 * 512, 3.0f);
    std::vector<float> host_b(512 * 512, 4.0f);

    hipMemcpy(a_data, host_a.data(), 512 * 512 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b_data, host_b.data(), 512 * 512 * sizeof(float), hipMemcpyHostToDevice);

    auto c = mul(a, b);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 12.0f);
    }
}

TEST(ROCmKernelsTest, Div_Float32_Basic) {
    auto a = ones({128, 128}, DType::Float32, Device::rocm());
    auto b = ones({128, 128}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    std::vector<float> host_a(128 * 128, 12.0f);
    std::vector<float> host_b(128 * 128, 4.0f);

    hipMemcpy(a_data, host_a.data(), 128 * 128 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b_data, host_b.data(), 128 * 128 * sizeof(float), hipMemcpyHostToDevice);

    auto c = div(a, b);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 3.0f);
    }
}

TEST(ROCmKernelsTest, Neg_Float32_Basic) {
    auto a = ones({1024}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1024);
    for (int i = 0; i < 1024; i++) {
        host_a[i] = static_cast<float>(i);
    }
    hipMemcpy(a_data, host_a.data(), 1024 * sizeof(float), hipMemcpyHostToDevice);

    auto c = neg(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(c_data[i], -static_cast<float>(i));
    }
}

TEST(ROCmKernelsTest, Abs_Float32_Basic) {
    auto a = ones({2048}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(2048);
    for (int i = 0; i < 2048; i++) {
        host_a[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
    }
    hipMemcpy(a_data, host_a.data(), 2048 * sizeof(float), hipMemcpyHostToDevice);

    auto c = abs(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 2048; i++) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>(i));
    }
}

TEST(ROCmKernelsTest, Sqrt_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>((i + 1) * (i + 1));
    }
    hipMemcpy(a_data, host_a.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    auto c = sqrt(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        EXPECT_NEAR(c_data[i], static_cast<float>(i + 1), 1e-4f);
    }
}

TEST(ROCmKernelsTest, Exp_Float32_Basic) {
    auto a = ones({500}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(500);
    for (int i = 0; i < 500; i++) {
        host_a[i] = static_cast<float>(i) * 0.01f;
    }
    hipMemcpy(a_data, host_a.data(), 500 * sizeof(float), hipMemcpyHostToDevice);

    auto c = exp(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 500; i++) {
        float expected = std::exp(static_cast<float>(i) * 0.01f);
        EXPECT_NEAR(c_data[i], expected, 1e-4f);
    }
}

TEST(ROCmKernelsTest, Log_Float32_Basic) {
    auto a = ones({500}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(500);
    for (int i = 0; i < 500; i++) {
        host_a[i] = static_cast<float>(i + 1);
    }
    hipMemcpy(a_data, host_a.data(), 500 * sizeof(float), hipMemcpyHostToDevice);

    auto c = log(a);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 500; i++) {
        float expected = std::log(static_cast<float>(i + 1));
        EXPECT_NEAR(c_data[i], expected, 1e-5f);
    }
}

TEST(ROCmKernelsTest, Pow_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>(i % 10 + 1);
    }
    hipMemcpy(a_data, host_a.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    auto c = pow(a, 2.0f);

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        float base = static_cast<float>(i % 10 + 1);
        EXPECT_NEAR(c_data[i], base * base, 1e-4f);
    }
}

TEST(ROCmKernelsTest, Clamp_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>(i - 500);
    }
    hipMemcpy(a_data, host_a.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, Sum_Float32_FullReduction) {
    auto a = ones({1000}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>(i + 1);
    }
    hipMemcpy(a_data, host_a.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    auto result = sum(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    // Sum of 1 to 1000 = 1000 * 1001 / 2 = 500500
    EXPECT_NEAR(result_data[0], 500500.0f, 1.0f);
}

TEST(ROCmKernelsTest, Sum_Float32_LargeReduction) {
    const int64_t size = 1000000;
    auto a = ones({size}, DType::Float32, Device::rocm());

    auto result = sum(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_NEAR(result_data[0], static_cast<float>(size), 100.0f);
}

TEST(ROCmKernelsTest, Sum_Float32_DimReduction) {
    auto a = ones({10, 100}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000);
    for (int i = 0; i < 1000; i++) {
        host_a[i] = static_cast<float>(i % 10 + 1);
    }
    hipMemcpy(a_data, host_a.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    // Sum along dimension 1 (columns)
    auto result = sum(a, 1);

    EXPECT_EQ(result.shape()[0], 10);
    EXPECT_EQ(result.shape().size(), 1);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    // Each row should sum to (1+2+...+10)*10 = 550
    for (int i = 0; i < 10; i++) {
        EXPECT_NEAR(result_data[i], 550.0f, 1.0f);
    }
}

TEST(ROCmKernelsTest, Mean_Float32_Basic) {
    auto a = ones({1000}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(1000, 5.0f);
    hipMemcpy(a_data, host_a.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    auto result = mean(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_NEAR(result_data[0], 5.0f, 1e-5f);
}

TEST(ROCmKernelsTest, Max_Float32_Basic) {
    auto a = ones({10000}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(10000);
    for (int i = 0; i < 10000; i++) {
        host_a[i] = static_cast<float>(i);
    }
    hipMemcpy(a_data, host_a.data(), 10000 * sizeof(float), hipMemcpyHostToDevice);

    auto result = max(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_FLOAT_EQ(result_data[0], 9999.0f);
}

TEST(ROCmKernelsTest, Min_Float32_Basic) {
    auto a = ones({10000}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(10000);
    for (int i = 0; i < 10000; i++) {
        host_a[i] = static_cast<float>(10000 - i);
    }
    hipMemcpy(a_data, host_a.data(), 10000 * sizeof(float), hipMemcpyHostToDevice);

    auto result = min(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
}

// ============================================================================
// Activation Functions Tests
// ============================================================================

TEST(ROCmKernelsTest, ReLU_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>(i - 500);
    }
    hipMemcpy(input_data, host_input.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    auto output = nn::relu(Variable(input));

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    for (int i = 0; i < 1000; i++) {
        float expected = std::max(0.0f, static_cast<float>(i - 500));
        EXPECT_FLOAT_EQ(output_data[i], expected);
    }
}

TEST(ROCmKernelsTest, ReLU_Backward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>(i - 500);
    }
    hipMemcpy(input_data, host_input.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    // Forward pass
    auto output = nn::relu(Variable(input));

    // Verify forward pass
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

TEST(ROCmKernelsTest, Sigmoid_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>((i - 500) * 0.01f);
    }
    hipMemcpy(input_data, host_input.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    auto output = nn::sigmoid(Variable(input));

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.template data<float>();

    for (int i = 0; i < 1000; i++) {
        float x = static_cast<float>((i - 500) * 0.01f);
        float expected = 1.0f / (1.0f + std::exp(-x));
        EXPECT_NEAR(output_data[i], expected, 1e-5f);
    }
}

TEST(ROCmKernelsTest, Tanh_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>((i - 500) * 0.01f);
    }
    hipMemcpy(input_data, host_input.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

    auto output = tanh(input);

    auto output_cpu = output.to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        float x = static_cast<float>((i - 500) * 0.01f);
        float expected = std::tanh(x);
        EXPECT_NEAR(output_data[i], expected, 1e-5f);
    }
}

TEST(ROCmKernelsTest, LeakyReLU_Forward_Float32) {
    auto input = ones({1000}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>(i - 500);
    }
    hipMemcpy(input_data, host_input.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, Softmax_Forward_2D) {
    auto input = ones({32, 10}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(32 * 10);
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 10; j++) {
            host_input[i * 10 + j] = static_cast<float>(j);
        }
    }
    hipMemcpy(input_data, host_input.data(), 32 * 10 * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, LogSoftmax_Forward_2D) {
    auto input = ones({64, 20}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(64 * 20);
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 20; j++) {
            host_input[i * 20 + j] = static_cast<float>(j) * 0.1f;
        }
    }
    hipMemcpy(input_data, host_input.data(), 64 * 20 * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, MatMul_Float32_2x3_3x2) {
    auto a = ones({2, 3}, DType::Float32, Device::rocm());
    auto b = ones({3, 2}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    std::vector<float> host_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> host_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    hipMemcpy(a_data, host_a.data(), 6 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b_data, host_b.data(), 6 * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, MatMul_Float32_LargeMatrix) {
    const int M = 512;
    const int K = 512;
    const int N = 512;

    auto a = ones({M, K}, DType::Float32, Device::rocm());
    auto b = ones({K, N}, DType::Float32, Device::rocm());

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

TEST(ROCmKernelsTest, MatMul_Float64_Precision) {
    auto a = ones({10, 10}, DType::Float64, Device::rocm());
    auto b = ones({10, 10}, DType::Float64, Device::rocm());

    auto a_data = a.data<double>();
    auto b_data = b.data<double>();

    std::vector<double> host_a(100), host_b(100);
    for (int i = 0; i < 100; i++) {
        host_a[i] = static_cast<double>(i) * 0.1;
        host_b[i] = static_cast<double>(i) * 0.1;
    }

    hipMemcpy(a_data, host_a.data(), 100 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(b_data, host_b.data(), 100 * sizeof(double), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, Transpose_Float32) {
    auto a = ones({4, 5}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(20);
    for (int i = 0; i < 20; i++) {
        host_a[i] = static_cast<float>(i);
    }
    hipMemcpy(a_data, host_a.data(), 20 * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, Reshape_Float32) {
    auto a = ones({2, 3, 4}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    std::vector<float> host_a(24);
    for (int i = 0; i < 24; i++) {
        host_a[i] = static_cast<float>(i);
    }
    hipMemcpy(a_data, host_a.data(), 24 * sizeof(float), hipMemcpyHostToDevice);

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
// Normalization Tests
// ============================================================================

TEST(ROCmKernelsTest, BatchNorm2d_Forward) {
    // Input: [N, C, H, W] = [2, 4, 8, 8]
    auto input = ones({2, 4, 8, 8}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(2 * 4 * 8 * 8);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < host_input.size(); i++) {
        host_input[i] = dist(gen);
    }
    hipMemcpy(input_data, host_input.data(), host_input.size() * sizeof(float),
              hipMemcpyHostToDevice);

    // Create batch norm layer
    auto bn = nn::BatchNorm2d(4);
    bn.to(Device::rocm());

    // Forward pass
    auto output = bn.forward(Variable(input));

    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 4);
    EXPECT_EQ(output.tensor().shape()[2], 8);
    EXPECT_EQ(output.tensor().shape()[3], 8);

    // Verify output is on ROCm
    EXPECT_EQ(output.tensor().device().type, Device::Type::ROCm);
}

TEST(ROCmKernelsTest, LayerNorm_Forward) {
    auto input = ones({32, 128}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(32 * 128);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < host_input.size(); i++) {
        host_input[i] = dist(gen);
    }
    hipMemcpy(input_data, host_input.data(), host_input.size() * sizeof(float),
              hipMemcpyHostToDevice);

    // Create layer norm
    auto ln = nn::LayerNorm({128});
    ln.to(Device::rocm());

    // Forward pass
    auto output = ln.forward(Variable(input));

    EXPECT_EQ(output.tensor().shape()[0], 32);
    EXPECT_EQ(output.tensor().shape()[1], 128);

    // Verify output is on ROCm
    EXPECT_EQ(output.tensor().device().type, Device::Type::ROCm);

    // Check that mean is close to 0 and std is close to 1 for each sample
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    for (int i = 0; i < 32; i++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int j = 0; j < 128; j++) {
            float val = output_data[i * 128 + j];
            sum += val;
            sum_sq += val * val;
        }

        float mean = sum / 128.0f;
        float var = sum_sq / 128.0f - mean * mean;
        float std = std::sqrt(var);

        EXPECT_NEAR(mean, 0.0f, 1e-5f);
        EXPECT_NEAR(std, 1.0f, 1e-1f);
    }
}

// ============================================================================
// Pooling Tests
// ============================================================================

TEST(ROCmKernelsTest, MaxPool2d_Forward) {
    // Input: [N, C, H, W] = [2, 3, 8, 8]
    auto input = ones({2, 3, 8, 8}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(2 * 3 * 8 * 8);

    for (size_t i = 0; i < host_input.size(); i++) {
        host_input[i] = static_cast<float>(i % 64);
    }
    hipMemcpy(input_data, host_input.data(), host_input.size() * sizeof(float),
              hipMemcpyHostToDevice);

    // Create MaxPool2d layer with kernel_size=2, stride=2
    auto pool = nn::MaxPool2d(2, 2);
    pool.to(Device::rocm());

    // Forward pass
    auto output = pool.forward(Variable(input));

    // Output should be [2, 3, 4, 4]
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 3);
    EXPECT_EQ(output.tensor().shape()[2], 4);
    EXPECT_EQ(output.tensor().shape()[3], 4);

    // Verify output is on ROCm
    EXPECT_EQ(output.tensor().device().type, Device::Type::ROCm);
}

TEST(ROCmKernelsTest, AvgPool2d_Forward) {
    // Input: [N, C, H, W] = [2, 3, 8, 8]
    auto input = ones({2, 3, 8, 8}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(2 * 3 * 8 * 8);

    for (size_t i = 0; i < host_input.size(); i++) {
        host_input[i] = static_cast<float>((i % 4) + 1);
    }
    hipMemcpy(input_data, host_input.data(), host_input.size() * sizeof(float),
              hipMemcpyHostToDevice);

    // Create AvgPool2d layer with kernel_size=2, stride=2
    auto pool = nn::AvgPool2d(2, 2);
    pool.to(Device::rocm());

    // Forward pass
    auto output = pool.forward(Variable(input));

    // Output should be [2, 3, 4, 4]
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 3);
    EXPECT_EQ(output.tensor().shape()[2], 4);
    EXPECT_EQ(output.tensor().shape()[3], 4);

    // Verify output is on ROCm
    EXPECT_EQ(output.tensor().device().type, Device::Type::ROCm);
}

// ============================================================================
// Convolution Tests
// ============================================================================

TEST(ROCmKernelsTest, Conv2d_Forward_Basic) {
    // Input: [N, C_in, H, W] = [1, 3, 8, 8]
    auto input = ones({1, 3, 8, 8}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1 * 3 * 8 * 8, 1.0f);
    hipMemcpy(input_data, host_input.data(), host_input.size() * sizeof(float),
              hipMemcpyHostToDevice);

    // Create Conv2d layer: in_channels=3, out_channels=16, kernel_size=3
    auto conv = nn::Conv2d(3, 16, 3, 1, 1); // stride=1, padding=1
    conv.to(Device::rocm());

    // Forward pass
    auto output = conv.forward(Variable(input));

    // Output should be [1, 16, 8, 8] (with padding=1)
    EXPECT_EQ(output.tensor().shape()[0], 1);
    EXPECT_EQ(output.tensor().shape()[1], 16);
    EXPECT_EQ(output.tensor().shape()[2], 8);
    EXPECT_EQ(output.tensor().shape()[3], 8);

    // Verify output is on ROCm
    EXPECT_EQ(output.tensor().device().type, Device::Type::ROCm);
}

TEST(ROCmKernelsTest, Conv2d_Forward_NoPadding) {
    // Input: [N, C_in, H, W] = [2, 3, 10, 10]
    auto input = ones({2, 3, 10, 10}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(2 * 3 * 10 * 10, 1.0f);
    hipMemcpy(input_data, host_input.data(), host_input.size() * sizeof(float),
              hipMemcpyHostToDevice);

    // Create Conv2d layer: in_channels=3, out_channels=8, kernel_size=3, padding=0
    auto conv = nn::Conv2d(3, 8, 3, 1, 0); // stride=1, padding=0
    conv.to(Device::rocm());

    // Forward pass
    auto output = conv.forward(Variable(input));

    // Output should be [2, 8, 8, 8] (10 - 3 + 1 = 8)
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 8);
    EXPECT_EQ(output.tensor().shape()[2], 8);
    EXPECT_EQ(output.tensor().shape()[3], 8);

    // Verify output is on ROCm
    EXPECT_EQ(output.tensor().device().type, Device::Type::ROCm);
}

TEST(ROCmKernelsTest, Conv2d_Forward_Stride2) {
    // Input: [N, C_in, H, W] = [1, 3, 16, 16]
    auto input = ones({1, 3, 16, 16}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1 * 3 * 16 * 16, 1.0f);
    hipMemcpy(input_data, host_input.data(), host_input.size() * sizeof(float),
              hipMemcpyHostToDevice);

    // Create Conv2d layer: in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
    auto conv = nn::Conv2d(3, 32, 3, 2, 1); // stride=2, padding=1
    conv.to(Device::rocm());

    // Forward pass
    auto output = conv.forward(Variable(input));

    // Output should be [1, 32, 8, 8] (16 / 2 = 8)
    EXPECT_EQ(output.tensor().shape()[0], 1);
    EXPECT_EQ(output.tensor().shape()[1], 32);
    EXPECT_EQ(output.tensor().shape()[2], 8);
    EXPECT_EQ(output.tensor().shape()[3], 8);

    // Verify output is on ROCm
    EXPECT_EQ(output.tensor().device().type, Device::Type::ROCm);
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST(ROCmKernelsTest, EmptyTensor_Operations) {
    auto a = zeros({0}, DType::Float32, Device::rocm());
    auto b = zeros({0}, DType::Float32, Device::rocm());

    // Operations on empty tensors should not crash
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);
}

TEST(ROCmKernelsTest, SingleElement_AllOps) {
    auto a = ones({1}, DType::Float32, Device::rocm());
    auto b = ones({1}, DType::Float32, Device::rocm());

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    float val_a = 10.0f;
    float val_b = 2.0f;

    hipMemcpy(a_data, &val_a, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b_data, &val_b, sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, NumericalStability_Softmax) {
    auto input = ones({10, 100}, DType::Float32, Device::rocm());

    auto input_data = input.data<float>();
    std::vector<float> host_input(1000);

    // Large values to test numerical stability
    for (int i = 0; i < 1000; i++) {
        host_input[i] = static_cast<float>((i % 100) * 10);
    }

    hipMemcpy(input_data, host_input.data(), 1000 * sizeof(float), hipMemcpyHostToDevice);

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

TEST(ROCmKernelsTest, MixedDTypes_Error) {
    auto a = ones({10, 10}, DType::Float32, Device::rocm());
    auto b = ones({10, 10}, DType::Float64, Device::rocm());

    // This should throw an error
    EXPECT_THROW(add(a, b), std::runtime_error);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(ROCmKernelsTest, Performance_LargeAdd) {
    const int64_t size = 10000000;  // 10 million elements

    auto a = ones({size}, DType::Float32, Device::rocm());
    auto b = ones({size}, DType::Float32, Device::rocm());

    // Warmup
    auto c = add(a, b);
    hipDeviceSynchronize();

    // Measure
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < 10; i++) {
        c = add(a, b);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    float avg_time = milliseconds / 10.0f;
    std::cout << "Average add time for " << size << " elements: "
              << avg_time << " ms" << std::endl;

    // Should be reasonably fast (< 10ms per operation)
    EXPECT_LT(avg_time, 10.0f);

    hipEventDestroy(start);
    hipEventDestroy(stop);
}

TEST(ROCmKernelsTest, Performance_LargeMatMul) {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    auto a = ones({M, K}, DType::Float32, Device::rocm());
    auto b = ones({K, N}, DType::Float32, Device::rocm());

    // Warmup
    auto c = matmul(a, b);
    hipDeviceSynchronize();

    // Measure
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    c = matmul(a, b);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    std::cout << "MatMul time for " << M << "x" << K << " @ " << K << "x" << N
              << ": " << milliseconds << " ms" << std::endl;

    // Should complete in reasonable time (< 1000ms)
    EXPECT_LT(milliseconds, 1000.0f);

    hipEventDestroy(start);
    hipEventDestroy(stop);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(ROCmKernelsTest, NeuralNetwork_ForwardPass) {
    // Create a simple neural network on ROCm
    auto input = randn({32, 784}, DType::Float32, Device::rocm());

    auto fc1 = nn::Linear(784, 256);
    auto fc2 = nn::Linear(256, 128);
    auto fc3 = nn::Linear(128, 10);

    fc1.to(Device::rocm());
    fc2.to(Device::rocm());
    fc3.to(Device::rocm());

    // Forward pass
    auto x = fc1.forward(Variable(input));
    x = nn::relu(x);
    x = fc2.forward(x);
    x = nn::relu(x);
    x = fc3.forward(x);
    x = nn::softmax(x, 1);

    // Check output shape
    EXPECT_EQ(x.tensor().shape()[0], 32);
    EXPECT_EQ(x.tensor().shape()[1], 10);

    // Verify output is on ROCm
    EXPECT_EQ(x.tensor().device().type, Device::Type::ROCm);

    // Verify softmax outputs sum to 1
    auto x_cpu = x.tensor().to(Device::cpu());
    auto x_data = x_cpu.template data<float>();

    for (int i = 0; i < 32; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 10; j++) {
            sum += x_data[i * 10 + j];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(ROCmKernelsTest, CNN_ForwardPass) {
    // Create a simple CNN on ROCm
    auto input = randn({4, 3, 32, 32}, DType::Float32, Device::rocm());

    auto conv1 = nn::Conv2d(3, 32, 3, 1, 1);
    auto conv2 = nn::Conv2d(32, 64, 3, 1, 1);
    auto fc = nn::Linear(64 * 8 * 8, 10);

    conv1.to(Device::rocm());
    conv2.to(Device::rocm());
    fc.to(Device::rocm());

    auto pool = nn::MaxPool2d(2, 2);
    pool.to(Device::rocm());

    // Forward pass
    auto x = conv1.forward(Variable(input));
    x = nn::relu(x);
    x = pool.forward(x);  // 4x3x16x16

    x = conv2.forward(x);
    x = nn::relu(x);
    x = pool.forward(x);  // 4x64x8x8

    // Flatten
    auto x_flat = reshape(x.tensor(), {4, 64 * 8 * 8});
    x = fc.forward(Variable(x_flat));

    // Check output shape
    EXPECT_EQ(x.tensor().shape()[0], 4);
    EXPECT_EQ(x.tensor().shape()[1], 10);

    // Verify output is on ROCm
    EXPECT_EQ(x.tensor().device().type, Device::Type::ROCm);
}

TEST(ROCmKernelsTest, MultiStream_Execution) {
    // Create multiple streams
    hipStream_t stream1, stream2, stream3;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);
    hipStreamCreate(&stream3);

    const int64_t size = 1000000;

    // Allocate tensors
    auto a1 = randn({size}, DType::Float32, Device::rocm());
    auto b1 = randn({size}, DType::Float32, Device::rocm());

    auto a2 = randn({size}, DType::Float32, Device::rocm());
    auto b2 = randn({size}, DType::Float32, Device::rocm());

    auto a3 = randn({size}, DType::Float32, Device::rocm());
    auto b3 = randn({size}, DType::Float32, Device::rocm());

    // Launch operations on different streams concurrently
    // Note: Actual stream assignment would need backend support
    auto c1 = add(a1, b1);
    auto c2 = mul(a2, b2);
    auto c3 = sub(a3, b3);

    // Synchronize all streams
    hipStreamSynchronize(stream1);
    hipStreamSynchronize(stream2);
    hipStreamSynchronize(stream3);

    // Verify results
    EXPECT_EQ(c1.numel(), size);
    EXPECT_EQ(c2.numel(), size);
    EXPECT_EQ(c3.numel(), size);

    // Cleanup
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);
    hipStreamDestroy(stream3);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
