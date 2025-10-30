#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

static bool isCudaAvailable() {
    try {
        Device test_device = Device::cuda(0);
        auto t = zeros({2}, DType::Float32, test_device);
        return true;
    } catch (...) {
        return false;
    }
}

TEST(CUDAScalarDebug, BasicSum) {
    if (!isCudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Device device = Device::cuda(0);

    // Create a simple tensor with known values on CUDA
    auto data = full({4}, 2.0f, DType::Float32, device);  // [2, 2, 2, 2]

    // Print the input to verify it's correct
    auto data_cpu = data.to(Device::cpu());
    const float* data_ptr = data_cpu.data<float>();
    std::cout << "Input tensor: [";
    for (int64_t i = 0; i < data_cpu.numel(); i++) {
        std::cout << data_ptr[i];
        if (i < data_cpu.numel() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Perform full reduction on CUDA
    auto result = sum(data);  // Should be scalar with value 8.0

    std::cout << "Result shape: {";
    auto shape = result.shape();
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << "Result numel: " << result.numel() << std::endl;
    std::cout << "Result device: " << result.device().to_string() << std::endl;

    // Transfer to CPU
    auto result_cpu = result.to(Device::cpu());
    std::cout << "After transfer - Result device: " << result_cpu.device().to_string() << std::endl;

    // Access the scalar value
    const float* result_data = result_cpu.data<float>();
    std::cout << "Scalar value: " << result_data[0] << std::endl;

    EXPECT_FLOAT_EQ(result_data[0], 8.0f);
}

TEST(CUDAScalarDebug, MeanReduction) {
    if (!isCudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Device device = Device::cuda(0);

    // Create tensor [1, 1, 1, 1]
    auto data = ones({4}, DType::Float32, device);

    // Mean should be 1.0
    auto result = mean(data);

    std::cout << "Mean result shape: {";
    auto shape = result.shape();
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

    auto result_cpu = result.to(Device::cpu());
    const float* result_data = result_cpu.data<float>();
    std::cout << "Mean value: " << result_data[0] << std::endl;

    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
}
