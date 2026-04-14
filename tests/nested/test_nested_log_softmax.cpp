/**
 * @file test_nested_log_softmax.cpp
 * @brief Backend parity tests for NestedLogSoftmax
 *
 * Verifies that NestedLogSoftmax produces identical results across
 * all GPU backends compared to the CPU reference implementation.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nested/nested_tensor.hpp>
#include <tenzor/nested/nested_ops.hpp>
#include <tenzor/ops/creation.hpp>

using namespace tenzor;

class NestedLogSoftmaxParityTest : public tenzor::testing::BackendTest {};

TEST_P(NestedLogSoftmaxParityTest, RaggedDimParity) {
    // Test log-softmax along the ragged dimension (dim=0).
    // This exercises the NestedLogSoftmax kernel directly.
    std::vector<Tensor> segments = {
        randn({3, 8}, DType::Float32, Device::cpu()),
        randn({5, 8}, DType::Float32, Device::cpu()),
        randn({2, 8}, DType::Float32, Device::cpu()),
    };
    auto nested_cpu = NestedTensor::from_tensor_list(segments);
    auto result_cpu = nested_log_softmax(nested_cpu, 0);
    Tensor cpu_values = result_cpu.values();

    if (device.type == Device::Type::CPU) {
        EXPECT_EQ(cpu_values.shape()[0], 10);
        EXPECT_EQ(cpu_values.shape()[1], 8);
        return;
    }

    auto nested_gpu = nested_cpu.to(device);
    auto result_gpu = nested_log_softmax(nested_gpu, 0);
    Tensor gpu_values = result_gpu.values().to(Device::cpu());

    ASSERT_EQ(cpu_values.shape()[0], gpu_values.shape()[0]);
    ASSERT_EQ(cpu_values.shape()[1], gpu_values.shape()[1]);

    const float* cp = cpu_values.template data<float>();
    const float* gp = gpu_values.template data<float>();
    int64_t n = cpu_values.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(cp[i], gp[i], 1e-4f)
            << "Mismatch at index " << i << " on " << device.to_string();
    }
}

TEST_P(NestedLogSoftmaxParityTest, SingleSegmentRaggedDim) {
    std::vector<Tensor> segments = {
        randn({4, 16}, DType::Float32, Device::cpu()),
    };
    auto nested_cpu = NestedTensor::from_tensor_list(segments);
    auto result_cpu = nested_log_softmax(nested_cpu, 0);

    if (device.type == Device::Type::CPU) {
        EXPECT_EQ(result_cpu.values().shape()[0], 4);
        return;
    }

    auto nested_gpu = nested_cpu.to(device);
    auto result_gpu = nested_log_softmax(nested_gpu, 0);
    Tensor gpu_values = result_gpu.values().to(Device::cpu());
    Tensor cpu_values = result_cpu.values();

    const float* cp = cpu_values.template data<float>();
    const float* gp = gpu_values.template data<float>();
    for (int64_t i = 0; i < cpu_values.numel(); ++i) {
        EXPECT_NEAR(cp[i], gp[i], 1e-4f)
            << "Mismatch at index " << i << " on " << device.to_string();
    }
}

TEST_P(NestedLogSoftmaxParityTest, OutputValuesNonPositive) {
    std::vector<Tensor> segments = {
        randn({6, 4}, DType::Float32, Device::cpu()),
    };
    auto nested = NestedTensor::from_tensor_list(segments);
    if (device.type != Device::Type::CPU) {
        nested = nested.to(device);
    }

    auto result = nested_log_softmax(nested, 0);
    Tensor values = result.values().to(Device::cpu());
    const float* vp = values.template data<float>();

    for (int64_t i = 0; i < values.numel(); ++i) {
        EXPECT_LE(vp[i], 1e-6f)
            << "Log-softmax value > 0 at index " << i << " on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(NestedLogSoftmaxParityTest);
