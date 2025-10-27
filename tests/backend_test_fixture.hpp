#pragma once

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <string>

namespace tenzor {
namespace testing {

/**
 * @brief Base test fixture for backend parity testing
 *
 * Usage:
 *   TEST_P(BackendTest, MyTestName) {
 *       auto tensor = ones({10, 10}, DType::Float32, device);
 *       // ... test operations on this device ...
 *   }
 *
 * The 'device' member is automatically set based on the test parameter.
 */
class BackendTest : public ::testing::TestWithParam<std::string> {
protected:
    Device device;

    void SetUp() override {
        std::string backend_name = GetParam();

        if (backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA backend not available";
            }
            device = Device::cuda(0);
        }
        else if (backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan backend not available";
            }
            device = Device::vulkan(0);
        }
        else if (backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI backend not available";
            }
            device = Device::oneapi(0);
        }
        else {
            FAIL() << "Unknown backend: " << backend_name;
        }
    }

    // Helper to check if a backend is available
    static bool isBackendAvailable(Device::Type backend_type, int32_t index = 0) {
        try {
            Device test_device{backend_type, index};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to compare tensors across devices
    void expectTensorNear(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
        ASSERT_EQ(a.numel(), b.numel()) << "Tensors have different number of elements";

        // Move both to CPU for comparison
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());

        auto* a_data = a_cpu.data<float>();
        auto* b_data = b_cpu.data<float>();

        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            EXPECT_NEAR(a_data[i], b_data[i], tolerance)
                << "Mismatch at index " << i
                << " on device " << device.to_string();
        }
    }
};

// Instantiate tests for all available backends
#define INSTANTIATE_BACKEND_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        AllBackends, \
        TestSuiteName, \
        ::testing::Values("cpu", "cuda", "vulkan", "oneapi"), \
        [](const ::testing::TestParamInfo<std::string>& info) { \
            return info.param; \
        } \
    )

} // namespace testing
} // namespace tenzor
