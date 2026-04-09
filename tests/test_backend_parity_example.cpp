#include <gtest/gtest.h>
#include "backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// Example: Convert a CPU-only test to a multi-backend test
class MathOpsBackendTest : public BackendTest {};

TEST_P(MathOpsBackendTest, Addition) {
    // Create tensors on the parameterized device
    auto a = ones({10, 10}, DType::Float32, device);
    auto b = ones({10, 10}, DType::Float32, device);

    // Perform operation
    auto c = a + b;

    // Verify result
    auto c_cpu = c.to(Device::cpu());
    auto* data = c_cpu.data<float>();
    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(data[i], 2.0f, 1e-5f) << "Failed on " << device.to_string();
    }
}

TEST_P(MathOpsBackendTest, Subtraction) {
    auto a = full({5, 5}, 10.0f, DType::Float32, device);
    auto b = full({5, 5}, 3.0f, DType::Float32, device);

    auto c = a - b;

    auto c_cpu = c.to(Device::cpu());
    auto* data = c_cpu.data<float>();
    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(data[i], 7.0f, 1e-5f) << "Failed on " << device.to_string();
    }
}

TEST_P(MathOpsBackendTest, Multiplication) {
    auto a = full({3, 3}, 4.0f, DType::Float32, device);
    auto b = full({3, 3}, 2.5f, DType::Float32, device);

    auto c = a * b;

    auto c_cpu = c.to(Device::cpu());
    auto* data = c_cpu.data<float>();
    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(data[i], 10.0f, 1e-5f) << "Failed on " << device.to_string();
    }
}

TEST_P(MathOpsBackendTest, SliceAndCompute) {
    // Create a 2D tensor
    auto tensor = zeros({4, 4}, DType::Float32, device);
    auto tensor_cpu = tensor.to(Device::cpu());
    auto* data = tensor_cpu.data<float>();

    // Fill with test data
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            data[i * 4 + j] = static_cast<float>(i * 4 + j);
        }
    }

    // Move back to device
    tensor = tensor_cpu.to(device);

    // Slice column 1 and column 3
    auto col1 = tensor.slice(1, 1, 2);  // Column 1
    auto col3 = tensor.slice(1, 3, 4);  // Column 3

    // Compute difference (should work with .contiguous())
    auto diff = col3 - col1;

    // Verify result (col3 - col1 should be [2, 2, 2, 2])
    auto diff_cpu = diff.to(Device::cpu());
    auto* diff_data = diff_cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(diff_data[i], 2.0f, 1e-5f)
            << "Slice operation failed on " << device.to_string()
            << " at index " << i;
    }
}

// Instantiate tests for all backends
INSTANTIATE_BACKEND_TESTS(MathOpsBackendTest);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
