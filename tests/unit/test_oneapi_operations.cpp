/**
 * @file test_oneapi_operations.cpp
 * @brief Test OneAPI backend tensor operations
 *
 * Tests basic tensor operations on OneAPI devices:
 * 1. Tensor creation (zeros, ones, fill)
 * 2. Element-wise operations (add, mul, etc.)
 * 3. Math operations (exp, log, sqrt)
 * 4. Copy operations between host and device
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <memory>

namespace tenzor {
namespace test {

class OneAPIOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip tests if OneAPI backend is not available
        try {
            [[maybe_unused]] Device device = Device::oneapi(0);
            oneapi_available_ = true;
        } catch (const std::exception& e) {
            GTEST_SKIP() << "OneAPI backend not available: " << e.what();
        }
    }

    bool oneapi_available_ = false;
};

TEST_F(OneAPIOperationsTest, CreateZerosTensor) {
    ASSERT_TRUE(oneapi_available_);

    try {
        auto t = zeros({2, 3}, DType::Float32, Device::oneapi(0));
        auto shape = t.shape();
        EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), std::vector<int64_t>({2, 3}));
        EXPECT_EQ(t.dtype(), DType::Float32);
        EXPECT_EQ(t.device().type, Device::Type::OneAPI);

        // Copy to CPU and verify
        auto cpu_t = t.to(Device::cpu());
        auto* data = static_cast<float*>(cpu_t.data_ptr());
        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(data[i], 0.0f) << "Element " << i << " should be 0";
        }

        std::cout << "Successfully created zeros tensor on OneAPI device" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Failed to create zeros tensor: " << e.what();
    }
}

TEST_F(OneAPIOperationsTest, CreateOnesTensor) {
    ASSERT_TRUE(oneapi_available_);

    try {
        auto t = ones({3, 2}, DType::Float32, Device::oneapi(0));
        auto shape = t.shape();
        EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), std::vector<int64_t>({3, 2}));

        // Copy to CPU and verify
        auto cpu_t = t.to(Device::cpu());
        auto* data = static_cast<float*>(cpu_t.data_ptr());
        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(data[i], 1.0f) << "Element " << i << " should be 1";
        }

        std::cout << "Successfully created ones tensor on OneAPI device" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Failed to create ones tensor: " << e.what();
    }
}

TEST_F(OneAPIOperationsTest, TensorAddition) {
    ASSERT_TRUE(oneapi_available_);

    try {
        auto a = ones({2, 3}, DType::Float32, Device::oneapi(0));
        auto b = ones({2, 3}, DType::Float32, Device::oneapi(0));

        auto c = a + b;

        // Copy to CPU and verify
        auto cpu_c = c.to(Device::cpu());
        auto* data = static_cast<float*>(cpu_c.data_ptr());
        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(data[i], 2.0f) << "Element " << i << " should be 2";
        }

        std::cout << "Successfully performed addition on OneAPI device" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Failed tensor addition: " << e.what();
    }
}

TEST_F(OneAPIOperationsTest, TensorMultiplication) {
    ASSERT_TRUE(oneapi_available_);

    try {
        auto a = full({2, 2}, 3.0f, DType::Float32, Device::oneapi(0));
        auto b = full({2, 2}, 4.0f, DType::Float32, Device::oneapi(0));

        auto c = a * b;

        // Copy to CPU and verify
        auto cpu_c = c.to(Device::cpu());
        auto* data = static_cast<float*>(cpu_c.data_ptr());
        for (int i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(data[i], 12.0f) << "Element " << i << " should be 12";
        }

        std::cout << "Successfully performed multiplication on OneAPI device" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Failed tensor multiplication: " << e.what();
    }
}

TEST_F(OneAPIOperationsTest, TensorSqrt) {
    ASSERT_TRUE(oneapi_available_);

    try {
        auto a = full({3, 3}, 9.0f, DType::Float32, Device::oneapi(0));
        auto b = sqrt(a);

        // Copy to CPU and verify
        auto cpu_b = b.to(Device::cpu());
        auto* data = static_cast<float*>(cpu_b.data_ptr());
        for (int i = 0; i < 9; ++i) {
            EXPECT_NEAR(data[i], 3.0f, 1e-5f) << "Element " << i << " should be ~3";
        }

        std::cout << "Successfully performed sqrt on OneAPI device" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Failed tensor sqrt: " << e.what();
    }
}

TEST_F(OneAPIOperationsTest, HostToDeviceCopy) {
    ASSERT_TRUE(oneapi_available_);

    try {
        // Create tensor on CPU
        auto cpu_t = full({4, 4}, 42.0f, DType::Float32, Device::cpu());

        // Copy to OneAPI device
        auto oneapi_t = cpu_t.to(Device::oneapi(0));
        EXPECT_EQ(oneapi_t.device().type, Device::Type::OneAPI);

        // Copy back to CPU and verify
        auto cpu_t2 = oneapi_t.to(Device::cpu());
        auto* data1 = static_cast<float*>(cpu_t.data_ptr());
        auto* data2 = static_cast<float*>(cpu_t2.data_ptr());

        for (int i = 0; i < 16; ++i) {
            EXPECT_FLOAT_EQ(data1[i], data2[i])
                << "Element " << i << " should match after round-trip";
        }

        std::cout << "Successfully performed host<->device copy" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Failed host to device copy: " << e.what();
    }
}

TEST_F(OneAPIOperationsTest, TensorClone) {
    ASSERT_TRUE(oneapi_available_);

    try {
        auto a = full({3, 3}, 7.5f, DType::Float32, Device::oneapi(0));
        auto b = a.clone();

        EXPECT_EQ(b.device().type, Device::Type::OneAPI);
        auto shape_a = a.shape();
        auto shape_b = b.shape();
        EXPECT_EQ(std::vector<int64_t>(shape_a.begin(), shape_a.end()),
                  std::vector<int64_t>(shape_b.begin(), shape_b.end()));

        // Verify data is copied
        auto cpu_a = a.to(Device::cpu());
        auto cpu_b = b.to(Device::cpu());

        auto* data_a = static_cast<float*>(cpu_a.data_ptr());
        auto* data_b = static_cast<float*>(cpu_b.data_ptr());

        for (int i = 0; i < 9; ++i) {
            EXPECT_FLOAT_EQ(data_a[i], data_b[i])
                << "Element " << i << " should match";
        }

        std::cout << "Successfully cloned tensor on OneAPI device" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Failed tensor clone: " << e.what();
    }
}

} // namespace test
} // namespace tenzor
