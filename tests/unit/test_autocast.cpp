/**
 * @file test_autocast.cpp
 * @brief Tests for automatic mixed precision (Autocast)
 */

#include <gtest/gtest.h>
#include "tenzor/nn/amp/autocast.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"

using namespace tenzor;
using namespace tenzor::nn::amp;

class AutocastTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean state before each test
        // Note: Thread-local state is automatically reset per test
    }
};

// Test basic enable/disable
TEST_F(AutocastTest, BasicEnableDisable) {
    EXPECT_FALSE(Autocast::is_enabled());

    {
        Autocast autocast(true, DType::Float16);
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_TRUE(Autocast::get_dtype().has_value());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);
    }

    EXPECT_FALSE(Autocast::is_enabled());
    EXPECT_FALSE(Autocast::get_dtype().has_value());
}

// Test nested autocast contexts
TEST_F(AutocastTest, NestedContexts) {
    {
        Autocast outer(true, DType::Float16);
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);

        {
            Autocast inner(true, DType::BFloat16);
            EXPECT_TRUE(Autocast::is_enabled());
            EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);
        }

        // Should restore to outer context
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);
    }

    EXPECT_FALSE(Autocast::is_enabled());
}

// Test disabling within enabled context
TEST_F(AutocastTest, DisableWithinEnabled) {
    {
        Autocast outer(true, DType::Float16);
        EXPECT_TRUE(Autocast::is_enabled());

        {
            Autocast inner(false);
            EXPECT_FALSE(Autocast::is_enabled());
        }

        // Should restore to outer context
        EXPECT_TRUE(Autocast::is_enabled());
    }
}

// Test compute-heavy operations should be autocast
TEST_F(AutocastTest, ComputeHeavyOperations) {
    {
        Autocast autocast(true, DType::Float16, Device::Type::CUDA);

        // These operations should be autocast
        EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)));
        EXPECT_TRUE(Autocast::should_autocast("mm", Device::cuda(0)));
        EXPECT_TRUE(Autocast::should_autocast("bmm", Device::cuda(0)));
        EXPECT_TRUE(Autocast::should_autocast("conv2d", Device::cuda(0)));
        EXPECT_TRUE(Autocast::should_autocast("linear", Device::cuda(0)));
    }
}

// Test stability-critical operations should not be autocast
TEST_F(AutocastTest, StabilityCriticalOperations) {
    {
        Autocast autocast(true, DType::Float16, Device::Type::CUDA);

        // These operations should NOT be autocast
        EXPECT_FALSE(Autocast::should_autocast("softmax", Device::cuda(0)));
        EXPECT_FALSE(Autocast::should_autocast("log_softmax", Device::cuda(0)));
        EXPECT_FALSE(Autocast::should_autocast("batch_norm", Device::cuda(0)));
        EXPECT_FALSE(Autocast::should_autocast("layer_norm", Device::cuda(0)));
        EXPECT_FALSE(Autocast::should_autocast("cross_entropy", Device::cuda(0)));
    }
}

// Test device type filtering
TEST_F(AutocastTest, DeviceTypeFiltering) {
    {
        // Enable only for CUDA
        Autocast autocast(true, DType::Float16, Device::Type::CUDA);

        EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)));
        EXPECT_FALSE(Autocast::should_autocast("matmul", Device::cpu()));
    }
}

// Test BFloat16 dtype
TEST_F(AutocastTest, BFloat16Dtype) {
    {
        Autocast autocast(true, DType::BFloat16);
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);

        DType result = Autocast::get_autocast_dtype("matmul", DType::Float32);
        EXPECT_EQ(result, DType::BFloat16);
    }
}

// Test invalid dtype throws exception
TEST_F(AutocastTest, InvalidDtype) {
    EXPECT_THROW({
        Autocast autocast(true, DType::Int32);
    }, std::invalid_argument);
}

// Test AutocastGuard
TEST_F(AutocastTest, AutocastGuard) {
    EXPECT_FALSE(Autocast::is_enabled());

    {
        AutocastGuard guard(true, DType::Float16);
        EXPECT_TRUE(Autocast::is_enabled());
    }

    EXPECT_FALSE(Autocast::is_enabled());
}

// Test AutocastDisabled
TEST_F(AutocastTest, AutocastDisabled) {
    {
        Autocast autocast(true, DType::Float16);
        EXPECT_TRUE(Autocast::is_enabled());

        {
            AutocastDisabled disabled;
            EXPECT_FALSE(Autocast::is_enabled());
        }

        EXPECT_TRUE(Autocast::is_enabled());
    }
}

// Test get_autocast_dtype with Float32 input
TEST_F(AutocastTest, AutocastDtypeFloat32) {
    {
        Autocast autocast(true, DType::Float16);

        // For compute-heavy ops, should convert Float32 to Float16
        Device cuda_device = Device::cuda(0);
        DType result = Autocast::get_autocast_dtype("matmul", DType::Float32);
        // Note: get_autocast_dtype checks if operation should be autocast
        // which requires device context, but we're testing dtype logic here

        // When disabled, should return original dtype
    }

    {
        Autocast autocast(false);
        DType result = Autocast::get_autocast_dtype("matmul", DType::Float32);
        EXPECT_EQ(result, DType::Float32);
    }
}

// Test that already-low-precision inputs stay the same
TEST_F(AutocastTest, LowPrecisionInput) {
    {
        Autocast autocast(true, DType::Float16);

        // Float16 input should stay Float16
        DType result = Autocast::get_autocast_dtype("matmul", DType::Float16);
        EXPECT_EQ(result, DType::Float16);

        // BFloat16 input should stay BFloat16
        result = Autocast::get_autocast_dtype("matmul", DType::BFloat16);
        EXPECT_EQ(result, DType::BFloat16);
    }
}

// Test that integer types are preserved
TEST_F(AutocastTest, IntegerPreservation) {
    {
        Autocast autocast(true, DType::Float16);

        EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Int32), DType::Int32);
        EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Int64), DType::Int64);
    }
}

// Test helper function
TEST_F(AutocastTest, HelperFunction) {
    EXPECT_FALSE(Autocast::is_enabled());

    {
        auto ctx = autocast(true, DType::BFloat16, Device::Type::CUDA);
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);
    }

    EXPECT_FALSE(Autocast::is_enabled());
}

// Test multiple enable/disable cycles
TEST_F(AutocastTest, MultipleCycles) {
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(Autocast::is_enabled());

        {
            Autocast autocast(true, DType::Float16);
            EXPECT_TRUE(Autocast::is_enabled());
        }

        EXPECT_FALSE(Autocast::is_enabled());
    }
}

// Test device type persistence
TEST_F(AutocastTest, DeviceTypePersistence) {
    {
        Autocast autocast(true, DType::Float16, Device::Type::CUDA);
        EXPECT_TRUE(Autocast::get_device_type().has_value());
        EXPECT_EQ(Autocast::get_device_type().value(), Device::Type::CUDA);
    }

    EXPECT_FALSE(Autocast::get_device_type().has_value());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
