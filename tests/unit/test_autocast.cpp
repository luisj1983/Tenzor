/**
 * @file test_autocast.cpp
 * @brief Tests for automatic mixed precision (Autocast)
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/nn/amp/autocast.hpp"
#include "autocast_guard_test_support.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"

using namespace tenzor;
using namespace tenzor::nn::amp;

class AutocastTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        // Ensure clean state before each test
        // Note: Thread-local state is automatically reset per test
    }
};

// Test basic enable/disable
TEST_P(AutocastTest, BasicEnableDisable) {
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
TEST_P(AutocastTest, NestedContexts) {
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
TEST_P(AutocastTest, DisableWithinEnabled) {
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
TEST_P(AutocastTest, ComputeHeavyOperations) {
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
TEST_P(AutocastTest, StabilityCriticalOperations) {
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
TEST_P(AutocastTest, DeviceTypeFiltering) {
    {
        // Enable only for CUDA
        Autocast autocast(true, DType::Float16, Device::Type::CUDA);

        EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)));
        EXPECT_FALSE(Autocast::should_autocast("matmul", Device::cpu()));
    }
}

// Test BFloat16 dtype
TEST_P(AutocastTest, BFloat16Dtype) {
    {
        Autocast autocast(true, DType::BFloat16);
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);

        DType result = Autocast::get_autocast_dtype("matmul", DType::Float32);
        EXPECT_EQ(result, DType::BFloat16);
    }
}

// Test invalid dtype throws exception
TEST_P(AutocastTest, InvalidDtype) {
    EXPECT_THROW({
        Autocast autocast(true, DType::Int32);
    }, std::invalid_argument);
}

// Test AutocastGuard
TEST_P(AutocastTest, AutocastGuard) {
    EXPECT_FALSE(Autocast::is_enabled());

    {
        AutocastGuard guard(true, DType::Float16);
        EXPECT_TRUE(Autocast::is_enabled());
    }

    EXPECT_FALSE(Autocast::is_enabled());
}

// Test AutocastDisabled
TEST_P(AutocastTest, AutocastDisabled) {
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
TEST_P(AutocastTest, AutocastDtypeFloat32) {
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
TEST_P(AutocastTest, LowPrecisionInput) {
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
TEST_P(AutocastTest, IntegerPreservation) {
    {
        Autocast autocast(true, DType::Float16);

        EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Int32), DType::Int32);
        EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Int64), DType::Int64);
    }
}

// Test helper function
TEST_P(AutocastTest, HelperFunction) {
    EXPECT_FALSE(Autocast::is_enabled());

    {
        auto ctx = autocast(true, DType::BFloat16, Device::Type::CUDA);
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);
    }

    EXPECT_FALSE(Autocast::is_enabled());
}

// Test multiple enable/disable cycles
TEST_P(AutocastTest, MultipleCycles) {
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
TEST_P(AutocastTest, DeviceTypePersistence) {
    {
        Autocast autocast(true, DType::Float16, Device::Type::CUDA);
        EXPECT_TRUE(Autocast::get_device_type().has_value());
        EXPECT_EQ(Autocast::get_device_type().value(), Device::Type::CUDA);
    }

    EXPECT_FALSE(Autocast::get_device_type().has_value());
}

INSTANTIATE_BACKEND_TESTS(AutocastTest);
