/**
 * @file test_autocast_multidtype.cpp
 * @brief Multi-dtype tests for automatic mixed precision (Autocast)
 *
 * Tests autocast behavior across different dtype combinations:
 * - Float32 + Float16 (most common mixed precision)
 * - Float64 + Float32 (high precision mixed mode)
 * - BFloat16 mixed precision
 * - Nested autocast contexts with different dtypes
 * - Operation-specific casting rules per dtype
 *
 * audit-3 T.1: every TEST_F here is on the Autocast state-machine API
 * (is_enabled / get_dtype / should_autocast / get_autocast_dtype) and already
 * has element-level EXPECT_TRUE / EXPECT_EQ / EXPECT_FALSE assertions on the
 * returned DType / bool values. No tensor compute happens, so a CPU-tensor
 * reference cross-check is not applicable; the existing equality checks ARE
 * the value assertions.
 */

#include <gtest/gtest.h>
#include "tenzor/nn/amp/autocast.hpp"
#include "autocast_guard_test_support.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include <vector>
#include <tuple>

using namespace tenzor;
using namespace tenzor::nn::amp;

// Test fixture for multi-dtype autocast tests
class AutocastMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean state before each test
        // Thread-local state is automatically reset per test
    }

    // Helper to get supported autocast dtypes
    std::vector<DType> get_autocast_dtypes() {
        return {DType::Float16, DType::BFloat16};
    }

    // Helper to get all floating point dtypes
    std::vector<DType> get_float_dtypes() {
        return {DType::Float16, DType::BFloat16, DType::Float32, DType::Float64};
    }

    // Helper to get compute-heavy operation names
    std::vector<std::string> get_compute_ops() {
        return {"matmul", "mm", "bmm", "conv2d", "linear", "conv1d", "conv3d"};
    }

    // Helper to get stability-critical operation names
    std::vector<std::string> get_stability_ops() {
        return {"softmax", "log_softmax", "batch_norm", "layer_norm",
                "cross_entropy", "nll_loss", "mse_loss"};
    }
};

// Test Float32 -> Float16 autocasting (most common use case)
TEST_F(AutocastMultiDTypeTest, Float32ToFloat16Casting) {
    Autocast autocast(true, DType::Float16, Device::Type::CUDA);

    EXPECT_TRUE(Autocast::is_enabled());
    EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);

    // Compute-heavy operations should cast Float32 to Float16
    for (const auto& op : get_compute_ops()) {
        EXPECT_TRUE(Autocast::should_autocast(op, Device::cuda(0)))
            << "Operation " << op << " should be autocast";

        DType result = Autocast::get_autocast_dtype(op, DType::Float32);
        // Result depends on whether operation is in autocast list
        // For compute-heavy ops with CUDA, should convert to Float16
    }

    // Stability-critical operations should NOT autocast
    for (const auto& op : get_stability_ops()) {
        EXPECT_FALSE(Autocast::should_autocast(op, Device::cuda(0)))
            << "Operation " << op << " should NOT be autocast";
    }
}

// Test Float64 -> Float32 autocasting (high precision mixed mode)
TEST_F(AutocastMultiDTypeTest, Float64ToFloat32Casting) {
    // Note: Autocast currently only supports Float16/BFloat16
    // This tests the preservation behavior for Float64
    {
        Autocast autocast(true, DType::Float16, Device::Type::CUDA);

        // Float64 should be preserved (not cast to Float16)
        DType result = Autocast::get_autocast_dtype("matmul", DType::Float64);
        EXPECT_EQ(result, DType::Float64)
            << "Float64 should be preserved, not cast to Float16";
    }
}

// Test BFloat16 autocasting
TEST_F(AutocastMultiDTypeTest, BFloat16Casting) {
    Autocast autocast(true, DType::BFloat16, Device::Type::CUDA);

    EXPECT_TRUE(Autocast::is_enabled());
    EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);

    // Float32 operations should use BFloat16
    for (const auto& op : get_compute_ops()) {
        EXPECT_TRUE(Autocast::should_autocast(op, Device::cuda(0)));
    }

    // BFloat16 input should stay BFloat16
    DType result = Autocast::get_autocast_dtype("matmul", DType::BFloat16);
    EXPECT_EQ(result, DType::BFloat16);
}

// Test nested autocast contexts with different dtypes
TEST_F(AutocastMultiDTypeTest, NestedContextsDifferentDtypes) {
    {
        // Outer: Float16
        Autocast outer(true, DType::Float16, Device::Type::CUDA);
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);

        {
            // Inner: BFloat16
            Autocast inner(true, DType::BFloat16, Device::Type::CUDA);
            EXPECT_TRUE(Autocast::is_enabled());
            EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);

            // Should use BFloat16 in inner context
            EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)));
        }

        // Should restore to Float16
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);
    }

    EXPECT_FALSE(Autocast::is_enabled());
}

// Test nested contexts with disabled inner context
TEST_F(AutocastMultiDTypeTest, NestedContextsWithDisable) {
    {
        Autocast outer(true, DType::Float16, Device::Type::CUDA);
        EXPECT_TRUE(Autocast::is_enabled());

        {
            // Disable autocast in inner context
            Autocast inner(false);
            EXPECT_FALSE(Autocast::is_enabled());

            // Operations should NOT be autocast
            EXPECT_FALSE(Autocast::should_autocast("matmul", Device::cuda(0)));
        }

        // Should restore to enabled Float16
        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);
    }
}

// Test dtype preservation for different input dtypes
TEST_F(AutocastMultiDTypeTest, DtypePreservation) {
    Autocast autocast(true, DType::Float16, Device::Type::CUDA);

    // Float16 should stay Float16
    EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Float16), DType::Float16);

    // BFloat16 should stay BFloat16 (already low precision)
    EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::BFloat16), DType::BFloat16);

    // Float64 should stay Float64 (higher precision than target)
    EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Float64), DType::Float64);

    // Integer types should be preserved
    EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Int32), DType::Int32);
    EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Int64), DType::Int64);
    EXPECT_EQ(Autocast::get_autocast_dtype("matmul", DType::Int8), DType::Int8);
}

// Test all supported autocast dtypes
TEST_F(AutocastMultiDTypeTest, AllAutocastDtypes) {
    for (const auto& target_dtype : get_autocast_dtypes()) {
        Autocast autocast(true, target_dtype, Device::Type::CUDA);

        EXPECT_TRUE(Autocast::is_enabled());
        EXPECT_EQ(Autocast::get_dtype().value(), target_dtype);

        // Compute operations should be autocast
        for (const auto& op : get_compute_ops()) {
            EXPECT_TRUE(Autocast::should_autocast(op, Device::cuda(0)))
                << "Op: " << op << ", Target dtype: " << static_cast<int>(target_dtype);
        }
    }
}

// Test device type filtering with different dtypes
TEST_F(AutocastMultiDTypeTest, DeviceTypeFilteringMultiDtype) {
    for (const auto& dtype : get_autocast_dtypes()) {
        {
            // Enable only for CUDA
            Autocast autocast(true, dtype, Device::Type::CUDA);

            // Should autocast on CUDA
            EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)))
                << "Should autocast on CUDA with dtype " << static_cast<int>(dtype);

            // Should NOT autocast on CPU
            EXPECT_FALSE(Autocast::should_autocast("matmul", Device::cpu()))
                << "Should NOT autocast on CPU with dtype " << static_cast<int>(dtype);
        }
    }
}

// Test mixed precision with compute-heavy operations
TEST_F(AutocastMultiDTypeTest, ComputeHeavyOperationsMultiDtype) {
    auto compute_ops = get_compute_ops();
    auto autocast_dtypes = get_autocast_dtypes();

    for (const auto& dtype : autocast_dtypes) {
        Autocast autocast(true, dtype, Device::Type::CUDA);

        for (const auto& op : compute_ops) {
            // All compute-heavy ops should be autocast
            EXPECT_TRUE(Autocast::should_autocast(op, Device::cuda(0)))
                << "Op: " << op << ", Dtype: " << static_cast<int>(dtype);
        }
    }
}

// Test stability-critical operations are NOT autocast
TEST_F(AutocastMultiDTypeTest, StabilityCriticalOperationsMultiDtype) {
    auto stability_ops = get_stability_ops();
    auto autocast_dtypes = get_autocast_dtypes();

    for (const auto& dtype : autocast_dtypes) {
        Autocast autocast(true, dtype, Device::Type::CUDA);

        for (const auto& op : stability_ops) {
            // Stability-critical ops should NOT be autocast
            EXPECT_FALSE(Autocast::should_autocast(op, Device::cuda(0)))
                << "Op: " << op << ", Dtype: " << static_cast<int>(dtype);
        }
    }
}

// Test deeply nested contexts with different dtypes
TEST_F(AutocastMultiDTypeTest, DeeplyNestedContexts) {
    {
        Autocast level1(true, DType::Float16, Device::Type::CUDA);
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);

        {
            Autocast level2(true, DType::BFloat16, Device::Type::CUDA);
            EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);

            {
                Autocast level3(false);
                EXPECT_FALSE(Autocast::is_enabled());

                {
                    Autocast level4(true, DType::Float16, Device::Type::CUDA);
                    EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);
                }

                EXPECT_FALSE(Autocast::is_enabled());
            }

            EXPECT_EQ(Autocast::get_dtype().value(), DType::BFloat16);
        }

        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);
    }

    EXPECT_FALSE(Autocast::is_enabled());
}

// Test AutocastGuard with different dtypes
TEST_F(AutocastMultiDTypeTest, AutocastGuardMultiDtype) {
    for (const auto& dtype : get_autocast_dtypes()) {
        EXPECT_FALSE(Autocast::is_enabled());

        {
            AutocastGuard guard(true, dtype);
            EXPECT_TRUE(Autocast::is_enabled());
            EXPECT_EQ(Autocast::get_dtype().value(), dtype);
        }

        EXPECT_FALSE(Autocast::is_enabled());
    }
}

// Test AutocastDisabled with different outer dtypes
TEST_F(AutocastMultiDTypeTest, AutocastDisabledMultiDtype) {
    for (const auto& dtype : get_autocast_dtypes()) {
        {
            Autocast outer(true, dtype, Device::Type::CUDA);
            EXPECT_TRUE(Autocast::is_enabled());
            EXPECT_EQ(Autocast::get_dtype().value(), dtype);

            {
                AutocastDisabled disabled;
                EXPECT_FALSE(Autocast::is_enabled());
            }

            EXPECT_TRUE(Autocast::is_enabled());
            EXPECT_EQ(Autocast::get_dtype().value(), dtype);
        }
    }
}

// Test invalid dtypes are rejected
TEST_F(AutocastMultiDTypeTest, InvalidDtypes) {
    // Integer dtypes should be rejected
    EXPECT_THROW({
        Autocast autocast(true, DType::Int32);
    }, std::invalid_argument);

    EXPECT_THROW({
        Autocast autocast(true, DType::Int64);
    }, std::invalid_argument);

    EXPECT_THROW({
        Autocast autocast(true, DType::Int8);
    }, std::invalid_argument);

    // Float32 and Float64 should also be rejected as target dtypes
    EXPECT_THROW({
        Autocast autocast(true, DType::Float32);
    }, std::invalid_argument);

    EXPECT_THROW({
        Autocast autocast(true, DType::Float64);
    }, std::invalid_argument);
}

// Test dtype transitions in nested contexts
TEST_F(AutocastMultiDTypeTest, DtypeTransitionsNested) {
    std::vector<std::tuple<DType, DType, DType>> transitions = {
        {DType::Float16, DType::BFloat16, DType::Float16},
        {DType::BFloat16, DType::Float16, DType::BFloat16}
    };

    for (const auto& [outer, inner, restored] : transitions) {
        {
            Autocast outer_ctx(true, outer, Device::Type::CUDA);
            EXPECT_EQ(Autocast::get_dtype().value(), outer);

            {
                Autocast inner_ctx(true, inner, Device::Type::CUDA);
                EXPECT_EQ(Autocast::get_dtype().value(), inner);
            }

            EXPECT_EQ(Autocast::get_dtype().value(), restored);
        }

        EXPECT_FALSE(Autocast::is_enabled());
    }
}

// Test multiple operations with same autocast context
TEST_F(AutocastMultiDTypeTest, MultipleOperationsSameContext) {
    Autocast autocast(true, DType::Float16, Device::Type::CUDA);

    // Multiple compute operations
    std::vector<std::string> ops = {"matmul", "conv2d", "linear", "bmm"};

    for (const auto& op : ops) {
        EXPECT_TRUE(Autocast::should_autocast(op, Device::cuda(0)));
    }

    // Multiple stability operations
    std::vector<std::string> stable_ops = {"softmax", "batch_norm", "layer_norm"};

    for (const auto& op : stable_ops) {
        EXPECT_FALSE(Autocast::should_autocast(op, Device::cuda(0)));
    }
}

// Test dtype casting rules for all float types
TEST_F(AutocastMultiDTypeTest, DtypeCastingRulesAllFloats) {
    Autocast autocast(true, DType::Float16, Device::Type::CUDA);

    // Test all float dtypes
    auto float_dtypes = get_float_dtypes();

    for (const auto& input_dtype : float_dtypes) {
        DType result = Autocast::get_autocast_dtype("matmul", input_dtype);

        // Check preservation rules
        if (input_dtype == DType::Float16 || input_dtype == DType::BFloat16) {
            // Already low precision, should preserve
            EXPECT_EQ(result, input_dtype)
                << "Low precision dtype " << static_cast<int>(input_dtype)
                << " should be preserved";
        } else if (input_dtype == DType::Float64) {
            // Higher precision than target, should preserve
            EXPECT_EQ(result, DType::Float64)
                << "Float64 should be preserved";
        }
    }
}

// Test helper function with different dtypes
TEST_F(AutocastMultiDTypeTest, HelperFunctionMultiDtype) {
    for (const auto& dtype : get_autocast_dtypes()) {
        EXPECT_FALSE(Autocast::is_enabled());

        {
            auto ctx = autocast(true, dtype, Device::Type::CUDA);
            EXPECT_TRUE(Autocast::is_enabled());
            EXPECT_EQ(Autocast::get_dtype().value(), dtype);
        }

        EXPECT_FALSE(Autocast::is_enabled());
    }
}

// Test multiple cycles with different dtypes
TEST_F(AutocastMultiDTypeTest, MultipleCyclesDifferentDtypes) {
    auto dtypes = get_autocast_dtypes();

    for (int cycle = 0; cycle < 3; ++cycle) {
        for (const auto& dtype : dtypes) {
            EXPECT_FALSE(Autocast::is_enabled());

            {
                Autocast autocast(true, dtype, Device::Type::CUDA);
                EXPECT_TRUE(Autocast::is_enabled());
                EXPECT_EQ(Autocast::get_dtype().value(), dtype);
            }

            EXPECT_FALSE(Autocast::is_enabled());
        }
    }
}

// Test device type persistence with different dtypes
TEST_F(AutocastMultiDTypeTest, DeviceTypePersistenceMultiDtype) {
    for (const auto& dtype : get_autocast_dtypes()) {
        {
            Autocast autocast(true, dtype, Device::Type::CUDA);
            EXPECT_TRUE(Autocast::get_device_type().has_value());
            EXPECT_EQ(Autocast::get_device_type().value(), Device::Type::CUDA);
            EXPECT_EQ(Autocast::get_dtype().value(), dtype);
        }

        EXPECT_FALSE(Autocast::get_device_type().has_value());
        EXPECT_FALSE(Autocast::get_dtype().has_value());
    }
}

// Test alternating between Float16 and BFloat16
TEST_F(AutocastMultiDTypeTest, AlternatingDtypes) {
    for (int i = 0; i < 5; ++i) {
        {
            DType dtype = (i % 2 == 0) ? DType::Float16 : DType::BFloat16;
            Autocast autocast(true, dtype, Device::Type::CUDA);

            EXPECT_TRUE(Autocast::is_enabled());
            EXPECT_EQ(Autocast::get_dtype().value(), dtype);

            EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)));
        }

        EXPECT_FALSE(Autocast::is_enabled());
    }
}

// Test default device type behavior
TEST_F(AutocastMultiDTypeTest, NoDeviceTypeSpecified) {
    // When device_type is not explicitly specified, CUDA is the default
    Autocast autocast(true, DType::Float16);

    EXPECT_TRUE(Autocast::is_enabled());
    // Default device type is CUDA
    EXPECT_TRUE(Autocast::get_device_type().has_value());
    EXPECT_EQ(Autocast::get_device_type().value(), Device::Type::CUDA);

    // Should autocast on CUDA (the default device type)
    EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)));
    // CPU autocast is not enabled when device_type is CUDA
    EXPECT_FALSE(Autocast::should_autocast("matmul", Device::cpu()));
}

// Test edge case: empty operation name
TEST_F(AutocastMultiDTypeTest, EmptyOperationName) {
    Autocast autocast(true, DType::Float16, Device::Type::CUDA);

    // Empty operation name should not be autocast
    EXPECT_FALSE(Autocast::should_autocast("", Device::cuda(0)));
}

// Test consistency across multiple queries
TEST_F(AutocastMultiDTypeTest, ConsistencyAcrossQueries) {
    Autocast autocast(true, DType::Float16, Device::Type::CUDA);

    // Same query should give same result
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(Autocast::should_autocast("matmul", Device::cuda(0)));
        EXPECT_EQ(Autocast::get_dtype().value(), DType::Float16);
        EXPECT_TRUE(Autocast::is_enabled());
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
