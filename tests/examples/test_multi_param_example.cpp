#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;
using namespace tenzor::testing;

/**
 * @file test_multi_param_example.cpp
 * @brief Example: Testing with BOTH backend AND dtype parameterization
 *
 * Demonstrates the canonical multi-backend × multi-dtype test pattern using
 * `MultiBackendDTypeTest` from `tests/multi_backend_dtype_fixture.hpp`. This
 * fixture parameterizes a single TEST_P over the backend×dtype matrix while
 * honoring TENZOR_SKIP_BACKENDS and TENZOR_REQUIRE_MULTI_BACKEND env vars.
 *
 * Three patterns shown below:
 *   Method 1 — backend × dtype (multi-dtype) using `MultiBackendDTypeTest`.
 *   Method 2 — dtype-specific tests using `BackendTest` (backend-only).
 *   Method 3 — type conversion tests using `BackendTest` (backend-only).
 *
 * NOTE: this file used to define a private `struct BackendDTypeParam` with a
 * hand-rolled SetUp() that duplicated the canonical fixture's behavior. That
 * pattern was banned by TESTING.md ("Fixture hygiene" §) because per-file
 * fixtures don't honor the env-var contract uniformly. Use the canonical
 * fixture instead — see this file as the migration template for the other
 * 23 offenders.
 */

// ============================================================================
// Method 1: Multi-Backend × Multi-DType (canonical fixture)
// ============================================================================

class MathOpsMultiParamTest : public MultiBackendDTypeTest {};

// Test addition across all available backends and a custom dtype set.
// We override the default FLOAT_DTYPES because this test specifically wants
// integer-dtype coverage (the canonical macro covers float dtypes only).
TEST_P(MathOpsMultiParamTest, AddBasic) {
    auto a = ones({100, 100}, dtype(), device());
    auto b = ones({100, 100}, dtype(), device());
    auto c = add(a, b);

    auto c_cpu = c.to(Device::cpu());

    if (dtype() == DType::Float32) {
        const float* data = c_cpu.data<float>();
        for (int i = 0; i < 100; ++i) EXPECT_FLOAT_EQ(data[i], 2.0f);
    } else if (dtype() == DType::Float64) {
        const double* data = c_cpu.data<double>();
        for (int i = 0; i < 100; ++i) EXPECT_DOUBLE_EQ(data[i], 2.0);
    } else if (dtype() == DType::Int32) {
        const int32_t* data = c_cpu.data<int32_t>();
        for (int i = 0; i < 100; ++i) EXPECT_EQ(data[i], 2);
    } else if (dtype() == DType::Int64) {
        const int64_t* data = c_cpu.data<int64_t>();
        for (int i = 0; i < 100; ++i) EXPECT_EQ(data[i], 2);
    } else {
        FAIL() << "Unexpected dtype in MathOpsMultiParamTest.AddBasic";
    }
}

// Custom INSTANTIATE — we want integer dtypes in addition to floats, which
// the default `INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS` macro doesn't provide.
// Reuses STANDARD_BACKENDS (the same backend list every multidtype test
// uses) and BackendDTypeParamName for consistent reporting.
INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    MathOpsMultiParamTest,
    ::testing::Combine(
        STANDARD_BACKENDS,
        ::testing::Values(DType::Float32, DType::Float64,
                          DType::Int32, DType::Int64)
    ),
    BackendDTypeParamName);

// ============================================================================
// Method 2: DType-Specific Operation Tests (BackendTest fixture)
// ============================================================================

class DTypeSpecificOpsTest : public BackendTest {};

// Test operations that ONLY make sense for specific dtypes
TEST_P(DTypeSpecificOpsTest, BooleanLogic) {
    // Bool dtype for logical operations
    auto a_cpu = zeros({100}, DType::Bool, Device::cpu());
    auto b_cpu = ones({100}, DType::Bool, Device::cpu());

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = (device.type == Device::Type::CPU) ? b_cpu : b_cpu.to(device);

    // Logical OR: false | true = true
    auto c = logical_or(a, b);

    auto c_cpu = c.to(Device::cpu());
    const bool* data = c_cpu.data<bool>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_P(DTypeSpecificOpsTest, IntegerDivision) {
    // Int32 for integer division behavior
    auto a_cpu = ones({100}, DType::Int32, Device::cpu());
    auto a_data = a_cpu.data<int32_t>();

    for (int i = 0; i < 100; ++i) {
        a_data[i] = 7;
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({100}, DType::Int32, device) * 3;

    auto c = div(a, b);  // 7 / 3 = 2 (integer division)

    auto c_cpu = c.to(Device::cpu());
    const int32_t* data = c_cpu.data<int32_t>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], 2);  // NOT 2.333...
    }
}

TEST_P(DTypeSpecificOpsTest, Float16Precision) {
    // Float16 for mixed precision training - Float16 only supported on GPU backends
    if (device.type == Device::Type::CPU) {
        GTEST_SKIP() << "Float16 not commonly supported on CPU backend";
    }

    auto a = ones({100, 100}, DType::Float16, device);
    auto b = ones({100, 100}, DType::Float16, device);

    auto c = add(a, b);

    // Verify it's still Float16
    EXPECT_EQ(c.dtype(), DType::Float16);

    // Convert to Float32 for verification
    auto c_f32 = c.to(DType::Float32);
    auto c_cpu = c_f32.to(Device::cpu());
    const float* data = c_cpu.data<float>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data[i], 2.0f, 1e-3f);  // Float16 has less precision
    }
}

TEST_P(DTypeSpecificOpsTest, IntegerOverflow) {
    // Test Int8 overflow behavior
    auto a_cpu = zeros({10}, DType::Int8, Device::cpu());
    auto a_data = a_cpu.data<int8_t>();

    for (int i = 0; i < 10; ++i) {
        a_data[i] = 127;  // Max Int8 value
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({10}, DType::Int8, device);

    auto c = add(a, b);  // 127 + 1 = overflow

    auto c_cpu = c.to(Device::cpu());
    const int8_t* data = c_cpu.data<int8_t>();

    // Document overflow behavior (may be -128 or saturate to 127)
    for (int i = 0; i < 10; ++i) {
        // Platform-dependent behavior
        EXPECT_TRUE(data[i] == -128 || data[i] == 127);
    }
}

INSTANTIATE_BACKEND_TESTS(DTypeSpecificOpsTest);

// ============================================================================
// Method 3: Type Conversion Tests
// ============================================================================

class TypeConversionTest : public BackendTest {};

TEST_P(TypeConversionTest, Float32ToFloat64) {
    auto a_f32 = ones({100, 100}, DType::Float32, device);
    auto a_f64 = a_f32.to(DType::Float64);

    EXPECT_EQ(a_f64.dtype(), DType::Float64);

    auto a_cpu = a_f64.to(Device::cpu());
    const double* data = a_cpu.data<double>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 1.0);
    }
}

TEST_P(TypeConversionTest, Float32ToInt32) {
    auto a_cpu = zeros({100}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    for (int i = 0; i < 100; ++i) {
        a_data[i] = 3.7f;
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_int = a.to(DType::Int32);

    auto int_cpu = a_int.to(Device::cpu());
    const int32_t* data = int_cpu.data<int32_t>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], 3);  // Truncation, not rounding
    }
}

TEST_P(TypeConversionTest, Int32ToFloat32) {
    auto a = ones({100, 100}, DType::Int32, device) * 5;
    auto a_float = a.to(DType::Float32);

    auto a_cpu = a_float.to(Device::cpu());
    const float* data = a_cpu.data<float>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(data[i], 5.0f);
    }
}

INSTANTIATE_BACKEND_TESTS(TypeConversionTest);
