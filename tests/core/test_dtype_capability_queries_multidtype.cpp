// Multi-backend multi-dtype tests for fp8_is_native() and int4_is_native()
// capability queries. Verifies backend matrix consistency across devices.

#include "../multi_backend_dtype_fixture.hpp"

#include <tenzor/ops/fp8_scaling.hpp>

namespace tenzor {
namespace testing {

class DTypeCapabilityQueriesMultiDTypeTest : public MultiBackendDTypeTest {};

// ---------------------------------------------------------------------------
// fp8_is_native() results must be consistent regardless of which device
// we are currently testing on.
// ---------------------------------------------------------------------------

TEST_P(DTypeCapabilityQueriesMultiDTypeTest, Fp8NativeBackendsConsistent) {
    // These are backend-wide queries (not per-device), so results should
    // be the same regardless of the current test device.
    EXPECT_TRUE(fp8_is_native(Device::Type::CPU));
    EXPECT_TRUE(fp8_is_native(Device::Type::CUDA));
    EXPECT_TRUE(fp8_is_native(Device::Type::OneAPI));
}

TEST_P(DTypeCapabilityQueriesMultiDTypeTest, Fp8FallbackBackendsConsistent) {
    EXPECT_FALSE(fp8_is_native(Device::Type::ROCm));
    EXPECT_FALSE(fp8_is_native(Device::Type::Vulkan));
    EXPECT_FALSE(fp8_is_native(Device::Type::MPS));
}

TEST_P(DTypeCapabilityQueriesMultiDTypeTest, Int4NativeBackendsConsistent) {
    EXPECT_TRUE(int4_is_native(Device::Type::CPU));
    EXPECT_TRUE(int4_is_native(Device::Type::CUDA));
    EXPECT_TRUE(int4_is_native(Device::Type::ROCm));
}

TEST_P(DTypeCapabilityQueriesMultiDTypeTest, Int4FallbackBackendsConsistent) {
    EXPECT_FALSE(int4_is_native(Device::Type::OneAPI));
    EXPECT_FALSE(int4_is_native(Device::Type::Vulkan));
    EXPECT_FALSE(int4_is_native(Device::Type::MPS));
}

// ---------------------------------------------------------------------------
// Verify that the current test device can create tensors of the test dtype
// ---------------------------------------------------------------------------

TEST_P(DTypeCapabilityQueriesMultiDTypeTest, DeviceSupportsDType) {
    // All backends should support Float32/Float64/Float16
    auto t = tenzor::zeros({2, 2}, dtype(), device());
    EXPECT_EQ(t.dtype(), dtype());
    EXPECT_EQ(t.device().type, device().type);
}

// ---------------------------------------------------------------------------
// Query the current device type and verify fp8/int4 results match known matrix
// ---------------------------------------------------------------------------

TEST_P(DTypeCapabilityQueriesMultiDTypeTest, CurrentDeviceFp8Query) {
    bool native = fp8_is_native(device().type);
    if (device().type == Device::Type::CPU ||
        device().type == Device::Type::CUDA ||
        device().type == Device::Type::OneAPI) {
        EXPECT_TRUE(native) << "Expected fp8 native for " << backend_name();
    } else {
        EXPECT_FALSE(native) << "Expected fp8 fallback for " << backend_name();
    }
}

TEST_P(DTypeCapabilityQueriesMultiDTypeTest, CurrentDeviceInt4Query) {
    bool native = int4_is_native(device().type);
    if (device().type == Device::Type::CPU ||
        device().type == Device::Type::CUDA ||
        device().type == Device::Type::ROCm) {
        EXPECT_TRUE(native) << "Expected int4 native for " << backend_name();
    } else {
        EXPECT_FALSE(native) << "Expected int4 fallback for " << backend_name();
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DTypeCapabilityQueriesMultiDTypeTest);

} // namespace testing
} // namespace tenzor
