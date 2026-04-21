/**
 * @file test_dtype_capability_queries.cpp
 * @brief P4.3 / P4.4: verify fp8_is_native() and int4_is_native()
 *        return values that match the documented backend matrix.
 *
 * These queries exist so user code can decide whether an FP8 / int4
 * tensor op will run natively on the current device or fall through
 * a CPU round-trip. Pin the backend matrix so a future change that
 * adds native support has to also update this test (which is the
 * only signal to downstream code).
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fp8_scaling.hpp>

namespace tenzor {
namespace {

class DTypeCapabilityQueriesTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(DTypeCapabilityQueriesTest, Fp8NativeBackends) {
    // ROCm gained native FP8 Cast kernels in kernels/transform.hip.cpp;
    // Vulkan ships cast_f32_fp8e*.comp / cast_fp8e*_f32.comp shaders.
    // Only MPS still falls back via CPU round-trip.
    EXPECT_TRUE(fp8_is_native(Device::Type::CPU));
    EXPECT_TRUE(fp8_is_native(Device::Type::CUDA));
    EXPECT_TRUE(fp8_is_native(Device::Type::OneAPI));
    EXPECT_TRUE(fp8_is_native(Device::Type::ROCm));
    EXPECT_TRUE(fp8_is_native(Device::Type::Vulkan));
}

TEST_F(DTypeCapabilityQueriesTest, Fp8FallbackBackends) {
    EXPECT_FALSE(fp8_is_native(Device::Type::MPS));
}

TEST_F(DTypeCapabilityQueriesTest, Int4NativeBackends) {
    EXPECT_TRUE(int4_is_native(Device::Type::CPU));
    EXPECT_TRUE(int4_is_native(Device::Type::CUDA));
    EXPECT_TRUE(int4_is_native(Device::Type::ROCm));
}

TEST_F(DTypeCapabilityQueriesTest, Int4FallbackBackends) {
    EXPECT_FALSE(int4_is_native(Device::Type::OneAPI));
    EXPECT_FALSE(int4_is_native(Device::Type::Vulkan));
    EXPECT_FALSE(int4_is_native(Device::Type::MPS));
}

} // namespace
} // namespace tenzor
