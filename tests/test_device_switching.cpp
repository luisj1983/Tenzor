// Finding 40: Backend::set_device()/get_current_device() are documented as
// "GPU backends must override this" (include/tenzor/backend/backend.hpp),
// and CUDA/ROCm/MPS all do -- but Vulkan and OneAPI inherited the CPU
// backend's no-op default despite genuinely supporting multiple physical
// devices (both override device_count() independently of this). On a
// multi-GPU Vulkan or OneAPI machine, WorkerPool/DeviceGuard's per-thread
// device binding (src/core/device_guard.cpp's switch_device(), the sole
// caller of Backend::set_device()) would silently no-op and
// get_current_device() would always report 0 regardless of the requested
// index. This test exercises set_device()/get_current_device() directly on
// every available non-CPU backend so the same behavior is verified
// uniformly instead of only for CUDA/ROCm/MPS.
#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/backend/loader_fwd.hpp>
#include <tenzor/core/device_guard.hpp>

using namespace tenzor;

namespace {

auto device_available(Device::Type type) -> bool {
    try {
        Device d{type, 0};
        auto t = zeros({2, 2}, DType::Float32, d);
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

struct BackendCase {
    Device::Type type;
    const char* name;
};

}  // namespace

class BackendSetDeviceTest : public ::testing::TestWithParam<BackendCase> {};

// Every GPU backend must honor a round-trip: set_device(0) followed by
// get_current_device() must report 0. This alone would already pass for
// Vulkan/OneAPI's old CPU-inherited no-op (both default to 0), so it isn't
// sufficient by itself -- OutOfRangeRejected and, where a second physical
// device exists, MultiDeviceRoundTrip below are what actually distinguish a
// real implementation from the no-op default.
TEST_P(BackendSetDeviceTest, DeviceZeroRoundTrip) {
    const auto& tc = GetParam();
    if (!device_available(tc.type)) {
        GTEST_SKIP() << tc.name << " backend not available";
    }
    auto* backend = try_get_backend(tc.type);
    ASSERT_NE(backend, nullptr) << tc.name;
    ASSERT_GT(backend->device_count(), 0) << tc.name;

    backend->set_device(0);
    EXPECT_EQ(backend->get_current_device(), 0) << tc.name;
}

// set_device() must reject an out-of-range index rather than silently
// accepting it (the CPU-inherited no-op default silently accepted any
// value, including garbage, since it never looked at device_id at all).
TEST_P(BackendSetDeviceTest, OutOfRangeRejected) {
    const auto& tc = GetParam();
    if (!device_available(tc.type)) {
        GTEST_SKIP() << tc.name << " backend not available";
    }
    auto* backend = try_get_backend(tc.type);
    ASSERT_NE(backend, nullptr) << tc.name;

    int32_t bogus = backend->device_count() + 100;
    EXPECT_THROW(backend->set_device(bogus), std::exception) << tc.name;
}

// On a machine with >1 physical device for this backend, set_device(1) must
// actually move the "current device" -- this is the case a no-op default
// cannot fake: DeviceZeroRoundTrip alone is satisfied by "always report 0".
TEST_P(BackendSetDeviceTest, MultiDeviceRoundTripIfAvailable) {
    const auto& tc = GetParam();
    if (!device_available(tc.type)) {
        GTEST_SKIP() << tc.name << " backend not available";
    }
    auto* backend = try_get_backend(tc.type);
    ASSERT_NE(backend, nullptr) << tc.name;
    if (backend->device_count() < 2) {
        GTEST_SKIP() << tc.name << " has only " << backend->device_count()
                     << " device(s); need >=2 to exercise a real switch";
    }

    backend->set_device(1);
    EXPECT_EQ(backend->get_current_device(), 1) << tc.name;
    backend->set_device(0);
    EXPECT_EQ(backend->get_current_device(), 0) << tc.name;
}

// DeviceGuard (the RAII wrapper WorkerPool uses) must round-trip through
// the same backend-level state, restoring the previous device on scope
// exit, for every backend -- not just CUDA/ROCm/MPS.
TEST_P(BackendSetDeviceTest, DeviceGuardRestoresPreviousDevice) {
    const auto& tc = GetParam();
    if (!device_available(tc.type)) {
        GTEST_SKIP() << tc.name << " backend not available";
    }
    auto* backend = try_get_backend(tc.type);
    ASSERT_NE(backend, nullptr) << tc.name;

    backend->set_device(0);
    {
        DeviceGuard guard(Device{tc.type, 0});
        EXPECT_EQ(backend->get_current_device(), 0) << tc.name;
    }
    EXPECT_EQ(backend->get_current_device(), 0) << tc.name;
}

INSTANTIATE_TEST_SUITE_P(
    AllGpuBackends, BackendSetDeviceTest,
    ::testing::Values(
        BackendCase{Device::Type::CUDA, "cuda"},
        BackendCase{Device::Type::ROCm, "rocm"},
        BackendCase{Device::Type::Vulkan, "vulkan"},
        BackendCase{Device::Type::OneAPI, "oneapi"},
        BackendCase{Device::Type::MPS, "mps"}
    ),
    [](const ::testing::TestParamInfo<BackendCase>& info) {
        return std::string(info.param.name);
    });

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
