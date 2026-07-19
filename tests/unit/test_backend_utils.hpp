#pragma once
/**
 * @file test_backend_utils.hpp
 * @brief Shared utilities for backend availability checks in tests
 *
 * This header provides safe backend availability checking that avoids
 * crashes on misconfigured systems (especially ROCm).
 */

#include <tenzor/tenzor.hpp>
#include <string>
#include <vector>
#include <cstdlib>

namespace tenzor::test {

struct BackendConfig {
    std::string name;
    Device::Type type;
    int device_id;
    bool is_available;

    std::string ToString() const {
        return name + "_" + std::to_string(device_id);
    }
};

inline void PrintTo(const BackendConfig& param, std::ostream* os) {
    *os << param.ToString();
}

/**
 * @brief Check if a backend is available for testing
 *
 * For ROCm, uses a safer check that doesn't launch kernels because
 * some ROCm configurations may crash on kernel launch. ROCm tests
 * are opt-in via TENZOR_TEST_ROCM=1 environment variable.
 */
inline bool is_backend_available(Device::Type type) {
    try {
        auto backend = backend_registry().get_backend(type);
        if (!backend) {
            return false;
        }

        // Check if the backend reports having devices
        if (backend->device_count() == 0) {
            return false;
        }

        // For ROCm, check environment variable to opt-in (avoids crashes on broken setups)
        if (type == Device::Type::ROCm) {
            const char* enable_rocm = std::getenv("TENZOR_TEST_ROCM");
            if (!enable_rocm || std::string(enable_rocm) != "1") {
                return false;  // Skip ROCm tests unless explicitly enabled
            }
            return true;  // Trust the device count, don't launch kernels
        }

        // For OneAPI, trust the device count — SYCL queue creation can segfault
        // during GTest's static initialization (test discovery phase)
        if (type == Device::Type::OneAPI) {
            return true;
        }

        // For CUDA and other backends, verify with actual tensor creation
        auto device = Device(type, 0);
        auto test_tensor = ones({2, 2}, DType::Float32, device);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Like is_backend_available() but without the TENZOR_TEST_ROCM env-var
 * gate. Useful for the small number of tests that need to know whether a
 * backend would *actually* work on this host (e.g. DeviceNotAvailable_* tests
 * that intentionally exercise the unavailable-device error path and must skip
 * when the hardware is genuinely present).
 */
inline bool backend_has_runtime_devices(Device::Type type) {
    try {
        auto backend = backend_registry().get_backend(type);
        if (!backend) return false;
        return backend->device_count() > 0;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Get all available backends for testing
 *
 * Returns all possible backend configs without initializing the library.
 * Actual availability is checked lazily in is_backend_available() during
 * test SetUp(), so INSTANTIATE_TEST_SUITE_P (which runs at static init
 * time, including during --gtest_list_tests) does not trigger heavy GPU
 * backend initialization that can segfault under concurrent test discovery.
 */
inline std::vector<BackendConfig> get_available_backends() {
    // Return all possible backends.  Tests should call
    //   tenzor::initialize() in SetUp() and GTEST_SKIP() when a
    //   backend is unavailable, rather than filtering here.
    return {
        {"CPU",   Device::Type::CPU,   0, true},
        {"CUDA",  Device::Type::CUDA,  0, true},
        {"OneAPI", Device::Type::OneAPI, 0, true},
        {"ROCm",  Device::Type::ROCm,  0, true},
        {"Vulkan", Device::Type::Vulkan, 0, true},
    };
}

} // namespace tenzor::test
