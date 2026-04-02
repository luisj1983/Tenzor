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
 * @brief Get all available backends for testing
 */
inline std::vector<BackendConfig> get_available_backends() {
    static bool initialized = false;
    if (!initialized) {
        tenzor::initialize();
        initialized = true;
    }

    std::vector<BackendConfig> configs;

    // Always include CPU
    configs.push_back({"CPU", Device::Type::CPU, 0, true});

    // Check CUDA
    if (is_backend_available(Device::Type::CUDA)) {
        configs.push_back({"CUDA", Device::Type::CUDA, 0, true});
    }

    // Check OneAPI
    if (is_backend_available(Device::Type::OneAPI)) {
        configs.push_back({"OneAPI", Device::Type::OneAPI, 0, true});
    }

    // Check ROCm (opt-in via TENZOR_TEST_ROCM=1)
    if (is_backend_available(Device::Type::ROCm)) {
        configs.push_back({"ROCm", Device::Type::ROCm, 0, true});
    }

    return configs;
}

} // namespace tenzor::test
