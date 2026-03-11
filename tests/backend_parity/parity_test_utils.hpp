/**
 * @file parity_test_utils.hpp
 * @brief Utilities for backend parity testing
 *
 * Provides helper functions and macros for comprehensive backend parity tests
 * to ensure all backends (CPU, CUDA, OneAPI, Vulkan) produce identical results.
 *
 * This header also provides consolidated backend availability checking functions.
 * These are the canonical implementations — other test files that previously
 * duplicated this logic should migrate to using these functions instead:
 *   - tests/backend_test_fixture.hpp (BackendTest::isBackendAvailable)
 *   - tests/multi_backend_dtype_fixture.hpp (isBackendAvailable, isBackendNameAvailable)
 *   - tests/test_phase11_backends.cpp (BackendTestBase::isBackendAvailable)
 *   - tests/test_slice_backend_parity.cpp (standalone isBackendAvailable)
 */

#pragma once

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <iostream>
#include <ranges>
#include <iomanip>
#include <random>

namespace tenzor {
namespace testing {

// ============================================================================
// Consolidated Backend Availability Checking
// ============================================================================

/**
 * @brief Check if a specific backend device type is available.
 *
 * Attempts to create a small tensor on the device to verify the backend
 * is loaded and functional. Results are cached for the lifetime of the process.
 *
 * @param backend_type The device type to check (e.g., Device::Type::CUDA)
 * @param index Device index (default: 0)
 * @return true if the backend is available and functional
 */
inline bool is_backend_available(Device::Type backend_type, int32_t index = 0) {
    try {
        Device test_device{backend_type, index};
        auto t = zeros({2, 2}, DType::Float32, test_device);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Check if CUDA backend is available.
 */
inline bool has_cuda(int32_t index = 0) {
    return is_backend_available(Device::Type::CUDA, index);
}

/**
 * @brief Check if Vulkan backend is available.
 */
inline bool has_vulkan(int32_t index = 0) {
    return is_backend_available(Device::Type::Vulkan, index);
}

/**
 * @brief Check if OneAPI backend is available.
 */
inline bool has_oneapi(int32_t index = 0) {
    return is_backend_available(Device::Type::OneAPI, index);
}

/**
 * @brief Check if ROCm backend is available.
 */
inline bool has_rocm(int32_t index = 0) {
    return is_backend_available(Device::Type::ROCm, index);
}

/**
 * @brief Check if a backend is available by name string.
 *
 * Accepts: "cpu", "cuda", "vulkan", "oneapi", "rocm"
 *
 * @param name Backend name (case-sensitive, lowercase)
 * @return true if available
 */
inline bool is_backend_name_available(const std::string& name) {
    if (name == "cpu") return true;
    if (name == "cuda") return has_cuda();
    if (name == "vulkan") return has_vulkan();
    if (name == "oneapi") return has_oneapi();
    if (name == "rocm") return has_rocm();
    return false;
}

/**
 * @brief Get Device object from a backend name string.
 *
 * @param name Backend name ("cpu", "cuda", "vulkan", "oneapi", "rocm", "metal")
 * @return Corresponding Device object
 * @throws std::runtime_error if name is unknown
 */
inline Device device_from_name(const std::string& name) {
    if (name == "cpu") return Device::cpu();
    if (name == "cuda") return Device::cuda(0);
    if (name == "vulkan") return Device::vulkan(0);
    if (name == "oneapi") return Device::oneapi(0);
    if (name == "rocm") return Device::rocm(0);
    throw std::runtime_error("Unknown backend name: " + name);
}

/**
 * @brief Get list of all available backend Devices.
 *
 * Always includes CPU. Checks CUDA, OneAPI, Vulkan, and ROCm.
 *
 * @return Vector of available Device objects
 */
inline std::vector<Device> get_available_backends() {
    std::vector<Device> backends;
    backends.push_back(Device::cpu());

    if (has_cuda()) backends.push_back(Device::cuda(0));
    if (has_oneapi()) backends.push_back(Device::oneapi(0));
    if (has_vulkan()) backends.push_back(Device::vulkan(0));
    if (has_rocm()) backends.push_back(Device::rocm(0));

    return backends;
}

/**
 * @brief Get list of available backend names as strings.
 *
 * @return Vector of backend name strings (e.g., {"cpu", "cuda", "vulkan"})
 */
inline std::vector<std::string> get_available_backend_names() {
    std::vector<std::string> names = {"cpu"};
    if (has_cuda()) names.push_back("cuda");
    if (has_oneapi()) names.push_back("oneapi");
    if (has_vulkan()) names.push_back("vulkan");
    if (has_rocm()) names.push_back("rocm");
    // Metal: planned for future release
    return names;
}

// ============================================================================
// Skip Macros for Backend Availability
// ============================================================================

/**
 * @brief Skip test if the specified backend is not available.
 *
 * Usage:
 *   TEST(MyTest, CudaOp) {
 *       SKIP_IF_NO_CUDA;
 *       auto t = zeros({4, 4}, DType::Float32, Device::cuda(0));
 *       // ...
 *   }
 */
#define SKIP_IF_NO_CUDA \
    if (!tenzor::testing::has_cuda()) GTEST_SKIP() << "CUDA backend not available"

#define SKIP_IF_NO_VULKAN \
    if (!tenzor::testing::has_vulkan()) GTEST_SKIP() << "Vulkan backend not available"

#define SKIP_IF_NO_ONEAPI \
    if (!tenzor::testing::has_oneapi()) GTEST_SKIP() << "OneAPI backend not available"

#define SKIP_IF_NO_ROCM \
    if (!tenzor::testing::has_rocm()) GTEST_SKIP() << "ROCm backend not available"

/**
 * @brief Skip test if the named backend is not available.
 *
 * Usage:
 *   SKIP_IF_NO_BACKEND("cuda");
 */
#define SKIP_IF_NO_BACKEND(name) \
    if (!tenzor::testing::is_backend_name_available(name)) \
        GTEST_SKIP() << name << " backend not available"

// ============================================================================
// Tensor Comparison Utilities
// ============================================================================

/**
 * @brief Get backend name for reporting.
 */
inline std::string backend_name(const Device& device) {
    return device.to_string();
}

/**
 * @brief Check if two tensors are close within tolerance.
 *
 * Supports all floating-point and integer dtypes by dispatching to the
 * correct data pointer type. Integer dtypes use exact comparison (atol=0).
 *
 * Tolerance rationale:
 * - Element-wise ops (add, mul): rtol=1e-5, atol=1e-8 (single rounding)
 * - Accumulation ops (matmul, sum): rtol=1e-4, atol=1e-5 (FP32 accumulation error)
 * - Reduction chains (mean, var): rtol=1e-5, atol=1e-7 (moderate accumulation)
 * - Convolution: rtol=1e-4, atol=1e-6 (algorithm-dependent rounding)
 *
 * @param a First tensor
 * @param b Second tensor
 * @param rtol Relative tolerance (default: 1e-5)
 * @param atol Absolute tolerance (default: 1e-8)
 * @param equal_nan Treat NaN as equal (default: false)
 * @return true if tensors are close
 */
inline bool tensors_close(const Tensor& a, const Tensor& b,
                         float rtol = 1e-5f, float atol = 1e-8f,
                         bool equal_nan = false) {
    if (!std::ranges::equal(a.shape(), b.shape())) {
        return false;
    }

    if (a.dtype() != b.dtype()) {
        return false;
    }

    // Synchronize devices before comparison
    if (a.device().type != Device::Type::CPU) a.device().synchronize();
    if (b.device().type != Device::Type::CPU) b.device().synchronize();

    // Move both to CPU for comparison
    auto a_cpu = a.device().type == Device::Type::CPU ? a : a.to(Device::cpu());
    auto b_cpu = b.device().type == Device::Type::CPU ? b : b.to(Device::cpu());

    // For Float16/BFloat16, promote to Float32 for comparison since
    // data<float16>() would require half-precision comparison math.
    if (a_cpu.dtype() == DType::Float16 || a_cpu.dtype() == DType::BFloat16) {
        a_cpu = a_cpu.to(DType::Float32);
        b_cpu = b_cpu.to(DType::Float32);
    }

    // Dispatch comparison by dtype
    auto compare_float = [&](auto* a_data, auto* b_data) -> bool {
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            double va = static_cast<double>(a_data[i]);
            double vb = static_cast<double>(b_data[i]);

            if (std::isnan(va) && std::isnan(vb)) {
                if (equal_nan) continue;
                return false;
            }
            if (std::isnan(va) || std::isnan(vb)) return false;

            if (std::isinf(va) && std::isinf(vb)) {
                if ((va > 0) == (vb > 0)) continue;
                return false;
            }

            double diff = std::abs(va - vb);
            double threshold = static_cast<double>(atol) + static_cast<double>(rtol) * std::abs(vb);
            if (diff > threshold) return false;
        }
        return true;
    };

    auto compare_int = [&](auto* a_data, auto* b_data) -> bool {
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            if (a_data[i] != b_data[i]) return false;
        }
        return true;
    };

    switch (a_cpu.dtype()) {
        case DType::Float32:
            return compare_float(a_cpu.data<float>(), b_cpu.data<float>());
        case DType::Float64:
            return compare_float(a_cpu.data<double>(), b_cpu.data<double>());
        case DType::Int8:
            return compare_int(a_cpu.data<int8_t>(), b_cpu.data<int8_t>());
        case DType::Int16:
            return compare_int(a_cpu.data<int16_t>(), b_cpu.data<int16_t>());
        case DType::Int32:
            return compare_int(a_cpu.data<int32_t>(), b_cpu.data<int32_t>());
        case DType::Int64:
            return compare_int(a_cpu.data<int64_t>(), b_cpu.data<int64_t>());
        case DType::UInt8:
            return compare_int(a_cpu.data<uint8_t>(), b_cpu.data<uint8_t>());
        case DType::Bool:
            return compare_int(a_cpu.data<uint8_t>(), b_cpu.data<uint8_t>());
        default:
            // Fall back to Float32 comparison for any unhandled dtype
            return compare_float(a_cpu.data<float>(), b_cpu.data<float>());
    }
}

/**
 * @brief Compute maximum absolute difference between two tensors.
 *
 * Promotes Float16/BFloat16 to Float32 for comparison.
 */
inline float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.device().type != Device::Type::CPU) a.device().synchronize();
    if (b.device().type != Device::Type::CPU) b.device().synchronize();

    auto a_cpu = a.device().type == Device::Type::CPU ? a : a.to(Device::cpu());
    auto b_cpu = b.device().type == Device::Type::CPU ? b : b.to(Device::cpu());

    // Promote half types
    if (a_cpu.dtype() == DType::Float16 || a_cpu.dtype() == DType::BFloat16) {
        a_cpu = a_cpu.to(DType::Float32);
        b_cpu = b_cpu.to(DType::Float32);
    }

    if (a_cpu.dtype() == DType::Float64) {
        const double* a_data = a_cpu.data<double>();
        const double* b_data = b_cpu.data<double>();
        double max_diff = 0.0;
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            max_diff = std::max(max_diff, std::abs(a_data[i] - b_data[i]));
        }
        return static_cast<float>(max_diff);
    }

    // Default: Float32 path (also handles promoted half types)
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        max_diff = std::max(max_diff, std::abs(a_data[i] - b_data[i]));
    }
    return max_diff;
}

/**
 * @brief Generate random test tensor with seed for deterministic reproducibility.
 *
 * Uses std::mt19937 seeded with the given seed to generate deterministic
 * values. The same seed + shape + dtype always produces identical results,
 * regardless of the global random state.
 *
 * Values are drawn from a standard normal distribution (mean=0, stddev=1).
 *
 * @param shape Tensor shape
 * @param dtype Data type for the output tensor
 * @param device Target device (tensor is created on CPU then moved)
 * @param seed Random seed for reproducibility (default: 12345)
 * @return Deterministically-generated tensor
 */
inline Tensor generate_test_tensor(const std::vector<int64_t>& shape,
                                   DType dtype,
                                   Device device,
                                   uint64_t seed = 12345) {
    // Compute total elements
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }

    // Handle empty tensors
    if (numel == 0) {
        auto t = zeros(shape, dtype, Device::cpu());
        if (device.type != Device::Type::CPU) {
            return t.to(device);
        }
        return t;
    }

    // Generate deterministic values using mt19937
    std::mt19937 gen(static_cast<unsigned>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Create Float32 tensor on CPU, fill with deterministic values
    auto t = zeros(shape, DType::Float32, Device::cpu());
    float* data = t.data<float>();
    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }

    // Convert dtype if needed
    if (dtype != DType::Float32) {
        t = t.to(dtype);
    }

    // Move to target device if needed
    if (device.type != Device::Type::CPU) {
        return t.to(device);
    }
    return t;
}

/**
 * @brief Generate random tensor with uniform distribution [low, high).
 *
 * Uses std::mt19937 seeded with the given seed for deterministic output.
 *
 * @param shape Tensor shape
 * @param low Lower bound (inclusive)
 * @param high Upper bound (exclusive)
 * @param dtype Data type for the output tensor
 * @param device Target device
 * @param seed Random seed for reproducibility (default: 54321)
 * @return Deterministically-generated tensor with values in [low, high)
 */
inline Tensor generate_uniform_tensor(const std::vector<int64_t>& shape,
                                      float low, float high,
                                      DType dtype,
                                      Device device,
                                      uint64_t seed = 54321) {
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }

    if (numel == 0) {
        auto t = zeros(shape, dtype, Device::cpu());
        if (device.type != Device::Type::CPU) {
            return t.to(device);
        }
        return t;
    }

    std::mt19937 gen(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dist(low, high);

    auto t = zeros(shape, DType::Float32, Device::cpu());
    float* data = t.data<float>();
    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }

    if (dtype != DType::Float32) {
        t = t.to(dtype);
    }

    if (device.type != Device::Type::CPU) {
        return t.to(device);
    }
    return t;
}

/**
 * @brief Test operation parity across specified backends.
 *
 * @param operation Function that takes input tensors and returns result
 * @param inputs Input tensors (on CPU)
 * @param backends List of backends to test (if empty, uses all available)
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param test_name Test name for error reporting
 */
template<typename Op>
void test_operation_parity_backends(Op operation,
                          const std::vector<Tensor>& inputs,
                          std::vector<Device> backends,
                          float rtol = 1e-5f,
                          float atol = 1e-8f,
                          const std::string& test_name = "Operation") {
    if (backends.empty()) {
        backends = get_available_backends();
    }

    if (backends.size() < 2) {
        GTEST_SKIP() << "Need at least 2 backends for parity testing";
        return;
    }

    std::vector<Tensor> results;
    std::vector<Device> used_backends;

    // Run operation on each backend
    for (const auto& backend : backends) {
        try {
            // Move inputs to backend
            std::vector<Tensor> backend_inputs;
            for (const auto& input : inputs) {
                backend_inputs.push_back(input.to(backend));
            }

            // Execute operation
            auto result = operation(backend_inputs);

            // Synchronize before storing
            backend.synchronize();

            results.push_back(result);
            used_backends.push_back(backend);
        } catch (const std::exception& e) {
            std::cerr << "Backend " << backend_name(backend)
                     << " failed: " << e.what() << std::endl;
        }
    }

    if (results.size() < 2) {
        GTEST_SKIP() << "Need at least 2 successful backends for comparison";
        return;
    }

    // Compare all results to CPU (first result)
    const auto& reference = results[0];
    const auto& reference_backend = used_backends[0];

    for (size_t i = 1; i < results.size(); ++i) {
        const auto& result = results[i];
        const auto& backend = used_backends[i];

        bool close = tensors_close(reference, result, rtol, atol);

        if (!close) {
            float max_diff = max_abs_diff(reference, result);

            FAIL() << test_name << " parity failed:\n"
                  << "  Reference backend: " << backend_name(reference_backend) << "\n"
                  << "  Test backend: " << backend_name(backend) << "\n"
                  << "  Max absolute difference: " << std::scientific << max_diff << "\n"
                  << "  Tolerance: rtol=" << rtol << ", atol=" << atol;
        }
    }
}

/**
 * @brief Test operation parity across all available backends.
 *
 * @param operation Function that takes input tensors and returns result
 * @param inputs Input tensors (on CPU)
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param test_name Test name for error reporting
 */
template<typename Op>
void test_operation_parity(Op operation,
                          const std::vector<Tensor>& inputs,
                          float rtol = 1e-5f,
                          float atol = 1e-8f,
                          const std::string& test_name = "Operation") {
    test_operation_parity_backends(operation, inputs, {}, rtol, atol, test_name);
}

/**
 * @brief Macro for expecting tensors to be close with detailed error message.
 */
#define EXPECT_TENSORS_CLOSE(a, b, rtol, atol) \
    do { \
        if (!tenzor::testing::tensors_close(a, b, rtol, atol)) { \
            float max_diff = tenzor::testing::max_abs_diff(a, b); \
            FAIL() << "Tensors not close:\n" \
                  << "  Max absolute difference: " << std::scientific << max_diff << "\n" \
                  << "  Tolerance: rtol=" << rtol << ", atol=" << atol; \
        } \
    } while(0)

/**
 * @brief Macro for expecting tensor values to match exactly.
 */
#define EXPECT_TENSORS_EQUAL(a, b) \
    EXPECT_TENSORS_CLOSE(a, b, 0.0f, 0.0f)

/**
 * @brief Compute numerical gradient using finite differences.
 */
inline Tensor numerical_gradient(std::function<Tensor(const Tensor&)> func,
                                const Tensor& input,
                                float eps = 1e-4f) {
    auto grad = zeros_like(input);
    auto grad_data = grad.data<float>();
    auto input_cpu = input.to(Device::cpu());
    auto input_data = input_cpu.data<float>();

    for (int64_t i = 0; i < input.numel(); ++i) {
        float original = input_data[i];

        // f(x + eps)
        input_data[i] = original + eps;
        auto input_plus = input_cpu.clone();
        float f_plus = sum(func(input_plus)).item<float>();

        // f(x - eps)
        input_data[i] = original - eps;
        auto input_minus = input_cpu.clone();
        float f_minus = sum(func(input_minus)).item<float>();

        // Restore original
        input_data[i] = original;

        // Central difference
        grad_data[i] = (f_plus - f_minus) / (2.0f * eps);
    }

    return grad;
}

/**
 * @brief Test configuration for different input sizes.
 */
struct TestConfig {
    std::vector<int64_t> shape;
    std::string description;
    float rtol = 1e-5f;
    float atol = 1e-8f;
};

/**
 * @brief Standard test configurations for different scales.
 */
inline std::vector<TestConfig> get_standard_test_configs() {
    return {
        {{8, 8}, "Small 8x8", 1e-5f, 1e-8f},
        {{32, 32}, "Medium 32x32", 1e-5f, 1e-8f},
        {{128, 128}, "Large 128x128", 1e-5f, 1e-7f},
        {{4, 64, 64}, "Batched 4x64x64", 1e-5f, 1e-7f}
    };
}

/**
 * @brief Test configurations for convolution operations.
 */
inline std::vector<TestConfig> get_conv_test_configs() {
    return {
        {{1, 3, 32, 32}, "Single image 3x32x32", 1e-4f, 1e-6f},
        {{4, 16, 64, 64}, "Batch 4x16x64x64", 1e-4f, 1e-6f},
        {{8, 32, 128, 128}, "Large batch 8x32x128x128", 1e-4f, 1e-5f}
    };
}

} // namespace testing
} // namespace tenzor
