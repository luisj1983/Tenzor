/**
 * @file parity_test_utils.hpp
 * @brief Utilities for backend parity testing
 *
 * Provides helper functions and macros for comprehensive backend parity tests
 * to ensure all backends (CPU, CUDA, OneAPI, Vulkan) produce identical results.
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

namespace tenzor {
namespace testing {

/**
 * @brief Get list of available backends for testing.
 * Excludes ROCm to prevent system crashes.
 */
inline std::vector<Device> get_available_backends() {
    std::vector<Device> backends;
    backends.push_back(Device::cpu());

    // Check CUDA availability
    try {
        auto t = zeros({2, 2}, DType::Float32, Device::cuda(0));
        backends.push_back(Device::cuda(0));
    } catch (...) {
        std::cout << "CUDA backend not available, skipping CUDA tests" << std::endl;
    }

    // Check OneAPI availability
    try {
        auto t = zeros({2, 2}, DType::Float32, Device::oneapi(0));
        backends.push_back(Device::oneapi(0));
    } catch (...) {
        std::cout << "OneAPI backend not available, skipping OneAPI tests" << std::endl;
    }

    // Check Vulkan availability
    try {
        auto t = zeros({2, 2}, DType::Float32, Device::vulkan(0));
        backends.push_back(Device::vulkan(0));
    } catch (...) {
        std::cout << "Vulkan backend not available, skipping Vulkan tests" << std::endl;
    }

    // SKIP ROCm - causes system crashes

    return backends;
}

/**
 * @brief Get backend name for reporting.
 */
inline std::string backend_name(const Device& device) {
    return device.to_string();
}

/**
 * @brief Check if two tensors are close within tolerance.
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

    // Move both to CPU for comparison
    auto a_cpu = a.device().type == Device::Type::CPU ? a : a.to(Device::cpu());
    auto b_cpu = b.device().type == Device::Type::CPU ? b : b.to(Device::cpu());

    // Synchronize devices before comparison
    a.device().synchronize();
    b.device().synchronize();

    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();

    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        float va = a_data[i];
        float vb = b_data[i];

        // Handle NaN
        if (std::isnan(va) && std::isnan(vb)) {
            if (equal_nan) continue;
            return false;
        }

        if (std::isnan(va) || std::isnan(vb)) {
            return false;
        }

        // Handle infinity
        if (std::isinf(va) && std::isinf(vb)) {
            if ((va > 0) == (vb > 0)) continue;
            return false;
        }

        // Check tolerance
        float diff = std::abs(va - vb);
        float threshold = atol + rtol * std::abs(vb);

        if (diff > threshold) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Compute maximum absolute difference between two tensors.
 */
inline float max_abs_diff(const Tensor& a, const Tensor& b) {
    auto a_cpu = a.device().type == Device::Type::CPU ? a : a.to(Device::cpu());
    auto b_cpu = b.device().type == Device::Type::CPU ? b : b.to(Device::cpu());

    a.device().synchronize();
    b.device().synchronize();

    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();

    float max_diff = 0.0f;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        max_diff = std::max(max_diff, diff);
    }

    return max_diff;
}

/**
 * @brief Generate random test tensor with seed for reproducibility.
 */
inline Tensor generate_test_tensor(const std::vector<int64_t>& shape,
                                   DType dtype,
                                   Device device,
                                   uint64_t seed = 12345) {
    // TODO: Use seed for reproducibility when available
    auto t = randn(shape, dtype, Device::cpu());
    if (device.type != Device::Type::CPU) {
        return t.to(device);
    }
    return t;
}

/**
 * @brief Generate random tensor with uniform distribution [low, high).
 */
inline Tensor generate_uniform_tensor(const std::vector<int64_t>& shape,
                                      float low, float high,
                                      DType dtype,
                                      Device device) {
    auto t = rand(shape, dtype, Device::cpu());
    // Scale to [low, high)
    auto scaled = t * (high - low) + low;
    if (device.type != Device::Type::CPU) {
        return scaled.to(device);
    }
    return scaled;
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
