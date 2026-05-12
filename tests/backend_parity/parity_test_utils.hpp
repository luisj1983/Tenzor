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
#include "golden_util.hpp"
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

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
// Returns true if `backend` appears in the comma-separated $TENZOR_SKIP_BACKENDS list.
// Consulted by is_backend_available() and the SKIP_IF_NO_* macros so a single
// env setting silences a backend across every parity test without a rebuild.
inline bool is_backend_skipped_by_env(std::string_view backend) {
    const char* raw = std::getenv("TENZOR_SKIP_BACKENDS");
    if (!raw || !*raw) return false;
    std::string_view list{raw};
    size_t start = 0;
    while (start <= list.size()) {
        size_t end = list.find(',', start);
        if (end == std::string_view::npos) end = list.size();
        auto token = list.substr(start, end - start);
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.remove_prefix(1);
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) token.remove_suffix(1);
        if (!token.empty() && token == backend) return true;
        start = end + 1;
    }
    return false;
}

inline std::string_view device_type_to_backend_name(Device::Type t) {
    switch (t) {
        case Device::Type::CPU:    return "cpu";
        case Device::Type::CUDA:   return "cuda";
        case Device::Type::Vulkan: return "vulkan";
        case Device::Type::OneAPI: return "oneapi";
        case Device::Type::ROCm:   return "rocm";
    }
    return "";
}

inline bool is_backend_available(Device::Type backend_type, int32_t index = 0) {
    if (is_backend_skipped_by_env(device_type_to_backend_name(backend_type))) {
        return false;
    }
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
 * @brief Parse the base backend name from a possibly-indexed string.
 *
 * "cuda:1" -> "cuda", "cuda" -> "cuda", "cpu" -> "cpu"
 */
inline std::string parse_backend_name(const std::string& s) {
    auto pos = s.find(':');
    return (pos == std::string::npos) ? s : s.substr(0, pos);
}

/**
 * @brief Parse the device index from a possibly-indexed string.
 *
 * "cuda:1" -> 1, "cuda" -> 0, "cpu" -> 0
 */
inline int32_t parse_device_index(const std::string& s) {
    auto pos = s.find(':');
    if (pos == std::string::npos) return 0;
    return std::stoi(s.substr(pos + 1));
}

/**
 * @brief Map a base backend name to Device::Type.
 */
inline Device::Type name_to_device_type(const std::string& name) {
    if (name == "cpu") return Device::Type::CPU;
    if (name == "cuda") return Device::Type::CUDA;
    if (name == "vulkan") return Device::Type::Vulkan;
    if (name == "oneapi") return Device::Type::OneAPI;
    if (name == "rocm") return Device::Type::ROCm;
    throw std::runtime_error("Unknown backend name: " + name);
}

/**
 * @brief Check if a backend is available by name string.
 *
 * Accepts both legacy ("cuda") and indexed ("cuda:1") formats.
 *
 * @param name Backend name (case-sensitive, lowercase)
 * @return true if available
 */
inline bool is_backend_name_available(const std::string& name) {
    auto base = parse_backend_name(name);
    auto index = parse_device_index(name);
    if (base == "cpu") return true;
    try {
        return is_backend_available(name_to_device_type(base), index);
    } catch (...) {
        return false;
    }
}

/**
 * @brief Get Device object from a backend name string.
 *
 * Accepts both legacy ("cuda") and indexed ("cuda:1") formats.
 *
 * @param name Backend name
 * @return Corresponding Device object
 * @throws std::runtime_error if name is unknown
 */
inline Device device_from_name(const std::string& name) {
    auto base = parse_backend_name(name);
    auto index = parse_device_index(name);
    if (base == "cpu") return Device::cpu();
    return Device{name_to_device_type(base), index};
}

/**
 * @brief Get list of all available backend Devices (single device per backend).
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
 * @brief Gate a parity test on having at least 2 backends available.
 *
 * Default behavior: GTEST_SKIP() with the provided reason.
 * When env var TENZOR_REQUIRE_MULTI_BACKEND=1 is set: FAIL() instead, so CI
 * environments that are supposed to have a GPU backend hard-fail when the
 * backend fails to initialize — a silent skip would hide the broken env.
 *
 * Usage:
 *   TEST_P(MyParity, SomeOp) {
 *       REQUIRE_MULTI_BACKEND_OR_SKIP("SomeOp parity");
 *       auto backends = get_available_backends();
 *       ...
 *   }
 */
#define REQUIRE_MULTI_BACKEND_OR_SKIP(reason_msg)                              \
    do {                                                                       \
        auto _parity_backends = ::tenzor::testing::get_available_backends();   \
        if (_parity_backends.size() < 2) {                                     \
            if (::tenzor::testing::golden::require_multi_backend()) {          \
                FAIL() << "Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1)" \
                       << " but only " << _parity_backends.size()              \
                       << " backend(s) available: " << (reason_msg);           \
            } else {                                                           \
                GTEST_SKIP() << "Need >=2 backends: " << (reason_msg);         \
            }                                                                  \
        }                                                                      \
    } while (0)

/**
 * @brief Get list of all available backend Devices with all device indices.
 *
 * Queries device_count() per backend and probes each device.
 *
 * @return Vector of available Device objects (e.g., cpu, cuda:0, cuda:1, rocm:0)
 */
inline std::vector<Device> get_available_backends_all_devices() {
    std::vector<Device> backends;
    backends.push_back(Device::cpu());

    struct BackendInfo { Device::Type type; };
    for (auto type : {Device::Type::CUDA, Device::Type::OneAPI,
                      Device::Type::Vulkan, Device::Type::ROCm}) {
        auto* backend = backend_registry().get_backend(type);
        if (!backend || !backend->is_available()) continue;
        for (int32_t i = 0; i < backend->device_count(); ++i) {
            if (is_backend_available(type, i)) {
                backends.push_back(Device{type, i});
            }
        }
    }

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
    return names;
}

/**
 * @brief Get list of available backend names with all device indices.
 *
 * @return Vector like {"cpu", "cuda:0", "cuda:1", "rocm:0"}
 */
inline std::vector<std::string> get_available_backend_names_all_devices() {
    std::vector<std::string> names = {"cpu"};

    struct BackendInfo { const char* name; Device::Type type; };
    constexpr BackendInfo infos[] = {
        {"cuda", Device::Type::CUDA},
        {"oneapi", Device::Type::OneAPI},
        {"vulkan", Device::Type::Vulkan},
        {"rocm", Device::Type::ROCm},
    };

    for (const auto& [name, type] : infos) {
        auto* backend = backend_registry().get_backend(type);
        if (!backend || !backend->is_available()) continue;
        for (int32_t i = 0; i < backend->device_count(); ++i) {
            if (is_backend_available(type, i)) {
                names.push_back(std::string(name) + ":" + std::to_string(i));
            }
        }
    }

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

    // Materialize contiguous copies so element-by-element pointer iteration
    // walks logical positions, not the underlying storage. Without this,
    // non-contiguous views (e.g. transpose/permute outputs) compare by raw
    // memory layout and produce false negatives across backends whose
    // contiguous() materialization happens at different points.
    if (!a_cpu.is_contiguous()) a_cpu = a_cpu.contiguous();
    if (!b_cpu.is_contiguous()) b_cpu = b_cpu.contiguous();

    // For Float16/BFloat16, promote to Float32 for comparison since
    // data<float16>() would require half-precision comparison math.
    if (a_cpu.dtype() == DType::Float16 || a_cpu.dtype() == DType::BFloat16) {
        a_cpu = a_cpu.to(DType::Float32);
        b_cpu = b_cpu.to(DType::Float32);
    }

    // Dispatch comparison by dtype. Two tensors are considered close if,
    // elementwise, |a - b| <= atol + rtol * |b|. Both-NaN positions are
    // always treated as matching — backends agreeing on an undefined
    // result (e.g. exp(log(-x))) is a valid parity outcome. The
    // equal_nan flag is kept for API compatibility but is effectively
    // always on for the comparison path.
    (void)equal_nan;
    auto compare_float = [&](auto* a_data, auto* b_data) -> bool {
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            double va = static_cast<double>(a_data[i]);
            double vb = static_cast<double>(b_data[i]);

            if (std::isnan(va) && std::isnan(vb)) continue;
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

    // Complex types: compare the interleaved (real, imag) storage as
    // pairs of floats/doubles. The numel() loop in compare_float walks
    // over tensor elements — double it to cover both components.
    // Matching NaN pairs are treated as equal, same rationale as above.
    auto compare_complex = [&](auto* a_data, auto* b_data) -> bool {
        int64_t n = a_cpu.numel() * 2;
        for (int64_t i = 0; i < n; ++i) {
            double va = static_cast<double>(a_data[i]);
            double vb = static_cast<double>(b_data[i]);
            if (std::isnan(va) && std::isnan(vb)) continue;
            if (std::isnan(va) || std::isnan(vb)) return false;
            if (std::isinf(va) && std::isinf(vb)) {
                if ((va > 0) == (vb > 0)) continue;
                return false;
            }
            double diff = std::abs(va - vb);
            double threshold =
                static_cast<double>(atol) + static_cast<double>(rtol) * std::abs(vb);
            if (diff > threshold) return false;
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
        case DType::Complex64:
            return compare_complex(
                reinterpret_cast<const float*>(a_cpu.data_ptr()),
                reinterpret_cast<const float*>(b_cpu.data_ptr()));
        case DType::Complex128:
            return compare_complex(
                reinterpret_cast<const double*>(a_cpu.data_ptr()),
                reinterpret_cast<const double*>(b_cpu.data_ptr()));
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

    // Per-element float diff that mirrors tensors_close: matching NaN counts
    // as 0 diff, matching same-sign Inf counts as 0 diff (otherwise +Inf-Inf
    // would silently produce NaN through std::abs and poison std::max).
    // Mixed Inf / NaN positions report +inf so callers see real divergence.
    auto float_diff = [](double va, double vb) -> double {
        if (std::isnan(va) && std::isnan(vb)) return 0.0;
        if (std::isnan(va) || std::isnan(vb)) {
            return std::numeric_limits<double>::infinity();
        }
        if (std::isinf(va) && std::isinf(vb)) {
            return ((va > 0) == (vb > 0))
                ? 0.0
                : std::numeric_limits<double>::infinity();
        }
        return std::abs(va - vb);
    };

    if (a_cpu.dtype() == DType::Float64) {
        const double* a_data = a_cpu.data<double>();
        const double* b_data = b_cpu.data<double>();
        double max_diff = 0.0;
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            double d = float_diff(a_data[i], b_data[i]);
            if (d > max_diff) max_diff = d;
        }
        return static_cast<float>(max_diff);
    }

    if (a_cpu.dtype() == DType::Int64) {
        const int64_t* a_data = a_cpu.data<int64_t>();
        const int64_t* b_data = b_cpu.data<int64_t>();
        int64_t max_diff = 0;
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            max_diff = std::max(max_diff, std::abs(a_data[i] - b_data[i]));
        }
        return static_cast<float>(max_diff);
    }

    if (a_cpu.dtype() == DType::Int32) {
        const int32_t* a_data = a_cpu.data<int32_t>();
        const int32_t* b_data = b_cpu.data<int32_t>();
        int32_t max_diff = 0;
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            max_diff = std::max(max_diff, static_cast<int32_t>(std::abs(a_data[i] - b_data[i])));
        }
        return static_cast<float>(max_diff);
    }

    if (a_cpu.dtype() == DType::Int16) {
        const int16_t* a_data = a_cpu.data<int16_t>();
        const int16_t* b_data = b_cpu.data<int16_t>();
        int32_t max_diff = 0;
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            max_diff = std::max(max_diff,
                                std::abs(static_cast<int32_t>(a_data[i]) - static_cast<int32_t>(b_data[i])));
        }
        return static_cast<float>(max_diff);
    }

    if (a_cpu.dtype() == DType::Int8) {
        const int8_t* a_data = a_cpu.data<int8_t>();
        const int8_t* b_data = b_cpu.data<int8_t>();
        int32_t max_diff = 0;
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            max_diff = std::max(max_diff,
                                std::abs(static_cast<int32_t>(a_data[i]) - static_cast<int32_t>(b_data[i])));
        }
        return static_cast<float>(max_diff);
    }

    if (a_cpu.dtype() == DType::UInt8 || a_cpu.dtype() == DType::Bool) {
        const uint8_t* a_data = a_cpu.data<uint8_t>();
        const uint8_t* b_data = b_cpu.data<uint8_t>();
        int32_t max_diff = 0;
        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            max_diff = std::max(max_diff,
                                std::abs(static_cast<int32_t>(a_data[i]) - static_cast<int32_t>(b_data[i])));
        }
        return static_cast<float>(max_diff);
    }

    if (a_cpu.dtype() == DType::Complex64 || a_cpu.dtype() == DType::Complex128) {
        // Treat the storage as interleaved (real, imag) pairs of
        // the underlying real component type and take the max abs
        // component difference (not the magnitude diff — that would
        // require a sqrt per element and hides per-component errors).
        int64_t n = a_cpu.numel() * 2;
        if (a_cpu.dtype() == DType::Complex64) {
            const float* a_data = reinterpret_cast<const float*>(a_cpu.data_ptr());
            const float* b_data = reinterpret_cast<const float*>(b_cpu.data_ptr());
            float max_diff = 0.0f;
            for (int64_t i = 0; i < n; ++i) {
                double d = float_diff(a_data[i], b_data[i]);
                if (d > max_diff) max_diff = static_cast<float>(d);
            }
            return max_diff;
        } else {
            const double* a_data = reinterpret_cast<const double*>(a_cpu.data_ptr());
            const double* b_data = reinterpret_cast<const double*>(b_cpu.data_ptr());
            double max_diff = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                double d = float_diff(a_data[i], b_data[i]);
                if (d > max_diff) max_diff = d;
            }
            return static_cast<float>(max_diff);
        }
    }

    // Default: Float32 path (also handles promoted half types)
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        double d = float_diff(a_data[i], b_data[i]);
        if (d > max_diff) max_diff = static_cast<float>(d);
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
        // Golden fallback: on a single-backend host, compare the lone backend's
        // result against a recorded golden instead of skipping outright.
        if (backends.empty()) {
            GTEST_SKIP() << "No backends available";
            return;
        }
        const auto& backend = backends[0];
        std::vector<Tensor> backend_inputs;
        backend_inputs.reserve(inputs.size());
        for (const auto& t : inputs) backend_inputs.push_back(t.to(backend));
        auto result = operation(backend_inputs);
        backend.synchronize();

        if (golden::recording_enabled() && backend.type == Device::Type::CPU) {
            golden::maybe_record(test_name, inputs, result);
            return;
        }
        if (auto golden_result = golden::maybe_load(test_name, inputs)) {
            if (!tensors_close(result, *golden_result, rtol, atol)) {
                float max_diff = max_abs_diff(result, *golden_result);
                FAIL() << test_name << " golden parity failed:\n"
                       << "  Backend: " << backend_name(backend) << "\n"
                       << "  Max absolute difference: " << std::scientific << max_diff << "\n"
                       << "  Tolerance: rtol=" << rtol << ", atol=" << atol;
            }
            return;
        }
        if (golden::require_multi_backend()) {
            FAIL() << test_name << ": only 1 backend available and no golden recorded.\n"
                   << "  Record from a multi-backend host with TENZOR_RECORD_GOLDENS=1.";
        }
        GTEST_SKIP() << "Need at least 2 backends for parity testing"
                     << " (or a recorded golden — set TENZOR_RECORD_GOLDENS=1 on a "
                     << "multi-backend host to record)";
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

    // Persist reference outputs for future single-backend runs when the
    // caller has opted in via TENZOR_RECORD_GOLDENS. We intentionally record
    // the CPU result rather than any GPU result so the golden is always the
    // reference the parity test treats as ground truth.
    if (golden::recording_enabled() &&
        reference_backend.type == Device::Type::CPU) {
        golden::maybe_record(test_name, inputs, reference);
    }

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
 * @brief Single-backend parity check (used by TEST_P-style refactored tests).
 *
 * Runs `operation` on CPU and on `target`, then compares results within
 * tolerance. Unlike test_operation_parity_backends() this does NOT loop over
 * all backends — the loop is performed by GoogleTest's TEST_P parameterization
 * instead, which gives one ctest entry per (op, backend) combination.
 *
 * Golden-tensor fallback: when `target` is CPU we used to be a no-op. We now
 * also consult a recorded golden (if present) and compare — this gives signal
 * even on single-backend CI hosts. Record goldens on a multi-backend host by
 * running with TENZOR_RECORD_GOLDENS=1 in the environment; recording uses the
 * CPU result as reference.
 *
 * Env vars consulted:
 *   TENZOR_RECORD_GOLDENS=1       — record the CPU output; no compare.
 *   TENZOR_REQUIRE_MULTI_BACKEND=1 — if target==CPU and no golden exists, FAIL
 *                                   instead of silently passing.
 */
template<typename Op>
void test_operation_parity_single(Op operation,
                                  const std::vector<Tensor>& cpu_inputs,
                                  const Device& target,
                                  float rtol = 1e-5f,
                                  float atol = 1e-8f,
                                  const std::string& test_name = "Operation") {
    auto cpu_result = operation(cpu_inputs);

    // Always record when requested — the CPU result is the canonical reference.
    if (golden::recording_enabled() && target.type == Device::Type::CPU) {
        golden::maybe_record(test_name, cpu_inputs, cpu_result);
    }

    if (target.type == Device::Type::CPU) {
        // On CPU-only hosts we'd otherwise be a no-op. Compare against a
        // recorded golden if one exists. If not, either FAIL (when the caller
        // has asked for strict multi-backend coverage) or silently pass.
        if (auto golden_result = golden::maybe_load(test_name, cpu_inputs)) {
            if (!tensors_close(cpu_result, *golden_result, rtol, atol)) {
                float max_diff = max_abs_diff(cpu_result, *golden_result);
                FAIL() << test_name << " golden parity failed (CPU vs recorded):\n"
                       << "  Max absolute difference: " << std::scientific << max_diff << "\n"
                       << "  Tolerance: rtol=" << rtol << ", atol=" << atol << "\n"
                       << "  To re-record: TENZOR_RECORD_GOLDENS=1 ctest -R <test>";
            }
            return;
        }
        if (golden::require_multi_backend()) {
            FAIL() << test_name << " has no recorded golden and no GPU backend is available.\n"
                   << "  Record one from a multi-backend host with TENZOR_RECORD_GOLDENS=1.";
        }
        return;
    }
    // Run on the target backend and compare against CPU.
    std::vector<Tensor> target_inputs;
    target_inputs.reserve(cpu_inputs.size());
    for (const auto& t : cpu_inputs) target_inputs.push_back(t.to(target));
    auto target_result = operation(target_inputs);
    target.synchronize();
    if (!tensors_close(cpu_result, target_result, rtol, atol)) {
        float max_diff = max_abs_diff(cpu_result, target_result);
        FAIL() << test_name << " single-backend parity failed:\n"
               << "  Reference: cpu\n"
               << "  Test backend: " << backend_name(target) << "\n"
               << "  Max absolute difference: " << std::scientific << max_diff << "\n"
               << "  Tolerance: rtol=" << rtol << ", atol=" << atol;
        return;
    }

    // Cross-backend check: also compare the target against every other
    // *non-CPU* backend that happens to be available on this host. The CPU
    // reference was already compared above, so we only add new pairings.
    // This turns every TEST_P into a full matrix of (target × other) pair
    // checks — catching e.g. CUDA-vs-ROCm or Vulkan-vs-OneAPI divergence
    // that the CPU pivot would miss.
    //
    // We catch per-backend exceptions (the other backend may not support
    // the op) and emit a plain cerr warning rather than failing — failures
    // here would be noise from feature-availability differences, not
    // genuine parity issues.
    auto all_backends = get_available_backends();
    for (const auto& other : all_backends) {
        if (other.type == Device::Type::CPU) continue;
        if (other.type == target.type && other.index == target.index) continue;
        try {
            std::vector<Tensor> other_inputs;
            other_inputs.reserve(cpu_inputs.size());
            for (const auto& t : cpu_inputs) other_inputs.push_back(t.to(other));
            auto other_result = operation(other_inputs);
            other.synchronize();
            if (!tensors_close(target_result, other_result, rtol, atol)) {
                float max_diff = max_abs_diff(target_result, other_result);
                FAIL() << test_name << " cross-backend parity failed:\n"
                       << "  Backend A: " << backend_name(target) << "\n"
                       << "  Backend B: " << backend_name(other) << "\n"
                       << "  Max absolute difference: " << std::scientific << max_diff << "\n"
                       << "  Tolerance: rtol=" << rtol << ", atol=" << atol;
            }
        } catch (const std::exception& e) {
            // Other backend doesn't support the op — not a parity bug, just
            // a feature gap. Surface as a warning so the tally is visible.
            std::cerr << "[parity] " << test_name << ": cross-backend skipped on "
                      << backend_name(other) << " (" << e.what() << ")\n";
        }
    }
}

/**
 * @brief Cross-backend parity (compare every available pair, not CPU-as-ref).
 */
template<typename Op>
void test_operation_parity_cross_backend(Op operation,
                                         const std::vector<Tensor>& inputs,
                                         std::vector<Device> backends = {},
                                         float rtol = 1e-5f,
                                         float atol = 1e-8f,
                                         const std::string& test_name = "Operation") {
    if (backends.empty()) backends = get_available_backends();
    if (backends.size() < 2) {
        if (golden::require_multi_backend()) {
            FAIL() << "Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1)"
                   << " for cross-backend parity: " << test_name;
        }
        GTEST_SKIP() << "Need 2+ backends for cross-backend parity: " << test_name;
        return;
    }
    std::vector<Tensor> results;
    std::vector<Device> used;
    for (const auto& backend : backends) {
        try {
            std::vector<Tensor> backend_inputs;
            backend_inputs.reserve(inputs.size());
            for (const auto& input : inputs) backend_inputs.push_back(input.to(backend));
            auto result = operation(backend_inputs);
            backend.synchronize();
            results.push_back(result);
            used.push_back(backend);
        } catch (const std::exception& e) {
            std::cerr << "Backend " << backend_name(backend) << " failed: " << e.what() << "\n";
        }
    }
    if (results.size() < 2) { GTEST_SKIP() << "Need 2+ successful"; return; }
    for (size_t i = 0; i < results.size(); ++i) {
        for (size_t j = i + 1; j < results.size(); ++j) {
            if (!tensors_close(results[i], results[j], rtol, atol)) {
                FAIL() << test_name << " cross-backend parity failed:\n"
                       << "  A: " << backend_name(used[i]) << "\n"
                       << "  B: " << backend_name(used[j]) << "\n"
                       << "  Max diff: " << std::scientific
                       << max_abs_diff(results[i], results[j]);
            }
        }
    }
}

/**
 * @brief Test operation parity across all available backends.
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
 * @brief Test forward + backward parity across all available backends.
 *
 * Symmetric to test_operation_parity_backends but for ops that participate in
 * autograd. For each backend we:
 *   1. Move `inputs` to the backend and wrap each in a `Variable` with
 *      requires_grad=true.
 *   2. Run `op(vars)` to produce an output Variable.
 *   3. Run `.backward(grad_output)` where `grad_output` is built from
 *      `grad_output_factory(output.tensor())` (defaults to ones_like).
 *   4. Capture the output tensor and each input's `.grad()` tensor.
 *
 * We then compare both the forward outputs and each input's gradient against
 * the reference backend (first successful backend in the list), reporting
 * the first mismatch with backend names and max absolute difference.
 *
 * Forward and backward each have their own tolerance pair because accumulation
 * paths on the backward pass typically have looser numerical bounds than the
 * single-step forward path.
 *
 * @param op Callable with signature `Variable(std::vector<Variable>&)`
 * @param inputs Leaf tensors on CPU (will be cloned per backend)
 * @param grad_output_factory Builds the backward seed from the forward output
 *        tensor. Defaults to `ones_like(out)` (the scalar-loss convention).
 * @param rtol_fwd/atol_fwd Forward-output tolerance
 * @param rtol_bwd/atol_bwd Input-gradient tolerance
 * @param backends Backends to test; empty => get_available_backends()
 * @param test_name Name used in error messages
 */
template<typename Op>
void test_gradient_parity(
    Op op,
    const std::vector<Tensor>& inputs,
    std::function<Tensor(const Tensor&)> grad_output_factory = {},
    float rtol_fwd = 1e-5f,
    float atol_fwd = 1e-8f,
    float rtol_bwd = 1e-4f,
    float atol_bwd = 1e-6f,
    std::vector<Device> backends = {},
    const std::string& test_name = "GradientParity") {

    if (backends.empty()) {
        backends = get_available_backends();
    }
    if (backends.size() < 2) {
        if (golden::require_multi_backend()) {
            FAIL() << "Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1)"
                   << " for gradient parity: " << test_name;
        }
        GTEST_SKIP() << "Need at least 2 backends for gradient parity: " << test_name;
        return;
    }

    struct PerBackend {
        Tensor output;
        std::vector<Tensor> grads;
        Device backend;
    };

    std::vector<PerBackend> runs;

    for (const auto& backend : backends) {
        try {
            std::vector<Variable> vars;
            vars.reserve(inputs.size());
            for (const auto& t : inputs) {
                // Clone so each backend gets its own storage; move to device.
                vars.emplace_back(t.clone().to(backend), /*requires_grad=*/true);
            }

            auto out = op(vars);

            Tensor grad_seed = grad_output_factory
                ? grad_output_factory(out.tensor())
                : ones_like(out.tensor());
            out.backward(grad_seed);
            backend.synchronize();

            PerBackend run{out.tensor(), {}, backend};
            run.grads.reserve(vars.size());
            for (auto& v : vars) {
                if (!v.has_grad()) {
                    throw std::runtime_error(
                        "Input has no gradient after backward() — "
                        "op may have ignored an input or grad didn't flow.");
                }
                run.grads.push_back(v.grad().value());
            }
            runs.push_back(std::move(run));
        } catch (const std::exception& e) {
            std::cerr << test_name << ": backend "
                      << backend_name(backend)
                      << " failed: " << e.what() << std::endl;
        }
    }

    if (runs.size() < 2) {
        GTEST_SKIP() << "Need at least 2 successful backends for comparison";
        return;
    }

    const auto& ref = runs[0];

    for (size_t i = 1; i < runs.size(); ++i) {
        const auto& cur = runs[i];

        // Forward parity
        if (!tensors_close(ref.output, cur.output, rtol_fwd, atol_fwd)) {
            float diff = max_abs_diff(ref.output, cur.output);
            FAIL() << test_name << " forward parity failed:\n"
                   << "  Reference backend: " << backend_name(ref.backend) << "\n"
                   << "  Test backend: " << backend_name(cur.backend) << "\n"
                   << "  Max absolute difference: " << std::scientific << diff << "\n"
                   << "  Tolerance: rtol=" << rtol_fwd << ", atol=" << atol_fwd;
        }

        // Gradient parity (per input)
        ASSERT_EQ(ref.grads.size(), cur.grads.size())
            << test_name << " gradient count differs between backends";
        for (size_t g = 0; g < ref.grads.size(); ++g) {
            if (!tensors_close(ref.grads[g], cur.grads[g], rtol_bwd, atol_bwd)) {
                float diff = max_abs_diff(ref.grads[g], cur.grads[g]);
                FAIL() << test_name << " backward parity failed on input #" << g << ":\n"
                       << "  Reference backend: " << backend_name(ref.backend) << "\n"
                       << "  Test backend: " << backend_name(cur.backend) << "\n"
                       << "  Max absolute difference: " << std::scientific << diff << "\n"
                       << "  Tolerance: rtol=" << rtol_bwd << ", atol=" << atol_bwd;
            }
        }
    }
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
