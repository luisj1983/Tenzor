#pragma once

/**
 * @file multi_backend_dtype_fixture.hpp
 * @brief Comprehensive test utilities for multi-backend and multi-dtype testing
 *
 * This header provides:
 * - MultiBackendDTypeTest: Parameterized fixture for testing across backends and dtypes
 * - Helper functions for dtype-safe tensor operations
 * - Tolerance settings per dtype
 * - Model conversion utilities
 * - Backend availability checking with GTEST_SKIP() support
 *
 * Usage:
 *   class MyModelTest : public MultiBackendDTypeTest {};
 *
 *   TEST_P(MyModelTest, ForwardPass) {
 *       auto model = createModel();
 *       convert_to_dtype(model);
 *       auto input = createInput({batch, channels, height, width});
 *       auto output = model->forward(input);
 *       EXPECT_EQ(output.tensor().dtype(), dtype());
 *   }
 *
 *   INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MyModelTest);
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/core/dtype.hpp>

// audit-2026-05-03 — disable TF32 BEFORE any matmul thread-local flag
// initialises. Static initialiser at file scope runs before main(), which
// is before any test runs, which is before the first cuBLAS call. This
// closes the LSTMCell-on-Cuda-Float32 gradcheck failures (the cuBLAS TF32
// default silently drops ~13 mantissa bits, breaking the 5e-3 tolerance).
namespace tenzor::testing::detail {
inline const int _disable_tf32_init = []() {
    ::setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/0);
    return 0;
}();
}
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace tenzor {
namespace testing {

// Bring Variable into scope for convenience
using Variable = tenzor::Variable;

// ============================================================================
// Skip Reason Taxonomy
// ============================================================================
/**
 * Machine-readable reasons a parity/unit test may skip. Passed to SKIP_WITH_REASON
 * so scripts/count_skips.py can trend skip-by-category over time. If you reach
 * for "untagged" you are almost certainly wrong — find or add the right enum.
 */
enum class SkipReason {
    BackendUnavailable,           // device genuinely absent on host
    BackendExcludedByEnv,         // TENZOR_SKIP_BACKENDS
    NumericalDivergence,          // algorithm loses precision below Float32
    DtypeUnsupportedOnBackend,    // kernel explicitly doesn't register dtype
    ComplexFP16Unrepresentable,   // no Float16 complex type
    GradcheckFDPrecision,         // finite-difference noise dominates at FP16
    KernelNotImplemented,         // feature is genuinely TODO
    RequiresMultiGPU,             // multi-device distributed coverage
    KnownBug,                     // track with issue #; do not paper over
};

inline const char* skip_reason_string(SkipReason r) {
    switch (r) {
        case SkipReason::BackendUnavailable:         return "BackendUnavailable";
        case SkipReason::BackendExcludedByEnv:       return "BackendExcludedByEnv";
        case SkipReason::NumericalDivergence:        return "NumericalDivergence";
        case SkipReason::DtypeUnsupportedOnBackend:  return "DtypeUnsupportedOnBackend";
        case SkipReason::ComplexFP16Unrepresentable: return "ComplexFP16Unrepresentable";
        case SkipReason::GradcheckFDPrecision:       return "GradcheckFDPrecision";
        case SkipReason::KernelNotImplemented:       return "KernelNotImplemented";
        case SkipReason::RequiresMultiGPU:           return "RequiresMultiGPU";
        case SkipReason::KnownBug:                   return "KnownBug";
    }
    return "Unknown";
}

/**
 * Emit a skip with a structured reason. The reason name appears in the skip
 * message so count_skips.py can tally categories via a simple grep.
 *
 * Usage:
 *   SKIP_WITH_REASON(SkipReason::GradcheckFDPrecision, "randn LU precision");
 */
#define SKIP_WITH_REASON(reason, detail) \
    GTEST_SKIP() << "[" << ::tenzor::testing::skip_reason_string(reason) << "] " << detail

// HONOR_BACKEND_ENV_VARS is defined in backend_test_fixture.hpp so it is
// reachable from custom fixtures that declare their own local
// `BackendDTypeParam` struct. (Importing this header's canonical
// `BackendDTypeParam` tuple alias via `using namespace tenzor::testing` would
// collide with those local structs.)

// ============================================================================
// Backend Name Parsing Utilities
// ============================================================================

/**
 * @brief Extract the base backend name from a possibly-indexed string.
 *
 * "cuda:1" -> "cuda", "cuda" -> "cuda", "cpu" -> "cpu"
 */
inline std::string parseBackendName(const std::string& s) {
    auto pos = s.find(':');
    return (pos == std::string::npos) ? s : s.substr(0, pos);
}

/**
 * @brief Extract the device index from a possibly-indexed string.
 *
 * "cuda:1" -> 1, "cuda" -> 0, "cpu" -> 0
 */
inline int32_t parseDeviceIndex(const std::string& s) {
    auto pos = s.find(':');
    if (pos == std::string::npos) return 0;
    return std::stoi(s.substr(pos + 1));
}

/**
 * @brief Map a base backend name to Device::Type.
 *
 * @throws std::runtime_error if name is unknown
 */
inline Device::Type nameToDeviceType(const std::string& name) {
    if (name == "cpu") return Device::Type::CPU;
    if (name == "cuda") return Device::Type::CUDA;
    if (name == "vulkan") return Device::Type::Vulkan;
    if (name == "oneapi") return Device::Type::OneAPI;
    if (name == "rocm") return Device::Type::ROCm;
    throw std::runtime_error("Unknown backend name: " + name);
}

/**
 * @brief Format a backend string as a GTest-safe test name.
 *
 * "cuda:1" -> "cuda1", "cpu" -> "cpu", "cuda" -> "cuda0"
 *
 * Lowercase is chosen deliberately so the generated CTest name (e.g.
 * `MultiBackendDType/MyTest.Foo/cuda0_Float32`) matches `ctest -R cuda`.
 * GTest parameter-name identifiers permit `[a-zA-Z0-9_]`, so leaving the
 * leading character lowercase is still valid.
 */
inline std::string formatBackendTestName(const std::string& backend) {
    auto base = parseBackendName(backend);
    auto index = parseDeviceIndex(backend);
    std::string result = base;  // already lowercase by convention
    if (base != "cpu") {
        result += std::to_string(index);
    }
    return result;
}

// ============================================================================
// Test Parameter Types
// ============================================================================

/**
 * @brief Parameter tuple for combined backend + dtype testing
 * First element: backend name (string)
 * Second element: dtype (DType)
 */
using BackendDTypeParam = std::tuple<std::string, DType>;

/**
 * @brief Get human-readable name for test parameter
 */
inline std::string BackendDTypeParamName(
    const ::testing::TestParamInfo<BackendDTypeParam>& info) {
    auto [backend, dtype] = info.param;
    std::string dtype_str;
    switch (dtype) {
        case DType::Float32: dtype_str = "Float32"; break;
        case DType::Float64: dtype_str = "Float64"; break;
        case DType::Float16: dtype_str = "Float16"; break;
        case DType::BFloat16: dtype_str = "BFloat16"; break;
        case DType::Int8: dtype_str = "Int8"; break;
        case DType::Int16: dtype_str = "Int16"; break;
        case DType::Int32: dtype_str = "Int32"; break;
        case DType::Int64: dtype_str = "Int64"; break;
        case DType::UInt8: dtype_str = "UInt8"; break;
        case DType::Bool: dtype_str = "Bool"; break;
        case DType::Complex64: dtype_str = "Complex64"; break;
        case DType::Complex128: dtype_str = "Complex128"; break;
        default: dtype_str = "Unknown"; break;
    }
    return formatBackendTestName(backend) + "_" + dtype_str;
}

/**
 * @brief Get human-readable name for dtype-only test parameter
 */
inline std::string DTypeParamName(const ::testing::TestParamInfo<DType>& info) {
    switch (info.param) {
        case DType::Float32: return "Float32";
        case DType::Float64: return "Float64";
        case DType::Float16: return "Float16";
        case DType::BFloat16: return "BFloat16";
        case DType::Int8: return "Int8";
        case DType::Int16: return "Int16";
        case DType::Int32: return "Int32";
        case DType::Int64: return "Int64";
        default: return "Unknown";
    }
}

// ============================================================================
// Backend Availability Utilities
// ============================================================================

/**
 * @brief Check if a backend is available
 */
inline bool isBackendAvailable(Device::Type backend_type, int32_t index = 0) {
    try {
        Device test_device{backend_type, index};
        auto t = zeros({2, 2}, DType::Float32, test_device);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Get Device from backend name string.
 *
 * Handles both legacy format ("cuda" -> device 0) and indexed format ("cuda:1").
 *
 * @throws std::runtime_error if backend name is unknown
 */
inline Device getDeviceFromName(const std::string& backend_name) {
    auto base = parseBackendName(backend_name);
    auto index = parseDeviceIndex(backend_name);
    if (base == "cpu") return Device::cpu();
    return Device{nameToDeviceType(base), index};
}

/**
 * @brief Check if backend name corresponds to an available backend.
 *
 * Handles both legacy format ("cuda") and indexed format ("cuda:1").
 */
// Returns true if `backend` appears in the comma-separated $TENZOR_SKIP_BACKENDS list.
// Matching is on the base backend name; lower-case, whitespace-trimmed tokens.
inline bool isBackendSkippedByEnv(const std::string& backend) {
    const char* raw = std::getenv("TENZOR_SKIP_BACKENDS");
    if (!raw || !*raw) return false;
    std::string_view target{backend};
    std::string_view list{raw};
    size_t start = 0;
    while (start <= list.size()) {
        size_t end = list.find(',', start);
        if (end == std::string_view::npos) end = list.size();
        auto token = list.substr(start, end - start);
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.remove_prefix(1);
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) token.remove_suffix(1);
        if (!token.empty() && token == target) return true;
        start = end + 1;
    }
    return false;
}

inline bool isBackendNameAvailable(const std::string& backend_name) {
    auto base = parseBackendName(backend_name);
    auto index = parseDeviceIndex(backend_name);
    if (base == "cpu") return true;
    try {
        return isBackendAvailable(nameToDeviceType(base), index);
    } catch (...) {
        return false;
    }
}

/**
 * @brief Get list of available backend names (single device per backend).
 */
inline std::vector<std::string> getAvailableBackends() {
    std::vector<std::string> backends = {"cpu"};  // CPU always available

    if (isBackendAvailable(Device::Type::CUDA)) {
        backends.push_back("cuda");
    }
    if (isBackendAvailable(Device::Type::OneAPI)) {
        backends.push_back("oneapi");
    }
    if (isBackendAvailable(Device::Type::Vulkan)) {
        backends.push_back("vulkan");
    }
    if (isBackendAvailable(Device::Type::ROCm)) {
        backends.push_back("rocm");
    }

    return backends;
}

/**
 * @brief Get list of available backends with all device indices enumerated.
 *
 * Queries each backend's device_count() and probes each device.
 * Returns strings like {"cpu", "cuda:0", "cuda:1", "rocm:0"}.
 */
inline std::vector<std::string> getAvailableBackendsWithDevices() {
    static std::once_flag init;
    std::call_once(init, []() { tenzor::initialize(); });

    std::vector<std::string> result = {"cpu"};

    struct BackendInfo { const char* name; Device::Type type; };
    constexpr BackendInfo backends[] = {
        {"cuda", Device::Type::CUDA},
        {"oneapi", Device::Type::OneAPI},
        {"vulkan", Device::Type::Vulkan},
        {"rocm", Device::Type::ROCm},
    };

    for (const auto& [name, type] : backends) {
        auto* backend = backend_registry().get_backend(type);
        if (!backend || !backend->is_available()) continue;
        int32_t count = backend->device_count();
        for (int32_t i = 0; i < count; ++i) {
            if (isBackendAvailable(type, i)) {
                result.push_back(std::string(name) + ":" + std::to_string(i));
            }
        }
    }

    return result;
}

// ============================================================================
// Multi-Backend + Multi-DType Test Fixture
// ============================================================================

/**
 * @brief Test fixture parameterized by both backend and dtype
 *
 * Provides:
 * - Automatic backend availability checking with GTEST_SKIP()
 * - Dtype-appropriate tolerances
 * - Helper methods for tensor/variable creation
 * - Model dtype conversion utilities
 * - Dtype-safe reduction operations (for Float16 compatibility)
 */
class MultiBackendDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    // device_ defaults to CPU so that derived SetUp() methods which run
    // after a GTEST_SKIP() returns from this base SetUp() observe a
    // well-defined Device::Type. Without this, Device::type is an
    // uninitialised uint8_t and code paths like
    //   if (device().type == Device::Type::CPU) GTEST_SKIP();
    // (in the DataParallel fixtures) read garbage and may hit downstream
    // backend dispatch with an invalid type, manifesting as
    // "Subprocess aborted" instead of a clean skip.
    Device device_ = Device::cpu();
    DType dtype_ = DType::Float32;
    float rtol_ = 1e-4f;
    float atol_ = 1e-5f;
    static bool initialized_;

    void SetUp() override {
        // Initialize Tenzor library (only once)
        if (!initialized_) {
            tenzor::initialize();
            initialized_ = true;
        }

        // Deterministic RNG seed per test — keeps randn() output stable across
        // process invocations so recorded golden tensors stay valid.
        {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            std::string key = info ? std::string(info->test_suite_name()) + "."
                                       + info->name()
                                   : std::string("default");
            uint32_t seed = 0x811c9dc5u;
            for (char c : key) {
                seed ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
                seed *= 0x01000193u;
            }
            tenzor::manual_seed(seed);
        }

        auto [backend_name, dtype] = GetParam();
        dtype_ = dtype;

        // Honor TENZOR_SKIP_BACKENDS opt-out first — matches on the base
        // backend name, so "cuda:1" is covered by "cuda".
        if (isBackendSkippedByEnv(parseBackendName(backend_name))) {
            GTEST_SKIP() << parseBackendName(backend_name)
                         << " excluded via TENZOR_SKIP_BACKENDS";
        }

        // Check backend availability. If TENZOR_REQUIRE_MULTI_BACKEND=1 is
        // set in the environment, an unavailable non-CPU backend becomes a
        // hard FAIL instead of a skip — surfaces broken CI environments.
        // TENZOR_SKIP_BACKENDS wins over this, already handled above.
        if (!isBackendNameAvailable(backend_name)) {
            const char* req = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");
            if (req && *req && *req != '0' &&
                parseBackendName(backend_name) != "cpu") {
                FAIL() << backend_name << " required by "
                          "TENZOR_REQUIRE_MULTI_BACKEND but unavailable";
            }
            GTEST_SKIP() << backend_name << " backend not available";
        }

        device_ = getDeviceFromName(backend_name);

        // Set tolerances based on dtype
        setTolerances();
    }

    /**
     * @brief Set relative and absolute tolerances based on dtype
     */
    void setTolerances() {
        switch (dtype_) {
            case DType::Float16:
                rtol_ = 1e-2f;
                atol_ = 1e-2f;
                break;
            case DType::BFloat16:
                rtol_ = 1e-2f;
                atol_ = 1e-2f;
                break;
            case DType::Float32:
                rtol_ = 1e-4f;
                atol_ = 1e-5f;
                break;
            case DType::Float64:
                rtol_ = 1e-6f;
                atol_ = 1e-7f;
                break;
            default:
                rtol_ = 1e-4f;
                atol_ = 1e-5f;
                break;
        }
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    Device device() const { return device_; }
    DType dtype() const { return dtype_; }
    float rtol() const { return rtol_; }
    float atol() const { return atol_; }

    std::string backend_name() const {
        return std::get<0>(GetParam());
    }

    // -------------------------------------------------------------------------
    // Tensor/Variable Creation Helpers
    // -------------------------------------------------------------------------

    /**
     * @brief Create a tensor with the test dtype and device
     */
    Tensor createTensor(const std::vector<int64_t>& shape) {
        return Tensor(shape, dtype_, device_);
    }

    /**
     * @brief Create a Variable (autograd-enabled tensor) with the test dtype and device
     *
     * Uses random initialization (randn) to ensure meaningful values for operations
     * like BatchNorm that require input variation.
     */
    Variable createInput(const std::vector<int64_t>& shape,
                                   bool requires_grad = true) {
        // Create random tensor in Float32 first (randn may not support all dtypes)
        auto t = tenzor::randn(shape, DType::Float32, device_);
        if (dtype_ != DType::Float32) {
            t = t.to(dtype_);
        }
        return Variable(t, requires_grad);
    }

    /**
     * @brief Create a zeros tensor with the test dtype and device
     */
    Tensor createZeros(const std::vector<int64_t>& shape) {
        return tenzor::zeros(shape, dtype_, device_);
    }

    /**
     * @brief Create a ones tensor with the test dtype and device
     */
    Tensor createOnes(const std::vector<int64_t>& shape) {
        return tenzor::ones(shape, dtype_, device_);
    }

    /**
     * @brief Create a randn tensor with the test dtype and device
     */
    Tensor createRandn(const std::vector<int64_t>& shape) {
        // randn may not support all dtypes, so create in Float32 and convert
        auto t = tenzor::randn(shape, DType::Float32, device_);
        if (dtype_ != DType::Float32) {
            t = t.to(dtype_);
        }
        return t;
    }

    // -------------------------------------------------------------------------
    // Model Conversion Helpers
    // -------------------------------------------------------------------------

    /**
     * @brief Convert a model to the test dtype and device
     * @tparam ModuleT The module type (shared_ptr or direct object)
     *
     * Uses if constexpr to detect pointer-like types and call appropriately.
     */
    template <typename ModuleT>
    void convert_model(ModuleT& model) {
        // First move to device, then convert dtype
        if constexpr (requires { model->to(device_); }) {
            // Pointer-like type (shared_ptr, unique_ptr, raw pointer)
            model->to(device_);
            if (dtype_ != DType::Float32) {
                model->to(dtype_);
            }
        } else {
            // Direct object type
            model.to(device_);
            if (dtype_ != DType::Float32) {
                model.to(dtype_);
            }
        }
    }

    /**
     * @brief Alias for convert_model (for backward compatibility)
     */
    template <typename ModuleT>
    void convert_to_dtype(ModuleT& model) {
        convert_model(model);
    }

    // -------------------------------------------------------------------------
    // Dtype-Safe Reduction Helpers (Float16 compatibility)
    // -------------------------------------------------------------------------

    /**
     * @brief Compute max(abs(tensor)) handling all dtypes including Float16
     *
     * Float16 tensors are converted to Float32 before reduction since
     * some reduction ops don't support Float16 directly.
     */
    float compute_max_abs(const Tensor& tensor) const {
        Tensor t = tensor.to(Device::cpu());  // Move to CPU for data access
        if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
            t = t.to(DType::Float32);
        }
        auto max_result = tenzor::max(tenzor::abs(t));
        if (max_result.dtype() == DType::Float64) {
            return static_cast<float>(max_result.item<double>());
        }
        return max_result.item<float>();
    }

    /**
     * @brief Compute max(tensor) handling all dtypes including Float16
     */
    float compute_max(const Tensor& tensor) const {
        Tensor t = tensor.to(Device::cpu());
        if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
            t = t.to(DType::Float32);
        }
        auto max_result = tenzor::max(t);
        if (max_result.dtype() == DType::Float64) {
            return static_cast<float>(max_result.item<double>());
        }
        return max_result.item<float>();
    }

    /**
     * @brief Compute min(tensor) handling all dtypes including Float16
     */
    float compute_min(const Tensor& tensor) const {
        Tensor t = tensor.to(Device::cpu());
        if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
            t = t.to(DType::Float32);
        }
        auto min_result = tenzor::min(t);
        if (min_result.dtype() == DType::Float64) {
            return static_cast<float>(min_result.item<double>());
        }
        return min_result.item<float>();
    }

    /**
     * @brief Compute mean(tensor) handling all dtypes including Float16
     */
    float compute_mean(const Tensor& tensor) const {
        Tensor t = tensor.to(Device::cpu());
        if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
            t = t.to(DType::Float32);
        }
        auto mean_result = tenzor::mean(t);
        if (mean_result.dtype() == DType::Float64) {
            return static_cast<float>(mean_result.item<double>());
        }
        return mean_result.item<float>();
    }

    // -------------------------------------------------------------------------
    // Tensor Comparison Helpers
    // -------------------------------------------------------------------------

    /**
     * @brief Compare two tensors element-wise with dtype-appropriate tolerance
     */
    void expectTensorNear(const Tensor& actual, const Tensor& expected) {
        expectTensorNear(actual, expected, atol_);
    }

    /**
     * @brief Compare two tensors element-wise with custom tolerance
     */
    void expectTensorNear(const Tensor& actual, const Tensor& expected,
                          float tolerance) {
        ASSERT_EQ(actual.numel(), expected.numel())
            << "Tensors have different number of elements";

        // Move both to CPU and Float32 for comparison
        auto a_cpu = actual.to(Device::cpu());
        auto b_cpu = expected.to(Device::cpu());

        if (a_cpu.dtype() != DType::Float32) {
            a_cpu = a_cpu.to(DType::Float32);
        }
        if (b_cpu.dtype() != DType::Float32) {
            b_cpu = b_cpu.to(DType::Float32);
        }

        auto* a_data = a_cpu.data<float>();
        auto* b_data = b_cpu.data<float>();

        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            EXPECT_NEAR(a_data[i], b_data[i], tolerance)
                << "Mismatch at index " << i
                << " on device " << device_.to_string()
                << " with dtype " << static_cast<int>(dtype_);
        }
    }

    /**
     * @brief Check that tensor has expected shape
     */
    void expectShape(const Tensor& tensor,
                     const std::vector<int64_t>& expected_shape) {
        auto shape = tensor.shape();
        EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), expected_shape)
            << "Shape mismatch on device " << device_.to_string();
    }

    /**
     * @brief Check that tensor has the test dtype
     */
    void expectDType(const Tensor& tensor) {
        EXPECT_EQ(tensor.dtype(), dtype_)
            << "DType mismatch: expected " << static_cast<int>(dtype_)
            << " but got " << static_cast<int>(tensor.dtype());
    }

    /**
     * @brief Check that tensor is on the test device
     */
    void expectDevice(const Tensor& tensor) {
        EXPECT_EQ(tensor.device().type, device_.type)
            << "Device mismatch";
    }

    // -------------------------------------------------------------------------
    // Parameter Counting Helper
    // -------------------------------------------------------------------------

    /**
     * @brief Count total number of parameters in a parameter list
     */
    size_t countParameters(
        const std::vector<std::shared_ptr<Variable>>& params) {
        size_t total = 0;
        for (const auto& p : params) {
            size_t param_size = 1;
            for (auto dim : p->tensor().shape()) {
                param_size *= dim;
            }
            total += param_size;
        }
        return total;
    }
};

// Static member initialization
inline bool MultiBackendDTypeTest::initialized_ = false;

// ============================================================================
// Multi-DType Only Test Fixture (for tests that only need dtype variation)
// ============================================================================

/**
 * @brief Test fixture parameterized by dtype only (uses CPU backend)
 *
 * Use this when you only need to test dtype variations on a single backend.
 * For full backend+dtype testing, use MultiBackendDTypeTest instead.
 */
class MultiDTypeTest : public ::testing::TestWithParam<DType> {
protected:
    Device device_ = Device::cpu();
    DType dtype_ = DType::Float32;
    float rtol_ = 1e-4f;
    float atol_ = 1e-5f;
    static bool initialized_;

    void SetUp() override {
        if (!initialized_) {
            tenzor::initialize();
            initialized_ = true;
        }

        // Deterministic RNG seed per test — keeps randn() output stable across
        // process invocations so recorded golden tensors stay valid.
        {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            std::string key = info ? std::string(info->test_suite_name()) + "."
                                       + info->name()
                                   : std::string("default");
            uint32_t seed = 0x811c9dc5u;
            for (char c : key) {
                seed ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
                seed *= 0x01000193u;
            }
            tenzor::manual_seed(seed);
        }

        device_ = Device::cpu();
        dtype_ = GetParam();
        setTolerances();
    }

    void setTolerances() {
        switch (dtype_) {
            case DType::Float16:
            case DType::BFloat16:
                rtol_ = 1e-2f;
                atol_ = 1e-2f;
                break;
            case DType::Float32:
                rtol_ = 1e-4f;
                atol_ = 1e-5f;
                break;
            case DType::Float64:
                rtol_ = 1e-6f;
                atol_ = 1e-7f;
                break;
            default:
                rtol_ = 1e-4f;
                atol_ = 1e-5f;
                break;
        }
    }

    // Same helper methods as MultiBackendDTypeTest
    Device device() const { return device_; }
    DType dtype() const { return dtype_; }
    float rtol() const { return rtol_; }
    float atol() const { return atol_; }

    Tensor createTensor(const std::vector<int64_t>& shape) {
        return Tensor(shape, dtype_, device_);
    }

    Variable createInput(const std::vector<int64_t>& shape,
                                   bool requires_grad = true) {
        // Create random tensor in Float32 first (randn may not support all dtypes)
        auto t = tenzor::randn(shape, DType::Float32, device_);
        if (dtype_ != DType::Float32) {
            t = t.to(dtype_);
        }
        return Variable(t, requires_grad);
    }

    template <typename ModuleT>
    void convert_model(ModuleT& model) {
        if constexpr (requires { model->to(dtype_); }) {
            if (dtype_ != DType::Float32) {
                model->to(dtype_);
            }
        } else {
            if (dtype_ != DType::Float32) {
                model.to(dtype_);
            }
        }
    }

    template <typename ModuleT>
    void convert_to_dtype(ModuleT& model) {
        convert_model(model);
    }

    float compute_max_abs(const Tensor& tensor) const {
        Tensor t = tensor;
        if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
            t = t.to(DType::Float32);
        }
        auto max_result = tenzor::max(tenzor::abs(t));
        if (max_result.dtype() == DType::Float64) {
            return static_cast<float>(max_result.item<double>());
        }
        return max_result.item<float>();
    }

    float compute_max(const Tensor& tensor) const {
        Tensor t = tensor;
        if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
            t = t.to(DType::Float32);
        }
        auto max_result = tenzor::max(t);
        if (max_result.dtype() == DType::Float64) {
            return static_cast<float>(max_result.item<double>());
        }
        return max_result.item<float>();
    }

    float compute_min(const Tensor& tensor) const {
        Tensor t = tensor;
        if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
            t = t.to(DType::Float32);
        }
        auto min_result = tenzor::min(t);
        if (min_result.dtype() == DType::Float64) {
            return static_cast<float>(min_result.item<double>());
        }
        return min_result.item<float>();
    }

    size_t countParameters(
        const std::vector<std::shared_ptr<Variable>>& params) {
        size_t total = 0;
        for (const auto& p : params) {
            size_t param_size = 1;
            for (auto dim : p->tensor().shape()) {
                param_size *= dim;
            }
            total += param_size;
        }
        return total;
    }
};

inline bool MultiDTypeTest::initialized_ = false;

// ============================================================================
// Test Instantiation Macros
// ============================================================================

/**
 * @brief Standard float dtypes for most tests.
 *
 * Defaults to Float32/Float64/Float16 to keep the default parity test
 * matrix at 3 dtypes × 5 backends = 15 cases. Define TENZOR_TEST_BFLOAT16
 * at build time (e.g. `cmake -DTENZOR_TEST_BFLOAT16=ON`) to additionally
 * include BFloat16 in every multi-dtype parity test — useful for
 * BF16-specialised CI runs. Setting this flag globally doubles the number
 * of `INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS` instantiations so it is an
 * intentional opt-in rather than the default (the full suite takes 15+ hrs
 * already and a 4-dtype default would be disproportionate).
 */
#ifdef TENZOR_TEST_BFLOAT16
#define FLOAT_DTYPES \
    ::testing::Values(DType::Float32, DType::Float64, DType::Float16, DType::BFloat16)
#else
#define FLOAT_DTYPES \
    ::testing::Values(DType::Float32, DType::Float64, DType::Float16)
#endif

/**
 * @brief All float dtypes including BFloat16 — unconditional.
 *
 * Use when a test specifically needs BFloat16 coverage regardless of the
 * build-time default.
 */
#define ALL_FLOAT_DTYPES \
    ::testing::Values(DType::Float32, DType::Float64, DType::Float16, DType::BFloat16)

/**
 * @brief Standard backends for testing.
 *
 * Hardcoded because `INSTANTIATE_TEST_SUITE_P` is evaluated at static-init
 * time, BEFORE `tenzor::initialize()` has run — so `getAvailableBackends()`
 * returns only "cpu" at that point and the test matrix collapses to CPU
 * for every macro-using test. Keeping the list static ensures every
 * backend×dtype combination is discovered; the fixture's SetUp() skips
 * unavailable ones at test time (honoring TENZOR_REQUIRE_MULTI_BACKEND).
 *
 * New backend (Metal, WebGPU, etc.): add to this list + teach SetUp how
 * to construct the Device. That is intentional — adding a backend is a
 * deliberate act, not a silent auto-discovery.
 */
#define STANDARD_BACKENDS \
    ::testing::Values("cpu", "cuda", "vulkan", "oneapi", "rocm")

/**
 * @brief All backends — same as STANDARD_BACKENDS for now; reserved for a
 * future expansion that enumerates per-device indices (cuda:0, cuda:1, ...)
 * once a mechanism exists to do so at static-init time without triggering
 * backend loading.
 */
#define ALL_BACKENDS \
    ::testing::Values("cpu", "cuda", "vulkan", "oneapi", "rocm")

/**
 * @brief Generate all combinations of backends and dtypes
 */
inline auto GenerateBackendDTypeParams(
    const std::vector<std::string>& backends = {"cpu", "cuda", "oneapi"},
    const std::vector<DType>& dtypes = {DType::Float32, DType::Float64, DType::Float16}) {

    std::vector<BackendDTypeParam> params;
    for (const auto& backend : backends) {
        for (const auto& dtype : dtypes) {
            params.emplace_back(backend, dtype);
        }
    }
    return ::testing::ValuesIn(params);
}

/**
 * @brief Instantiate tests for all backend + dtype combinations
 *
 * Usage:
 *   INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MyTestFixture);
 */
#define INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        MultiBackendDType, \
        TestSuiteName, \
        ::testing::Combine( \
            STANDARD_BACKENDS, \
            FLOAT_DTYPES \
        ), \
        [](const ::testing::TestParamInfo<BackendDTypeParam>& info) { \
            return BackendDTypeParamName(info); \
        } \
    )

/**
 * @brief Instantiate tests for all backends + all dtypes (including BFloat16)
 */
#define INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        MultiBackendAllDType, \
        TestSuiteName, \
        ::testing::Combine( \
            STANDARD_BACKENDS, \
            ALL_FLOAT_DTYPES \
        ), \
        [](const ::testing::TestParamInfo<BackendDTypeParam>& info) { \
            return BackendDTypeParamName(info); \
        } \
    )

/**
 * @brief Instantiate tests for dtype only (CPU backend)
 *
 * Usage:
 *   INSTANTIATE_MULTI_DTYPE_TESTS(MyTestFixture);
 */
#define INSTANTIATE_MULTI_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        MultiDType, \
        TestSuiteName, \
        FLOAT_DTYPES, \
        DTypeParamName \
    )

/**
 * @brief Instantiate tests for all dtypes including BFloat16 (CPU backend)
 */
#define INSTANTIATE_ALL_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        AllDTypes, \
        TestSuiteName, \
        ALL_FLOAT_DTYPES, \
        DTypeParamName \
    )

/**
 * @brief Instantiate tests for backends only (Float32 dtype)
 */
#define INSTANTIATE_MULTI_BACKEND_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        MultiBackend, \
        TestSuiteName, \
        STANDARD_BACKENDS, \
        [](const ::testing::TestParamInfo<std::string>& info) { \
            return info.param; \
        } \
    )

// ============================================================================
// Multi-Device Test Instantiation Macros
// ============================================================================

/**
 * @brief Runtime-discovered backends with per-device enumeration.
 *
 * Unlike STANDARD_BACKENDS which lists static backend names,
 * this queries device_count() at runtime and emits entries like
 * "cuda:0", "cuda:1", "rocm:0", etc.
 */
#define DISCOVERED_BACKENDS_WITH_DEVICES \
    ::testing::ValuesIn(::tenzor::testing::getAvailableBackendsWithDevices())

/**
 * @brief Instantiate tests for all discovered devices + dtype combinations.
 *
 * Generates test instances like Cuda0_Float32, Cuda1_Float32, Rocm0_Float16, etc.
 */
#define INSTANTIATE_MULTI_DEVICE_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        MultiDeviceDType, \
        TestSuiteName, \
        ::testing::Combine( \
            DISCOVERED_BACKENDS_WITH_DEVICES, \
            FLOAT_DTYPES \
        ), \
        [](const ::testing::TestParamInfo<BackendDTypeParam>& info) { \
            return BackendDTypeParamName(info); \
        } \
    )

/**
 * @brief Instantiate tests for all discovered devices + all dtypes (including BFloat16).
 */
#define INSTANTIATE_MULTI_DEVICE_ALL_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        MultiDeviceAllDType, \
        TestSuiteName, \
        ::testing::Combine( \
            DISCOVERED_BACKENDS_WITH_DEVICES, \
            ALL_FLOAT_DTYPES \
        ), \
        [](const ::testing::TestParamInfo<BackendDTypeParam>& info) { \
            return BackendDTypeParamName(info); \
        } \
    )

/**
 * @brief Instantiate tests for all discovered devices only (Float32 dtype).
 */
#define INSTANTIATE_MULTI_DEVICE_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        MultiDevice, \
        TestSuiteName, \
        DISCOVERED_BACKENDS_WITH_DEVICES, \
        [](const ::testing::TestParamInfo<std::string>& info) { \
            return ::tenzor::testing::formatBackendTestName(info.param); \
        } \
    )

// ============================================================================
// Global opt-in: -DTENZOR_TEST_ALL_DEVICES
//
// When TENZOR_TEST_ALL_DEVICES is defined (via CMake option), the standard
// single-device macros are redirected to their multi-device equivalents.
// This lets you enable multi-device testing for the entire test suite from
// CMake without touching any test files:
//
//   cmake -B build -DTENZOR_TEST_ALL_DEVICES=ON ...
// ============================================================================
#ifdef TENZOR_TEST_ALL_DEVICES

#undef INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS
#define INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_MULTI_DEVICE_DTYPE_TESTS(TestSuiteName)

#undef INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS
#define INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(TestSuiteName) \
    INSTANTIATE_MULTI_DEVICE_ALL_DTYPE_TESTS(TestSuiteName)

#undef INSTANTIATE_MULTI_BACKEND_TESTS
#define INSTANTIATE_MULTI_BACKEND_TESTS(TestSuiteName) \
    INSTANTIATE_MULTI_DEVICE_TESTS(TestSuiteName)

#endif // TENZOR_TEST_ALL_DEVICES

} // namespace testing
} // namespace tenzor
