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
#include <tuple>
#include <vector>
#include <string>
#include <memory>

namespace tenzor {
namespace testing {

// Bring Variable into scope for convenience
using Variable = tenzor::Variable;

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
        default: dtype_str = "Unknown"; break;
    }
    // Capitalize first letter of backend
    std::string backend_cap = backend;
    if (!backend_cap.empty()) {
        backend_cap[0] = std::toupper(backend_cap[0]);
    }
    return backend_cap + "_" + dtype_str;
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
 * @brief Get Device from backend name string
 * @throws std::runtime_error if backend name is unknown
 */
inline Device getDeviceFromName(const std::string& backend_name) {
    if (backend_name == "cpu") {
        return Device::cpu();
    } else if (backend_name == "cuda") {
        return Device::cuda(0);
    } else if (backend_name == "vulkan") {
        return Device::vulkan(0);
    } else if (backend_name == "oneapi") {
        return Device::oneapi(0);
    } else if (backend_name == "rocm") {
        return Device::rocm(0);
    } else if (backend_name == "metal") {
        return Device::metal(0);
    }
    throw std::runtime_error("Unknown backend: " + backend_name);
}

/**
 * @brief Check if backend name corresponds to an available backend
 */
inline bool isBackendNameAvailable(const std::string& backend_name) {
    if (backend_name == "cpu") {
        return true;  // CPU is always available
    } else if (backend_name == "cuda") {
        return isBackendAvailable(Device::Type::CUDA);
    } else if (backend_name == "vulkan") {
        return isBackendAvailable(Device::Type::Vulkan);
    } else if (backend_name == "oneapi") {
        return isBackendAvailable(Device::Type::OneAPI);
    } else if (backend_name == "rocm") {
        return isBackendAvailable(Device::Type::ROCm);
    } else if (backend_name == "metal") {
        return isBackendAvailable(Device::Type::Metal);
    }
    return false;
}

/**
 * @brief Get list of available backend names
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
    if (isBackendAvailable(Device::Type::Metal)) {
        backends.push_back("metal");
    }

    return backends;
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
    Device device_;
    DType dtype_;
    float rtol_;
    float atol_;
    static bool initialized_;

    void SetUp() override {
        // Initialize Tenzor library (only once)
        if (!initialized_) {
            tenzor::initialize();
            initialized_ = true;
        }

        auto [backend_name, dtype] = GetParam();
        dtype_ = dtype;

        // Check backend availability and skip if not available
        if (!isBackendNameAvailable(backend_name)) {
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
    Device device_;
    DType dtype_;
    float rtol_;
    float atol_;
    static bool initialized_;

    void SetUp() override {
        if (!initialized_) {
            tenzor::initialize();
            initialized_ = true;
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
 * @brief Standard float dtypes for most tests
 */
#define FLOAT_DTYPES \
    ::testing::Values(DType::Float32, DType::Float64, DType::Float16)

/**
 * @brief All float dtypes including BFloat16
 */
#define ALL_FLOAT_DTYPES \
    ::testing::Values(DType::Float32, DType::Float64, DType::Float16, DType::BFloat16)

/**
 * @brief Standard backends for testing
 */
#define STANDARD_BACKENDS \
    ::testing::Values("cpu", "cuda", "oneapi")

/**
 * @brief All backends including experimental ones
 */
#define ALL_BACKENDS \
    ::testing::Values("cpu", "cuda", "oneapi", "vulkan", "rocm")

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

} // namespace testing
} // namespace tenzor
