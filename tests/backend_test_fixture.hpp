#pragma once

// ============================================================================
// IMPORTANT: no test file may call `tenzor::initialize()` directly.
// ----------------------------------------------------------------------------
// Use `tenzor::testing::EnsureInitialized()` (declared below) from any
// SetUp / SetUpTestSuite / Environment::SetUp that needs the runtime to be
// initialised. It wraps the canonical `std::call_once` so the underlying
// `tenzor::initialize()` runs exactly once per process across all test
// suites, regardless of which order Google Test instantiates them in. A
// direct call risks reinitialising backends in the middle of another
// suite's run, and bypasses the TF32-disable defaults this helper applies.
// ============================================================================

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>

namespace tenzor {
namespace testing {

// ----------------------------------------------------------------------------
// EnsureInitialized: process-wide one-shot library bring-up for test fixtures.
// ----------------------------------------------------------------------------
// Thread-safe via std::call_once. Sets TENZOR_DISABLE_TF32=1 (only if the
// caller has not already set it) so cuBLAS does not silently downgrade FP32
// matmul to TF32 on Ampere+, which would blow through the parity tolerances
// the test suite uses. Idempotent: subsequent calls are no-ops.
//
// Call this from BackendTest::SetUp, any TestEnvironment::SetUp, or any
// custom fixture's SetUp / SetUpTestSuite. Never call `tenzor::initialize()`
// directly from a test (see header banner above).
inline void EnsureInitialized() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        // Force IEEE 754 FP32 on CUDA matmul — cuBLAS defaults to TF32 on
        // Ampere+, which silently drops ~13 mantissa bits and blows through
        // the 1e-4/1e-5 tolerances most parity tests use. Respect the
        // caller's explicit value if they set it themselves.
        setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/0);
        tenzor::initialize();
    });
}

// Forward declarations for parsing utilities (defined in multi_backend_dtype_fixture.hpp).
// Duplicated here to keep this header self-contained.
namespace detail {

inline std::string parseBackendName(const std::string& s) {
    auto pos = s.find(':');
    return (pos == std::string::npos) ? s : s.substr(0, pos);
}

inline int32_t parseDeviceIndex(const std::string& s) {
    auto pos = s.find(':');
    if (pos == std::string::npos) return 0;
    return std::stoi(s.substr(pos + 1));
}

inline Device::Type nameToDeviceType(const std::string& name) {
    if (name == "cpu") return Device::Type::CPU;
    if (name == "cuda") return Device::Type::CUDA;
    if (name == "vulkan") return Device::Type::Vulkan;
    if (name == "oneapi") return Device::Type::OneAPI;
    if (name == "rocm") return Device::Type::ROCm;
    throw std::runtime_error("Unknown backend name: " + name);
}

// Returns true if `backend` appears in the comma-separated $TENZOR_SKIP_BACKENDS list.
// Matching is on the base backend name ("cuda"), not "cuda:0". Case-sensitive, lower-case.
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

inline std::string formatBackendTestName(const std::string& backend) {
    auto base = parseBackendName(backend);
    auto index = parseDeviceIndex(backend);
    std::string result = base;
    if (!result.empty()) {
        result[0] = std::toupper(result[0]);
    }
    if (base != "cpu") {
        result += std::to_string(index);
    }
    return result;
}

/// Probe whether a backend name ("cuda", "cuda:0", "vulkan") is available.
/// Returns true for "cpu" unconditionally; for other backends attempts a
/// trivial allocation on the target device to confirm the backend loaded.
inline bool isBackendNameAvailable(const std::string& backend_name) {
    auto base = parseBackendName(backend_name);
    auto index = parseDeviceIndex(backend_name);
    if (base == "cpu") return true;
    try {
        Device test_device{nameToDeviceType(base), index};
        auto t = zeros({2, 2}, DType::Float32, test_device);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace detail

// ----------------------------------------------------------------------------
// HONOR_BACKEND_ENV_VARS: canonical env-var handling for custom-fixture SetUp()
// ----------------------------------------------------------------------------
// Call from a SetUp() that resolves its own backend name from GetParam() —
// i.e., any custom BackendDTypeParam fixture that does not inherit
// BackendTest / MultiBackendDTypeTest. Invoke with the backend-name string
// AFTER reading GetParam() and BEFORE constructing the Device. It:
//   1. Honors TENZOR_SKIP_BACKENDS (matches on base name, skips this case).
//   2. For non-CPU backends that are genuinely unavailable, escalates to
//      FAIL when TENZOR_REQUIRE_MULTI_BACKEND=1 is set, else GTEST_SKIP.
//   3. Otherwise returns normally so SetUp continues.
// The macro form is required because GTEST_SKIP/FAIL use `return` internally
// and must execute in the test method's scope, not a nested function.
// Defined here (backend_test_fixture.hpp) rather than multi_backend_dtype_
// fixture.hpp so it is reachable from custom fixtures that declare their own
// local `BackendDTypeParam` struct — pulling in the canonical fixture's
// `BackendDTypeParam` tuple alias would conflict with those locals.
#define HONOR_BACKEND_ENV_VARS(backend_name_str)                              \
    do {                                                                      \
        std::string __tenzor_base =                                           \
            ::tenzor::testing::detail::parseBackendName(backend_name_str);    \
        if (::tenzor::testing::detail::isBackendSkippedByEnv(__tenzor_base)) {\
            GTEST_SKIP() << __tenzor_base                                     \
                         << " excluded via TENZOR_SKIP_BACKENDS";             \
        }                                                                     \
        if (!::tenzor::testing::detail::isBackendNameAvailable(               \
                backend_name_str)) {                                          \
            const char* __tenzor_req =                                        \
                std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");                  \
            if (__tenzor_req && *__tenzor_req && *__tenzor_req != '0' &&      \
                __tenzor_base != "cpu") {                                     \
                FAIL() << backend_name_str                                    \
                       << " required by TENZOR_REQUIRE_MULTI_BACKEND"         \
                          " but unavailable";                                 \
            }                                                                 \
            GTEST_SKIP() << backend_name_str << " backend not available";     \
        }                                                                     \
    } while (0)

/**
 * @brief Base test fixture for backend parity testing
 *
 * Supports both legacy backend names ("cuda") and indexed names ("cuda:1").
 *
 * Usage:
 *   TEST_P(BackendTest, MyTestName) {
 *       auto tensor = ones({10, 10}, DType::Float32, device);
 *       // ... test operations on this device ...
 *   }
 *
 * The 'device' member is automatically set based on the test parameter.
 */
class BackendTest : public ::testing::TestWithParam<std::string> {
protected:
    Device device;
    // NOTE: init_flag is retained for ABI/compat with any out-of-tree fixture
    // that may have referenced it, but is no longer used by SetUp — the
    // one-shot guard now lives inside EnsureInitialized() above.
    static std::once_flag init_flag;

    // Set TENZOR_REQUIRE_MULTI_BACKEND=1 in CI to turn every "backend not
    // available" skip below into a hard FAIL. Use this in jobs where a GPU
    // backend is supposed to be present — a silent skip would hide the
    // broken environment. TENZOR_SKIP_BACKENDS still wins: an explicit opt-
    // out skip never escalates to a failure regardless of the require flag.
    static bool require_multi_backend() {
        const char* v = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");
        return v && *v && *v != '0';
    }

    void SetUp() override {
        // Bring up the Tenzor runtime exactly once per process. Test files
        // MUST NOT call tenzor::initialize() directly — the helper applies
        // the TENZOR_DISABLE_TF32 default required by FP32 parity tests.
        ::tenzor::testing::EnsureInitialized();

        // Deterministic RNG seed per test. Parity tests that use randn() would
        // otherwise produce different inputs on every process invocation, which
        // makes recorded golden tensors mismatch the next run's inputs. Seeding
        // per (suite, test) gives each test a stable but unique input sequence.
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

        std::string backend_param = GetParam();
        auto base = detail::parseBackendName(backend_param);
        auto index = detail::parseDeviceIndex(backend_param);

        if (detail::isBackendSkippedByEnv(base)) {
            GTEST_SKIP() << base << " excluded via TENZOR_SKIP_BACKENDS";
        }

        if (base == "cpu") {
            device = Device::cpu();
        } else {
            auto type = detail::nameToDeviceType(base);
            if (!isBackendAvailable(type, index)) {
                if (require_multi_backend()) {
                    FAIL() << backend_param << " required by "
                              "TENZOR_REQUIRE_MULTI_BACKEND but unavailable";
                } else {
                    GTEST_SKIP() << backend_param << " not available";
                }
            }
            device = Device{type, index};
        }
    }

    // Helper to check if a backend is available
    static bool isBackendAvailable(Device::Type backend_type, int32_t index = 0) {
        try {
            Device test_device{backend_type, index};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to compare tensors across devices
    void expectTensorNear(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
        ASSERT_EQ(a.numel(), b.numel()) << "Tensors have different number of elements";

        // Move both to CPU for comparison
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());

        auto* a_data = a_cpu.data<float>();
        auto* b_data = b_cpu.data<float>();

        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            EXPECT_NEAR(a_data[i], b_data[i], tolerance)
                << "Mismatch at index " << i
                << " on device " << device.to_string();
        }
    }
};

// Define static member with inline to avoid multiple definition errors
inline std::once_flag BackendTest::init_flag;

// Instantiate tests for all available backends (legacy, device index 0 only)
#define INSTANTIATE_BACKEND_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        AllBackends, \
        TestSuiteName, \
        ::testing::Values("cpu", "cuda", "vulkan", "oneapi", "rocm"), \
        [](const ::testing::TestParamInfo<std::string>& info) { \
            return info.param; \
        } \
    )

// Instantiate tests for all discovered devices across all backends
#define INSTANTIATE_MULTI_DEVICE_BACKEND_TESTS(TestSuiteName) \
    INSTANTIATE_TEST_SUITE_P( \
        AllDevices, \
        TestSuiteName, \
        ::testing::ValuesIn([]() -> std::vector<std::string> { \
            ::tenzor::testing::EnsureInitialized(); \
            std::vector<std::string> result = {"cpu"}; \
            struct Info { const char* name; Device::Type type; }; \
            constexpr Info backends[] = { \
                {"cuda", Device::Type::CUDA}, \
                {"oneapi", Device::Type::OneAPI}, \
                {"vulkan", Device::Type::Vulkan}, \
                {"rocm", Device::Type::ROCm}, \
            }; \
            for (const auto& [n, t] : backends) { \
                auto* b = backend_registry().get_backend(t); \
                if (!b || !b->is_available()) continue; \
                for (int32_t i = 0; i < b->device_count(); ++i) { \
                    try { \
                        Device d{t, i}; \
                        zeros({2, 2}, DType::Float32, d); \
                        result.push_back(std::string(n) + ":" + std::to_string(i)); \
                    } catch (...) {} \
                } \
            } \
            return result; \
        }()), \
        [](const ::testing::TestParamInfo<std::string>& info) { \
            return ::tenzor::testing::detail::formatBackendTestName(info.param); \
        } \
    )

#ifdef TENZOR_TEST_ALL_DEVICES

#undef INSTANTIATE_BACKEND_TESTS
#define INSTANTIATE_BACKEND_TESTS(TestSuiteName) \
    INSTANTIATE_MULTI_DEVICE_BACKEND_TESTS(TestSuiteName)

#endif // TENZOR_TEST_ALL_DEVICES

} // namespace testing
} // namespace tenzor
