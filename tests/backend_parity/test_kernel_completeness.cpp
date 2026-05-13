/**
 * @file test_kernel_completeness.cpp
 * @brief Verify that all required operations have registered kernels on each backend.
 *
 * Phase 4A: For each compute backend (CPU, CUDA, ROCm, Vulkan, OneAPI), checks that
 * a curated set of "required" OpIds are present in the dispatch table.  Specialized,
 * fused, backward-only, in-place, and creation ops are excluded from the required set.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include "parity_test_utils.hpp"
#include "required_ops.hpp"
#include <vector>
#include <string>
#include <sstream>

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// The required-op floor lives in required_ops.hpp so new parity tests can
// include the same list when asserting "this op must exist everywhere" before
// running. See that header for the grow-the-floor convention.

// ---------------------------------------------------------------------------
// Helper: join a vector of strings with ", "
// ---------------------------------------------------------------------------
std::string join(const std::vector<std::string>& items) {
    std::ostringstream oss;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << items[i];
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Generic checker used by each per-backend TEST()
// ---------------------------------------------------------------------------
void check_backend_completeness(Device::Type device_type, const char* backend_name) {
    tenzor::initialize();

    const auto& table = DispatchTableRegistry::get_table_const(device_type);
    const auto required = get_required_ops();

    std::vector<std::string> missing;
    for (auto op : required) {
        if (!table.has_kernel(op)) {
            missing.emplace_back(std::string(op_id_to_name(op)));
        }
    }

    EXPECT_TRUE(missing.empty())
        << backend_name << " backend is missing " << missing.size()
        << " of " << required.size() << " required kernels:\n  "
        << join(missing);
}

}  // namespace

// ===========================================================================
// Per-backend completeness tests
// ===========================================================================

// Lower-case case names so `ctest -R cuda` (lowercase) matches the
// generated CTest entry KernelCompleteness.cuda. The user-facing strings
// passed to check_backend_completeness keep their canonical capitalised
// form for log messages.
TEST(KernelCompleteness, cpu) {
    check_backend_completeness(Device::Type::CPU, "CPU");
}

TEST(KernelCompleteness, cuda) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA backend not available";
    check_backend_completeness(Device::Type::CUDA, "CUDA");
}

TEST(KernelCompleteness, rocm) {
    if (!has_rocm()) GTEST_SKIP() << "ROCm backend not available";
    check_backend_completeness(Device::Type::ROCm, "ROCm");
}

TEST(KernelCompleteness, vulkan) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan backend not available";
    check_backend_completeness(Device::Type::Vulkan, "Vulkan");
}

TEST(KernelCompleteness, oneapi) {
    if (!has_oneapi()) GTEST_SKIP() << "OneAPI backend not available";
    check_backend_completeness(Device::Type::OneAPI, "OneAPI");
}
