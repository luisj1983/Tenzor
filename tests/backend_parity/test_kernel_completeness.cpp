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
#include "../backend_test_fixture.hpp"
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

const char* device_type_name(Device::Type t) {
    switch (t) {
        case Device::Type::CPU:    return "CPU";
        case Device::Type::CUDA:   return "CUDA";
        case Device::Type::ROCm:   return "ROCm";
        case Device::Type::Vulkan: return "Vulkan";
        case Device::Type::OneAPI: return "OneAPI";
        default:                   return "Unknown";
    }
}

}  // namespace

// ===========================================================================
// II.17: collapsed the 5 hand-rolled per-backend TEST() cases into a single
// TEST_P that runs over the BackendTest matrix. BackendTest::SetUp() handles
// the "skip if backend not available" / "fail if TENZOR_REQUIRE_MULTI_BACKEND"
// logic uniformly with the rest of the parity suite.
// ===========================================================================
class KernelCompletenessParity : public BackendTest {};

TEST_P(KernelCompletenessParity, AllOpsRegistered) {
    check_backend_completeness(device.type, device_type_name(device.type));
}

INSTANTIATE_BACKEND_TESTS(KernelCompletenessParity);
