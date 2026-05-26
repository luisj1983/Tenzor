// Cross-backend attention contract audit.
//
// Per docs/internals/attention-contract.md every backend must register a
// kernel for the contract OpIds — FlashAttention, FlashAttentionBackward,
// FusedAttention, FusedLayerNorm, FusedRMSNorm, NestedAttention,
// NestedAttentionBackward, FlexAttention, FlexAttentionBackward.
//
// This test is the CI gate that catches "silently absent on this backend"
// gaps — the audit caught FlexAttention missing on every GPU backend, which
// would have been blocked here. Adding a new attention OpId? Add it to the
// kRequiredAttentionOpIds list below; this test then enforces registration
// across every enabled backend.
//
// The test is *not* a numerical check — it only verifies registration
// presence. Cross-backend numerical parity is in test_cross_backend.cpp and
// the per-backend unit tests.

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Contract OpIds. Adding to this list and re-running ratchets the
// requirement across every enabled backend.
struct OpEntry {
    tenzor::OpId id;
    const char* name;
};

constexpr OpEntry kRequiredAttentionOpIds[] = {
    {tenzor::OpId::FlashAttention,         "FlashAttention"},
    {tenzor::OpId::FlashAttentionBackward, "FlashAttentionBackward"},
    {tenzor::OpId::FusedAttention,         "FusedAttention"},
    {tenzor::OpId::FusedLayerNorm,         "FusedLayerNorm"},
    {tenzor::OpId::FusedRMSNorm,           "FusedRMSNorm"},
    {tenzor::OpId::NestedAttention,        "NestedAttention"},
    {tenzor::OpId::NestedAttentionBackward,"NestedAttentionBackward"},
    {tenzor::OpId::FlexAttention,          "FlexAttention"},
    {tenzor::OpId::FlexAttentionBackward,  "FlexAttentionBackward"},
};

struct DeviceEntry {
    tenzor::Device::Type type;
    const char* name;
};

constexpr DeviceEntry kCheckedDevices[] = {
    {tenzor::Device::Type::CPU,    "cpu"},
    {tenzor::Device::Type::CUDA,   "cuda"},
    {tenzor::Device::Type::ROCm,   "rocm"},
    {tenzor::Device::Type::OneAPI, "oneapi"},
    {tenzor::Device::Type::Vulkan, "vulkan"},
};

// Whether a given backend was loaded successfully on this run. Drives
// "skip silently if this backend isn't available" semantics, but flips to
// hard-fail under TENZOR_REQUIRE_MULTI_BACKEND=1 (per CLAUDE.md memory).
bool backend_loaded(tenzor::Device::Type type) {
    return tenzor::DispatchTableRegistry::has_backend(type);
}

bool require_multi_backend() {
    const char* env = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");
    return env != nullptr && env[0] == '1';
}

}  // anonymous namespace

class AttentionContractTest : public ::testing::TestWithParam<DeviceEntry> {
protected:
    static void SetUpTestSuite() {
        // Initialize the runtime so backend .so plugins load and register
        // their dispatch tables. Without this, has_backend() returns false
        // for every device because nothing has triggered the loader.
        tenzor::initialize();
    }
};

TEST_P(AttentionContractTest, AllRequiredOpIdsRegistered) {
    auto dev = GetParam();
    if (!backend_loaded(dev.type)) {
        if (require_multi_backend()) {
            FAIL() << "Backend " << dev.name << " not loaded but TENZOR_REQUIRE_MULTI_BACKEND=1";
        }
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "Backend " << dev.name << " not loaded (set TENZOR_REQUIRE_MULTI_BACKEND=1 "
            "to make this a hard fail)");
        return;
    }

    std::vector<std::string> missing;
    for (const auto& entry : kRequiredAttentionOpIds) {
        if (!tenzor::is_op_supported(entry.id, dev.type)) {
            missing.emplace_back(entry.name);
        }
    }

    if (!missing.empty()) {
        std::string msg = "Backend " + std::string(dev.name) +
                          " is missing required attention OpIds: ";
        for (size_t i = 0; i < missing.size(); ++i) {
            if (i > 0) msg += ", ";
            msg += missing[i];
        }
        msg += ". Per docs/internals/attention-contract.md every backend must "
               "register all contract OpIds — see the attention audit (M9) for "
               "the gates this test enforces.";
        FAIL() << msg;
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    AttentionContractTest,
    ::testing::ValuesIn(kCheckedDevices),
    [](const ::testing::TestParamInfo<DeviceEntry>& info) {
        return std::string(info.param.name);
    });
