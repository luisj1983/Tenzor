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
//
// Backend selection / availability (skip-vs-FAIL under
// TENZOR_REQUIRE_MULTI_BACKEND, TENZOR_SKIP_BACKENDS opt-out) is handled by
// the canonical BackendTest::SetUp — this file deliberately keeps no local
// availability logic so it tracks the project-wide policy.

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/op_id.hpp"
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

// Rebased onto the canonical BackendTest fixture: `device` is resolved (and
// availability/skip/require policy applied) by the base SetUp. We only check
// IsSkipped() to bail out cleanly when the base decided to skip this backend.
class AttentionContractTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(AttentionContractTest, AllRequiredOpIdsRegistered) {
    std::vector<std::string> missing;
    for (const auto& entry : kRequiredAttentionOpIds) {
        if (!tenzor::is_op_supported(entry.id, device.type)) {
            missing.emplace_back(entry.name);
        }
    }

    if (!missing.empty()) {
        std::string msg = "Backend " + device.to_string() +
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

INSTANTIATE_BACKEND_TESTS(AttentionContractTest);

}  // anonymous namespace
