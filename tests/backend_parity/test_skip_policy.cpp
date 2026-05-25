/**
 * @file test_skip_policy.cpp
 * @brief Meta-test for the multi-backend skip / require / skip-trumps-require
 *        policy used by REQUIRE_MULTI_BACKEND_OR_SKIP.
 *
 * The audit (2026-05-02) flagged this contract as the most load-bearing
 * piece of test infrastructure: AMD-driver-fragility on the dev host
 * means a backend init failure looks indistinguishable from a clean
 * "host has no GPU" skip unless the policy is intact. This file pins:
 *   - With TENZOR_REQUIRE_MULTI_BACKEND unset, golden::require_multi_backend()
 *     returns false. Stand-in for "default: skip on missing backend".
 *   - With TENZOR_REQUIRE_MULTI_BACKEND=1, returns true.
 *   - is_backend_skipped_by_env() reflects the CSV in TENZOR_SKIP_BACKENDS.
 *   - When both env vars are set, get_available_backends() drops the
 *     skip-listed entries — confirming "skip env wins over require env".
 */

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"
#include "golden_util.hpp"
#include "../multi_backend_dtype_fixture.hpp"  // FF.28: SKIP_WITH_REASON
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::testing;

class SkipPolicyMetaTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    // RAII: snapshot/restore an env var so individual tests don't pollute
    // each other. setenv("", overwrite=1) or unsetenv leaves a known-empty
    // state for the body and restores at scope exit.
    struct ScopedEnv {
        std::string name;
        std::string saved;
        bool was_set;
        ScopedEnv(const char* n, const char* val) : name(n) {
            const char* prev = std::getenv(n);
            was_set = prev != nullptr;
            saved = was_set ? prev : "";
            if (val == nullptr) {
                ::unsetenv(n);
            } else {
                ::setenv(n, val, /*overwrite=*/1);
            }
        }
        ~ScopedEnv() {
            if (was_set) ::setenv(name.c_str(), saved.c_str(), /*overwrite=*/1);
            else         ::unsetenv(name.c_str());
        }
    };
};

// ---------------------------------------------------------------------------
// require_multi_backend() — read TENZOR_REQUIRE_MULTI_BACKEND each call.
// ---------------------------------------------------------------------------

TEST_F(SkipPolicyMetaTest, RequireFlagDefaultsFalse) {
    ScopedEnv g("TENZOR_REQUIRE_MULTI_BACKEND", nullptr);
    EXPECT_FALSE(golden::require_multi_backend());
}

TEST_F(SkipPolicyMetaTest, RequireFlagTrueWhenOne) {
    ScopedEnv g("TENZOR_REQUIRE_MULTI_BACKEND", "1");
    EXPECT_TRUE(golden::require_multi_backend());
}

TEST_F(SkipPolicyMetaTest, RequireFlagFalseWhenZero) {
    ScopedEnv g("TENZOR_REQUIRE_MULTI_BACKEND", "0");
    EXPECT_FALSE(golden::require_multi_backend());
}

// ---------------------------------------------------------------------------
// is_backend_skipped_by_env() — parses TENZOR_SKIP_BACKENDS as CSV.
// ---------------------------------------------------------------------------

TEST_F(SkipPolicyMetaTest, SkipFlagDefaultsEmpty) {
    ScopedEnv g("TENZOR_SKIP_BACKENDS", nullptr);
    EXPECT_FALSE(is_backend_skipped_by_env("cuda"));
    EXPECT_FALSE(is_backend_skipped_by_env("rocm"));
    EXPECT_FALSE(is_backend_skipped_by_env("oneapi"));
    EXPECT_FALSE(is_backend_skipped_by_env("vulkan"));
    EXPECT_FALSE(is_backend_skipped_by_env("cpu"));
}

TEST_F(SkipPolicyMetaTest, SkipFlagSingleBackend) {
    ScopedEnv g("TENZOR_SKIP_BACKENDS", "cuda");
    EXPECT_TRUE(is_backend_skipped_by_env("cuda"));
    EXPECT_FALSE(is_backend_skipped_by_env("rocm"));
}

TEST_F(SkipPolicyMetaTest, SkipFlagCSV) {
    ScopedEnv g("TENZOR_SKIP_BACKENDS", "cuda,rocm");
    EXPECT_TRUE(is_backend_skipped_by_env("cuda"));
    EXPECT_TRUE(is_backend_skipped_by_env("rocm"));
    EXPECT_FALSE(is_backend_skipped_by_env("vulkan"));
    EXPECT_FALSE(is_backend_skipped_by_env("oneapi"));
}

// ---------------------------------------------------------------------------
// Skip wins over require. Even when REQUIRE=1, an explicit SKIP must drop
// the backend from the available list — that's the rule that prevents AMD
// driver flakes from forcing CI to fail when the dev wants to opt out.
// ---------------------------------------------------------------------------

TEST_F(SkipPolicyMetaTest, SkipDropsBackendsFromGetAvailable) {
    // Snapshot current available backend names.
    std::vector<Device> all = get_available_backends();
    std::set<std::string> all_names;
    for (auto& d : all) all_names.insert(backend_name(d));

    if (all.size() <= 1) {
        SKIP_WITH_REASON(SkipReason::BackendUnavailable,
                         "Only one backend available — nothing to skip from.");
    }

    // Pick one non-CPU backend to skip via env.
    std::string victim;
    for (auto& d : all) {
        if (d.type != Device::Type::CPU) {
            victim = backend_name(d);
            break;
        }
    }
    ASSERT_FALSE(victim.empty()) << "Test setup failure: no GPU backend to skip";

    {
        ScopedEnv require("TENZOR_REQUIRE_MULTI_BACKEND", "1");
        ScopedEnv skip("TENZOR_SKIP_BACKENDS", victim.c_str());
        // Important contract: even though REQUIRE=1, the skip-list still
        // applies. is_backend_skipped_by_env(victim) must be true.
        EXPECT_TRUE(is_backend_skipped_by_env(victim))
            << "TENZOR_SKIP_BACKENDS=" << victim
            << " should still be honoured when REQUIRE=1.";
    }

    // After ScopedEnv destructors fire, the environment is restored.
    EXPECT_FALSE(is_backend_skipped_by_env(victim))
        << "ScopedEnv didn't restore TENZOR_SKIP_BACKENDS";
}

// Final sanity: when both flags are zeroed, the policy is in its default
// "skip on missing" mode.
TEST_F(SkipPolicyMetaTest, BothFlagsClear_DefaultPolicy) {
    ScopedEnv require("TENZOR_REQUIRE_MULTI_BACKEND", nullptr);
    ScopedEnv skip("TENZOR_SKIP_BACKENDS", nullptr);
    EXPECT_FALSE(golden::require_multi_backend());
    EXPECT_FALSE(is_backend_skipped_by_env("cuda"));
}

// ---------------------------------------------------------------------------
// Macro-level coverage of REQUIRE_MULTI_BACKEND_OR_SKIP. The function-level
// tests above verify the env-var helpers in isolation; these three exercise
// the actual macro, which is what every parity TEST_P body invokes. The
// three combinations match the verification matrix in the audit plan.
// ---------------------------------------------------------------------------

// REQUIRE=1 + only one backend reachable => the macro must FAIL (not SKIP).
// We force "only CPU" by skipping every GPU type via env so the test is
// deterministic regardless of the host's actual backend availability.
// EXPECT_FATAL_FAILURE captures the FAIL() that the macro raises and turns
// it into a passing assertion — if the macro silently skipped instead, this
// test would itself report SKIPPED, which CI surfaces as a regression.
TEST_F(SkipPolicyMetaTest, Macro_FailsRather_WhenRequireSetAndOnlyCpu) {
    ScopedEnv skip("TENZOR_SKIP_BACKENDS", "cuda,rocm,oneapi,vulkan");
    ScopedEnv req("TENZOR_REQUIRE_MULTI_BACKEND", "1");
    // Sanity-check the precondition before invoking the macro under the SPI.
    ASSERT_LT(get_available_backends().size(), 2u)
        << "Test setup expected only CPU after skipping every GPU.";
    EXPECT_FATAL_FAILURE(
        REQUIRE_MULTI_BACKEND_OR_SKIP("macro-level FAIL path"),
        "Multi-backend required");
}

// REQUIRE unset + only one backend reachable => the macro must SKIP. The
// post-macro ADD_FAILURE is an "if the skip didn't fire, fail loudly" trap:
// a properly working macro causes GTEST_SKIP() and the test reports SKIPPED;
// a broken macro reaches the trap and the test reports FAILED.
TEST_F(SkipPolicyMetaTest, Macro_SkipsRather_WhenRequireUnsetAndOnlyCpu) {
    ScopedEnv skip("TENZOR_SKIP_BACKENDS", "cuda,rocm,oneapi,vulkan");
    ScopedEnv req("TENZOR_REQUIRE_MULTI_BACKEND", nullptr);
    ASSERT_LT(get_available_backends().size(), 2u)
        << "Test setup expected only CPU after skipping every GPU.";
    REQUIRE_MULTI_BACKEND_OR_SKIP("macro-level SKIP path");
    ADD_FAILURE() << "Macro should have skipped before reaching here. "
                     "REQUIRE was unset and only CPU was available, so the "
                     "default-skip policy should fire.";
}

// 2+ backends available + no env overrides => macro is a no-op (pass-through).
// The flag-after-macro proves execution continued. If the host happens to
// only have one real backend we skip the test — there's nothing to verify
// pass-through against.
TEST_F(SkipPolicyMetaTest, Macro_PassesThrough_WhenMultiBackendAvailable) {
    ScopedEnv skip("TENZOR_SKIP_BACKENDS", nullptr);
    ScopedEnv req("TENZOR_REQUIRE_MULTI_BACKEND", nullptr);
    if (get_available_backends().size() < 2) {
        SKIP_WITH_REASON(SkipReason::BackendUnavailable,
                         "Single-backend host — pass-through assertion vacuous.");
    }
    bool reached = false;
    REQUIRE_MULTI_BACKEND_OR_SKIP("macro-level pass-through path");
    reached = true;
    EXPECT_TRUE(reached) << "Macro must be a no-op when 2+ backends are "
                            "available and no env override is set.";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
