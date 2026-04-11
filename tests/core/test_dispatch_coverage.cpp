// Test for Phase 3.3: dispatch startup coverage validation.
//
// The real `DispatchTableRegistry::validate_coverage()` runs once at
// `tenzor::initialize()` time and reports any OpId with no kernel in any
// active backend. Verifying it directly against the live tables is the
// cleanest way to catch a regression: we query coverage for all loaded
// backends, confirm it passes, and additionally simulate a missing
// kernel by temporarily clobbering one active kernel entry and checking
// the report fires.
//
// CPU-only and does not depend on any particular backend being loaded
// beyond the standard CPU dispatch table.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include <tenzor/ops/op_id.hpp>

#include <cstdlib>
#include <exception>

namespace tenzor {
namespace {

class DispatchCoverageTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// The process-wide coverage check that fires inside initialize() should
// find every named OpId covered by at least one backend. If this ever
// starts failing, something was added to the OpId enum without a
// corresponding kernel registration in any backend — the fix is to add
// the kernel, not to relax the test.
TEST_F(DispatchCoverageTest, AllNamedOpsAreCovered) {
    const bool ok = DispatchTableRegistry::validate_coverage(/*strict=*/false);
    EXPECT_TRUE(ok)
        << "At least one named OpId has no kernel in any loaded backend. "
           "Check the WARNING emitted during tenzor::initialize() for the "
           "list; add a kernel (or remove the enum entry) to restore "
           "coverage.";
}

// Strict mode must not throw when coverage is complete. This is the
// path CI would run with TENZOR_DISPATCH_STRICT=1 — we invoke it
// programmatically so the test works regardless of the env var state
// in the host shell.
TEST_F(DispatchCoverageTest, StrictModeNoThrowOnFullCoverage) {
    EXPECT_NO_THROW({
        (void)DispatchTableRegistry::validate_coverage(/*strict=*/true);
    });
}

// Simulate a missing kernel: temporarily clobber the CPU kernel entry
// for a well-known op, confirm that strict validation throws (so CI
// would catch a regression), restore the kernel, then re-validate to
// make sure we left the dispatch table in a good state. This test is
// deliberately intrusive; it exercises the exact failure path the
// warning is supposed to catch.
TEST_F(DispatchCoverageTest, StrictModeThrowsOnMissingKernel) {
    // MatMul is registered in every backend, so we need to clobber it
    // across all *ready* backends to make it actually uncovered. We
    // snapshot every pointer per backend so we can restore regardless
    // of which slot was populated.
    const OpId victim = OpId::MatMul;
    const size_t idx = static_cast<size_t>(victim);

    struct Saved {
        Device::Type type;
        KernelFn kernel;
        SingleOutputKernelFn single;
        InplaceKernelFn inplace;
    };
    std::vector<Saved> snapshots;

    for (size_t d = 0; d < DEVICE_TYPE_COUNT; ++d) {
        const auto type = static_cast<Device::Type>(d);
        auto& table = DispatchTableRegistry::get_table(type);
        if (!table.ready.load(std::memory_order_acquire)) continue;
        snapshots.push_back({type,
                             table.kernels[idx],
                             table.single_output_kernels[idx],
                             table.inplace_kernels[idx]});
        table.kernels[idx] = nullptr;
        table.single_output_kernels[idx] = nullptr;
        table.inplace_kernels[idx] = nullptr;
    }
    ASSERT_FALSE(snapshots.empty())
        << "Precondition: at least one backend must be ready.";

    // Restore on scope exit so that a failing assertion doesn't leave
    // the dispatch tables broken for subsequent tests.
    struct Restore {
        const std::vector<Saved>& snapshots;
        size_t idx;
        ~Restore() {
            for (const auto& s : snapshots) {
                auto& table = DispatchTableRegistry::get_table(s.type);
                table.kernels[idx] = s.kernel;
                table.single_output_kernels[idx] = s.single;
                table.inplace_kernels[idx] = s.inplace;
            }
        }
    } restore{snapshots, idx};

    // Strict mode must now throw because no ready backend has a MatMul
    // kernel. We deliberately don't match the exception message text.
    EXPECT_THROW(
        (void)DispatchTableRegistry::validate_coverage(/*strict=*/true),
        std::runtime_error);

    // Non-strict mode must return false and not throw.
    bool non_strict_ok = true;
    EXPECT_NO_THROW({
        non_strict_ok = DispatchTableRegistry::validate_coverage(/*strict=*/false);
    });
    EXPECT_FALSE(non_strict_ok);
}

// After the clobbering test restores state, full coverage must hold
// again. Declared as a separate test so the Restore RAII in the
// previous test is guaranteed to have run before this one starts.
TEST_F(DispatchCoverageTest, RestoredAfterClobberIsStillFull) {
    EXPECT_TRUE(DispatchTableRegistry::validate_coverage(/*strict=*/false));
}

} // namespace
} // namespace tenzor
