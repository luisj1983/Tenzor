/**
 * @file test_dispatch_coverage_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for dispatch coverage validation
 *
 * These tests verify dispatch table coverage across backends. The coverage
 * validation itself is backend-level machinery, but we parameterize to ensure
 * the checks hold for every available backend + dtype combination.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/ops/op_id.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DispatchCoverageMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(DispatchCoverageMultiDTypeTest, AllNamedOpsAreCovered) {
    const bool ok = DispatchTableRegistry::validate_coverage(/*strict=*/false);
    EXPECT_TRUE(ok)
        << "At least one named OpId has no kernel in any loaded backend.";
}

TEST_P(DispatchCoverageMultiDTypeTest, StrictModeNoThrowOnFullCoverage) {
    EXPECT_NO_THROW({
        (void)DispatchTableRegistry::validate_coverage(/*strict=*/true);
    });
}

TEST_P(DispatchCoverageMultiDTypeTest, BackendDispatchTableIsReady) {
    // Verify the dispatch table for the current backend is marked ready
    auto& table = DispatchTableRegistry::get_table(device().type);
    EXPECT_TRUE(table.ready.load(std::memory_order_acquire))
        << "Dispatch table for " << backend_name() << " should be ready";
}

TEST_P(DispatchCoverageMultiDTypeTest, BasicOpDispatchesCorrectly) {
    // Verify a basic op (Add) works on the current backend + dtype
    auto a = createOnes({4});
    auto b = createOnes({4});
    auto result = tenzor::dispatch(OpId::Add, std::array<Tensor, 2>{a, b});
    ASSERT_FALSE(result.empty());
    expectDevice(result[0]);
    expectDType(result[0]);

    // Verify the result is correct: 1 + 1 = 2
    auto r_cpu = result[0].to(Device::cpu()).to(DType::Float32);
    const float* data = r_cpu.data<float>();
    EXPECT_NEAR(data[0], 2.0f, atol() + 1e-2f);
}

TEST_P(DispatchCoverageMultiDTypeTest, RestoredAfterClobberIsStillFull) {
    // Verifies coverage remains intact (regression guard for any test
    // that might clobber dispatch entries)
    EXPECT_TRUE(DispatchTableRegistry::validate_coverage(/*strict=*/false));
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DispatchCoverageMultiDTypeTest);
