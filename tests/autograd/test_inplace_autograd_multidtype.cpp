/**
 * @file test_inplace_autograd_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for in-place ops + autograd version counters
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class InplaceAutogradMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(InplaceAutogradMultiDTypeTest, NewTensorVersionIsZero) {
    auto t = tenzor::ones({3, 3}, dtype(), device());
    EXPECT_EQ(t.version(), 0u);
}

TEST_P(InplaceAutogradMultiDTypeTest, FillBumpsVersion) {
    auto t = tenzor::zeros({4, 4}, dtype(), device());
    uint64_t v0 = t.version();

    t.fill_(1.0);
    uint64_t v1 = t.version();
    EXPECT_GT(v1, v0) << "fill_() should bump version counter";

    t.fill_(2.0);
    uint64_t v2 = t.version();
    EXPECT_GT(v2, v1) << "Second fill_() should bump version again";
}

TEST_P(InplaceAutogradMultiDTypeTest, ZeroBumpsVersion) {
    auto t = tenzor::ones({3}, dtype(), device());
    uint64_t v0 = t.version();
    t.zero_();
    EXPECT_GT(t.version(), v0);
}

TEST_P(InplaceAutogradMultiDTypeTest, InplaceAddBumpsVersion) {
    auto t = tenzor::ones({4}, dtype(), device());
    auto other = tenzor::ones({4}, dtype(), device());
    uint64_t v0 = t.version();
    t += other;
    EXPECT_GT(t.version(), v0);
}

TEST_P(InplaceAutogradMultiDTypeTest, InplaceMulBumpsVersion) {
    auto t = tenzor::ones({4}, dtype(), device());
    auto other = tenzor::full({4}, 2.0f, dtype(), device());
    uint64_t v0 = t.version();
    t *= other;
    EXPECT_GT(t.version(), v0);
}

TEST_P(InplaceAutogradMultiDTypeTest, CloneDoesNotShareVersion) {
    auto t = tenzor::ones({3}, dtype(), device());
    auto c = t.clone();

    t.fill_(5.0);
    EXPECT_GT(t.version(), 0u);
    EXPECT_EQ(c.version(), 0u) << "Clone should have independent version counter";
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(InplaceAutogradMultiDTypeTest);
