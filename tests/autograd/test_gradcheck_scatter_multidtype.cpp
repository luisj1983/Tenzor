/**
 * @file test_gradcheck_scatter_multidtype.cpp
 * @brief Multi-backend numerical gradcheck for scatter and scatter_add.
 *
 * Phase 8-followup #28: existing scatter_add gradcheck in
 * test_gradcheck_comprehensive.cpp:607 only runs on CPU. The audit flagged
 * this as a gap because scatter family backward is index-aware and any
 * stride/index-handling backend bug would be invisible without per-backend
 * gradcheck. Small inputs to keep per-element FD perturbation fast on GPU.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckScatterTest : public MultiBackendDTypeTest {
protected:
    bool should_skip() {
        return dtype() != DType::Float32 && dtype() != DType::Float64;
    }
    double eps() const { return dtype() == DType::Float64 ? 1e-6 : 5e-4; }
    double tol() const { return dtype() == DType::Float64 ? 1e-5 : 1e-2; }
};

TEST_P(GradCheckScatterTest, Scatter) {
    if (should_skip()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "gradcheck supports only Float32/Float64");
    }
    auto x = Variable(randn({4, 8}, dtype(), device()), /*requires_grad=*/true);
    auto src = Variable(randn({4, 3}, dtype(), device()), false);
    auto idx = randint(0, 8, {4, 3}, DType::Int64, device());
    auto f = [&src, &idx](const Variable& v) -> Variable {
        return tenzor::sum(scatter(v, /*dim=*/1, idx, src));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "scatter gradcheck failed on " << device().to_string();
}


TEST_P(GradCheckScatterTest, ScatterAdd) {
    if (should_skip()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "gradcheck supports only Float32/Float64");
    }
    auto x = Variable(randn({4, 8}, dtype(), device()), /*requires_grad=*/true);
    auto src = Variable(randn({4, 3}, dtype(), device()), false);
    auto idx = randint(0, 8, {4, 3}, DType::Int64, device());
    auto f = [&src, &idx](const Variable& v) -> Variable {
        return tenzor::sum(scatter_add(v, /*dim=*/1, idx, src));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "scatter_add gradcheck failed on " << device().to_string();
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckScatterTest);
