/**
 * @file test_gradcheck_direct_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for numerical gradient checking
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// Macro (not a method) so that GTEST_SKIP's internal `return`
// statement returns from the TEST_P body rather than from a helper
// method — otherwise the test continues and fails on the first op
// that doesn't support Float16.
#define skipIfHalf() \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            GTEST_SKIP() << "Gradcheck needs higher precision than Float16"; \
        } \
    } while (0)

class GradCheckDirectMultiDTypeTest : public MultiBackendDTypeTest {
protected:
};

TEST_P(GradCheckDirectMultiDTypeTest, QuadraticPasses) {
    skipIfHalf();
    auto f = [](const Variable& x) -> Variable { return x * x; };
    auto x = createInput({3}, true);
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckDirectMultiDTypeTest, LinearPasses) {
    skipIfHalf();
    auto f = [](const Variable& x) -> Variable { return x * 2.0f + 3.0f; };
    auto x = createInput({4}, true);
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckDirectMultiDTypeTest, SumReductionPasses) {
    skipIfHalf();
    auto f = [](const Variable& x) -> Variable { return tenzor::sum(x); };
    auto x = createInput({4}, true);
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckDirectMultiDTypeTest, ExpPasses) {
    skipIfHalf();
    auto f = [](const Variable& x) -> Variable { return tenzor::exp(x); };
    auto x = createInput({3}, true);
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckDirectMultiDTypeTest, DetailedResult) {
    skipIfHalf();
    auto f = [](const Variable& x) -> Variable { return x * x; };
    auto x = createInput({2}, true);
    auto result = gradcheck_detailed(f, x);
    EXPECT_TRUE(result.passed);
    EXPECT_GT(result.max_rel_error, 0.0);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckDirectMultiDTypeTest);
