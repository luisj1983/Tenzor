/**
 * @file test_comparison_operators_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for comparison operations (gt, lt, eq, etc.)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ComparisonOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    Tensor fromValues(const std::vector<float>& vals) {
        auto t = tenzor::full({static_cast<int64_t>(vals.size())},
                              0.0f, DType::Float32, Device::cpu());
        auto* d = t.data<float>();
        for (size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
        return t.to(device()).to(dtype());
    }
};

TEST_P(ComparisonOpsMultiDTypeTest, EqualSelf) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto result = tenzor::eq(a, a);
    expectShape(result, {3});
    expectDevice(result);
}

TEST_P(ComparisonOpsMultiDTypeTest, NotEqual) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto b = fromValues({1.0f, 5.0f, 3.0f});
    auto result = tenzor::ne(a, b);
    expectShape(result, {3});
}

TEST_P(ComparisonOpsMultiDTypeTest, LessThan) {
    auto a = fromValues({1.0f, 5.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 4.0f});
    auto result = tenzor::lt(a, b);
    expectShape(result, {3});
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterThan) {
    auto a = fromValues({5.0f, 1.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 4.0f});
    auto result = tenzor::gt(a, b);
    expectShape(result, {3});
}

TEST_P(ComparisonOpsMultiDTypeTest, LessEqual) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 2.0f});
    auto result = tenzor::le(a, b);
    expectShape(result, {3});
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterEqual) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 2.0f});
    auto result = tenzor::ge(a, b);
    expectShape(result, {3});
}

TEST_P(ComparisonOpsMultiDTypeTest, BroadcastComparison) {
    auto a = createRandn({3, 4});
    auto b = createRandn({4});  // broadcasts
    auto result = tenzor::gt(a, b);
    expectShape(result, {3, 4});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ComparisonOpsMultiDTypeTest);
