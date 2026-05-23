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

    // Audit-T.1: helper to check bool tensor element-by-element against
    // an inline expected vector.  Comparison ops return Bool regardless of
    // input dtype, so this works for every parameter combination.
    void expectBoolTensorEquals(const Tensor& actual,
                                const std::vector<bool>& expected) {
        auto cpu = actual.to(Device::cpu());
        ASSERT_EQ(cpu.dtype(), DType::Bool)
            << "Comparison op must produce a Bool tensor";
        ASSERT_EQ(static_cast<size_t>(cpu.numel()), expected.size());
        const bool* d = static_cast<const bool*>(cpu.data_ptr());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(d[i], expected[i])
                << "Mismatch at index " << i
                << " on device " << device().to_string();
        }
    }
};

TEST_P(ComparisonOpsMultiDTypeTest, EqualSelf) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto result = tenzor::eq(a, a);
    expectShape(result, {3});
    expectDevice(result);
    // Audit-T.1: x == x must be all-true for finite values.
    expectBoolTensorEquals(result, {true, true, true});
}

TEST_P(ComparisonOpsMultiDTypeTest, NotEqual) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto b = fromValues({1.0f, 5.0f, 3.0f});
    auto result = tenzor::ne(a, b);
    expectShape(result, {3});
    // Audit-T.1: only middle element differs.
    expectBoolTensorEquals(result, {false, true, false});
}

TEST_P(ComparisonOpsMultiDTypeTest, LessThan) {
    auto a = fromValues({1.0f, 5.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 4.0f});
    auto result = tenzor::lt(a, b);
    expectShape(result, {3});
    // Audit-T.1: 1<2 true, 5<2 false, 3<4 true.
    expectBoolTensorEquals(result, {true, false, true});
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterThan) {
    auto a = fromValues({5.0f, 1.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 4.0f});
    auto result = tenzor::gt(a, b);
    expectShape(result, {3});
    // Audit-T.1: 5>2 true, 1>2 false, 3>4 false.
    expectBoolTensorEquals(result, {true, false, false});
}

TEST_P(ComparisonOpsMultiDTypeTest, LessEqual) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 2.0f});
    auto result = tenzor::le(a, b);
    expectShape(result, {3});
    // Audit-T.1: 1<=2 true, 2<=2 true, 3<=2 false.
    expectBoolTensorEquals(result, {true, true, false});
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterEqual) {
    auto a = fromValues({1.0f, 2.0f, 3.0f});
    auto b = fromValues({2.0f, 2.0f, 2.0f});
    auto result = tenzor::ge(a, b);
    expectShape(result, {3});
    // Audit-T.1: 1>=2 false, 2>=2 true, 3>=2 true.
    expectBoolTensorEquals(result, {false, true, true});
}

TEST_P(ComparisonOpsMultiDTypeTest, BroadcastComparison) {
    // Audit-T.1: use deterministic inputs so the expected value-pattern is
    // exact, not "shape only".  a is row-broadcast against vector b along
    // the last axis.  Both stored on CPU first to derive the expected
    // tensor, then pushed to device under test.
    auto a_cpu = tenzor::full({3, 4}, 0.0f, DType::Float32, Device::cpu());
    auto* ap = a_cpu.data<float>();
    // Row 0 entirely less than b, row 1 entirely greater, row 2 mixed.
    for (int c = 0; c < 4; ++c) ap[0 * 4 + c] = -1.0f;
    for (int c = 0; c < 4; ++c) ap[1 * 4 + c] =  5.0f;
    ap[2 * 4 + 0] = 0.0f;
    ap[2 * 4 + 1] = 3.0f;
    ap[2 * 4 + 2] = 2.0f;
    ap[2 * 4 + 3] = 1.0f;

    auto b_cpu = tenzor::full({4}, 0.0f, DType::Float32, Device::cpu());
    auto* bp = b_cpu.data<float>();
    bp[0] = 0.0f; bp[1] = 1.0f; bp[2] = 2.0f; bp[3] = 3.0f;

    auto a = a_cpu.to(device()).to(dtype());
    auto b = b_cpu.to(device()).to(dtype());
    auto result = tenzor::gt(a, b);
    expectShape(result, {3, 4});

    // Row-major expected: row 0 all false, row 1 all true,
    // row 2 = (0>0, 3>1, 2>2, 1>3) = (false, true, false, false).
    std::vector<bool> expected = {
        false, false, false, false,
        true,  true,  true,  true,
        false, true,  false, false,
    };
    expectBoolTensorEquals(result, expected);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ComparisonOpsMultiDTypeTest);
