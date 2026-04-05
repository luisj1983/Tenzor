/**
 * @file test_broadcasting_gradients.cpp
 * @brief Tests that gradients have correct shapes when broadcasting occurs
 *
 * Verifies that reduce_grad_for_broadcasting() is properly applied in all
 * binary backward operations. Each test creates inputs with different shapes
 * that broadcast together, runs backward, and checks gradient shapes.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

class BroadcastGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
        set_grad_enabled(true);
    }

    void check_grad_shape(const Variable& var, const std::vector<int64_t>& expected_shape) {
        ASSERT_TRUE(var.grad().has_value()) << "Gradient not computed";
        auto grad = var.grad().value();
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        EXPECT_EQ(grad_shape, expected_shape)
            << "Gradient shape mismatch: got " << shape_to_string(grad_shape)
            << " expected " << shape_to_string(expected_shape);
    }

    static std::string shape_to_string(const std::vector<int64_t>& s) {
        std::string result = "(";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(s[i]);
        }
        return result + ")";
    }
};

// ---------- Addition ----------

TEST_F(BroadcastGradientTest, AddMatrixPlusRow) {
    Variable a(ones({3, 4}, DType::Float32), true);
    Variable b(ones({4}, DType::Float32), true);
    auto c = a + b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 4});
    check_grad_shape(b, {4});
}

TEST_F(BroadcastGradientTest, AddMatrixPlusScalar) {
    Variable a(ones({3, 4}, DType::Float32), true);
    Variable b(ones({1}, DType::Float32), true);
    auto c = a + b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 4});
    check_grad_shape(b, {1});
}

TEST_F(BroadcastGradientTest, AddColumnPlusRow) {
    Variable a(ones({3, 1}, DType::Float32), true);
    Variable b(ones({1, 4}, DType::Float32), true);
    auto c = a + b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 1});
    check_grad_shape(b, {1, 4});
}

// ---------- Subtraction ----------

TEST_F(BroadcastGradientTest, SubMatrixMinusRow) {
    Variable a(ones({3, 4}, DType::Float32), true);
    Variable b(ones({4}, DType::Float32), true);
    auto c = a - b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 4});
    check_grad_shape(b, {4});
}

// ---------- Multiplication ----------

TEST_F(BroadcastGradientTest, MulMatrixTimesRow) {
    Variable a(ones({3, 4}, DType::Float32), true);
    Variable b(ones({4}, DType::Float32), true);
    auto c = a * b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 4});
    check_grad_shape(b, {4});
}

TEST_F(BroadcastGradientTest, Mul3DTimesBroadcast) {
    Variable a(ones({2, 3, 4}, DType::Float32), true);
    Variable b(ones({1, 1, 4}, DType::Float32), true);
    auto c = a * b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {2, 3, 4});
    check_grad_shape(b, {1, 1, 4});
}

// ---------- Division ----------

TEST_F(BroadcastGradientTest, DivMatrixByRow) {
    Variable a(ones({3, 4}, DType::Float32), true);
    Variable b(ones({4}, DType::Float32) * 2.0f, true);  // Avoid div by 1 for interesting grad
    auto c = a / b;
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(a, {3, 4});
    check_grad_shape(b, {4});
}

// ---------- Atan2 ----------

TEST_F(BroadcastGradientTest, Atan2MatrixAndRow) {
    Variable y(ones({3, 4}, DType::Float32), true);
    Variable x(ones({4}, DType::Float32), true);
    auto c = tenzor::atan2(y, x);
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(y, {3, 4});
    check_grad_shape(x, {4});
}

TEST_F(BroadcastGradientTest, Atan2ColumnAndRow) {
    Variable y(ones({3, 1}, DType::Float32), true);
    Variable x(ones({1, 4}, DType::Float32), true);
    auto c = tenzor::atan2(y, x);
    auto loss = tenzor::sum(c);
    loss.backward();
    check_grad_shape(y, {3, 1});
    check_grad_shape(x, {1, 4});
}
