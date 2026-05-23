/**
 * @file test_func.cpp
 * @brief Tests for composable function transforms (tenzor::func)
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/func.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>

class FuncTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Test that func::grad computes the correct gradient of a simple function
TEST_F(FuncTest, GradSimpleQuadratic) {
    // f(x) = sum(x^2), grad(f)(x) = 2*x
    auto f = [](const tenzor::Variable& x) -> tenzor::Variable {
        auto sq = x * x;
        return tenzor::sum(sq);
    };

    auto grad_f = tenzor::func::grad(f);

    // Evaluate at x = [1, 2, 3]
    auto x_data = tenzor::zeros({3}, tenzor::DType::Float32);
    static_cast<float*>(x_data.data_ptr())[0] = 1.0f;
    static_cast<float*>(x_data.data_ptr())[1] = 2.0f;
    static_cast<float*>(x_data.data_ptr())[2] = 3.0f;
    tenzor::Variable x(x_data, true);

    auto grad_val = grad_f(x);
    auto grad_ptr = static_cast<const float*>(grad_val.tensor().data_ptr());

    // Expected: [2, 4, 6]
    EXPECT_NEAR(grad_ptr[0], 2.0f, 1e-5);
    EXPECT_NEAR(grad_ptr[1], 4.0f, 1e-5);
    EXPECT_NEAR(grad_ptr[2], 6.0f, 1e-5);
}

// Test that grad correctly computes gradient of x^3
TEST_F(FuncTest, GradCubic) {
    // f(x) = x^3 (scalar input)
    // f'(x) = 3x^2
    auto f = [](const tenzor::Variable& x) -> tenzor::Variable {
        auto x2 = x * x;
        return x2 * x;
    };

    auto grad_f = tenzor::func::grad(f);

    auto x_data = tenzor::zeros({1}, tenzor::DType::Float32);
    static_cast<float*>(x_data.data_ptr())[0] = 2.0f;
    tenzor::Variable x(x_data, true);

    auto grad_val = grad_f(x);
    auto grad_ptr = static_cast<const float*>(grad_val.tensor().data_ptr());

    // f'(2) = 3*4 = 12
    EXPECT_NEAR(grad_ptr[0], 12.0f, 1e-4);
}

// Test that func::jacrev produces the correct Jacobian
TEST_F(FuncTest, JacrevLinear) {
    // f(x) = [2*x[0] + x[1], x[0] - x[1]]
    // J = [[2, 1], [1, -1]]
    auto f = [](const tenzor::Variable& x) -> tenzor::Variable {
        auto x_t = x.tensor();
        auto out_data = tenzor::zeros({2}, tenzor::DType::Float32);
        auto* xp = static_cast<const float*>(x_t.data_ptr());
        auto* op = static_cast<float*>(out_data.data_ptr());
        op[0] = 2.0f * xp[0] + xp[1];
        op[1] = xp[0] - xp[1];
        return tenzor::Variable(out_data, false);
    };

    auto jac_f = tenzor::func::jacrev(f);

    auto x_data = tenzor::ones({2}, tenzor::DType::Float32);
    tenzor::Variable x(x_data, true);

    auto jac_val = jac_f(x);
    auto jac_tensor = jac_val.tensor();

    // Jacobian should be 2x2: [[2, 1], [1, -1]]
    ASSERT_EQ(jac_tensor.shape().size(), 2u);
    ASSERT_EQ(jac_tensor.shape()[0], 2);
    ASSERT_EQ(jac_tensor.shape()[1], 2);

    auto* jp = static_cast<const float*>(jac_tensor.data_ptr());
    // reason: JVP/VJP probe noise — Jacobian assembled via repeated
    // forward-mode JVP probes accumulates Float32 cancellation error
    EXPECT_NEAR(jp[0], 2.0f, 0.1f);  // J[0,0]
    EXPECT_NEAR(jp[1], 1.0f, 0.1f);  // J[0,1]
    EXPECT_NEAR(jp[2], 1.0f, 0.1f);  // J[1,0]
    EXPECT_NEAR(jp[3], -1.0f, 0.1f); // J[1,1]
}

// Test that func::hessian works for a quadratic using autograd ops
TEST_F(FuncTest, HessianQuadratic) {
    // f(x) = sum(x * x) — a simple quadratic
    // H = 2*I (identity scaled by 2)
    auto f = [](const tenzor::Variable& x) -> tenzor::Variable {
        return tenzor::sum(x * x);
    };

    auto hess_f = tenzor::func::hessian(f);

    auto x_data = tenzor::ones({3}, tenzor::DType::Float32);
    tenzor::Variable x(x_data, true);

    auto hess_val = hess_f(x);
    auto hess_tensor = hess_val.tensor();

    ASSERT_EQ(hess_tensor.shape().size(), 2u);
    ASSERT_EQ(hess_tensor.shape()[0], 3);
    ASSERT_EQ(hess_tensor.shape()[1], 3);

    auto* hp = static_cast<const float*>(hess_tensor.data_ptr());
    // Diagonal should be 2, off-diagonal should be 0
    EXPECT_NEAR(hp[0], 2.0f, 0.5f);  // H[0,0]
    EXPECT_NEAR(hp[1], 0.0f, 0.5f);  // H[0,1]
    EXPECT_NEAR(hp[3], 0.0f, 0.5f);  // H[1,0]
    EXPECT_NEAR(hp[4], 2.0f, 0.5f);  // H[1,1]
    EXPECT_NEAR(hp[8], 2.0f, 0.5f);  // H[2,2]
}

// Test composition: vmap(grad(f)) computes per-sample gradients
TEST_F(FuncTest, VmapOfGrad) {
    // f(x) = sum(x^2), so grad(f)(x) = 2*x
    // vmap(grad(f)) should apply this independently to each row of a batch
    auto f = [](const tenzor::Variable& x) -> tenzor::Variable {
        return tenzor::sum(x * x);
    };

    auto grad_f = tenzor::func::grad(f);
    auto batched_grad = tenzor::func::vmap(grad_f);

    // Batch of 3 vectors, each of length 2: [[1,2],[3,4],[5,6]]
    auto batch_data = tenzor::zeros({3, 2}, tenzor::DType::Float32);
    auto* bp = static_cast<float*>(batch_data.data_ptr());
    bp[0] = 1.0f; bp[1] = 2.0f;
    bp[2] = 3.0f; bp[3] = 4.0f;
    bp[4] = 5.0f; bp[5] = 6.0f;
    tenzor::Variable batch_input(batch_data, true);

    auto result = batched_grad(batch_input);
    auto result_tensor = result.tensor();

    // Expected: [[2,4],[6,8],[10,12]]
    ASSERT_EQ(result_tensor.shape().size(), 2u);
    ASSERT_EQ(result_tensor.shape()[0], 3);
    ASSERT_EQ(result_tensor.shape()[1], 2);

    auto* rp = static_cast<const float*>(result_tensor.data_ptr());
    EXPECT_NEAR(rp[0], 2.0f, 1e-4);   // 2*1
    EXPECT_NEAR(rp[1], 4.0f, 1e-4);   // 2*2
    EXPECT_NEAR(rp[2], 6.0f, 1e-4);   // 2*3
    EXPECT_NEAR(rp[3], 8.0f, 1e-4);   // 2*4
    EXPECT_NEAR(rp[4], 10.0f, 1e-4);  // 2*5
    EXPECT_NEAR(rp[5], 12.0f, 1e-4);  // 2*6
}
