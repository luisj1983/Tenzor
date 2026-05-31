/**
 * @file test_autograd_hvp_vhp.cpp
 * @brief Unit tests for Hessian-vector product (hvp) and vector-Hessian product (vhp)
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <functional>

using namespace tenzor;

class HvpVhpTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// HVP tests
// ============================================================================

TEST_P(HvpVhpTest, HvpQuadraticIdentityHessian) {
    // f(x) = 0.5 * x^T * I * x = 0.5 * sum(x^2)
    // Hessian = I (identity)
    // hvp(f, x, v) = I @ v = v
    auto x = Variable(ones({4}, DType::Float64, device), true);
    float v_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto v = from_data(v_data, {4}).to(DType::Float64).to(device);

    auto func = [](const Variable& input) -> Variable {
        // 0.5 * sum(x * x)
        auto sq = input * input;
        return sum(sq) * 0.5;
    };

    auto [output, hv] = hvp(func, x, v);

    // H @ v = I @ v = v
    auto hv_cpu = hv.cpu();
    auto hv_data = hv_cpu.data<double>();
    EXPECT_NEAR(hv_data[0], 1.0, 1e-5);
    EXPECT_NEAR(hv_data[1], 2.0, 1e-5);
    EXPECT_NEAR(hv_data[2], 3.0, 1e-5);
    EXPECT_NEAR(hv_data[3], 4.0, 1e-5);
}

TEST_P(HvpVhpTest, HvpQuadraticScaledHessian) {
    // f(x) = 0.5 * (2*x0^2 + 4*x1^2) = x0^2 + 2*x1^2
    // Hessian = diag(2, 4)
    // hvp(f, x, v) = [2*v0, 4*v1]
    float x_data[] = {1.0f, 1.0f};
    auto x = Variable(from_data(x_data, {2}).to(DType::Float64).to(device), true);

    float v_data[] = {3.0f, 5.0f};
    auto v = from_data(v_data, {2}).to(DType::Float64).to(device);

    // Build the diagonal A = diag(2, 4) via element-wise multiply
    auto func = [this](const Variable& input) -> Variable {
        // Manually compute: x0^2 + 2*x1^2
        // which is 0.5 * x^T * diag(2,4) * x
        auto sq = input * input;  // [x0^2, x1^2]
        // We need weights [1, 2] to get [x0^2, 2*x1^2] then sum
        float w[] = {1.0f, 2.0f};
        auto weights = Variable(from_data(w, {2}).to(DType::Float64).to(device), false);
        return sum(sq * weights);
    };

    auto [output, hv] = hvp(func, x, v);

    auto hv_cpu = hv.cpu();
    auto hv_data = hv_cpu.data<double>();
    // Hessian of (x0^2 + 2*x1^2) = diag(2, 4)
    EXPECT_NEAR(hv_data[0], 2.0 * 3.0, 1e-5);  // 6
    EXPECT_NEAR(hv_data[1], 4.0 * 5.0, 1e-5);  // 20
}

// ============================================================================
// VHP tests
// ============================================================================

TEST_P(HvpVhpTest, VhpQuadraticIdentityHessian) {
    // f(x) = 0.5 * sum(x^2), Hessian = I
    // vhp(f, x, v) = v^T @ I = v
    auto x = Variable(ones({4}, DType::Float64, device), true);
    float v_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto v = from_data(v_data, {4}).to(DType::Float64).to(device);

    auto func = [](const Variable& input) -> Variable {
        auto sq = input * input;
        return sum(sq) * 0.5;
    };

    auto [output, vh] = vhp(func, x, v);

    auto vh_cpu = vh.cpu();
    auto vh_data = vh_cpu.data<double>();
    EXPECT_NEAR(vh_data[0], 1.0, 1e-5);
    EXPECT_NEAR(vh_data[1], 2.0, 1e-5);
    EXPECT_NEAR(vh_data[2], 3.0, 1e-5);
    EXPECT_NEAR(vh_data[3], 4.0, 1e-5);
}

TEST_P(HvpVhpTest, VhpQuadraticScaledHessian) {
    // f(x) = x0^2 + 2*x1^2, Hessian = diag(2, 4)
    // vhp(f, x, v) = v^T @ H = [2*v0, 4*v1] (H is symmetric so same as hvp)
    float x_data[] = {1.0f, 1.0f};
    auto x = Variable(from_data(x_data, {2}).to(DType::Float64).to(device), true);

    float v_data[] = {3.0f, 5.0f};
    auto v = from_data(v_data, {2}).to(DType::Float64).to(device);

    auto func = [this](const Variable& input) -> Variable {
        auto sq = input * input;
        float w[] = {1.0f, 2.0f};
        auto weights = Variable(from_data(w, {2}).to(DType::Float64).to(device), false);
        return sum(sq * weights);
    };

    auto [output, vh] = vhp(func, x, v);

    auto vh_cpu = vh.cpu();
    auto vh_data = vh_cpu.data<double>();
    // v^T @ diag(2,4) = [2*3, 4*5] = [6, 20]
    EXPECT_NEAR(vh_data[0], 6.0, 1e-5);
    EXPECT_NEAR(vh_data[1], 20.0, 1e-5);
}

TEST_P(HvpVhpTest, HvpAndVhpAgreeForSymmetricHessian) {
    // For symmetric H, hvp(f,x,v) and vhp(f,x,v) should give the same result
    // f(x) = 0.5 * sum(x^2), H = I (symmetric)
    auto x = Variable(ones({3}, DType::Float64, device) * 2.0, true);
    float v_data[] = {1.0f, -1.0f, 2.0f};
    auto v = from_data(v_data, {3}).to(DType::Float64).to(device);

    auto func = [](const Variable& input) -> Variable {
        return sum(input * input) * 0.5;
    };

    auto [out_hvp, hv] = hvp(func, x, v);
    auto [out_vhp, vh] = vhp(func, x, v);

    auto hv_cpu = hv.cpu();
    auto vh_cpu = vh.cpu();
    auto hv_data = hv_cpu.data<double>();
    auto vh_data = vh_cpu.data<double>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(hv_data[i], vh_data[i], 1e-5) << "index " << i;
    }
}

INSTANTIATE_BACKEND_TESTS(HvpVhpTest);
