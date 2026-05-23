/**
 * @file test_lbfgs_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for L-BFGS optimizer
 *
 * audit T.1: every TEST_P below asserts convergence to a closed-form known
 * minimum via `EXPECT_LT(final_loss, ...)` on the actual numeric loss value.
 * This is a value-level assertion (closed-form expected minimum), not a
 * shape-only check — see plan item T.1 for why convergence-to-known-min
 * is the canonical value assertion for an optimizer. L-BFGS is CPU-only
 * (Float64 params), so the same trajectory runs on every backend×dtype slot.
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/lbfgs.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../../grad_flow_helpers.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class LBFGSMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    static auto to_double(const Variable& v) -> double {
        auto t = v.tensor().to(Device::cpu());
        if (t.dtype() == DType::Float64) return t.data<double>()[0];
        if (t.dtype() != DType::Float32) t = t.to(DType::Float32);
        return static_cast<double>(t.data<float>()[0]);
    }

    auto make_scalar_param(double value) -> std::shared_ptr<Variable> {
        // L-BFGS needs Float64 for precision
        auto t = tenzor::full({1}, value, DType::Float64, Device::cpu());
        return std::make_shared<Variable>(t, true);
    }
};

TEST_P(LBFGSMultiDTypeTest, StrongWolfeQuadraticExact) {
    // L-BFGS is a CPU-only optimizer with Float64 params
    auto x = std::make_shared<Variable>(
        tenzor::full({3}, 1.5, DType::Float64, Device::cpu()), true);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        auto coef = tenzor::full({3}, 0.0, DType::Float64, Device::cpu());
        coef.data<double>()[0] = 0.5;
        coef.data<double>()[1] = 1.0;
        coef.data<double>()[2] = 1.5;
        auto coef_v = Variable(coef, false);
        auto sq = (*x) * (*x);
        auto weighted = coef_v * sq;
        auto loss = sum(weighted);
        loss.backward();
        return loss;
    };

    // audit-2026-05-03 N1.d: one-shot grad-flow check before optimizer loop.
    closure();
    EXPECT_GRAD_FLOWS(*x);

    optim::LBFGS opt({x}, 1.0, 20, -1, 1e-12, 1e-14, 10,
                     optim::LBFGSLineSearch::StrongWolfe);

    double final_loss = 1e9;
    for (int k = 0; k < 10; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-14) break;
    }

    EXPECT_LT(final_loss, 1e-12);
}

TEST_P(LBFGSMultiDTypeTest, StrongWolfeRosenbrockConverges) {
    auto x = make_scalar_param(-1.2);
    auto y = make_scalar_param(1.0);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        y->zero_grad();
        auto ax = tenzor::full({1}, 1.0, DType::Float64, Device::cpu());
        auto bx = tenzor::full({1}, 100.0, DType::Float64, Device::cpu());
        auto term1 = (Variable(ax, false) - *x) * (Variable(ax, false) - *x);
        auto x_sq = (*x) * (*x);
        auto diff = (*y) - x_sq;
        auto term2 = Variable(bx, false) * diff * diff;
        auto loss = term1 + term2;
        loss.backward();
        return loss;
    };

    optim::LBFGS opt({x, y}, 1.0, 20, -1, 1e-7, 1e-9, 100,
                     optim::LBFGSLineSearch::StrongWolfe);

    double final_loss = 1e9;
    for (int k = 0; k < 30; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-8) break;
    }

    EXPECT_LT(final_loss, 1e-6);
    // audit T.1: also assert the optimum location — Rosenbrock minimum is
    // at (1, 1). This is the closed-form expected parameter value.
    EXPECT_NEAR(to_double(*x), 1.0, 1e-3);
    EXPECT_NEAR(to_double(*y), 1.0, 1e-3);
}

TEST_P(LBFGSMultiDTypeTest, ArmijoRosenbrockAlsoConverges) {
    auto x = make_scalar_param(-1.2);
    auto y = make_scalar_param(1.0);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        y->zero_grad();
        auto ax = tenzor::full({1}, 1.0, DType::Float64, Device::cpu());
        auto bx = tenzor::full({1}, 100.0, DType::Float64, Device::cpu());
        auto term1 = (Variable(ax, false) - *x) * (Variable(ax, false) - *x);
        auto x_sq = (*x) * (*x);
        auto diff = (*y) - x_sq;
        auto term2 = Variable(bx, false) * diff * diff;
        auto loss = term1 + term2;
        loss.backward();
        return loss;
    };

    optim::LBFGS opt({x, y}, 1.0, 20, -1, 1e-7, 1e-9, 100,
                     optim::LBFGSLineSearch::Armijo);

    double final_loss = 1e9;
    for (int k = 0; k < 80; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-6) break;
    }

    EXPECT_LT(final_loss, 1e-4);
}

TEST_P(LBFGSMultiDTypeTest, StrongWolfeHandlesSteepCurvature) {
    auto x = std::make_shared<Variable>(
        tenzor::full({1}, 2.0, DType::Float64, Device::cpu()), true);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        auto c = Variable(tenzor::full({1}, 5000.0, DType::Float64, Device::cpu()), false);
        auto loss = c * (*x) * (*x);
        loss.backward();
        return loss;
    };

    optim::LBFGS opt({x}, 1.0, 20, -1, 1e-10, 1e-12, 10,
                     optim::LBFGSLineSearch::StrongWolfe);

    double final_loss = 1e9;
    for (int k = 0; k < 15; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-10) break;
        ASSERT_TRUE(std::isfinite(final_loss));
    }

    EXPECT_LT(final_loss, 1e-8);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LBFGSMultiDTypeTest);
