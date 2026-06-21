// Tests for the L-BFGS optimizer, with specific coverage of the
// strong-Wolfe line search that Phase 1.4 of typed-meandering-dragon added.
//
// Ported to the cross-backend parity suite: each case runs on every available
// backend via BackendTest. Parameters and closures construct their tensors on
// the per-test `device`; host-side assertions read back through .cpu().

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/lbfgs.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>

#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

#include <cmath>

namespace tenzor {
namespace {

class LBFGSTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Scalar loss to plain double for assertions.
    static auto to_double(const Variable& v) -> double {
        auto t = (v.tensor().device() == Device::cpu())
            ? v.tensor() : v.tensor().to(Device::cpu());
        if (t.dtype() == DType::Float64) return t.data<double>()[0];
        if (t.dtype() != DType::Float32) t = t.to(DType::Float32);
        return static_cast<double>(t.data<float>()[0]);
    }

    // f(x,y) = (a - x)^2 + b * (y - x^2)^2, classic Rosenbrock.
    // Unique minimizer at (a, a^2) with f = 0. Non-quadratic, non-trivial
    // curvature — the canonical quasi-Newton benchmark.
    static auto rosenbrock(const std::shared_ptr<Variable>& x,
                           const std::shared_ptr<Variable>& y,
                           const Device& device,
                           double a = 1.0,
                           double b = 100.0) -> Variable {
        auto ax = tenzor::full({1}, a, DType::Float64, device);
        auto bx = tenzor::full({1}, b, DType::Float64, device);
        auto term1 = (Variable(ax, false) - *x) * (Variable(ax, false) - *x);
        auto x_sq = (*x) * (*x);
        auto diff = (*y) - x_sq;
        auto term2 = Variable(bx, false) * diff * diff;
        return term1 + term2;
    }

    static auto make_scalar_param(double value, const Device& device)
        -> std::shared_ptr<Variable> {
        auto t = tenzor::full({1}, value, DType::Float64, device);
        return std::make_shared<Variable>(t, /*requires_grad=*/true);
    }
};

// ---------------------------------------------------------------------------
// Strong Wolfe converges on Rosenbrock starting from (-1.2, 1.0).
// ---------------------------------------------------------------------------

TEST_P(LBFGSTest, StrongWolfeRosenbrockConverges) {
    auto x = make_scalar_param(-1.2, device);
    auto y = make_scalar_param(1.0, device);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        y->zero_grad();
        auto loss = rosenbrock(x, y, device);
        loss.backward();
        return loss;
    };

    // One-shot grad-flow check before entering the optimizer loop
    // (audit-2026-05-03 N1.d). Catches grad_fn severance / closure regressions
    // that would otherwise be invisible when the optimizer happens to converge
    // anyway (e.g. via numerical line search).
    closure();
    EXPECT_GRAD_FLOWS(*x);
    EXPECT_GRAD_FLOWS(*y);

    optim::LBFGS opt({x, y}, /*lr=*/1.0, /*max_iter=*/20, /*max_eval=*/-1,
                     /*tolerance_grad=*/1e-7, /*tolerance_change=*/1e-9,
                     /*history_size=*/100,
                     optim::LBFGSLineSearch::StrongWolfe);

    double final_loss = 1e9;
    for (int k = 0; k < 30; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-8) break;
    }

    const double fx = to_double(*x);
    const double fy = to_double(*y);
    EXPECT_LT(final_loss, 1e-6)
        << "StrongWolfe failed to drive Rosenbrock to the global min "
        << "(f=" << final_loss << ", x=" << fx << ", y=" << fy << ")";
    EXPECT_NEAR(fx, 1.0, 1e-3);
    EXPECT_NEAR(fy, 1.0, 1e-3);
}

// ---------------------------------------------------------------------------
// Armijo converges too, but typically takes more outer iterations — we just
// require that it also reaches the minimum under the same wall-clock budget.
// ---------------------------------------------------------------------------

TEST_P(LBFGSTest, ArmijoRosenbrockAlsoConverges) {
    auto x = make_scalar_param(-1.2, device);
    auto y = make_scalar_param(1.0, device);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        y->zero_grad();
        auto loss = rosenbrock(x, y, device);
        loss.backward();
        return loss;
    };

    // Deliberately use the legacy Armijo line search to make sure it still
    // works after the refactor.
    optim::LBFGS opt({x, y}, /*lr=*/1.0, /*max_iter=*/20, /*max_eval=*/-1,
                     /*tolerance_grad=*/1e-7, /*tolerance_change=*/1e-9,
                     /*history_size=*/100,
                     optim::LBFGSLineSearch::Armijo);

    double final_loss = 1e9;
    for (int k = 0; k < 80; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-6) break;
    }

    EXPECT_LT(final_loss, 1e-4)
        << "Armijo L-BFGS failed to converge on Rosenbrock "
        << "(f=" << final_loss << ")";
}

// ---------------------------------------------------------------------------
// Strong Wolfe on a simple convex quadratic should converge to machine
// precision in O(dim) iterations — this is the line-search reliability test.
// ---------------------------------------------------------------------------

TEST_P(LBFGSTest, StrongWolfeQuadraticExact) {
    // f(x) = 0.5 * (x0^2 + 2*x1^2 + 3*x2^2) — diagonal PSD.
    auto x = std::make_shared<Variable>(
        tenzor::full({3}, 1.5, DType::Float64, device),
        /*requires_grad=*/true);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        // Host write of the diagonal coefficients, then move onto device.
        auto coef_cpu = tenzor::full({3}, 0.0, DType::Float64, Device::cpu());
        coef_cpu.data<double>()[0] = 0.5;
        coef_cpu.data<double>()[1] = 1.0;
        coef_cpu.data<double>()[2] = 1.5;
        auto coef = coef_cpu.to(device);
        auto coef_v = Variable(coef, false);
        auto sq = (*x) * (*x);
        auto weighted = coef_v * sq;
        auto loss = sum(weighted);
        loss.backward();
        return loss;
    };

    optim::LBFGS opt({x}, /*lr=*/1.0, /*max_iter=*/20, /*max_eval=*/-1,
                     /*tolerance_grad=*/1e-12, /*tolerance_change=*/1e-14,
                     /*history_size=*/10,
                     optim::LBFGSLineSearch::StrongWolfe);

    double final_loss = 1e9;
    for (int k = 0; k < 10; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-14) break;
    }

    EXPECT_LT(final_loss, 1e-12)
        << "Strong Wolfe L-BFGS should drive this diagonal quadratic to "
        << "machine zero quickly (f=" << final_loss << ")";
}

// ---------------------------------------------------------------------------
// Cubic interpolation correctness: on a known cubic, strong Wolfe should
// accept in one bracketing step rather than many. We can't observe function
// eval counts directly, but we can verify it never diverges on a contrived
// steep-curvature quadratic where bisection would spin.
// ---------------------------------------------------------------------------

TEST_P(LBFGSTest, StrongWolfeHandlesSteepCurvature) {
    // f(x) = 0.5 * 1e4 * x^2 — very steep quadratic. The default lr=1 is
    // massively too large; the line search must shrink aggressively.
    auto x = std::make_shared<Variable>(
        tenzor::full({1}, 2.0, DType::Float64, device),
        /*requires_grad=*/true);

    auto closure = [&]() -> Variable {
        x->zero_grad();
        auto c = Variable(tenzor::full({1}, 5000.0, DType::Float64, device), false);
        auto loss = c * (*x) * (*x);
        loss.backward();
        return loss;
    };

    optim::LBFGS opt({x}, /*lr=*/1.0, /*max_iter=*/20, /*max_eval=*/-1,
                     /*tolerance_grad=*/1e-10, /*tolerance_change=*/1e-12,
                     /*history_size=*/10,
                     optim::LBFGSLineSearch::StrongWolfe);

    double final_loss = 1e9;
    for (int k = 0; k < 15; ++k) {
        auto loss_var = opt.step(closure);
        final_loss = to_double(loss_var);
        if (final_loss < 1e-10) break;
        ASSERT_TRUE(std::isfinite(final_loss))
            << "Line search diverged — loss became non-finite at step " << k;
    }

    EXPECT_LT(final_loss, 1e-8);
}

// ---------------------------------------------------------------------------
// Regression: a checkpointed L-BFGS run must resume correctly. state_dict()
// previously dropped prev_flat_params_, so the first post-reload step formed
// the curvature pair s = params - prev_flat_params_ against an empty tensor,
// either throwing on the shape mismatch or poisoning the (s, y) history. This
// test takes a few steps, snapshots, reloads into a fresh optimizer, and
// asserts the resumed run keeps making progress (and never throws / NaNs).
// ---------------------------------------------------------------------------
TEST_P(LBFGSTest, StateDictRoundTripResumesAfterStep) {
    auto make_x = [&]() {
        return std::make_shared<Variable>(
            tenzor::full({3}, 1.5, DType::Float64, device),
            /*requires_grad=*/true);
    };

    auto make_closure = [&](std::shared_ptr<Variable> x) {
        return [x, this]() -> Variable {
            x->zero_grad();
            auto coef_cpu = tenzor::full({3}, 0.0, DType::Float64, Device::cpu());
            coef_cpu.data<double>()[0] = 0.5;
            coef_cpu.data<double>()[1] = 1.0;
            coef_cpu.data<double>()[2] = 1.5;
            auto coef_v = Variable(coef_cpu.to(device), false);
            auto loss = sum(coef_v * (*x) * (*x));
            loss.backward();
            return loss;
        };
    };

    auto x = make_x();
    auto closure = make_closure(x);

    optim::LBFGS opt({x}, /*lr=*/1.0, /*max_iter=*/5, /*max_eval=*/-1,
                     /*tolerance_grad=*/1e-12, /*tolerance_change=*/1e-14,
                     /*history_size=*/10,
                     optim::LBFGSLineSearch::StrongWolfe);

    // A couple of steps to populate prev_flat_params_/prev_flat_grad_ + history.
    opt.step(closure);
    double loss_before_save = to_double(opt.step(closure));

    // Snapshot, then reload into a fresh optimizer over a parameter that we set
    // to the same current value as the live one.
    auto state = opt.state_dict();
    ASSERT_GT(state.count("prev_flat_params"), 0u)
        << "state_dict() must persist prev_flat_params for a resumable checkpoint";

    auto x2 = std::make_shared<Variable>(x->tensor().clone(), /*requires_grad=*/true);
    auto closure2 = make_closure(x2);
    optim::LBFGS opt2({x2}, /*lr=*/1.0, /*max_iter=*/5, /*max_eval=*/-1,
                      /*tolerance_grad=*/1e-12, /*tolerance_change=*/1e-14,
                      /*history_size=*/10,
                      optim::LBFGSLineSearch::StrongWolfe);
    ASSERT_NO_THROW(opt2.load_state_dict(state));

    // First post-reload step must not throw and must keep improving the loss.
    double resumed_loss = 1e9;
    ASSERT_NO_THROW({ resumed_loss = to_double(opt2.step(closure2)); });
    ASSERT_TRUE(std::isfinite(resumed_loss));
    EXPECT_LT(resumed_loss, loss_before_save + 1e-9)
        << "Resumed L-BFGS step regressed the loss (curvature pair corrupted by "
        << "missing prev_flat_params_)";

    // Drive to convergence to confirm the resumed run is healthy.
    for (int k = 0; k < 10 && resumed_loss > 1e-12; ++k) {
        resumed_loss = to_double(opt2.step(closure2));
        ASSERT_TRUE(std::isfinite(resumed_loss));
    }
    EXPECT_LT(resumed_loss, 1e-10);
}

INSTANTIATE_BACKEND_TESTS(LBFGSTest);

} // namespace
} // namespace tenzor
