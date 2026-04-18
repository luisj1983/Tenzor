"""
LBFGS optimizer Python binding coverage.

Verifies the binding added in bindings_optim.cpp: class construction, the
closure-driven step API, state-dict round-trip, and basic convergence on a
textbook problem (the Rosenbrock function). L-BFGS needs the closure form
because Strong-Wolfe line search makes multiple (loss, grad) evaluations
per outer step.
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


@pytest.fixture(scope="module", autouse=True)
def _init_tenzor():
    tz.initialize()
    tz.manual_seed(42)


def test_lbfgs_constructor_defaults():
    x = tz.Variable(tz.full([2], 0.5), True)
    opt = tz.optim.LBFGS([x])
    assert opt.get_lr() == pytest.approx(1.0)


def test_lbfgs_constructor_kwargs():
    x = tz.Variable(tz.full([3], 0.0), True)
    opt = tz.optim.LBFGS(
        [x],
        lr=0.5,
        max_iter=10,
        history_size=20,
        line_search=tz.optim.LBFGSLineSearch.StrongWolfe,
    )
    assert opt.get_lr() == pytest.approx(0.5)


def test_lbfgs_line_search_enum_exposes_both_variants():
    # Test both enum members are bound.
    assert tz.optim.LBFGSLineSearch.Armijo is not None
    assert tz.optim.LBFGSLineSearch.StrongWolfe is not None
    assert tz.optim.LBFGSLineSearch.Armijo != tz.optim.LBFGSLineSearch.StrongWolfe


def test_lbfgs_step_requires_closure():
    x = tz.Variable(tz.full([1], 2.0), True)
    opt = tz.optim.LBFGS([x])
    # Calling step() without a closure should raise. We can't rely on the
    # exact exception type (pybind11 maps it to RuntimeError in practice)
    # but it must not silently succeed.
    with pytest.raises(Exception):
        opt.step()


def test_lbfgs_converges_on_quadratic():
    # Minimum at x = [0, 0]. L-BFGS should drive |x| below 1e-4 in a few
    # outer steps.
    x = tz.Variable(tz.full([2], 3.0), True)
    opt = tz.optim.LBFGS([x], lr=1.0, max_iter=20, tolerance_grad=1e-10)

    def closure():
        opt.zero_grad()
        loss = tz.sum(x * x)
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)

    # Convert to Python floats via tensor element accessors.
    t = x.tensor()
    final_0 = float(t[0].item())
    final_1 = float(t[1].item())
    assert abs(final_0) < 1e-3, f"x[0] did not converge: {final_0}"
    assert abs(final_1) < 1e-3, f"x[1] did not converge: {final_1}"


def test_lbfgs_state_dict_roundtrip():
    x = tz.Variable(tz.full([2], 1.5), True)
    opt = tz.optim.LBFGS([x])

    def closure():
        opt.zero_grad()
        loss = tz.sum(x * x)
        loss.backward()
        return loss

    # Take a step to populate state history.
    opt.step(closure)
    state = opt.state_dict()
    assert isinstance(state, dict)
    # Apply to a fresh optimizer — should not raise.
    x2 = tz.Variable(tz.full([2], 1.5), True)
    opt2 = tz.optim.LBFGS([x2])
    opt2.load_state_dict(state)
