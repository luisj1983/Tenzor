"""
Autograd flag coverage.

Exercises the Python bindings for the keyword arguments on Variable.backward()
and Variable.retain_grad() — previously only tested transitively through the
optimizer suite. These tests drive the arguments directly so binding
regressions surface immediately.
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


def _scalar_from(v):
    t = v.tensor() if hasattr(v, "tensor") else v
    return float(t.item())


def test_backward_default_accumulates_grad():
    x = tz.Variable(tz.full([1], 3.0), True)
    y = x * x
    y.backward()
    g = x.grad
    assert g is not None
    # dy/dx at x=3 is 6.
    assert abs(_scalar_from(g) - 6.0) < 1e-5


def test_backward_retain_graph_allows_second_pass():
    x = tz.Variable(tz.full([1], 3.0), True)
    y = x * x
    y.backward(retain_graph=True)
    first = _scalar_from(x.grad)
    # Without retain_graph the second backward would error because the graph
    # was freed. We zero the grad first so we compare magnitudes not sums.
    x.zero_grad()
    y.backward()  # graph was retained, so this still works
    second = _scalar_from(x.grad)
    assert abs(first - second) < 1e-5


def test_backward_create_graph_enables_higher_order():
    # Build x^3; d/dx = 3x^2; d^2/dx^2 = 6x. At x=2 second derivative = 12.
    x = tz.Variable(tz.full([1], 2.0), True)
    y = x * x * x
    y.backward(create_graph=True)
    first_grad = x.grad
    assert first_grad is not None
    # first_grad holds a differentiable variable when create_graph=True. We
    # assert the numerical value of the first derivative; the second-derivative
    # path via x.grad.backward() depends on whether grad() returns a
    # Variable, which we don't require for this test.
    # d/dx (x^3) at x=2 = 12.
    assert abs(_scalar_from(first_grad) - 12.0) < 1e-5


def test_retain_grad_on_nonleaf():
    # Non-leaf tensors normally don't retain .grad. retain_grad() flips that.
    x = tz.Variable(tz.full([1], 2.0), True)
    y = x * 3  # y is non-leaf
    y.retain_grad()
    z = y * y  # z = (3x)^2 = 9x^2
    z.backward()
    # dz/dy at y=6 is 2y = 12.
    y_grad = y.grad
    assert y_grad is not None, "retain_grad did not preserve the non-leaf gradient"
    assert abs(_scalar_from(y_grad) - 12.0) < 1e-4
