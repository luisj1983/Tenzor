"""Tensor-native ``tenzor.distributions`` smoke + autograd tests (S19).

Runs as a plain Python script (no pytest dependency) — matches the
existing ``tests/python/test_*.py`` pattern wired through
``tests/CMakeLists.txt`` ``add_test``.

Coverage:

1. Every reparameterised location-scale distribution returns an
   autograd-aware Variable from ``rsample`` and produces non-None
   gradients for its parameters after ``backward()``.
2. ``Bernoulli.sample`` and ``Categorical.sample`` return Tenzor
   Tensors with the correct dtype.
3. ``Normal(0, 1).log_prob(0)`` and ``Exponential(1).log_prob(1)``
   match closed-form values to within 1e-3.
4. ``has_rsample`` honestly reflects whether ``rsample`` has a working
   autograd path: every ``has_rsample = False`` distribution raises
   ``NotImplementedError`` from ``rsample``.

Exit non-zero on the first failed assertion.
"""
from __future__ import annotations

import math
import os
import sys
import traceback

# Match the import pattern used by tests/python/test_autograd.py.
_BUILD_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
)
if os.path.isdir(_BUILD_PY):
    sys.path.insert(0, _BUILD_PY)

import numpy as np  # noqa: E402

import tenzor as tz  # noqa: E402
tz.initialize()

from tenzor.tenzor_core import Tensor, Variable  # noqa: E402
from tenzor.distributions import (  # noqa: E402
    Normal, Uniform, Laplace, Exponential, Cauchy, Gumbel,
    Bernoulli, Categorical,
    Binomial, Poisson, Geometric, NegativeBinomial,
    Multinomial, Dirichlet,
    Gamma, Beta, StudentT, Chi2, LogNormal, HalfNormal, Weibull, VonMises,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_failures = []


def _fail(name, exc):
    _failures.append((name, exc))
    print(f"  FAIL: {name}\n        {exc}")
    traceback.print_exc()


def _run(name, fn):
    print(f"[run] {name}")
    try:
        fn()
        print(f"  ok: {name}")
    except AssertionError as e:
        _fail(name, e)
    except Exception as e:  # pragma: no cover
        _fail(name, e)


def _var(values, requires_grad: bool = False) -> Variable:
    """Build a CPU float32 Variable from a Python scalar / list."""
    if isinstance(values, (list, tuple)):
        t = Tensor.from_numpy(np.asarray(values, dtype=np.float32))
    else:
        t = tz.full([], float(values), tz.dtype.float32)
    return Variable(t, requires_grad)


def _np(t) -> np.ndarray:
    """Convert a Tenzor Tensor/Variable to a numpy array."""
    if isinstance(t, Variable):
        return np.asarray(t.tensor())
    return np.asarray(t)


# ---------------------------------------------------------------------------
# 1) Location-scale rsample autograd
# ---------------------------------------------------------------------------

def _check_rsample_autograd(cls, param_kwargs, scalar_loc_attr,
                            scalar_scale_attr, sample_shape=(8,)):
    """Construct ``cls`` with the given grad-enabled params, draw
    ``rsample``, ``sum().backward()`` and assert non-None param grads.
    """
    params = {k: _var(v, requires_grad=True) for k, v in param_kwargs.items()}
    dist = cls(**params)

    assert dist.has_rsample is True, f"{cls.__name__}.has_rsample should be True"

    x = dist.rsample(sample_shape)
    assert isinstance(x, Variable), f"{cls.__name__}.rsample must return Variable"
    assert x.requires_grad, f"{cls.__name__}.rsample lost requires_grad"
    assert tuple(x.shape) == tuple(sample_shape), (
        f"{cls.__name__}.rsample shape {x.shape} != {sample_shape}")

    loss = tz.sum(x)
    loss.backward()

    # The first scalar parameter must have a non-None grad.
    loc = params[scalar_loc_attr]
    scale = params[scalar_scale_attr]
    assert loc.grad is not None, f"{cls.__name__}: {scalar_loc_attr}.grad is None"
    assert scale.grad is not None, f"{cls.__name__}: {scalar_scale_attr}.grad is None"


def test_normal_rsample():
    _check_rsample_autograd(Normal, {"loc": 0.0, "scale": 1.0}, "loc", "scale")


def test_uniform_rsample():
    _check_rsample_autograd(Uniform, {"low": 0.0, "high": 1.0}, "low", "high")


def test_laplace_rsample():
    _check_rsample_autograd(Laplace, {"loc": 0.0, "scale": 1.0}, "loc", "scale")


def test_exponential_rsample():
    # Exponential has only one parameter; check the rate.grad path.
    rate = _var(1.0, requires_grad=True)
    dist = Exponential(rate)
    assert dist.has_rsample
    x = dist.rsample((8,))
    assert isinstance(x, Variable) and x.requires_grad
    tz.sum(x).backward()
    assert rate.grad is not None


def test_cauchy_rsample():
    _check_rsample_autograd(Cauchy, {"loc": 0.0, "scale": 1.0}, "loc", "scale")


def test_gumbel_rsample():
    _check_rsample_autograd(Gumbel, {"loc": 0.0, "scale": 1.0}, "loc", "scale")


# ---------------------------------------------------------------------------
# 2) Discrete sample returns Tensor
# ---------------------------------------------------------------------------

def test_bernoulli_sample_is_tensor():
    b = Bernoulli(_var(0.5))
    s = b.sample((100,))
    assert isinstance(s, Tensor), f"Bernoulli.sample should return Tensor, got {type(s).__name__}"
    s_np = _np(s)
    assert s_np.shape == (100,), f"shape {s_np.shape}"
    assert set(np.unique(s_np)).issubset({0.0, 1.0}), f"values {np.unique(s_np)}"


def test_categorical_sample_is_int64():
    probs = Tensor.from_numpy(np.array([0.1, 0.2, 0.7], dtype=np.float32))
    c = Categorical(probs)
    s = c.sample((50,))
    assert isinstance(s, Tensor), f"Categorical.sample should return Tensor, got {type(s).__name__}"
    assert s.dtype == tz.dtype.int64, f"Categorical.sample dtype {s.dtype} != int64"
    s_np = _np(s)
    assert s_np.shape == (50,) or s_np.shape == (50, 1)
    assert s_np.min() >= 0 and s_np.max() < 3


# ---------------------------------------------------------------------------
# 3) log_prob numerical sanity
# ---------------------------------------------------------------------------

def test_normal_log_prob_at_zero():
    n = Normal(_var(0.0), _var(1.0))
    lp = n.log_prob(_var(0.0))
    lp_val = float(_np(lp))
    expected = -0.5 * math.log(2.0 * math.pi)
    assert abs(lp_val - expected) < 1e-3, f"Normal log_prob(0) = {lp_val}, expected {expected}"


def test_exponential_log_prob_at_one():
    e = Exponential(_var(1.0))
    lp = e.log_prob(_var(1.0))
    lp_val = float(_np(lp))
    expected = 0.0 - 1.0  # log(1) - 1*1 = -1
    assert abs(lp_val - expected) < 1e-3, f"Exp log_prob(1) = {lp_val}, expected {expected}"


# ---------------------------------------------------------------------------
# 4) has_rsample honesty
# ---------------------------------------------------------------------------

_NO_RSAMPLE = [
    (Bernoulli, (0.5,)),
    (Categorical, (Tensor.from_numpy(np.array([0.5, 0.5], dtype=np.float32)),)),
    (Binomial, (5, 0.3)),
    (Poisson, (2.0,)),
    (Geometric, (0.4,)),
    (NegativeBinomial, (3.0, 0.5)),
    (Multinomial, (10, Tensor.from_numpy(np.array([0.3, 0.7], dtype=np.float32)))),
    # Continuous reparameterisable distributions (Dirichlet/Gamma/Beta/StudentT/
    # Chi2/LogNormal/HalfNormal/Weibull) DO implement a real reparameterised
    # rsample, so they correctly report has_rsample=True and are NOT part of the
    # no-rsample honesty set.
    (VonMises, (0.0, 1.0)),  # rejection-sampled; no straightforward reparam
]


def test_no_rsample_honesty():
    for cls, args in _NO_RSAMPLE:
        dist = cls(*args)
        assert dist.has_rsample is False, (
            f"{cls.__name__}.has_rsample should be False")
        try:
            dist.rsample()
        except NotImplementedError:
            continue
        raise AssertionError(
            f"{cls.__name__}.rsample silently succeeded — should raise NotImplementedError")


# ---------------------------------------------------------------------------
# 5) Bonus: rsample value is Tensor/Variable (never numpy)
# ---------------------------------------------------------------------------

def test_rsample_never_returns_numpy():
    for cls, args in [
        (Normal, (0.0, 1.0)),
        (Uniform, (0.0, 1.0)),
        (Laplace, (0.0, 1.0)),
        (Exponential, (1.0,)),
        (Cauchy, (0.0, 1.0)),
        (Gumbel, (0.0, 1.0)),
    ]:
        d = cls(*args)
        s = d.rsample((4,))
        assert isinstance(s, (Tensor, Variable)), (
            f"{cls.__name__}.rsample returned {type(s).__name__}, not Tensor/Variable")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        ("normal_rsample", test_normal_rsample),
        ("uniform_rsample", test_uniform_rsample),
        ("laplace_rsample", test_laplace_rsample),
        ("exponential_rsample", test_exponential_rsample),
        ("cauchy_rsample", test_cauchy_rsample),
        ("gumbel_rsample", test_gumbel_rsample),
        ("bernoulli_sample_tensor", test_bernoulli_sample_is_tensor),
        ("categorical_sample_int64", test_categorical_sample_is_int64),
        ("normal_log_prob_at_zero", test_normal_log_prob_at_zero),
        ("exponential_log_prob_at_one", test_exponential_log_prob_at_one),
        ("no_rsample_honesty", test_no_rsample_honesty),
        ("rsample_never_returns_numpy", test_rsample_never_returns_numpy),
    ]
    for name, fn in cases:
        _run(name, fn)
    if _failures:
        print(f"\n{len(_failures)} test(s) failed:")
        for name, exc in _failures:
            print(f"  - {name}: {exc}")
        sys.exit(1)
    print(f"\nAll {len(cases)} distribution tests passed.")
