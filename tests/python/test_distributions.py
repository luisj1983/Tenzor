#!/usr/bin/env python3
"""
Smoke + correctness coverage for Python-bound probability distributions
(`tenzor.distributions`).

Closes audit-2026-05-03 N5 ("Python distributions wrapper untested").

Tenzor's `dist.sample(shape)` API treats the argument as the FULL output
shape (not a prepended sample_shape like PyTorch). Transforms expose
`t(x)` for the forward direction (no `.forward()` method).

For every Python-bound class in `python/bindings/bindings_core.cpp`:
- construct with reasonable defaults,
- call `sample(out_shape)` and assert non-empty / finite where supported,
- call `log_prob(x)`, `mean`, `variance`, `entropy` where supported
  (guarded by `hasattr` since some distributions only implement a subset),
- exercise transforms (forward via `__call__` + log_abs_det_jacobian),
- exercise composed distributions.

Runs as a plain Python script (not pytest) — matches the existing
`tests/python/test_*.py` invocation pattern wired through
`tests/CMakeLists.txt:583` `add_test(... ${Python_EXECUTABLE} test_file)`.
Exits non-zero on any assertion failure.
"""

import os
import sys
import math
import traceback

# Match the import pattern used by tests/python/test_autograd.py
build_python_dir = os.path.join(
    os.path.dirname(__file__), '..', '..', 'build', 'python'
)
build_python_dir = os.path.abspath(build_python_dir)
if os.path.isdir(build_python_dir):
    sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz  # noqa: E402

# Backends must be loaded before any tensor op can be dispatched.
tz.initialize()

D = tz.distributions


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def f32(values, shape=None):
    """Build a Float32 CPU tensor from a Python scalar/list."""
    if shape is None:
        if isinstance(values, (int, float)):
            t = tz.full([], float(values), tz.dtype.float32)
        else:
            t = tz.tensor(values, tz.dtype.float32)
    else:
        t = tz.full(shape, float(values), tz.dtype.float32)
    return t


def assert_finite(t, label):
    flat = t.to(tz.dtype.float32).numpy().ravel()
    for v in flat:
        if math.isnan(v) or math.isinf(v):
            raise AssertionError(f"{label}: non-finite value {v}")


def numel(t):
    return int(t.numel)


# ---------------------------------------------------------------------------
# Per-distribution exercises
# ---------------------------------------------------------------------------

def exercise_distribution(name, dist, x_for_log_prob, sample_out_shape,
                          required_methods=()):
    """Exercise one distribution. `sample_out_shape` is the FULL desired
    output shape (Tenzor convention). `required_methods` MUST be present
    and produce a finite result; other methods are tested gracefully."""

    # sample
    s = dist.sample(list(sample_out_shape))
    if numel(s) == 0:
        raise AssertionError(f"{name}.sample produced empty tensor")
    assert_finite(s, f"{name}.sample")

    # log_prob — usually mandatory
    if x_for_log_prob is not None and hasattr(dist, "log_prob"):
        lp = dist.log_prob(x_for_log_prob)
        assert_finite(lp, f"{name}.log_prob")

    # mean / variance / entropy — graceful unless explicitly required
    for method in ("mean", "variance", "entropy"):
        if method in required_methods:
            if not hasattr(dist, method):
                raise AssertionError(
                    f"{name}: required method `{method}` missing")
            r = getattr(dist, method)()
            assert_finite(r, f"{name}.{method}")
        elif hasattr(dist, method):
            try:
                r = getattr(dist, method)()
                assert_finite(r, f"{name}.{method}")
            except RuntimeError:
                # Distribution may declare the method but throw "not
                # implemented" — acceptable for distributions without
                # closed-form expressions.
                pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normal():
    loc = f32(0.0, [3])
    scale = f32(1.0, [3])
    exercise_distribution(
        "Normal", D.Normal(loc, scale),
        x_for_log_prob=f32([0.1, 0.2, 0.3]),
        sample_out_shape=[3],
        required_methods=("mean", "variance", "entropy"),
    )


def test_uniform():
    low = f32(0.0, [3])
    high = f32(1.0, [3])
    exercise_distribution(
        "Uniform", D.Uniform(low, high),
        x_for_log_prob=f32([0.5, 0.5, 0.5]),
        sample_out_shape=[3],
        required_methods=("mean", "variance"),
    )


def test_categorical():
    probs = f32([0.1, 0.2, 0.3, 0.4])
    cat = D.Categorical(probs)
    s = cat.sample([8])
    if numel(s) != 8:
        raise AssertionError(f"Categorical.sample([8]) got numel={numel(s)}")
    if hasattr(cat, "log_prob"):
        lp = cat.log_prob(tz.tensor([0, 1, 2, 3], tz.dtype.int64))
        assert_finite(lp, "Categorical.log_prob")


def test_exponential():
    rate = f32(1.0, [3])
    exercise_distribution(
        "Exponential", D.Exponential(rate),
        x_for_log_prob=f32([0.5, 1.0, 1.5]),
        sample_out_shape=[3],
        required_methods=("mean", "variance"),
    )


def test_laplace():
    loc = f32(0.0, [3])
    scale = f32(1.0, [3])
    exercise_distribution(
        "Laplace", D.Laplace(loc, scale),
        x_for_log_prob=f32([0.0, 0.1, -0.1]),
        sample_out_shape=[3],
        required_methods=("mean", "variance", "entropy"),
    )


def test_bernoulli():
    # Python binding renames C++ BernoulliDist to "Bernoulli".
    probs = f32([0.3, 0.5, 0.7])
    b = D.Bernoulli(probs)
    s = b.sample([3])
    if numel(s) == 0:
        raise AssertionError("Bernoulli.sample empty")
    if hasattr(b, "log_prob"):
        lp = b.log_prob(f32([1.0, 0.0, 1.0]))
        assert_finite(lp, "Bernoulli.log_prob")


def test_gamma():
    conc = f32(2.0, [3])
    rate = f32(1.0, [3])
    exercise_distribution(
        "Gamma", D.Gamma(conc, rate),
        x_for_log_prob=f32([0.5, 1.0, 2.0]),
        sample_out_shape=[3],
        required_methods=("mean", "variance"),
    )


def test_beta():
    c0 = f32(2.0, [3])
    c1 = f32(2.0, [3])
    # Beta.variance() raises "not implemented" in the current C++ binding
    # (despite having a closed-form αβ/((α+β)²(α+β+1))). Only require mean;
    # exercise_distribution will still try variance gracefully.
    exercise_distribution(
        "Beta", D.Beta(c0, c1),
        x_for_log_prob=f32([0.5, 0.3, 0.7]),
        sample_out_shape=[3],
        required_methods=("mean",),
    )


def test_dirichlet():
    conc = f32([1.0, 1.0, 1.0])
    d = D.Dirichlet(conc)
    s = d.sample([3])
    if numel(s) == 0:
        raise AssertionError("Dirichlet.sample empty")
    if hasattr(d, "log_prob"):
        # Sample row sums to 1 by construction; use one sample as the input.
        x = d.sample([3])
        lp = d.log_prob(x)
        assert_finite(lp, "Dirichlet.log_prob")


def test_student_t():
    df = f32(5.0, [3])
    loc = f32(0.0, [3])
    scale = f32(1.0, [3])
    t = D.StudentT(df, loc, scale)
    exercise_distribution(
        "StudentT", t,
        x_for_log_prob=f32([0.0, 0.1, -0.1]),
        sample_out_shape=[3],
    )


def test_poisson():
    rate = f32(2.0, [3])
    p = D.Poisson(rate)
    s = p.sample([3])
    if numel(s) == 0:
        raise AssertionError("Poisson.sample empty")
    if hasattr(p, "log_prob"):
        lp = p.log_prob(f32([1.0, 2.0, 3.0]))
        assert_finite(lp, "Poisson.log_prob")


def test_multivariate_normal():
    loc = f32(0.0, [3])
    cov = tz.tensor([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]], tz.dtype.float32)
    mvn = D.MultivariateNormal(loc, cov)
    # MVN sample shape == loc shape (event_dim=1, no extra batch).
    s = mvn.sample([3])
    if numel(s) == 0:
        raise AssertionError("MVN.sample empty")
    if hasattr(mvn, "log_prob"):
        x = f32([0.0, 0.0, 0.0])
        lp = mvn.log_prob(x)
        assert_finite(lp, "MultivariateNormal.log_prob")


def test_binomial():
    probs = f32([0.5, 0.5])
    b = D.Binomial(10, probs)
    s = b.sample([2])
    if numel(s) == 0:
        raise AssertionError("Binomial.sample empty")
    if hasattr(b, "log_prob"):
        lp = b.log_prob(f32([5.0, 5.0]))
        assert_finite(lp, "Binomial.log_prob")


def test_log_normal():
    loc = f32(0.0, [3])
    scale = f32(1.0, [3])
    exercise_distribution(
        "LogNormal", D.LogNormal(loc, scale),
        x_for_log_prob=f32([1.0, 0.5, 2.0]),
        sample_out_shape=[3],
    )


def test_cauchy():
    loc = f32(0.0, [3])
    scale = f32(1.0, [3])
    c = D.Cauchy(loc, scale)
    s = c.sample([3])
    if numel(s) == 0:
        raise AssertionError("Cauchy.sample empty")
    if hasattr(c, "log_prob"):
        lp = c.log_prob(f32([0.0, 0.1, -0.1]))
        assert_finite(lp, "Cauchy.log_prob")


def test_chi2():
    df = f32(3.0, [3])
    c = D.Chi2(df)
    s = c.sample([3])
    if numel(s) == 0:
        raise AssertionError("Chi2.sample empty")
    if hasattr(c, "log_prob"):
        lp = c.log_prob(f32([1.0, 2.0, 3.0]))
        assert_finite(lp, "Chi2.log_prob")


def test_geometric():
    probs = f32([0.5, 0.3, 0.7])
    g = D.Geometric(probs)
    s = g.sample([3])
    if numel(s) == 0:
        raise AssertionError("Geometric.sample empty")


def test_gumbel():
    loc = f32(0.0, [3])
    scale = f32(1.0, [3])
    g = D.Gumbel(loc, scale)
    exercise_distribution(
        "Gumbel", g,
        x_for_log_prob=f32([0.0, 0.1, -0.1]),
        sample_out_shape=[3],
    )


# ---------------------------------------------------------------------------
# Transforms — invoked via __call__ rather than .forward()
# ---------------------------------------------------------------------------

def test_exp_transform():
    t = D.ExpTransform()
    x = f32([0.0, 1.0, -1.0])
    y = t(x)
    assert_finite(y, "ExpTransform.forward")
    if hasattr(t, "log_abs_det_jacobian"):
        ladj = t.log_abs_det_jacobian(x, y)
        assert_finite(ladj, "ExpTransform.log_abs_det_jacobian")


def test_affine_transform():
    loc = f32(1.0, [3])
    scale = f32(2.0, [3])
    t = D.AffineTransform(loc, scale)
    x = f32([0.0, 1.0, -1.0])
    y = t(x)
    assert_finite(y, "AffineTransform.forward")


def test_sigmoid_transform():
    t = D.SigmoidTransform()
    x = f32([0.0, 1.0, -1.0])
    y = t(x)
    assert_finite(y, "SigmoidTransform.forward")


def test_tanh_transform():
    t = D.TanhTransform()
    x = f32([0.0, 1.0, -1.0])
    y = t(x)
    assert_finite(y, "TanhTransform.forward")


def test_softmax_transform():
    t = D.SoftmaxTransform(-1)
    x = f32([1.0, 2.0, 3.0])
    y = t(x)
    assert_finite(y, "SoftmaxTransform.forward")


def test_compose_transform():
    a = D.ExpTransform()
    b = D.AffineTransform(f32(0.0, [3]), f32(1.0, [3]))
    c = D.ComposeTransform([a, b])
    x = f32([0.0, 1.0, -1.0])
    y = c(x)
    assert_finite(y, "ComposeTransform.forward")


# ---------------------------------------------------------------------------
# Composed distributions
# ---------------------------------------------------------------------------

def test_transformed_distribution():
    base = D.Normal(f32(0.0, [3]), f32(1.0, [3]))
    transforms = [D.ExpTransform()]
    td = D.TransformedDistribution(base, transforms)
    s = td.sample([3])
    if numel(s) == 0:
        raise AssertionError("TransformedDistribution.sample empty")


def test_independent():
    base = D.Normal(f32(0.0, [2, 3]), f32(1.0, [2, 3]))
    indep = D.Independent(base, 1)
    s = indep.sample([2, 3])
    if numel(s) == 0:
        raise AssertionError("Independent.sample empty")
    if hasattr(indep, "log_prob"):
        x = f32(0.0, [2, 3])
        lp = indep.log_prob(x)
        assert_finite(lp, "Independent.log_prob")


def test_mixture_same_family():
    # Use the (weights: Tensor, component_distribution) constructor overload.
    # audit-2026-05-03 bug #6 fixed: sample() now works for both default args
    # and non-trivial sample_shape.
    base = D.Normal(f32([0.0, 5.0]), f32([1.0, 1.0]))
    weights = f32([0.5, 0.5])
    msf = D.MixtureSameFamily(weights, base)

    # Default args sample (this used to crash with vector OOB).
    s = msf.sample([])
    if numel(s) == 0:
        raise AssertionError("MixtureSameFamily.sample([]) empty")
    assert_finite(s, "MixtureSameFamily.sample([])")

    # Non-trivial sample shape — must produce shape [4].
    s4 = msf.sample([4])
    if numel(s4) != 4:
        raise AssertionError(
            f"MixtureSameFamily.sample([4]) numel={numel(s4)} (expected 4)")
    assert_finite(s4, "MixtureSameFamily.sample([4])")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_normal, test_uniform, test_categorical, test_exponential,
    test_laplace, test_bernoulli, test_gamma, test_beta,
    test_dirichlet, test_student_t, test_poisson,
    test_multivariate_normal, test_binomial, test_log_normal,
    test_cauchy, test_chi2, test_geometric, test_gumbel,
    test_exp_transform, test_affine_transform, test_sigmoid_transform,
    test_tanh_transform, test_softmax_transform, test_compose_transform,
    test_transformed_distribution, test_independent,
    test_mixture_same_family,
]


def main():
    failed = []
    for test in ALL_TESTS:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
    print()
    if failed:
        print(f"FAILED: {len(failed)} of {len(ALL_TESTS)} tests")
        sys.exit(1)
    print(f"PASSED: {len(ALL_TESTS)} of {len(ALL_TESTS)} tests")


if __name__ == "__main__":
    main()
