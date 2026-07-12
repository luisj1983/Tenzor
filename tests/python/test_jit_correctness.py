"""JIT trace correctness — verify the traced graph reproduces eager output.

The existing test_jit_tracing.py only checks that constructors don't throw
and that `graph.to_string()` returns a string (it even acknowledges the
graph "may have 0 nodes"). This file fills the gap by:
  - Tracing a Linear and a tiny Sequential, then re-running the traced
    graph on a fresh input and asserting the output matches eager mode
    within fp32 tolerance.
  - Verifying that JIT optimization preserves output (optimizer must be
    semantically a no-op).

If JIT tracing returns an empty graph (a known limitation noted in the
original smoke test), the equivalence assertion is skipped with a
SkipTest message — this signals a binding gap rather than a test bug.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


def _as_tensor(x):
    return x.tensor() if hasattr(x, 'tensor') else x


def _allclose(a, b, atol=1e-4, rtol=1e-4):
    """allclose that accepts Tensor or Variable."""
    a_t = _as_tensor(a).to('cpu').to(tz.dtype.float32)
    b_t = _as_tensor(b).to('cpu').to(tz.dtype.float32)
    if list(a_t.shape) != list(b_t.shape):
        return False
    diff = (a_t - b_t).abs()
    max_diff = float(diff.max().item())
    max_b = float(b_t.abs().max().item())
    return max_diff <= atol + rtol * max_b


# ----------------------------------------------------------------------------
# Linear: trace a single Linear and re-run.
# ----------------------------------------------------------------------------

def test_traced_linear_matches_eager():
    tz.manual_seed(0)
    model = tz.nn.Linear(8, 4)
    model.eval()

    sample = tz.Variable(tz.randn([1, 8]), False)
    eager_y = model(sample)

    graph = tz.jit.trace(model, sample)
    assert graph.num_nodes() > 0, (
        "JIT trace returned an empty graph — the dispatch interceptor is not "
        "recording ops. Was TracingGuard wired correctly in bindings_jit.cpp?"
    )

    # Run the traced graph on the SAME input first to verify replay.
    out_list = graph.forward([sample])
    assert len(out_list) >= 1, "Traced graph produced no outputs"
    traced_y = out_list[0]
    assert _allclose(traced_y, eager_y.tensor()), (
        "Traced Linear output diverged from eager Linear output."
    )


def test_traced_sequential_matches_eager():
    tz.manual_seed(0)
    model = tz.nn.Sequential(
        tz.nn.Linear(16, 32),
        tz.nn.ReLU(),
        tz.nn.Linear(32, 4),
    )
    model.eval()
    sample = tz.Variable(tz.randn([2, 16]), False)
    eager_y = model(sample)

    graph = tz.jit.trace(model, sample)
    assert graph.num_nodes() > 0, "JIT trace returned empty graph"

    out_list = graph.forward([sample])
    assert len(out_list) >= 1
    traced_y = out_list[0]
    assert _allclose(traced_y, eager_y.tensor()), (
        "Traced Sequential output diverged from eager Sequential output."
    )


def test_optimized_graph_preserves_output():
    """Compiler.optimize must be a semantic no-op — output must still match."""
    tz.manual_seed(0)
    model = tz.nn.Linear(8, 4)
    model.eval()
    sample = tz.Variable(tz.randn([1, 8]), False)
    eager_y = model(sample)

    graph = tz.jit.trace(model, sample)
    assert graph.num_nodes() > 0, "JIT trace returned empty graph"

    compiler = tz.jit.Compiler(enable_default_passes=True)
    # optimize() mutates `graph` in place and returns the number of
    # passes applied (an int) — not a new Graph object.
    passes_applied = compiler.optimize(graph, max_iterations=5)

    out_list = graph.forward([sample])
    assert len(out_list) > 0, "Optimized graph dropped its output"
    assert _allclose(out_list[0], eager_y), (
        f"Optimized JIT graph (after {passes_applied} passes) diverged from "
        f"eager output."
    )


def test_traced_graph_with_fresh_input():
    """Trace on one input, then run with a different input — must match eager."""
    tz.manual_seed(0)
    model = tz.nn.Linear(8, 4)
    model.eval()

    trace_in = tz.Variable(tz.randn([1, 8]), False)
    fresh_in = tz.randn([1, 8])  # bare tensor; different values

    graph = tz.jit.trace(model, trace_in)
    assert graph.num_nodes() > 0, "JIT trace returned empty graph"

    fresh_var = tz.Variable(fresh_in, False)
    eager_y = model(fresh_var)
    out_list = graph.forward([fresh_var])
    assert len(out_list) > 0, (
        "Traced graph dropped output on fresh input — JIT is specializing on "
        "the trace-time input pointer."
    )
    assert _allclose(out_list[0], eager_y.tensor()), (
        "Traced graph diverged from eager when fed a fresh input — JIT is "
        "specializing on the trace input rather than treating it as a "
        "placeholder."
    )


# ----------------------------------------------------------------------------
# @tz.jit wrapper regressions (JIT-070 family): the decorator must run the
# user's function EXACTLY ONCE per call, keep temp inputs alive for show_*, and
# key static args by value.
# ----------------------------------------------------------------------------

def test_jit_no_double_exec_on_error():
    """A side-effecting fn that raises must run its body ONCE. The MLIR path
    used to re-run fn_ on its eager fallback (and re-run it again in the Python
    wrapper's error hint), double-executing side effects (JIT-070)."""
    calls = {"n": 0}

    @tz.jit
    def bad(x):
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        bad(tz.randn([4, 8]))
    assert calls["n"] == 1, f"fn body ran {calls['n']} times (double-exec)"


def test_jit_no_double_exec_on_success():
    """A successful call must also run the traced fn exactly once, whether it
    compiles or degrades to eager."""
    calls = {"n": 0}

    @tz.jit
    def f(x):
        calls["n"] += 1
        return tz.sum(x, 1)

    y = f(tz.randn([4, 8]))
    assert y is not None
    assert calls["n"] == 1, f"fn body ran {calls['n']} times (double-exec)"


def test_jit_grad_path_runs_once_and_grads_correct():
    """A @tz.jit fn called with requires_grad inputs goes through grad_invoke.
    It must run the traced body EXACTLY ONCE (the grad-replay fallback used to
    re-run fn_ on a cache-miss whose replay failed — JIT-071) and produce grads
    matching eager autograd."""
    import numpy as np
    calls = {"n": 0}

    @tz.jit
    def f(x):
        calls["n"] += 1
        return tz.sum(x * x, 1)

    x = tz.Variable(tz.randn([4, 8]), True)
    y = f(x)
    assert calls["n"] == 1, f"grad-path fn body ran {calls['n']} times"

    y.backward(tz.Variable(tz.ones([4]), False))
    g = x.grad
    gnp = np.asarray(g.tensor().numpy() if hasattr(g, "tensor") else g.numpy())
    expect = 2.0 * np.asarray(x.tensor().numpy())   # d/dx sum(x^2) = 2x
    assert np.allclose(gnp, expect, atol=1e-4), "jit grad diverged from eager 2x"


def test_jit_show_graph_after_temp_input():
    """The natural `f(temp); show_graph(f)` idiom must not raise — the wrapper
    keeps a STRONG ref to the last call's tensor operands."""
    @tz.jit
    def f(x):
        return tz.sum(x, 1)

    _ = f(tz.randn([4, 8]))          # temp input, dropped after the call
    s = tz.jit.show_graph(f)         # must NOT raise
    assert isinstance(s, str)


def test_jit_show_graph_single_specialization_accepts_new_example():
    """R1-07: a function with static args but only ONE real-call
    specialization so far has no ambiguity — show_graph must accept a brand
    new tensor example for it without raising, and _tz_variants must not be
    inflated by the initial (never-called) seed variant."""
    @tz.jit
    def f(x, n):
        return x * n

    x1 = tz.randn([4])
    f(x1, 3)
    assert len(f._tz_variants) == 1, (
        "the seed (pre-first-call) variant must not count toward the "
        "real specialization count")

    x2 = tz.randn([4])
    s = tz.jit.show_graph(f, x2)   # must NOT raise
    assert isinstance(s, str)


def test_jit_show_graph_multi_specialization_explicit_example_raises():
    """R1-07 regression: once a static-argument function has been called
    with TWO OR MORE distinct static-value combinations, show_graph's
    example arguments (tensor-only) cannot express which specialization the
    caller wants — passing an explicit example must raise a clear error
    instead of silently pairing it with whichever specialization happened to
    run most recently."""
    @tz.jit
    def f(x, n):
        return x * n

    x = tz.randn([4])
    f(x, 3)
    f(x, 5)
    assert len(f._tz_variants) == 2

    with pytest.raises(ValueError, match="different variants"):
        tz.jit.show_graph(f, x)

    # No examples (reusing the last real call's stash) must still work —
    # the ambiguity only applies to NEW explicit examples.
    s = tz.jit.show_graph(f)
    assert isinstance(s, str)


def test_jit_static_arg_value_key_and_rejects_nonprimitive():
    """Scalar static args are keyed by value (not id-based repr); a
    non-primitive static arg raises a clear TypeError rather than mis-keying."""
    x = tz.randn([4, 8])

    @tz.jit
    def g(x, dim):
        return tz.sum(x, dim)

    assert _allclose(g(x, 1), tz.sum(x, 1))

    class C:  # default (id-based) __repr__ — a value-key must not use repr
        def __init__(self, v):
            self.v = v

    @tz.jit
    def h(x, obj):
        return x

    with pytest.raises(TypeError, match="not supported"):
        h(x, C(1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
