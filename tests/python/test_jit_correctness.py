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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
