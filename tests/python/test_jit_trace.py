#!/usr/bin/env python3
"""
audit-4 W.31: trace/profiler regression — confirm tenzor.jit.trace captures
high-level autograd ops (Gelu) at Variable composition granularity rather
than lowering them into their primitive Tensor decomposition.

Before this regression test landed, a refactor of the GELU autograd
backward to build the result via raw-tensor arithmetic (instead of
Variable-level composition over the Gelu Function) would silently change
the traced graph from a single GELU node to its Mul/Add/Tanh
decomposition. The trace would still produce a numerically correct
forward but optimisation passes that match on OpType::GELU (fused
LayerNorm+GELU patterns, the OneAPI MLP fast path, IREE codegen) would
all miss.
"""
import os
import sys

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def _op_types_in(graph):
    """Return the set of OpType enums present in the traced graph."""
    return {n.op_type for n in graph.nodes()}


def test_gelu_traces_as_single_gelu_node():
    """tz.nn.gelu must trace as one GELU op, not a Mul/Add/Tanh decomposition.

    The Variable-level Gelu Function survives tracing because every dispatch
    path (Eager Function -> JIT Tracer -> Graph) routes through OpType::GELU.
    If the autograd implementation regresses to raw-tensor decomposition,
    the trace contains Mul/Add/Tanh/Pow nodes instead and this test fails.
    """
    print("Testing GELU traces as a single GELU node...")

    x = tz.Variable(tz.randn([2, 8]), requires_grad=True)

    tracer = tz.jit.Tracer()
    tracer.start_trace()
    try:
        y = tz.nn.gelu(x)
    finally:
        graph = tracer.end_trace([x], [y])

    op_types = _op_types_in(graph)
    print(f"  traced op types: {[str(t) for t in op_types]}")

    # The trace must contain OpType.GELU.
    assert tz.jit.OpType.GELU in op_types, (
        "tz.nn.gelu traced without a GELU node — Variable-level autograd "
        "composition has regressed (likely a raw-tensor decomposition crept "
        f"into the forward).  op_types seen: {sorted(str(t) for t in op_types)}"
    )

    # And it must not have lowered to the decomposition (any of these
    # primitives appearing alongside GELU would mean we lost composition).
    # The canonical tanh-based GELU decomposition emits Pow/Tanh/Mul/Add.
    # Mul/Add appear in many legitimate graphs so we only flag the
    # decomposition-only ops (Tanh and Pow) — a fresh trace of nn.gelu
    # alone should never produce them.  If a future graph-rewrite pass
    # legitimately expands GELU into its decomposition downstream, add the
    # pass-specific OpType to the whitelist below with a comment.
    decomposition_primitives = {
        tz.jit.OpType.Tanh,
        tz.jit.OpType.Pow,
    }
    leak = op_types & decomposition_primitives
    assert leak == set(), (
        "tz.nn.gelu trace leaked decomposition primitives "
        f"{[str(t) for t in leak]} — Variable-level autograd composition "
        "has regressed (raw-tensor decomposition crept into the GELU "
        f"forward).  Full op_types seen: {sorted(str(t) for t in op_types)}"
    )

    print("  GELU trace OK")


def test_relu_traces_as_single_relu_node():
    """Sanity check: ReLU also traces as a single node (control for GELU)."""
    print("Testing ReLU traces as a single ReLU node...")
    x = tz.Variable(tz.randn([2, 8]), requires_grad=True)

    tracer = tz.jit.Tracer()
    tracer.start_trace()
    try:
        y = tz.nn.relu(x)
    finally:
        graph = tracer.end_trace([x], [y])

    op_types = _op_types_in(graph)
    print(f"  traced op types: {[str(t) for t in op_types]}")
    assert tz.jit.OpType.ReLU in op_types, (
        f"tz.nn.relu trace missing OpType.ReLU; got "
        f"{sorted(str(t) for t in op_types)}"
    )
    print("  ReLU trace OK")


def test_end_trace_out_of_lifo_order_raises_instead_of_silently_dropping_ops():
    """JIT-R196: ending an OUTER Tracer while an INNER one (started later on
    the same thread) is still active must raise instead of silently popping
    the inner tracer's interceptor and discarding its recorded ops with no
    error at all.
    """
    print("Testing end_trace() LIFO-violation detection...")

    x1 = tz.Variable(tz.randn([2, 8]), requires_grad=False)
    x2 = tz.Variable(tz.randn([2, 8]), requires_grad=False)

    t1 = tz.jit.Tracer()
    t2 = tz.jit.Tracer()
    t1.start_trace()
    t2.start_trace()  # owner stack is now [t1, t2]
    y2 = tz.nn.relu(x2)  # recorded onto t2 (the top of the stack)

    raised = False
    try:
        # t1 is BELOW t2 on the owner stack -- ending it now would silently
        # pop t2's interceptor too, discarding y2's op.
        t1.end_trace([x1], [x1])
    except RuntimeError as e:
        raised = True
        print(f"  correctly raised: {e}")
    assert raised, (
        "Tracer.end_trace() on an outer tracer with a still-active inner "
        "tracer must raise a RuntimeError (JIT-R196), not silently succeed "
        "while discarding the inner tracer's recorded ops"
    )

    # The interceptor stack must still have been cleaned up (both popped) so
    # the process isn't left with a permanently broken/leaked interceptor --
    # start a fresh, unrelated trace and confirm it works normally.
    t3 = tz.jit.Tracer()
    t3.start_trace()
    y3 = tz.nn.relu(x1)
    graph3 = t3.end_trace([x1], [y3])
    assert tz.jit.OpType.ReLU in {n.op_type for n in graph3.nodes()}, (
        "tracer stack was left in a broken state after the LIFO-violation "
        "error -- a fresh trace afterward must still record ops normally"
    )
    print("  end_trace() LIFO-violation detection OK")


def main():
    print("=" * 60)
    print("audit-4 W.31: JIT trace Variable-composition regression")
    print("=" * 60)

    tz.initialize()
    try:
        test_gelu_traces_as_single_gelu_node()
        test_relu_traces_as_single_relu_node()
        test_end_trace_out_of_lifo_order_raises_instead_of_silently_dropping_ops()
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED FAILURE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All W.31 trace regression tests PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
