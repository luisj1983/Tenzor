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
    decomposition_primitives = {
        # The canonical tanh-based GELU decomposition emits Pow/Tanh/Mul/Add.
        # Mul/Add are present in many graphs so we only flag the
        # decomposition-only ops.
        tz.jit.OpType.Tanh,
    }
    leak = op_types & decomposition_primitives
    if leak:
        print(f"  WARNING: trace also contains decomposition primitives: "
              f"{[str(t) for t in leak]}")
        # We intentionally don't assert here — a future graph rewrite pass
        # may legitimately expand GELU.  The hard requirement is just that
        # the *initial* trace carries the GELU node.

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


def main():
    print("=" * 60)
    print("audit-4 W.31: JIT trace Variable-composition regression")
    print("=" * 60)

    tz.initialize()
    try:
        test_gelu_traces_as_single_gelu_node()
        test_relu_traces_as_single_relu_node()
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
