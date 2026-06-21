#!/usr/bin/env python3
"""
Test JIT tracing Python bindings: Tracer, Graph, and trace().
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_tracer_lifecycle():
    """Test Tracer start/is_tracing lifecycle."""
    print("Testing tracer lifecycle...")
    tracer = tz.jit.Tracer()
    assert not tracer.is_tracing(), "Should not be tracing initially"

    tracer.start_trace()
    assert tracer.is_tracing(), "Should be tracing after start_trace"

    # End trace to clean up (clear may not stop tracing)
    dummy_in = tz.Variable(tz.randn([1, 4]))
    dummy_out = tz.Variable(tz.randn([1, 2]))
    tracer.end_trace([dummy_in], [dummy_out])
    assert not tracer.is_tracing(), "Should not be tracing after end_trace"
    print("  tracer lifecycle OK")


def test_trace_linear_model():
    """Trace a Linear model and verify a graph is returned."""
    print("Testing trace linear model...")
    model = tz.nn.Linear(10, 5)
    dummy = tz.Variable(tz.randn([1, 10]))

    graph = tz.jit.trace(model, dummy)
    assert graph is not None, "trace() returned None"
    print(f"  traced graph: {graph.num_nodes()} nodes, {graph.num_values()} values")
    # Graph may have 0 nodes if tracing only captures inputs/outputs
    text = graph.to_string()
    assert isinstance(text, str), "to_string() didn't return string"
    print("  trace linear model OK")


def test_jit_graph_string_repr():
    """Verify graph has a string representation."""
    print("Testing graph string representation...")
    model = tz.nn.Linear(4, 2)
    dummy = tz.Variable(tz.randn([1, 4]))

    graph = tz.jit.trace(model, dummy)
    text = graph.to_string()
    assert isinstance(text, str), "to_string() didn't return string"
    # Also test __repr__
    repr_text = repr(graph)
    assert isinstance(repr_text, str)
    print(f"  graph text length: {len(text)} chars")
    print("  graph string repr OK")


def test_jit_graph_execution():
    """Trace a model and attempt graph execution."""
    print("Testing graph execution...")
    model = tz.nn.Linear(8, 3)
    dummy = tz.Variable(tz.randn([2, 8]))

    graph = tz.jit.trace(model, dummy)

    # Execute with new input — may return empty if graph has no nodes.
    # A raised exception here is a real failure: let it propagate.
    # Graph.forward takes Variables (its C++ signature is
    # forward(std::vector<Variable>)), consistent with the rest of the JIT API.
    new_input = tz.Variable(tz.randn([2, 8]))
    outputs = graph.forward([new_input])
    if len(outputs) > 0:
        print(f"  got {len(outputs)} outputs, shape: {outputs[0].shape}")
    else:
        print("  graph returned empty (no traced ops)")
    print("  graph execution OK")


def test_jit_compiler():
    """Test Compiler creation and optimization pass."""
    print("Testing JIT compiler...")
    compiler = tz.jit.Compiler(enable_default_passes=True)

    model = tz.nn.Linear(6, 4)
    dummy = tz.Variable(tz.randn([1, 6]))
    graph = tz.jit.trace(model, dummy)

    original_nodes = graph.num_nodes()
    # Compiler.optimize mutates `graph` in place (its C++ signature is
    # optimize(Graph&, int) -> int) and returns the number of optimizations
    # applied. A raised exception is a real failure: let it propagate.
    num_opts = compiler.optimize(graph, max_iterations=5)
    assert isinstance(num_opts, int), "optimize() should return an int count"
    assert num_opts >= 0, "optimization count must be non-negative"
    print(f"  nodes before: {original_nodes}, after: {graph.num_nodes()}, "
          f"optimizations applied: {num_opts}")
    print("  JIT compiler OK")


def test_jit_optype_enum():
    """Test OpType enum values are accessible."""
    print("Testing OpType enum...")
    assert tz.jit.OpType.Add is not None
    assert tz.jit.OpType.MatMul is not None
    assert tz.jit.OpType.ReLU is not None
    assert tz.jit.OpType.Linear is not None
    assert tz.jit.OpType.Input is not None
    assert tz.jit.OpType.Output is not None
    print("  OpType enum OK")


def main():
    print("=" * 60)
    print("Testing JIT Tracing Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        test_tracer_lifecycle()
        test_trace_linear_model()
        test_jit_graph_string_repr()
        test_jit_graph_execution()
        test_jit_compiler()
        test_jit_optype_enum()

        print("\n" + "=" * 60)
        print("All JIT tracing tests PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
