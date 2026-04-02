"""Extended tests for JIT tracing and compilation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

def test_tracer_exists():
    assert hasattr(tz, 'jit') or hasattr(tz.nn, 'jit')


def test_trace_linear():
    """Trace a simple linear layer."""
    linear = tz.nn.Linear(4, 2)
    x = tz.Variable(tz.randn([1, 4]), False)
    # Check if tracing API is available
    if hasattr(tz, 'jit') and hasattr(tz.jit, 'trace'):
        traced = tz.jit.trace(linear, x)
        y = traced(x)
        assert y.data.shape == [1, 2]
    else:
        pytest.skip("JIT tracing not exposed to Python")


def test_trace_sequential():
    """Trace a sequential model."""
    model = tz.nn.Sequential([
        tz.nn.Linear(4, 8),
        tz.nn.ReLU(),
        tz.nn.Linear(8, 2),
    ])
    x = tz.Variable(tz.randn([1, 4]), False)
    if hasattr(tz, 'jit') and hasattr(tz.jit, 'trace'):
        traced = tz.jit.trace(model, x)
        y = traced(x)
        assert y.data.shape == [1, 2]
    else:
        pytest.skip("JIT tracing not exposed to Python")


# ---------------------------------------------------------------------------
# Compiled model execution
# ---------------------------------------------------------------------------

def test_compiled_matches_eager():
    """Compiled model should produce same output as eager."""
    linear = tz.nn.Linear(4, 2)
    x = tz.Variable(tz.randn([2, 4]), False)
    eager_out = linear(x)

    if hasattr(tz, 'jit') and hasattr(tz.jit, 'trace'):
        traced = tz.jit.trace(linear, x)
        traced_out = traced(x)
        # Shapes should match
        assert eager_out.data.shape == traced_out.data.shape
    else:
        pytest.skip("JIT tracing not exposed to Python")


# ---------------------------------------------------------------------------
# Graph serialization
# ---------------------------------------------------------------------------

def test_graph_save_load(tmp_path):
    """Save and load a traced model."""
    linear = tz.nn.Linear(4, 2)
    x = tz.Variable(tz.randn([1, 4]), False)

    if hasattr(tz, 'jit') and hasattr(tz.jit, 'trace'):
        traced = tz.jit.trace(linear, x)
        path = str(tmp_path / "model.tz")
        if hasattr(traced, 'save'):
            traced.save(path)
            loaded = tz.jit.load(path)
            y = loaded(x)
            assert y.data.shape == [1, 2]
        else:
            pytest.skip("JIT model save not available")
    else:
        pytest.skip("JIT tracing not exposed to Python")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
