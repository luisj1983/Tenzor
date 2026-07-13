"""Extended tests for JIT tracing and compilation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


def _as_tensor(x):
    return x.tensor() if hasattr(x, 'tensor') else x


def _allclose(a, b, atol=1e-4, rtol=1e-4):
    """allclose that accepts Tensor or Variable (mirrors test_jit_trace.py)."""
    a_t = _as_tensor(a).to('cpu').to(tz.dtype.float32)
    b_t = _as_tensor(b).to('cpu').to(tz.dtype.float32)
    if list(a_t.shape) != list(b_t.shape):
        return False
    diff = (a_t - b_t).abs()
    max_diff = float(diff.max().item())
    max_b = float(b_t.abs().max().item())
    return max_diff <= atol + rtol * max_b


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

def test_tracer_exists():
    assert hasattr(tz, 'jit') or hasattr(tz.nn, 'jit')


def test_trace_linear():
    """Trace a simple linear layer — check that trace builds a Graph."""
    linear = tz.nn.Linear(4, 2)
    x = tz.Variable(tz.randn([1, 4]), False)
    traced = tz.jit.trace(linear, x)
    # Tracer returns a jit.Graph — not directly callable. Exercise its
    # forward() entry point (evaluates the traced op graph) instead.
    assert traced.num_nodes() > 0
    assert traced.num_values() > 0


def test_trace_sequential():
    """Trace a sequential model."""
    model = tz.nn.Sequential(
        tz.nn.Linear(4, 8),
        tz.nn.ReLU(),
        tz.nn.Linear(8, 2),
    )
    x = tz.Variable(tz.randn([1, 4]), False)
    traced = tz.jit.trace(model, x)
    assert traced.num_nodes() > 0


# ---------------------------------------------------------------------------
# Compiled model execution
# ---------------------------------------------------------------------------

def test_compiled_matches_eager():
    """Compiled model must produce the SAME VALUES as eager, not just a
    plausible-looking topology. Graph.forward(list[Variable]) -> list[Tensor]
    is the established replay entry point (see test_jit_tracing.py's
    test_jit_graph_execution and test_jit_trace.py's
    test_traced_linear_matches_eager)."""
    tz.manual_seed(0)
    linear = tz.nn.Linear(4, 2)
    linear.eval()
    x = tz.Variable(tz.randn([2, 4]), False)
    eager_out = linear(x)

    traced = tz.jit.trace(linear, x)
    assert traced.num_nodes() > 0
    assert eager_out.shape == [2, 2]

    out_list = traced.forward([x])
    assert len(out_list) > 0, "traced graph produced no outputs on replay"
    assert _allclose(out_list[0], eager_out), (
        "Traced graph output diverged numerically from eager output — "
        "topology looked fine but forward() replay is wrong"
    )


# ---------------------------------------------------------------------------
# Graph serialization
# ---------------------------------------------------------------------------

def test_graph_save_load(tmp_path):
    """Save and load a traced model via save_graph/load_graph, and confirm
    the deserialized graph reproduces correct VALUES on forward(), not just
    a matching node count (a corrupted-but-same-size round-trip would pass
    the old node-count-only check)."""
    tz.manual_seed(0)
    linear = tz.nn.Linear(4, 2)
    linear.eval()
    x = tz.Variable(tz.randn([1, 4]), False)
    eager_out = linear(x)

    traced = tz.jit.trace(linear, x)
    path = str(tmp_path / "model.tz")
    tz.jit.save_graph(traced, path)
    loaded = tz.jit.load_graph(path)
    assert loaded.num_nodes() == traced.num_nodes()

    out_list = loaded.forward([x])
    assert len(out_list) > 0, "loaded graph produced no outputs on replay"
    assert _allclose(out_list[0], eager_out), (
        "Loaded graph diverged numerically from eager output — save/load "
        "round-trip corrupted the graph despite matching node count"
    )


# ---------------------------------------------------------------------------
# Cross-device coverage (JIT-R030)
#
# Every other test in the Python-facing JIT suite constructs tensors on the
# implicit default device (CPU), so @tz.jit's target="auto" resolution (which
# picks the compile target from the INPUT tensor's device) never got
# exercised from Python for a non-CPU device. This mirrors the C++
# AllBackends parity pattern (tests/backend_parity/parity_test_utils.hpp) via
# the shared `device` fixture from tests/python/conftest.py — the same
# ALL_DEVICES / indirect-parametrize convention used throughout
# tests/python/ (e.g. test_multibackend_ops.py) — which cleanly skips any
# device whose backend isn't compiled in / available on this host.
# ---------------------------------------------------------------------------

# JIT-R044: import the single source of truth from conftest.py rather than
# duplicating the device list (a prior verbatim duplication across 5 files
# was how the MPS omission propagated undetected).
from conftest import ALL_DEVICES


@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
def test_jit_matches_eager_on_device(device):
    """@tz.jit output must match eager within tolerance on every device this
    build actually has a backend for — not just CPU. Uses target="auto" (the
    @tz.jit default), which is exactly the resolution path JIT-R030 flagged
    as having zero non-CPU Python coverage.

    OneAPI and MPS have no IREE HAL target at all (documented, intentional
    capability limit — see compile.cpp's mlir_invoke C1 branch), so @tz.jit's
    default strict=True is REQUIRED to raise there rather than silently
    degrade to eager (the same "JIT coverage gaps throw loudly" contract
    CUDA/ROCm lowering gaps get). This test verifies that documented
    RuntimeError on those two devices instead of asserting output equality,
    and separately verifies the fallback_to_eager=True escape hatch actually
    produces correct output there — closing JIT-R021/R022's C++-side OneAPI/
    MPS coverage gap from the Python side too, on real hardware."""
    def eager_fn(x):
        return x * 2.0 + 1.0

    x = tz.randn([4, 8], device=device)
    eager_out = eager_fn(x)

    if device in ("oneapi", "mps"):
        @tz.jit
        def jit_fn_strict(x):
            return x * 2.0 + 1.0

        with pytest.raises(RuntimeError, match="no IREE target backend"):
            jit_fn_strict(x)

        @tz.jit(fallback_to_eager=True)
        def jit_fn_fallback(x):
            return x * 2.0 + 1.0

        jit_out = jit_fn_fallback(x)
        assert _allclose(jit_out, eager_out), (
            f"@tz.jit(fallback_to_eager=True) output diverged from eager on "
            f"device={device!r}"
        )
        return

    @tz.jit
    def jit_fn(x):
        return x * 2.0 + 1.0

    # JIT-R129: a compiled-vs-eager numeric check is vacuous if the compiled
    # call silently fell back to eager -- mirrors the C++ suite's
    # assert_jit_used() anti-vacuity pattern (mlir_target_util.hpp), which
    # this Python-side test never had.
    tz.jit.reset_cache_stats()
    jit_out = jit_fn(x)
    stats = tz.jit.cache_stats()

    assert stats["misses"] >= 1, (
        f"@tz.jit did NOT run through IREE on device={device!r} (silent "
        "eager fallback makes the check below vacuous)"
    )
    assert stats["eager_fallbacks"] == 0, (
        f"@tz.jit silently fell back to eager on device={device!r} "
        "(compiled path never actually ran; the diff check below is vacuous)"
    )

    assert _allclose(jit_out, eager_out), (
        f"@tz.jit output diverged from eager on device={device!r} "
        "(target=\"auto\" resolution from the input tensor's device)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
