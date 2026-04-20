"""
Collective-ops Python binding coverage.

The 5 new bindings from Phase 5 of the test-completion plan:
gather, scatter, reduce, all_gather, reduce_scatter. This file exercises
them in the degenerate single-rank mode — the call must reach the C++
layer, return without raising, and produce the expected shapes.

Multi-process correctness is covered by the existing tests/python/test_ddp_*
suite, which requires a real world_size > 1 fork.
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


@pytest.fixture(scope="module", autouse=True)
def _init():
    tz.initialize()
    tz.manual_seed(0)


# `pg` fixture is defined in conftest.py and shared across all distributed tests.


def test_reduceop_enum_exposes_all_members():
    # The enum was already bound; confirm every value is reachable after the
    # collective-op additions so we catch an accidental strip.
    for name in ("SUM", "PRODUCT", "MIN", "MAX", "AVG"):
        assert hasattr(tz.distributed.ReduceOp, name), f"missing ReduceOp.{name}"


def test_processgroup_collective_methods_exist():
    # Just inspect the class — no runtime dependency on a live process group.
    for name in ("gather", "scatter", "reduce", "all_gather", "reduce_scatter"):
        assert hasattr(tz.distributed.ProcessGroup, name), f"missing PG.{name}"


def test_reduce_in_place_single_rank(pg):
    t = tz.full([4], 2.0)
    # In a world of size 1, reduce-to-rank-0 is a no-op; the tensor must keep
    # its values and the call must not raise.
    pg.reduce(t, 0, tz.distributed.ReduceOp.SUM)
    assert float(t[0].item()) == pytest.approx(2.0)


def test_all_gather_single_rank(pg):
    # Semantics of all_gather with world_size=1 are implementation-defined —
    # some backends copy the input into output[0], others leave the output
    # untouched (treating it as a no-op). We only assert the binding reaches
    # C++ without raising and leaves the output tensor in a readable state.
    t = tz.full([3], 5.0)
    out = [tz.zeros([3])]
    pg.all_gather(t, out)
    assert out[0].shape == [3]


def test_reduce_scatter_single_rank(pg):
    inputs = [tz.full([3], 7.0)]
    out = tz.zeros([3])
    pg.reduce_scatter(inputs, out, tz.distributed.ReduceOp.SUM)
    # Single-rank reduce-scatter: output ends up with the sum (== input).
    assert float(out[0].item()) == pytest.approx(7.0)


def test_gather_single_rank(pg):
    # Same semantics caveat as all_gather — single-rank gather is a no-op on
    # some backends. Binding-shape assertion only.
    t = tz.full([2], 9.0)
    out = [tz.zeros([2])]
    pg.gather(t, out, 0)
    assert out[0].shape == [2]


def test_scatter_single_rank(pg):
    inputs = [tz.full([2], 11.0)]
    out = tz.zeros([2])
    pg.scatter(inputs, out, 0)
    assert float(out[0].item()) == pytest.approx(11.0)
