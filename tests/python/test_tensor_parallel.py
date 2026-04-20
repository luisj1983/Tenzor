"""
Python binding coverage for tensor-parallel layers.

ColumnParallelLinear, RowParallelLinear and ParallelAttention require a
ProcessGroup. Single-rank tests verify construction + metadata. End-to-end
forward/backward parity against an un-sharded equivalent requires a real
multi-rank launcher and lives in the distributed integration suite.
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


def test_column_parallel_linear_constructs(pg):
    layer = tz.distributed.ColumnParallelLinear(
        in_features=16, out_features=32, process_group=pg,
        bias=True, gather_output=True,
    )
    assert layer.in_features == 16
    assert layer.out_features == 32
    # world_size = 1 → local_out_features == out_features
    assert layer.local_out_features == 32


def test_row_parallel_linear_constructs(pg):
    layer = tz.distributed.RowParallelLinear(
        in_features=32, out_features=8, process_group=pg,
        bias=True, input_is_parallel=False,
    )
    # Only asserting construction — the class exposes fewer Python
    # properties than the column variant. A full forward test requires
    # sharded inputs across ≥2 ranks.
    assert layer is not None


def test_parallel_attention_constructs(pg):
    attn = tz.distributed.ParallelAttention(
        embed_dim=64, num_heads=8, process_group=pg,
    )
    assert attn is not None


def _flat(t):
    """Return list of elements from a 2D tensor via item() accessors."""
    rows, cols = t.shape
    return [float(t[i][j].item()) for i in range(rows) for j in range(cols)]


def test_column_parallel_forward_shape_matches_linear(pg):
    # In world_size=1 the column-parallel layer should produce the same
    # output shape (and, with matched weights, same values) as a regular
    # Linear. We only assert shape match here — numerical equivalence
    # requires weight-copying across the internal ParameterShard, which
    # depends on internal storage layout we don't want to couple to.
    layer = tz.distributed.ColumnParallelLinear(
        in_features=8, out_features=4, process_group=pg,
        bias=True, gather_output=True,
    )
    # Construct an input as a Variable, run forward, check shape.
    x = tz.Variable(tz.randn([1, 8]), False)
    y = layer.forward(x)
    assert y.shape == [1, 4]
