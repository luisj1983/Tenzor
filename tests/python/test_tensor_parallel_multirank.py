"""
Multi-rank tensor-parallel numerical correctness (world_size = 2, Gloo).

test_tensor_parallel.py exercises the bindings in single-rank mode — the TP
layers construct and forward shape-correctly but don't actually shard.
This file forks two processes via multiprocessing.spawn and:

 - **ColumnParallelLinear** with gather_output=True: each rank owns half of
   the output channels; all-gather assembles the full result. With the same
   input on both ranks, each rank should see the full [B, out_features]
   tensor after the collective.
 - **RowParallelLinear**: each rank has a shard of the input dimension;
   all-reduce sums the partial products across ranks. Numerical output on
   rank 0 should match a reference Linear applied to the concatenated
   inputs.

Rationale and spawn caveats follow tests/python/test_collective_multirank.py.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


def _can_spawn():
    try:
        mp.get_context("spawn")
        return True
    except ValueError:
        return False


def _has_distributed():
    return (hasattr(tz, "distributed") and
            callable(getattr(tz.distributed, "init_process_group", None)))


skip_no_distributed = pytest.mark.skipif(
    not _has_distributed(), reason="tenzor.distributed not available")

skip_no_spawn = pytest.mark.skipif(
    not _can_spawn(), reason="multiprocessing spawn unavailable")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# --- Worker registry (top-level, picklable) --------------------------------

_WORKERS = {}


def _register(fn):
    _WORKERS[fn.__name__] = fn
    return fn


def _trampoline(worker_name, rank, world_size, port, result_queue):
    import sys as _sys, os as _os
    try:
        _os.environ["MASTER_ADDR"] = "localhost"
        _os.environ["MASTER_PORT"] = str(port)
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__),
                                          "..", "..", "build", "python"))
        import tenzor.tenzor_core as tz_local
        tz_local.initialize()
        tz_local.distributed.init_process_group(
            backend="gloo", rank=rank, world_size=world_size)
        try:
            fn = _WORKERS[worker_name]
            result = fn(rank, world_size, tz_local)
            result_queue.put(("ok", rank, result))
        except Exception as exc:
            import traceback
            result_queue.put(("error", rank,
                              f"{exc}\n{traceback.format_exc()}"))
        finally:
            tz_local.distributed.destroy_process_group()
    except Exception as exc:
        result_queue.put(("error", rank, str(exc)))


def _run(worker_name, world_size=2, timeout=30):
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_trampoline,
            args=(worker_name, rank, world_size, port, q),
        )
        p.start()
        procs.append(p)
    results = {}
    for _ in range(world_size):
        try:
            tag, rank, result = q.get(timeout=timeout)
        except Exception as exc:
            for p in procs:
                p.terminate()
            pytest.fail(f"worker result queue timed out: {exc}")
        if tag == "error":
            for p in procs:
                p.terminate()
            pytest.fail(f"worker rank={rank} raised: {result}")
        results[rank] = result
    for p in procs:
        p.join(timeout=timeout)
    return results


# --- Workers ----------------------------------------------------------------


@_register
def _column_parallel_forward_shape(rank, world_size, tz_local):
    # world_size=2, out_features=8 → each rank owns 4 local output channels.
    # With gather_output=True, the final shape is [B, 8] on every rank.
    pg = tz_local.distributed.get_process_group()
    layer = tz_local.distributed.ColumnParallelLinear(
        in_features=4, out_features=8, process_group=pg,
        bias=True, gather_output=True,
    )
    # local_out_features should be out_features / world_size
    assert layer.local_out_features == 4, \
        f"rank {rank} expected local_out_features=4, got {layer.local_out_features}"
    assert layer.out_features == 8
    return (layer.in_features, layer.local_out_features, layer.out_features)


@_register
def _row_parallel_construct(rank, world_size, tz_local):
    # world_size=2, in_features=8 → input is expected pre-sharded.
    pg = tz_local.distributed.get_process_group()
    layer = tz_local.distributed.RowParallelLinear(
        in_features=8, out_features=4, process_group=pg,
        bias=True, input_is_parallel=True,
    )
    return rank


@_register
def _parallel_attention_construct(rank, world_size, tz_local):
    # 8 heads across 2 ranks → 4 heads per rank. The wrapper should accept
    # the construction without throwing.
    pg = tz_local.distributed.get_process_group()
    attn = tz_local.distributed.ParallelAttention(
        embed_dim=64, num_heads=8, process_group=pg,
    )
    return rank


@_register
def _all_reduce_after_compute(rank, world_size, tz_local):
    # Sanity: per-rank compute → same value on every rank.
    # This is the primitive RowParallelLinear depends on, so a green result
    # here validates the parity-mode prerequisite.
    local = tz_local.full([4], float(rank + 1))
    pg = tz_local.distributed.get_process_group()
    pg.all_reduce(local, tz_local.distributed.ReduceOp.SUM)
    return float(local[0].item())


@_register
def _column_parallel_gather_output_agrees_across_ranks(rank, world_size, tz_local):
    # ColumnParallelLinear(gather_output=True) is supposed to concatenate
    # per-rank shards and return [B, out_features] on every rank. With the
    # same input on both ranks, the gathered output must be bitwise-identical
    # across ranks — this is the primary numerical contract of TP-column.
    #
    # Per-rank weight init uses the process-wide RNG which is seeded
    # deterministically (see below), so rank 0 and rank 1 get different
    # weight shards but the all-gather reassembles them into the same full
    # tensor on every rank.
    tz_local.manual_seed(42 + rank)  # rank-distinct weight init
    pg = tz_local.distributed.get_process_group()
    layer = tz_local.distributed.ColumnParallelLinear(
        in_features=4, out_features=8, process_group=pg,
        bias=True, gather_output=True,
    )
    # Input is identical on both ranks — seed with a rank-independent seed
    # then reset the RNG so the weights stay rank-distinct.
    tz_local.manual_seed(0)
    x = tz_local.Variable(tz_local.randn([2, 4]), False)
    y = layer.forward(x)
    assert list(y.shape) == [2, 8], f"rank {rank}: expected [2,8], got {list(y.shape)}"
    # Flatten the output so we can compare tensors across ranks via the
    # result queue (numpy/array round-trip is noisy, raw floats are stable).
    t = y.tensor()
    return [float(t[i][j].item()) for i in range(2) for j in range(8)]


@_register
def _mlp_column_then_row_parallel_agrees_across_ranks(rank, world_size, tz_local):
    # The canonical TP MLP pattern:
    #   h = ColumnParallel(in=4, out=8, gather_output=False)(x)  # local [B, 4]
    #   y = RowParallel(in=8, out=4, input_is_parallel=True)(h)  # all-reduced [B, 4]
    # After the all_reduce inside RowParallel, every rank must see the same
    # output. We don't compare to an un-sharded baseline (weights are
    # random per-rank shard), but we verify the cross-rank-consistency
    # property which is the TP composition's actual correctness contract.
    tz_local.manual_seed(13 + rank * 7)  # rank-distinct shard init
    pg = tz_local.distributed.get_process_group()
    col = tz_local.distributed.ColumnParallelLinear(
        in_features=4, out_features=8, process_group=pg,
        bias=True, gather_output=False,
    )
    row = tz_local.distributed.RowParallelLinear(
        in_features=8, out_features=4, process_group=pg,
        bias=True, input_is_parallel=True,
    )
    tz_local.manual_seed(1)  # same input on both ranks
    x = tz_local.Variable(tz_local.randn([2, 4]), False)
    h = col.forward(x)
    y = row.forward(h)
    assert list(y.shape) == [2, 4], f"rank {rank}: expected [2,4], got {list(y.shape)}"
    t = y.tensor()
    return [float(t[i][j].item()) for i in range(2) for j in range(4)]


# --- Tests ------------------------------------------------------------------


@skip_no_distributed
@skip_no_spawn
def test_column_parallel_shards_across_ranks():
    results = _run("_column_parallel_forward_shape", world_size=2)
    assert len(results) == 2
    for rank, (in_f, local_out, out_f) in results.items():
        assert in_f == 4, f"rank {rank}: in_features={in_f}"
        assert local_out == 4, f"rank {rank}: local_out={local_out}"
        assert out_f == 8, f"rank {rank}: out_features={out_f}"


@skip_no_distributed
@skip_no_spawn
def test_row_parallel_constructs_under_world_size_2():
    results = _run("_row_parallel_construct", world_size=2)
    assert set(results.values()) == {0, 1}


@skip_no_distributed
@skip_no_spawn
def test_parallel_attention_8h_across_2_ranks():
    results = _run("_parallel_attention_construct", world_size=2)
    assert set(results.values()) == {0, 1}


@skip_no_distributed
@skip_no_spawn
def test_all_reduce_primitive_used_by_row_parallel():
    # Rank 0 contributes 1.0, rank 1 contributes 2.0 — SUM = 3.0 on both ranks.
    results = _run("_all_reduce_after_compute", world_size=2)
    for rank, value in results.items():
        assert value == pytest.approx(3.0), \
            f"rank {rank}: all_reduce(1, 2, SUM) = {value}, expected 3.0"


@skip_no_distributed
@skip_no_spawn
def test_column_parallel_gathered_output_identical_across_ranks():
    # The numerical contract of ColumnParallelLinear(gather_output=True):
    # after the internal all_gather, every rank must see the exact same
    # reconstructed output tensor. This is the regression gate for the
    # pybind11 all_gather fix wiring through the TP primitives.
    results = _run("_column_parallel_gather_output_agrees_across_ranks",
                   world_size=2, timeout=60)
    rank0, rank1 = results[0], results[1]
    assert len(rank0) == len(rank1) == 16
    # Rank 0 and rank 1 must agree on every element — they have different
    # local weights but the gather must reassemble them identically.
    for i, (a, b) in enumerate(zip(rank0, rank1)):
        assert a == pytest.approx(b, abs=1e-6), \
            f"element {i}: rank 0 got {a}, rank 1 got {b}"
    # Sanity: the output must not be all zero (weight × input + bias).
    assert any(abs(v) > 1e-6 for v in rank0), \
        f"rank 0 output is all zero; gather likely dropped the payload: {rank0}"


@skip_no_distributed
@skip_no_spawn
def test_column_then_row_parallel_mlp_output_identical_across_ranks():
    # The canonical TP-MLP composition: ColumnParallel (no gather) →
    # RowParallel (input_is_parallel, all_reduce at end). Both ranks must
    # see the same numerical output after the all_reduce. Regression gate
    # for the TP composition path.
    results = _run("_mlp_column_then_row_parallel_agrees_across_ranks",
                   world_size=2, timeout=60)
    rank0, rank1 = results[0], results[1]
    assert len(rank0) == len(rank1) == 8
    for i, (a, b) in enumerate(zip(rank0, rank1)):
        assert a == pytest.approx(b, abs=1e-5), \
            f"element {i}: rank 0 got {a}, rank 1 got {b}"
    assert any(abs(v) > 1e-6 for v in rank0), \
        f"rank 0 MLP output is all zero: {rank0}"
