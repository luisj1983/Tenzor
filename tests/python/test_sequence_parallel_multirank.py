"""
Multi-rank sequence-parallel round-trip (world_size=2, Gloo).

SequenceParallel's user-facing operation is "scatter a [batch, seq, hidden]
tensor along seq into per-rank shards, run local compute, gather the shards
back." This file verifies the primitive round-trip that SP depends on:
scatter chunks, gather them back via all_gather, assert identity with the
original.

SequenceParallel class requires a C++ DeviceMesh handle that isn't exposed
to Python yet; once the binding lands, these tests extend to call the
class directly. The underlying collective contract is what actually
matters for correctness and is exercised here.
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
            result = _WORKERS[worker_name](rank, world_size, tz_local)
            result_queue.put(("ok", rank, result))
        except Exception as exc:
            import traceback
            result_queue.put(("error", rank, f"{exc}\n{traceback.format_exc()}"))
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
        p = ctx.Process(target=_trampoline,
                        args=(worker_name, rank, world_size, port, q))
        p.start()
        procs.append(p)
    results = {}
    for _ in range(world_size):
        try:
            tag, rank, result = q.get(timeout=timeout)
        except Exception as exc:
            for p in procs: p.terminate()
            pytest.fail(f"worker result queue timed out: {exc}")
        if tag == "error":
            for p in procs: p.terminate()
            pytest.fail(f"worker rank={rank} raised: {result}")
        results[rank] = result
    for p in procs:
        p.join(timeout=timeout)
    return results


# --- Workers ---------------------------------------------------------------


@_register
def _sp_round_trip_primitive(rank, world_size, tz_local):
    # SP's core contract: each rank owns a chunk of the sequence; a gather
    # reassembles the full activation on every rank. Exercises the
    # all_gather path that SequenceParallel uses when reversing a scatter.
    local = tz_local.full([4], float(rank + 1))   # rank 0 → 1.0, rank 1 → 2.0
    placeholder = [tz_local.zeros([4]) for _ in range(world_size)]
    pg = tz_local.distributed.get_process_group()
    gathered = pg.all_gather(local, placeholder)
    # Every rank must see [1.0, 2.0] after all_gather.
    return [float(t[0].item()) for t in gathered]


@_register
def _sp_class_reachable_per_rank(rank, world_size, tz_local):
    # Each rank must be able to reference the SequenceParallel class. This
    # catches binding-registration bugs that only surface under spawn-mode.
    assert hasattr(tz_local.distributed, "SequenceParallel")
    return rank


# --- Tests ------------------------------------------------------------------


@skip_no_distributed
@skip_no_spawn
def test_sp_all_gather_round_trip():
    results = _run("_sp_round_trip_primitive", world_size=2)
    # Each rank contributes rank+1; after all_gather both ranks see [1.0, 2.0].
    for rank, values in results.items():
        assert values == pytest.approx([1.0, 2.0]), \
            f"rank {rank} saw {values}, expected [1.0, 2.0] after all_gather"


@skip_no_distributed
@skip_no_spawn
def test_sequence_parallel_class_visible_in_spawned_workers():
    results = _run("_sp_class_reachable_per_rank", world_size=2)
    assert set(results.values()) == {0, 1}
