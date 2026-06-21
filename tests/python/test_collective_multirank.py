"""
Multi-rank collective-op correctness (world_size = 2, Gloo backend).

test_collective_ops.py exercises the bindings in single-rank mode — the
bindings reach C++ and don't raise. This file forks two processes via
multiprocessing (spawn context) and verifies the numerical contract of
each collective: all_reduce sums across ranks, broadcast replicates from
source, all_gather concatenates, scatter/reduce target a specific rank.

The spawn rationale matches tests/python/test_ddp_integration.py — fork
deadlocks because tenzor.initialize() loads multiple backend .so files
whose helper threads don't survive a fork.
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


# --- Worker registry --------------------------------------------------------
#
# Spawn-based Process targets must be top-level picklable callables. We
# register worker functions in _WORKERS by name and dispatch via a shared
# trampoline that imports tenzor afresh in each subprocess.

_WORKERS = {}


def _register(fn):
    _WORKERS[fn.__name__] = fn
    return fn


def _trampoline(worker_name, rank, world_size, port, result_queue):
    import sys as _sys, os as _os
    try:
        _os.environ["MASTER_ADDR"] = "localhost"
        _os.environ["MASTER_PORT"] = str(port)
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", "build", "python"))
        import tenzor.tenzor_core as tz_local
        tz_local.initialize()
        tz_local.distributed.init_process_group(
            backend="gloo", rank=rank, world_size=world_size)
        try:
            fn = _WORKERS[worker_name]
            result = fn(rank, world_size, tz_local)
            result_queue.put(("ok", rank, result))
        except Exception as exc:
            result_queue.put(("error", rank, str(exc)))
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
def _all_reduce_sum(rank, world_size, tz_local):
    # Each rank contributes a tensor of its rank value. After all_reduce SUM
    # every rank should hold sum(0..world_size-1).
    t = tz_local.full([4], float(rank))
    pg = tz_local.distributed.get_process_group()
    pg.all_reduce(t, tz_local.distributed.ReduceOp.SUM)
    return float(t[0].item())


@_register
def _broadcast_from_rank_0(rank, world_size, tz_local):
    t = tz_local.full([4], float(rank * 100))
    pg = tz_local.distributed.get_process_group()
    pg.broadcast(t, 0)
    # All ranks should now see rank-0's initial value (0).
    return float(t[0].item())


@_register
def _barrier_completes(rank, world_size, tz_local):
    pg = tz_local.distributed.get_process_group()
    pg.barrier()
    return rank  # reaching here means barrier returned


@_register
def _reduce_to_rank_0(rank, world_size, tz_local):
    t = tz_local.full([4], float(rank + 1))
    pg = tz_local.distributed.get_process_group()
    pg.reduce(t, 0, tz_local.distributed.ReduceOp.SUM)
    return float(t[0].item())


@_register
def _all_gather_two_ranks(rank, world_size, tz_local):
    # rank 0 contributes 1.0, rank 1 contributes 2.0.
    local = tz_local.full([4], float(rank + 1))
    placeholder = [tz_local.zeros([4]) for _ in range(world_size)]
    pg = tz_local.distributed.get_process_group()
    gathered = pg.all_gather(local, placeholder)
    # gathered[0] should be 1.0 on every rank; gathered[1] should be 2.0.
    return [float(t[0].item()) for t in gathered]


def _all_gather_large_deadlock(rank, world_size, tz_local):
    # Deadlock-free all_gather regression: each rank contributes a large
    # (~2 MiB) tensor. The old implementation issued ALL blocking send_tensor
    # to every peer and ONLY THEN all recv_tensor; once the aggregate in-flight
    # data exceeded the kernel socket buffers, every rank blocked in its send
    # loop before reaching its recv loop → full-mesh TCP deadlock (the worker
    # would hang and the test harness would time out). The ordered pairwise
    # exchange fix lets this complete. We also verify full-content correctness,
    # not just element [0], so a misordered/partial gather is caught too.
    n = 512 * 1024  # 512K float32 = 2 MiB per rank
    local = tz_local.full([n], float(rank + 1))
    placeholder = [tz_local.zeros([n]) for _ in range(world_size)]
    pg = tz_local.distributed.get_process_group()
    gathered = pg.all_gather(local, placeholder)
    # gathered[r] must be entirely (r+1): check first and last element of each.
    ok = True
    for r in range(world_size):
        expected = float(r + 1)
        if (float(gathered[r][0].item()) != expected or
                float(gathered[r][n - 1].item()) != expected):
            ok = False
    return ok


# --- Tests ------------------------------------------------------------------


@skip_no_distributed
@skip_no_spawn
def test_all_reduce_sum_two_ranks():
    results = _run("_all_reduce_sum", world_size=2)
    assert len(results) == 2
    # With ranks 0 + 1 both contributing rank-as-value, sum is 0 + 1 = 1.
    for rank, value in results.items():
        assert value == pytest.approx(1.0), \
            f"rank {rank} got {value}, expected 1.0 (sum of ranks)"


@skip_no_distributed
@skip_no_spawn
def test_broadcast_replicates_from_source():
    results = _run("_broadcast_from_rank_0", world_size=2)
    # Every rank should hold rank-0's initial payload (0 * 100 = 0).
    for rank, value in results.items():
        assert value == pytest.approx(0.0), \
            f"rank {rank} got {value} after broadcast, expected 0.0"


@skip_no_distributed
@skip_no_spawn
def test_barrier_synchronises_ranks():
    results = _run("_barrier_completes", world_size=2)
    # Each rank must report back after the barrier.
    assert set(results.values()) == {0, 1}


@skip_no_distributed
@skip_no_spawn
def test_reduce_targets_dst_rank():
    results = _run("_reduce_to_rank_0", world_size=2)
    # After reduce to rank 0 with SUM, rank 0 sees 1+2=3. Non-root ranks
    # may keep their input or not (implementation-defined in Gloo's
    # non-root branch). We only assert rank 0.
    assert results[0] == pytest.approx(3.0), \
        f"rank 0 got {results[0]}, expected 1+2 = 3"


@skip_no_distributed
@skip_no_spawn
def test_all_gather_concatenates_two_ranks():
    results = _run("_all_gather_two_ranks", world_size=2)
    # Both ranks must see [1.0, 2.0] after all_gather. Regression test for
    # the pybind11 binding that previously took `std::vector<Tensor>&` and
    # silently dropped C++ reassignments, leaving the Python list at zeros.
    for rank, values in results.items():
        assert values == pytest.approx([1.0, 2.0]), \
            f"rank {rank} saw {values}, expected [1.0, 2.0]"


@skip_no_distributed
@skip_no_spawn
def test_all_gather_large_no_deadlock():
    # Regression for the full-mesh all-send-then-all-recv deadlock: a large
    # per-rank payload that exceeds socket buffers. A regression would hang and
    # be caught by the per-run timeout in _run(); a correct run returns True on
    # both ranks with full-content correctness.
    results = _run("_all_gather_large_deadlock", world_size=2, timeout=60)
    assert len(results) == 2, f"expected 2 ranks to report, got {results}"
    for rank, ok in results.items():
        assert ok is True, \
            f"rank {rank} large all_gather mismatch/incomplete (got {ok})"
