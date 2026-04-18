"""
Multi-rank pipeline-parallel test (world_size=2, Gloo).

PipelineStage wraps a module with stage-id + num-stages metadata. Each
rank holds a different stage. This test verifies:

 - A PipelineStage can be constructed on each rank with its own module.
 - Running the stage's local forward pass produces the expected output
   shape at the end of the pipeline.

Full inter-stage send/recv + micro-batch scheduling lives in the
GPipe / 1F1B schedulers in C++; those aren't bound to Python yet and
are covered by the C++ integration tests in tests/distributed/.
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


# --- Workers ----------------------------------------------------------------


@_register
def _two_stage_pipeline_forward(rank, world_size, tz_local):
    # Each rank owns a different sub-module:
    #   rank 0: Linear(8 -> 16)    (stage 0, first stage)
    #   rank 1: Linear(16 -> 4)    (stage 1, last stage)
    # In the absence of the send/recv scheduler on the Python side, each
    # rank runs its own forward on a rank-specific input and asserts shape.
    if rank == 0:
        mod = tz_local.nn.Linear(8, 16)
        stage = tz_local.distributed.PipelineStage(mod, stage_id=0, num_stages=2)
        x = tz_local.Variable(tz_local.randn([2, 8]), False)
        out = stage.forward(x)
        return list(out.shape)
    else:
        mod = tz_local.nn.Linear(16, 4)
        stage = tz_local.distributed.PipelineStage(mod, stage_id=1, num_stages=2)
        x = tz_local.Variable(tz_local.randn([2, 16]), False)
        out = stage.forward(x)
        return list(out.shape)


@_register
def _pipeline_stage_metadata_matches_construction(rank, world_size, tz_local):
    mod = tz_local.nn.Linear(4, 4)
    stage = tz_local.distributed.PipelineStage(mod, stage_id=rank, num_stages=world_size)
    # The binding doesn't yet expose stage_id / num_stages readers, so the
    # claim here is binding-level only: construction succeeds on every rank
    # with its own stage id.
    return rank


# --- Tests ------------------------------------------------------------------


@skip_no_distributed
@skip_no_spawn
def test_two_stage_pipeline_forward_shapes():
    results = _run("_two_stage_pipeline_forward", world_size=2)
    assert results[0] == [2, 16], f"stage 0 forward shape: {results[0]}"
    assert results[1] == [2, 4], f"stage 1 forward shape: {results[1]}"


@skip_no_distributed
@skip_no_spawn
def test_pipeline_stage_constructs_per_rank():
    results = _run("_pipeline_stage_metadata_matches_construction", world_size=2)
    assert set(results.values()) == {0, 1}
