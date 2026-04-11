"""
Integration tests for Distributed Data Parallel (DDP).

Tests actual multi-process gradient synchronization, allreduce,
broadcast, and barrier operations using the gloo backend (CPU-only,
no GPU required).

Skipped when multi-process spawning is not available.
"""
import sys
import os
import multiprocessing as mp

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Minimal pytest shim for running without pytest
    class _PytestShim:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(fn):
                    if condition:
                        def skipped(*a, **kw):
                            print(f"  SKIPPED: {reason}")
                        skipped.__name__ = fn.__name__
                        return skipped
                    return fn
                return decorator
        @staticmethod
        def main(args):
            pass
    pytest = _PytestShim()

sys.path.insert(0, "python")
# Also expose the build-tree layout so spawn workers (which re-import
# this file from a clean interpreter) can find tenzor_core.
_BUILD_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "python"))
if os.path.isdir(_BUILD_PY):
    sys.path.insert(0, _BUILD_PY)

import tenzor as tz

tz.initialize()


def _find_free_port():
    """Find a free port for distributed communication."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _can_spawn():
    """Check if multiprocessing spawn is available.

    We use spawn, not fork: tz.initialize() loads multiple backend .so
    files that start helper threads at import time, and fork()ing a
    multi-threaded process deadlocks in the children (they inherit
    mutexes held by threads that don't survive the fork). Spawn starts
    a fresh Python interpreter that re-imports and re-initializes
    Tenzor cleanly in each worker.
    """
    try:
        ctx = mp.get_context("spawn")
        return True
    except ValueError:
        return False


def _has_distributed():
    """Check if distributed module has init_process_group."""
    return (hasattr(tz, "distributed") and
            callable(getattr(tz.distributed, "init_process_group", None)))


skip_no_distributed = pytest.mark.skipif(
    not _has_distributed(),
    reason="tenzor.distributed not available"
)

skip_no_spawn = pytest.mark.skipif(
    not _can_spawn(),
    reason="multiprocessing spawn not available"
)


# ---------------------------------------------------------------------------
# Module-level worker dispatch table
# ---------------------------------------------------------------------------
#
# Spawn-context multiprocessing requires all Process target functions
# to be picklable — closures defined inside pytest test bodies are not,
# so each test registers its worker in _WORKERS by name and the parent
# passes that name through to a module-level dispatcher.

_WORKERS = {}


def _register_worker(fn):
    """Decorator: register *fn* by its __name__ in the module worker table."""
    _WORKERS[fn.__name__] = fn
    return fn


def _module_level_worker(worker_name, rank, world_size, port, result_queue):
    """Module-level trampoline used as the Process target under spawn.

    A fresh Python interpreter starts per worker, so we re-import tenzor
    and re-init the distributed process group from scratch. The actual
    per-test logic lives in _WORKERS, looked up by *worker_name*.
    """
    import sys as _sys
    import os as _os
    try:
        _os.environ["MASTER_ADDR"] = "localhost"
        _os.environ["MASTER_PORT"] = str(port)
        _sys.path.insert(0, "python")
        _sys.path.insert(
            0,
            _os.path.join(_os.path.dirname(__file__), "..", "..", "build", "python"),
        )
        import tenzor as tz_local
        tz_local.initialize()
        tz_local.distributed.init_process_group(
            backend="gloo", rank=rank, world_size=world_size
        )
        try:
            fn = _WORKERS[worker_name]
            result = fn(rank, world_size, tz_local)
            result_queue.put(("ok", rank, result))
        except Exception as e:
            result_queue.put(("error", rank, str(e)))
        finally:
            tz_local.distributed.destroy_process_group()
    except Exception as e:
        result_queue.put(("error", rank, str(e)))


def _run_distributed(worker_name, world_size=2, timeout=30):
    """Run a distributed test with multiple spawn-context processes.

    *worker_name* is the string name of a function in _WORKERS that
    accepts (rank, world_size, tz_local) and returns a JSON-serialisable
    result. We pass strings/ints/Queue across the Process boundary so
    spawn's pickling requirements are satisfied trivially.
    """
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_module_level_worker,
            args=(worker_name, rank, world_size, port, result_queue)
        )
        p.start()
        processes.append(p)

    # Collect results
    results = {}
    for _ in range(world_size):
        try:
            status, rank, data = result_queue.get(timeout=timeout)
            results[rank] = (status, data)
        except Exception:
            pass

    # Wait for processes
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Verify all workers succeeded
    for rank in range(world_size):
        assert rank in results, f"Worker {rank} did not report results"
        status, data = results[rank]
        assert status == "ok", f"Worker {rank} failed: {data}"

    return {rank: data for rank, (_, data) in results.items()}


# ============================================================================
# Tests
# ============================================================================

@_register_worker
def _ddp_allreduce_sum_worker(rank, world_size, tz_local):
    t = tz_local.ones([4], dtype=tz_local.dtype.float32, device="cpu")
    tz_local.distributed.all_reduce(t, tz_local.distributed.ReduceOp.SUM)
    values = [float(t[i].item()) for i in range(4)]
    expected = float(world_size)
    for v in values:
        assert abs(v - expected) < 1e-5, f"Expected {expected}, got {v}"
    return values


@skip_no_distributed
@skip_no_spawn
def test_allreduce_sum():
    """All-reduce with SUM: each rank has ones, result should be world_size."""
    _run_distributed("_ddp_allreduce_sum_worker", world_size=2)


@_register_worker
def _ddp_allreduce_avg_worker(rank, world_size, tz_local):
    val = float(rank + 1)
    t = tz_local.full([4], val, dtype=tz_local.dtype.float32, device="cpu")
    tz_local.distributed.all_reduce(t, tz_local.distributed.ReduceOp.AVG)
    expected = sum(r + 1 for r in range(world_size)) / world_size
    result = float(t[0].item())
    assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"
    return result


@skip_no_distributed
@skip_no_spawn
def test_allreduce_avg():
    """All-reduce with AVG: each rank has rank+1, result should be mean."""
    _run_distributed("_ddp_allreduce_avg_worker", world_size=2)


@_register_worker
def _ddp_broadcast_worker(rank, world_size, tz_local):
    if rank == 0:
        t = tz_local.full([4], 42.0, dtype=tz_local.dtype.float32, device="cpu")
    else:
        t = tz_local.zeros([4], dtype=tz_local.dtype.float32, device="cpu")
    tz_local.distributed.broadcast(t, src_rank=0)
    result = float(t[0].item())
    assert abs(result - 42.0) < 1e-5, f"Rank {rank}: expected 42.0, got {result}"
    return result


@skip_no_distributed
@skip_no_spawn
def test_broadcast():
    """Broadcast: rank 0 sends data, all ranks should have same values."""
    _run_distributed("_ddp_broadcast_worker", world_size=2)


@_register_worker
def _ddp_barrier_worker(rank, world_size, tz_local):
    tz_local.distributed.barrier()
    return True


@skip_no_distributed
@skip_no_spawn
def test_barrier():
    """Barrier: verify synchronization completes without deadlock."""
    _run_distributed("_ddp_barrier_worker", world_size=2, timeout=30)


@_register_worker
def _ddp_gradient_sync_worker(rank, world_size, tz_local):
    model = tz_local.nn.Linear(4, 2)
    ddp_model = tz_local.distributed.DistributedDataParallel(
        model, tz_local.distributed.get_process_group())

    if rank == 0:
        x = tz_local.ones([2, 4], dtype=tz_local.dtype.float32, device="cpu")
    else:
        x = tz_local.full([2, 4], 2.0, dtype=tz_local.dtype.float32, device="cpu")

    import tenzor.nn.functional as F
    x_var = tz_local.Variable(x, True)
    out = ddp_model.forward(x_var)
    zero_target = tz_local.Variable(
        tz_local.zeros([int(s) for s in out.shape]), False)
    loss = F.mse_loss(out, zero_target)
    loss.backward()
    ddp_model.synchronize_gradients()

    params = list(model.parameters())
    grad_sum = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.to("cpu").contiguous().reshape([p.grad.numel])
            for i in range(g.numel):
                v = float(g[i].item())
                grad_sum += v * v
    return grad_sum


@skip_no_distributed
@skip_no_spawn
def test_ddp_gradient_sync():
    """DDP gradient synchronization: gradients should be identical across ranks."""
    results = _run_distributed("_ddp_gradient_sync_worker", world_size=2)
    assert abs(results[0] - results[1]) < 1e-4, \
        f"Gradient norms differ: rank0={results[0]}, rank1={results[1]}"


@_register_worker
def _ddp_training_loop_worker(rank, world_size, tz_local):
    model = tz_local.nn.Linear(4, 2)
    ddp_model = tz_local.distributed.DistributedDataParallel(
        model, tz_local.distributed.get_process_group())
    optimizer = tz_local.optim.SGD(model.parameters(), lr=0.01)

    import tenzor.nn.functional as F
    for step in range(5):
        optimizer.zero_grad()
        x = tz_local.randn([2, 4], dtype=tz_local.dtype.float32, device="cpu")
        out = ddp_model.forward(tz_local.Variable(x, True))
        zero_target = tz_local.Variable(
            tz_local.zeros([int(s) for s in out.shape]), False)
        loss = F.mse_loss(out, zero_target)
        loss.backward()
        ddp_model.synchronize_gradients()
        optimizer.step()

    params = list(model.parameters())
    param_sum = 0.0
    for p in params:
        t = p.tensor().to("cpu").contiguous().reshape([p.tensor().numel])
        for i in range(t.numel):
            param_sum += float(t[i].item())
    return param_sum


@skip_no_distributed
@skip_no_spawn
def test_ddp_training_loop():
    """DDP training loop: parameters should stay synchronized across ranks."""
    results = _run_distributed("_ddp_training_loop_worker", world_size=2)
    assert abs(results[0] - results[1]) < 1e-4, \
        f"Parameters diverged: rank0={results[0]}, rank1={results[1]}"


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v", "-x"])
    else:
        # Run tests manually
        test_fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
        passed = 0
        failed = 0
        for fn in test_fns:
            name = fn.__name__
            try:
                print(f"  {name} ...", end=" ", flush=True)
                fn()
                print("PASSED")
                passed += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
