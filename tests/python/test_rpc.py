"""
Integration tests for the RPC framework.

Runs two spawn-context processes. Worker 0 registers a Python callable
under a name; worker 1 calls it via rpc_sync. Tensors flow both ways.
"""
import sys
import os
import multiprocessing as mp

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
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
_BUILD_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "python"))
if os.path.isdir(_BUILD_PY):
    sys.path.insert(0, _BUILD_PY)

import tenzor as tz

tz.initialize()


def _can_spawn():
    try:
        mp.get_context("spawn")
        return True
    except ValueError:
        return False


def _has_rpc():
    return (hasattr(tz, "distributed")
            and hasattr(tz.distributed, "rpc")
            and callable(getattr(tz.distributed.rpc, "init_rpc", None))
            and callable(getattr(tz.distributed.rpc, "register_function", None)))


skip_no_rpc = pytest.mark.skipif(
    not _has_rpc(), reason="tenzor.distributed.rpc not available")
skip_no_spawn = pytest.mark.skipif(
    not _can_spawn(), reason="multiprocessing spawn not available")


# Module-level worker dispatch
_WORKERS = {}


def _register_worker(fn):
    _WORKERS[fn.__name__] = fn
    return fn


def _module_level_worker(worker_name, rank, world_size, result_queue):
    """Fresh-interpreter entry point under spawn. Re-imports tenzor, boots
    RPC, then dispatches to the per-test worker function by name.
    """
    import sys as _sys
    import os as _os
    try:
        _sys.path.insert(0, "python")
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(__file__), "..", "..", "build", "python"))
        import tenzor as tz_local
        tz_local.initialize()
        tz_local.distributed.rpc.init_rpc(
            name=f"worker_{rank}", rank=rank, world_size=world_size)
        try:
            fn = _WORKERS[worker_name]
            result = fn(rank, world_size, tz_local)
            result_queue.put(("ok", rank, result))
        except Exception as e:
            result_queue.put(("error", rank, f"{type(e).__name__}: {e}"))
        finally:
            tz_local.distributed.rpc.shutdown_rpc()
    except Exception as e:
        result_queue.put(("error", rank, f"{type(e).__name__}: {e}"))


def _run_rpc_test(worker_name, world_size=2, timeout=60):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_module_level_worker,
            args=(worker_name, rank, world_size, result_queue))
        p.start()
        processes.append(p)

    results = {}
    for _ in range(world_size):
        try:
            status, rank, data = result_queue.get(timeout=timeout)
            results[rank] = (status, data)
        except Exception:
            pass

    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    for rank in range(world_size):
        assert rank in results, f"Worker {rank} did not report results"
        status, data = results[rank]
        assert status == "ok", f"Worker {rank} failed: {data}"

    return {rank: data for rank, (_, data) in results.items()}


# ==========================================================================
# Worker: each rank registers `double_it`; rank 0 calls rank 1's copy of it.
# ==========================================================================

@_register_worker
def _rpc_has_function_worker(rank, world_size, tz_local):
    assert not tz_local.distributed.rpc.has_function("my_fn")
    tz_local.distributed.rpc.register_function("my_fn", lambda ts: ts)
    assert tz_local.distributed.rpc.has_function("my_fn")
    return {"ok": True}


@skip_no_rpc
@skip_no_spawn
def test_rpc_register_and_has_function():
    """register_function + has_function round-trip on each worker."""
    results = _run_rpc_test("_rpc_has_function_worker", world_size=2)
    for rank, r in results.items():
        assert r["ok"] is True


# ==========================================================================
# Cross-rank rpc_sync round-trip. Rank 0 calls a function on rank 1; rank 1
# serves it. Exercises the full TCP transport: listen/accept/connect, frame
# write/read, response delivery, function-registry lookup.
# ==========================================================================

@_register_worker
def _rpc_double_worker(rank, world_size, tz_local):
    import time

    def double_it(tensors):
        t = tensors[0]
        return [t + t]

    tz_local.distributed.rpc.register_function("double_it", double_it)

    # Brief pause to let both ranks finish registering and let the mesh
    # establish inbound sockets as the first send() dials out.
    time.sleep(0.5)

    if rank == 0:
        x = tz_local.ones([3], dtype=tz_local.dtype.float32, device="cpu")
        result = tz_local.distributed.rpc.rpc_sync(
            dst=1, func_name="double_it", args=[x])
        assert len(result) == 1
        out = result[0]
        values = [float(out[i].item()) for i in range(3)]
        for v in values:
            assert abs(v - 2.0) < 1e-6, f"expected 2.0, got {v}"
        return {"values": values}
    else:
        # Rank 1 stays up long enough to serve the call.
        time.sleep(5.0)
        return {"served": True}


@skip_no_rpc
@skip_no_spawn
def test_rpc_sync_loopback():
    """rpc_sync cross-rank round-trip: rank 0 calls a function on rank 1."""
    results = _run_rpc_test("_rpc_double_worker", world_size=2, timeout=30)
    assert results[0]["values"] == [2.0, 2.0, 2.0]
    assert results[1]["served"] is True


if __name__ == "__main__":
    if HAS_PYTEST:
        sys.exit(pytest.main([__file__, "-v", "-x"]))
    else:
        test_rpc_register_and_has_function()
        test_rpc_sync_loopback()
        print("All RPC tests passed.")
