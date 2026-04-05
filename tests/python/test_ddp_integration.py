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
import tenzor as tz

tz.initialize()


def _find_free_port():
    """Find a free port for distributed communication."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _can_spawn():
    """Check if multiprocessing spawn is available."""
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


def _worker_fn(rank, world_size, port, test_fn, result_queue):
    """Worker process for distributed tests."""
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        # Re-initialize tenzor in worker process
        sys.path.insert(0, "python")
        import tenzor as tz
        tz.initialize()

        tz.distributed.init_process_group(
            backend="gloo", rank=rank, world_size=world_size
        )

        try:
            result = test_fn(rank, world_size)
            result_queue.put(("ok", rank, result))
        except Exception as e:
            result_queue.put(("error", rank, str(e)))
        finally:
            tz.distributed.destroy_process_group()
    except Exception as e:
        result_queue.put(("error", rank, str(e)))


def _run_distributed(test_fn, world_size=2, timeout=30):
    """Run a distributed test with multiple processes."""
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_fn,
            args=(rank, world_size, port, test_fn, result_queue)
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

@skip_no_distributed
@skip_no_spawn
def test_allreduce_sum():
    """All-reduce with SUM: each rank has ones, result should be world_size."""
    def worker(rank, world_size):
        t = tz.ones([4], dtype=tz.dtype.float32, device=tz.device("cpu"))
        tz.distributed.all_reduce(t, tz.distributed.ReduceOp.SUM)
        values = [t.data_ptr_float(i) for i in range(4)]
        expected = float(world_size)
        for v in values:
            assert abs(v - expected) < 1e-5, f"Expected {expected}, got {v}"
        return values

    _run_distributed(worker, world_size=2)


@skip_no_distributed
@skip_no_spawn
def test_allreduce_avg():
    """All-reduce with AVG: each rank has rank+1, result should be mean."""
    def worker(rank, world_size):
        val = float(rank + 1)
        t = tz.full([4], val, dtype=tz.dtype.float32, device=tz.device("cpu"))
        tz.distributed.all_reduce(t, tz.distributed.ReduceOp.AVG)
        expected = sum(r + 1 for r in range(world_size)) / world_size
        result = t.data_ptr_float(0)
        assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"
        return result

    _run_distributed(worker, world_size=2)


@skip_no_distributed
@skip_no_spawn
def test_broadcast():
    """Broadcast: rank 0 sends data, all ranks should have same values."""
    def worker(rank, world_size):
        if rank == 0:
            t = tz.full([4], 42.0, dtype=tz.dtype.float32, device=tz.device("cpu"))
        else:
            t = tz.zeros([4], dtype=tz.dtype.float32, device=tz.device("cpu"))
        tz.distributed.broadcast(t, src=0)
        result = t.data_ptr_float(0)
        assert abs(result - 42.0) < 1e-5, f"Rank {rank}: expected 42.0, got {result}"
        return result

    _run_distributed(worker, world_size=2)


@skip_no_distributed
@skip_no_spawn
def test_barrier():
    """Barrier: verify synchronization completes without deadlock."""
    def worker(rank, world_size):
        tz.distributed.barrier()
        return True

    _run_distributed(worker, world_size=2, timeout=10)


@skip_no_distributed
@skip_no_spawn
def test_ddp_gradient_sync():
    """DDP gradient synchronization: gradients should be identical across ranks."""
    def worker(rank, world_size):
        # Create a simple linear module
        model = tz.nn.Linear(4, 2)
        ddp_model = tz.distributed.DistributedDataParallel(model)

        # Each rank uses different input data
        if rank == 0:
            x = tz.ones([2, 4], dtype=tz.dtype.float32, device=tz.device("cpu"))
        else:
            x = tz.full([2, 4], 2.0, dtype=tz.dtype.float32, device=tz.device("cpu"))

        # Forward + backward
        out = ddp_model.forward(tz.Variable(x, True))
        loss = out.sum()
        loss.backward()

        # Synchronize gradients
        ddp_model.synchronize_gradients()

        # Return gradient norm for verification
        params = list(model.parameters())
        grad_sum = 0.0
        for p in params:
            if p.grad is not None:
                g = p.grad
                for i in range(g.numel()):
                    grad_sum += g.data_ptr_float(i) ** 2
        return grad_sum

    results = _run_distributed(worker, world_size=2)
    # Gradients should be identical across ranks after sync
    assert abs(results[0] - results[1]) < 1e-4, \
        f"Gradient norms differ: rank0={results[0]}, rank1={results[1]}"


@skip_no_distributed
@skip_no_spawn
def test_ddp_training_loop():
    """DDP training loop: parameters should stay synchronized across ranks."""
    def worker(rank, world_size):
        model = tz.nn.Linear(4, 2)
        ddp_model = tz.distributed.DistributedDataParallel(model)
        optimizer = tz.optim.SGD(model.parameters(), lr=0.01)

        for step in range(5):
            optimizer.zero_grad()
            x = tz.randn([2, 4], dtype=tz.dtype.float32, device=tz.device("cpu"))
            out = ddp_model.forward(tz.Variable(x, True))
            loss = out.sum()
            loss.backward()
            ddp_model.synchronize_gradients()
            optimizer.step()

        # Return parameter values for cross-rank comparison
        params = list(model.parameters())
        param_sum = 0.0
        for p in params:
            t = p.tensor()
            for i in range(t.numel()):
                param_sum += t.data_ptr_float(i)
        return param_sum

    results = _run_distributed(worker, world_size=2)
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
