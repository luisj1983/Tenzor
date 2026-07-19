"""
FINDING 9: NCCL Python binding coverage.

bindings_distributed.cpp's init_process_group() defaults its backend arg to
"nccl" -- a real, Python-exposed backend -- but before this file no Python
test ever constructed a process group with it; tests/python/conftest.py's
`pg` fixture (used by test_collective_ops.py and others) hardcodes
backend="gloo". This mirrors test_collective_ops.py's single-rank pattern
but through the NCCL/RCCL-backed process group (the `pg_nccl` fixture in
conftest.py), with GPU tensors instead of CPU ones. Skips cleanly on a
CPU-only host -- NCCL/RCCL require a CUDA or ROCm GPU.
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")

from conftest import gpu_device  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _init():
    tz.initialize()
    tz.manual_seed(0)


# `pg_nccl` fixture is defined in conftest.py.


def test_init_process_group_nccl_backend_reports_nccl(pg_nccl):
    # DistributedContext::initialize()/get_process_group() must actually
    # route "nccl" to an NCCL-backed ProcessGroup, not silently substitute
    # something else.
    assert tz.distributed.is_initialized()
    assert tz.distributed.get_world_size() == 1
    assert tz.distributed.get_rank() == 0


def test_all_reduce_single_rank_on_gpu(pg_nccl):
    dev = gpu_device()
    t = tz.full([4], 3.0).to(dev)
    pg_nccl.all_reduce(t, tz.distributed.ReduceOp.SUM)
    t_cpu = t.to(tz.Device("cpu"))
    assert float(t_cpu[0].item()) == pytest.approx(3.0)


def test_broadcast_single_rank_on_gpu(pg_nccl):
    dev = gpu_device()
    t = tz.full([8], 5.0).to(dev)
    pg_nccl.broadcast(t, 0)
    t_cpu = t.to(tz.Device("cpu"))
    assert float(t_cpu[0].item()) == pytest.approx(5.0)


def test_reduce_single_rank_on_gpu(pg_nccl):
    dev = gpu_device()
    t = tz.full([4], 2.0).to(dev)
    pg_nccl.reduce(t, 0, tz.distributed.ReduceOp.SUM)
    t_cpu = t.to(tz.Device("cpu"))
    assert float(t_cpu[0].item()) == pytest.approx(2.0)


def test_all_gather_single_rank_on_gpu(pg_nccl):
    dev = gpu_device()
    t = tz.full([3], 5.0).to(dev)
    out = [tz.zeros([3]).to(dev)]
    pg_nccl.all_gather(t, out)
    assert out[0].shape == [3]
    out_cpu = out[0].to(tz.Device("cpu"))
    assert float(out_cpu[0].item()) == pytest.approx(5.0)


def test_reduce_scatter_single_rank_on_gpu(pg_nccl):
    dev = gpu_device()
    inputs = [tz.full([3], 7.0).to(dev)]
    out = tz.zeros([3]).to(dev)
    pg_nccl.reduce_scatter(inputs, out, tz.distributed.ReduceOp.SUM)
    out_cpu = out.to(tz.Device("cpu"))
    assert float(out_cpu[0].item()) == pytest.approx(7.0)


def test_barrier_single_rank(pg_nccl):
    # Must not raise / hang in single-rank mode.
    pg_nccl.barrier()
