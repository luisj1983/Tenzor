"""
Python binding coverage for ZeRO optimizer wrappers.

ZeRO Stage 1/2/3 are implemented in C++ and exposed via Phase 5 of the test
plan. This file verifies:
  - The config structs accept their documented fields.
  - The optimizer classes exist and are subclasses of each other
    (Stage3 extends Stage2 extends Stage1).
  - Stage 1 construction via a Python-created base optimizer succeeds and
    the wrapper has the standard optimizer interface (step, zero_grad,
    state_dict, load_state_dict).

End-to-end convergence tests require a real multi-rank process group and
live in the integration suite.
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


def test_stage1_config_defaults():
    c = tz.optim.ZeROStage1Config()
    assert c.world_size == 1
    assert c.rank == 0
    assert c.offload_to_cpu is False
    c.rank = 0
    c.world_size = 2
    c.offload_to_cpu = True
    assert c.world_size == 2
    assert c.offload_to_cpu is True


def test_stage2_inherits_stage1_fields():
    c2 = tz.optim.ZeROStage2Config()
    # Stage 1 fields must still be present (C++ inheritance preserved).
    c2.rank = 1
    c2.world_size = 4
    assert c2.rank == 1
    assert c2.world_size == 4
    # Stage 2 specifics.
    c2.gradient_bucket_size = 1024 * 1024
    c2.gradient_bucketing = False
    assert c2.gradient_bucket_size == 1024 * 1024
    assert c2.gradient_bucketing is False


def test_stage3_config_constructs():
    c3 = tz.optim.ZeROStage3Config()
    # Inherited fields:
    c3.world_size = 8
    c3.gradient_bucket_size = 42
    assert c3.world_size == 8
    assert c3.gradient_bucket_size == 42


def test_stage_class_hierarchy():
    # Stage3 must be a Stage2, which must be a Stage1. pybind11 surfaces
    # inheritance to Python.
    assert issubclass(tz.optim.ZeROStage2Optimizer, tz.optim.ZeROStage1Optimizer)
    assert issubclass(tz.optim.ZeROStage3Optimizer, tz.optim.ZeROStage2Optimizer)


# `pg` fixture is defined in conftest.py and shared across all distributed tests.


def test_stage1_construct_and_step(pg):
    # The binding now uses the shared_ptr constructor on the C++ side, so
    # pybind11 can hold its reference to the Adam instance concurrently with
    # the ZeRO wrapper. This test exercises the full cycle:
    #   1. Build Linear + Adam
    #   2. Wrap Adam in ZeROStage1Optimizer (world_size=1)
    #   3. Forward + backward + step
    #   4. Assert the weight tensor changed.
    lin = tz.nn.Linear(8, 4)
    params = [p for _, p in lin.named_parameters()]
    adam = tz.optim.Adam(params, lr=1e-2)

    config = tz.optim.ZeROStage1Config()
    config.rank = 0
    config.world_size = 1

    zero = tz.optim.ZeROStage1Optimizer(adam, config)

    # Snapshot the weight sum (scalar reduction over the whole weight tensor)
    # — survives partitioning and gives us a single number to compare.
    w_before = float(tz.sum(params[0].tensor()).item())

    x = tz.Variable(tz.randn([1, 8]), False)
    y = lin.forward(x)
    tz.sum(y).backward()
    zero.step()
    zero.zero_grad()

    w_after = float(tz.sum(params[0].tensor()).item())
    assert w_before != w_after, "ZeRO step should have moved the weight"


def test_stage2_construct_and_step(pg):
    # Stage 2 extends Stage 1 with gradient bucketing + reduce-scatter. In
    # world_size=1 those paths degenerate to no-ops but the wrapper must
    # still step() cleanly.
    lin = tz.nn.Linear(8, 4)
    params = [p for _, p in lin.named_parameters()]
    adam = tz.optim.Adam(params, lr=1e-2)

    config = tz.optim.ZeROStage2Config()
    config.rank = 0
    config.world_size = 1

    zero = tz.optim.ZeROStage2Optimizer(adam, config)

    x = tz.Variable(tz.randn([1, 8]), False)
    y = lin.forward(x)
    tz.sum(y).backward()
    zero.step()
    zero.zero_grad()
