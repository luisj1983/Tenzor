"""
Python binding coverage for pipeline-parallel components.

The full pipeline-parallel scheduler (GPipe / 1F1B) requires a task-graph
composed in C++; at the Python layer we only surface the PipelineStage
class so callers can hold a handle. This test verifies the class is
importable and reachable through tz.distributed.
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


def test_pipeline_stage_class_exists():
    assert hasattr(tz.distributed, "PipelineStage")
    assert isinstance(tz.distributed.PipelineStage, type)


def test_pipeline_stage_module_accessible_from_distributed():
    # Canonical resolution: tz.distributed.PipelineStage must work alongside
    # tz.distributed.get_process_group() in the same import namespace.
    # Multi-stage execution requires a scheduler beyond this scope.
    assert tz.distributed.PipelineStage.__module__.endswith("distributed")
