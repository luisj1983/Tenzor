"""Type stubs for tenzor.profiler (CC.11).

`tenzor.profiler` is a pybind11 submodule exposed by `tenzor_core`. There
is no Python-level `profiler.py`; signatures below mirror the docstrings
emitted by pybind11 (verified via `help(tz.profiler.<name>)`).
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional


class Phase(Enum):
    """Forward / backward phase tag used to filter recorded profile events."""
    Forward = 0
    Backward = 1


def enable() -> None:
    """Enable profiling for both forward and backward passes."""
    ...


def disable() -> None:
    """Disable profiling."""
    ...


def enable_trace() -> None:
    """Enable trace mode — records per-invocation events suitable for the
    Chrome Trace Event Format export."""
    ...


def is_enabled() -> bool:
    """Whether profiling is currently enabled."""
    ...


def reset() -> None:
    """Clear all recorded profile events."""
    ...


def summary() -> str:
    """Return a human-readable forward + backward summary string."""
    ...


def profiles(phase: Optional[Phase] = None) -> list[dict[str, Any]]:
    """Return profiling data as a list of dicts. If `phase` is provided,
    only events from that phase are returned."""
    ...


def export_chrome_trace(path: str) -> None:
    """Export recorded trace events to a Chrome Trace Event Format JSON
    file at `path`."""
    ...
