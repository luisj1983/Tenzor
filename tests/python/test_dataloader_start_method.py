"""Audit-11 QQ.17: DataLoader must choose ``spawn`` when ANY GPU backend
is active, not just CUDA.

Previously the probe only checked ``cuda_is_initialized`` /
``cuda_is_available``, so ROCm / Vulkan / OneAPI users silently
``fork()``'d into a corrupted driver context.
"""
from __future__ import annotations

from unittest import mock

import pytest

import tenzor as tz
from tenzor.data import DataLoader, Dataset


class _NullDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):  # pragma: no cover - never called
        raise IndexError


def _make_loader():
    return DataLoader(_NullDataset(), batch_size=1, num_workers=1)


def _probe_module():
    """Return the tenzor_core module the loader probes."""
    from tenzor import tenzor_core  # type: ignore[attr-defined]
    return tenzor_core


@pytest.mark.parametrize(
    "active_probe",
    [
        "cuda_is_initialized",
        "rocm_is_initialized",
        "oneapi_is_initialized",
        "vulkan_is_initialized",
    ],
)
def test_each_gpu_backend_forces_spawn(active_probe, monkeypatch):
    """If ANY GPU backend reports active, the start method must be 'spawn'."""
    core = _probe_module()

    # Force every probe to report inactive except the one under test.
    for name in (
        "cuda_is_initialized", "cuda_is_available",
        "rocm_is_initialized", "rocm_is_available",
        "oneapi_is_initialized", "oneapi_is_available",
        "vulkan_is_initialized", "vulkan_is_available",
    ):
        monkeypatch.setattr(
            core, name,
            (lambda v=(name == active_probe): v),
            raising=False,
        )

    import multiprocessing as mp
    monkeypatch.setattr(mp, "get_start_method", lambda allow_none=False: None)

    captured: dict = {}
    real_get_context = mp.get_context

    def fake_get_context(name):
        captured["name"] = name
        return real_get_context(name)

    monkeypatch.setattr(mp, "get_context", fake_get_context)

    loader = _make_loader()
    try:
        loader._start_workers()
    finally:
        loader._stop_workers()

    assert captured["name"] == "spawn", (
        f"Expected spawn when {active_probe}=True, got {captured['name']!r}"
    )


def test_no_gpu_backend_allows_fork(monkeypatch):
    core = _probe_module()
    for name in (
        "cuda_is_initialized", "cuda_is_available",
        "rocm_is_initialized", "rocm_is_available",
        "oneapi_is_initialized", "oneapi_is_available",
        "vulkan_is_initialized", "vulkan_is_available",
    ):
        monkeypatch.setattr(core, name, lambda: False, raising=False)

    import multiprocessing as mp
    monkeypatch.setattr(mp, "get_start_method", lambda allow_none=False: None)

    captured: dict = {}
    real_get_context = mp.get_context

    def fake_get_context(name):
        captured["name"] = name
        return real_get_context(name)

    monkeypatch.setattr(mp, "get_context", fake_get_context)

    loader = _make_loader()
    try:
        loader._start_workers()
    finally:
        loader._stop_workers()

    # On POSIX with no GPU backend active, fork is the default.
    import os
    expected = "fork" if hasattr(os, "fork") else "spawn"
    assert captured["name"] == expected


def test_user_set_start_method_is_honored(monkeypatch):
    """If the user pinned a start method, respect it (warn on conflict)."""
    core = _probe_module()
    # GPU active to make the conflict meaningful.
    monkeypatch.setattr(core, "cuda_is_initialized", lambda: True, raising=False)

    import multiprocessing as mp
    monkeypatch.setattr(mp, "get_start_method", lambda allow_none=False: "fork")

    captured: dict = {}
    real_get_context = mp.get_context

    def fake_get_context(name):
        captured["name"] = name
        return real_get_context(name)

    monkeypatch.setattr(mp, "get_context", fake_get_context)

    loader = _make_loader()
    with pytest.warns(RuntimeWarning, match="fork"):
        try:
            loader._start_workers()
        finally:
            loader._stop_workers()

    assert captured["name"] == "fork"
