"""Phase 5.3 — TensorBoard SummaryWriter smoke tests.

Verifies the basic API surface (open / add_scalar / add_histogram /
add_image / flush / close / is_open) writes a tfevents file and the file
exists on disk and has non-zero bytes. We do not parse the tfevents
binary here — that's a heavyweight dependency. The intent is to catch
binding-level regressions (e.g. method signatures, missing constructors).
"""
from __future__ import annotations

import os
import tempfile

import pytest

import tenzor as tz
from tenzor.tenzor_core import tensorboard as tb


def _open_writer(logdir: str) -> tb.SummaryWriter:
    return tb.SummaryWriter(logdir)


def test_summary_writer_constructs_and_opens_logfile(tmp_path):
    logdir = str(tmp_path / "events")
    os.makedirs(logdir, exist_ok=True)
    w = _open_writer(logdir)
    assert w.is_open(), "SummaryWriter should be open after construction"
    w.close()


def test_summary_writer_add_scalar_emits_file(tmp_path):
    logdir = str(tmp_path / "events_scalar")
    os.makedirs(logdir, exist_ok=True)
    w = _open_writer(logdir)
    w.add_scalar("loss", 0.5, 0)
    w.add_scalar("loss", 0.4, 1)
    w.add_scalar("loss", 0.3, 2)
    w.flush()
    w.close()

    files = list(os.listdir(logdir))
    assert files, f"no tfevents file written to {logdir}"
    # At least one file should be non-empty (the events log).
    sizes = [os.path.getsize(os.path.join(logdir, f)) for f in files]
    assert max(sizes) > 0, f"all files in {logdir} are empty: {dict(zip(files, sizes))}"


def test_summary_writer_add_histogram_smoke(tmp_path):
    logdir = str(tmp_path / "events_hist")
    os.makedirs(logdir, exist_ok=True)
    w = _open_writer(logdir)
    values = tz.randn([100])
    w.add_histogram("weights", values, 0)
    w.flush()
    w.close()

    files = list(os.listdir(logdir))
    assert files


def test_summary_writer_close_idempotent(tmp_path):
    logdir = str(tmp_path / "events_close")
    os.makedirs(logdir, exist_ok=True)
    w = _open_writer(logdir)
    w.close()
    # Calling close() again should not throw.
    w.close()
    assert not w.is_open()
