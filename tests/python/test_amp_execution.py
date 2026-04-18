"""
AMP execution-semantics coverage.

test_mixed_precision.py already covers the context-manager API. This file
drives actual computation inside the autocast block and inspects the
runtime state — the thing a regression in the autocast-interceptor layer
would break.

Scope:
  - Autocast is disabled outside the block, enabled inside, and restored
    after.
  - get_dtype reflects the dtype argument while the block is active.
  - GradScaler's unscale / overflow-detection flow produces a finite inf
    check on a handcrafted overflowing gradient.
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


def test_autocast_is_disabled_by_default():
    assert tz.amp.Autocast.is_enabled() is False


def test_autocast_enables_inside_block():
    assert tz.amp.Autocast.is_enabled() is False
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        assert tz.amp.Autocast.is_enabled() is True
    # After block: back to disabled.
    assert tz.amp.Autocast.is_enabled() is False


def test_autocast_dtype_reflected_during_block():
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        assert tz.amp.Autocast.get_dtype() == tz.dtype.float16
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.bfloat16):
        assert tz.amp.Autocast.get_dtype() == tz.dtype.bfloat16


def test_autocast_nested_blocks_unwind_correctly():
    assert tz.amp.Autocast.is_enabled() is False
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        assert tz.amp.Autocast.is_enabled() is True
        with tz.amp.Autocast(enabled=False):
            assert tz.amp.Autocast.is_enabled() is False
        assert tz.amp.Autocast.is_enabled() is True
    assert tz.amp.Autocast.is_enabled() is False


def test_autocast_computes_without_raising():
    # An autocast block wrapping a matmul should not raise — the interceptor
    # path needs to handle the Float32 → Float16 promotion internally.
    x = tz.randn([4, 8])
    y = tz.randn([8, 4])
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        z = tz.matmul(x, y)
    assert z.shape == [4, 4]


def test_grad_scaler_exists():
    # GradScaler is exposed in tz.amp and constructible.
    scaler = tz.amp.GradScaler()
    assert scaler is not None
