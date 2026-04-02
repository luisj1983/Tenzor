"""Extended tests for mixed precision training (AMP, GradScaler)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


# ---------------------------------------------------------------------------
# GradScaler
# ---------------------------------------------------------------------------

def test_grad_scaler_creation():
    scaler = tz.amp.GradScaler()
    assert scaler is not None


def test_grad_scaler_scale():
    scaler = tz.amp.GradScaler()
    loss = tz.Variable(tz.full([1], 1.0), True)
    scaled = scaler.scale(loss)
    assert scaled.data.shape == [1]


def test_grad_scaler_step_update():
    scaler = tz.amp.GradScaler()
    linear = tz.nn.Linear(4, 2)
    optimizer = tz.optim.SGD(linear.parameters(), lr=0.01)

    x = tz.Variable(tz.randn([2, 4]), True)
    y = linear(x)
    loss = tz.mean(y)

    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()


def test_grad_scaler_get_scale():
    scaler = tz.amp.GradScaler(init_scale=1024.0)
    scale = scaler.get_scale()
    assert scale > 0


# ---------------------------------------------------------------------------
# Autocast context
# ---------------------------------------------------------------------------

def test_autocast_exists():
    assert hasattr(tz.amp, 'autocast') or hasattr(tz.amp, 'Autocast')


# ---------------------------------------------------------------------------
# BFloat16 operations
# ---------------------------------------------------------------------------

def test_bfloat16_creation():
    x = tz.randn([4, 4], dtype=tz.dtype.bfloat16)
    assert x.dtype == tz.dtype.bfloat16


def test_bfloat16_to_float32():
    x = tz.randn([4, 4], dtype=tz.dtype.bfloat16)
    y = x.to(tz.dtype.float32)
    assert y.dtype == tz.dtype.float32


def test_float16_creation():
    x = tz.randn([4, 4], dtype=tz.dtype.float16)
    assert x.dtype == tz.dtype.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
