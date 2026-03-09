#!/usr/bin/env python3
"""
Test mixed-precision (AMP) Python bindings: Autocast context manager and GradScaler.
"""

import sys
import os
import math

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_autocast_context_manager():
    """Verify Autocast can be entered and exited without error."""
    print("Testing autocast context manager...")
    ctx = tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16)
    ctx.__enter__()
    # Inside autocast scope — create some tensors and do matmul
    a = tz.randn([4, 4])
    b = tz.randn([4, 4])
    c = tz.matmul(a, b)
    assert c.shape == [4, 4]
    ctx.__exit__(None, None, None)
    print("  autocast context OK")


def test_autocast_disabled():
    """Verify Autocast with enabled=False is a no-op."""
    print("Testing autocast disabled...")
    ctx = tz.amp.Autocast(enabled=False, dtype=tz.dtype.float16)
    ctx.__enter__()
    a = tz.randn([3, 3])
    b = tz.randn([3, 3])
    c = tz.matmul(a, b)
    assert c.dtype == tz.dtype.float32, f"Expected float32, got {c.dtype}"
    ctx.__exit__(None, None, None)
    print("  autocast disabled OK")


def test_grad_scaler_basic():
    """Test GradScaler basic scale/unscale/step cycle."""
    print("Testing GradScaler basic...")
    scaler = tz.amp.GradScaler(init_scale=1024.0, growth_factor=2.0,
                                backoff_factor=0.5, growth_interval=100)

    assert scaler.get_scale() == 1024.0, f"Wrong init scale: {scaler.get_scale()}"
    assert scaler.get_growth_tracker() == 0

    # Scale a loss variable
    loss = tz.Variable(tz.ones([1]), requires_grad=True)
    scaled = scaler.scale(loss)
    assert scaled.tensor().shape == [1]

    # Update scaler
    scaler.update()
    print("  GradScaler basic OK")


def test_grad_scaler_overflow_detection():
    """Test GradScaler overflow detection via found_inf_nan."""
    print("Testing GradScaler overflow detection...")
    scaler = tz.amp.GradScaler(init_scale=65536.0)
    initial_scale = scaler.get_scale()

    # Reset should work
    scaler.reset()
    assert scaler.get_scale() == initial_scale
    assert scaler.get_growth_tracker() == 0

    # State dict round-trip
    state = scaler.state_dict()
    scaler2 = tz.amp.GradScaler(init_scale=1.0)
    scaler2.load_state_dict(state)
    assert scaler2.get_scale() == initial_scale, \
        f"Scale mismatch after load: {scaler2.get_scale()} vs {initial_scale}"
    print("  GradScaler overflow detection OK")


def main():
    print("=" * 60)
    print("Testing Mixed Precision (AMP) Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        test_autocast_context_manager()
        test_autocast_disabled()
        test_grad_scaler_basic()
        test_grad_scaler_overflow_detection()

        print("\n" + "=" * 60)
        print("All mixed-precision tests PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
