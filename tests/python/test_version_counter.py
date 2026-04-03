#!/usr/bin/env python3
"""Tests that in-place operations bump the tensor version counter by exactly 1.

dispatch_inplace() already calls bump_version(), so Python bindings must NOT
call it again (which would cause a double bump).
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def _version(var):
    """Get version counter from a Variable's underlying tensor."""
    return var.data.version()


def test_iadd_variable():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    b = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    a += b
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_isub_variable():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    b = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    a -= b
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_imul_variable():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    b = tz.Variable(tz.full([2, 2], 2.0, tz.dtype.float32), False)
    v0 = _version(a)
    a *= b
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_itruediv_variable():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    b = tz.Variable(tz.full([2, 2], 2.0, tz.dtype.float32), False)
    v0 = _version(a)
    a /= b
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_iadd_scalar():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    a += 1.0
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_isub_scalar():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    a -= 1.0
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_imul_scalar():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    a *= 2.0
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_itruediv_scalar():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    a /= 2.0
    assert _version(a) == v0 + 1, f"Expected version {v0+1}, got {_version(a)}"


def test_multiple_inplace_ops():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    b = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    a += b
    a *= 2.0
    a -= b
    a /= 3.0
    assert _version(a) == v0 + 4, f"Expected version {v0+4}, got {_version(a)}"


def test_no_bump_on_non_inplace():
    a = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    b = tz.Variable(tz.ones([2, 2], tz.dtype.float32), False)
    v0 = _version(a)
    _ = a + b  # non-inplace
    assert _version(a) == v0, f"Expected version {v0}, got {_version(a)}"


if __name__ == "__main__":
    tz.initialize()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
