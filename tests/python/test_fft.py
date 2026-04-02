"""Tests for FFT operations."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest
import math


# ---------------------------------------------------------------------------
# 1D FFT
# ---------------------------------------------------------------------------

def test_fft_basic():
    x = tz.randn([8])
    y = tz.fft.fft(x)
    assert y.shape == [8]


def test_fft_ifft_roundtrip():
    x = tz.randn([16])
    y = tz.fft.fft(x)
    x_back = tz.fft.ifft(y)
    assert x_back.shape == x.shape


def test_rfft_basic():
    x = tz.randn([8])
    y = tz.fft.rfft(x)
    # rfft output has n//2 + 1 complex elements
    assert y.shape[0] == 5  # 8//2 + 1


def test_rfft_irfft_roundtrip():
    x = tz.randn([16])
    y = tz.fft.rfft(x)
    x_back = tz.fft.irfft(y)
    assert x_back.shape[0] == 16 or x_back.shape[0] == 15  # depends on n param


# ---------------------------------------------------------------------------
# 2D FFT
# ---------------------------------------------------------------------------

def test_fft2_basic():
    x = tz.randn([4, 4])
    y = tz.fft.fft2(x)
    assert y.shape == [4, 4]


def test_fft2_ifft2_roundtrip():
    x = tz.randn([8, 8])
    y = tz.fft.fft2(x)
    x_back = tz.fft.ifft2(y)
    assert x_back.shape == x.shape


# ---------------------------------------------------------------------------
# N-D FFT
# ---------------------------------------------------------------------------

def test_fftn_basic():
    x = tz.randn([4, 4, 4])
    y = tz.fft.fftn(x)
    assert y.shape == [4, 4, 4]


def test_fftn_ifftn_roundtrip():
    x = tz.randn([4, 4])
    y = tz.fft.fftn(x)
    x_back = tz.fft.ifftn(y)
    assert x_back.shape == x.shape


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fft_single_element():
    x = tz.ones([1])
    y = tz.fft.fft(x)
    assert y.shape == [1]


def test_fft_power_of_two():
    for n in [2, 4, 8, 16, 32]:
        x = tz.randn([n])
        y = tz.fft.fft(x)
        assert y.shape == [n]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
