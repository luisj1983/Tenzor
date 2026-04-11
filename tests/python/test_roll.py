"""Tests for tenzor.roll() operation."""
import sys
sys.path.insert(0, "python")
import tenzor as tz

tz.initialize()

def test_roll_basic():
    t = tz.arange(0, 6, dtype=tz.dtype.float32)  # [0,1,2,3,4,5]
    r = tz.roll(t, 2, 0)
    # Should be [4,5,0,1,2,3]
    assert r.shape == [6]
    data = [r[i].item() for i in range(6)]
    assert data == [4.0, 5.0, 0.0, 1.0, 2.0, 3.0], f"Got {data}"

def test_roll_negative_shift():
    t = tz.arange(0, 6, dtype=tz.dtype.float32)
    r = tz.roll(t, -2, 0)
    # Should be [2,3,4,5,0,1]
    data = [r[i].item() for i in range(6)]
    assert data == [2.0, 3.0, 4.0, 5.0, 0.0, 1.0], f"Got {data}"

def test_roll_2d():
    t = tz.arange(0, 6, dtype=tz.dtype.float32).reshape([2, 3])
    r = tz.roll(t, 1, 1)
    # Each row rolled by 1 along dim 1
    assert r.shape == [2, 3]

def test_roll_zero_shift():
    t = tz.randn([4, 5], dtype=tz.dtype.float32)
    r = tz.roll(t, 0, 0)
    # Should be identical
    diff = tz.sub(t, r)
    assert tz.abs(diff).max().item() < 1e-6

if __name__ == "__main__":
    test_roll_basic()
    test_roll_negative_shift()
    test_roll_2d()
    test_roll_zero_shift()
    print("All roll tests passed!")
