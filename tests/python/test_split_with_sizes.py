"""Tests for tenzor.split_with_sizes() operation."""
import sys
sys.path.insert(0, "python")
import tenzor as tz

tz.initialize()

def test_split_basic():
    t = tz.arange(0, 10, dtype=tz.dtype.float32)
    parts = tz.split_with_sizes(t, [3, 3, 4], 0)
    assert len(parts) == 3
    assert parts[0].shape == [3]
    assert parts[1].shape == [3]
    assert parts[2].shape == [4]

def test_split_2d():
    t = tz.randn([6, 4], dtype=tz.dtype.float32)
    parts = tz.split_with_sizes(t, [2, 4], 0)
    assert len(parts) == 2
    assert parts[0].shape == [2, 4]
    assert parts[1].shape == [4, 4]

def test_split_dim1():
    t = tz.randn([3, 10], dtype=tz.dtype.float32)
    parts = tz.split_with_sizes(t, [2, 3, 5], 1)
    assert len(parts) == 3
    assert parts[0].shape == [3, 2]
    assert parts[1].shape == [3, 3]
    assert parts[2].shape == [3, 5]

def test_split_single():
    t = tz.randn([5], dtype=tz.dtype.float32)
    parts = tz.split_with_sizes(t, [5], 0)
    assert len(parts) == 1
    assert parts[0].shape == [5]

def test_split_sum_mismatch():
    t = tz.randn([5], dtype=tz.dtype.float32)
    try:
        tz.split_with_sizes(t, [2, 2], 0)
        assert False, "Should have raised"
    except (RuntimeError, ValueError):
        pass

if __name__ == "__main__":
    test_split_basic()
    test_split_2d()
    test_split_dim1()
    test_split_single()
    test_split_sum_mismatch()
    print("All split_with_sizes tests passed!")
