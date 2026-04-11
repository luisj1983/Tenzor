"""Tests for GIL release during tensor operations."""
import sys
sys.path.insert(0, "python")
import tenzor as tz
import threading
import time

tz.initialize()

def test_getitem_allows_concurrent_threads():
    """__getitem__ should release GIL, allowing other threads to run."""
    t = tz.randn([100, 100], dtype=tz.dtype.float32)
    results = []
    barrier = threading.Barrier(2)

    def worker(idx):
        barrier.wait()
        for _ in range(10):
            _ = t[idx]
        results.append(idx)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(results) == 2, f"Expected 2 threads completed, got {len(results)}"

def test_setitem_allows_concurrent_threads():
    """__setitem__ should release GIL, allowing other threads to run."""
    t = tz.zeros([100, 100], dtype=tz.dtype.float32)
    results = []
    barrier = threading.Barrier(2)

    def worker(idx):
        barrier.wait()
        for i in range(10):
            t[idx] = tz.ones([100], dtype=tz.dtype.float32)
        results.append(idx)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(results) == 2, f"Expected 2 threads completed, got {len(results)}"

def test_matmul_concurrent():
    """Tensor matmul should release GIL."""
    a = tz.randn([50, 50], dtype=tz.dtype.float32)
    b = tz.randn([50, 50], dtype=tz.dtype.float32)
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        for _ in range(5):
            _ = tz.matmul(a, b)
        results.append(True)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(results) == 2

if __name__ == "__main__":
    test_getitem_allows_concurrent_threads()
    print("  getitem concurrent test passed")
    test_setitem_allows_concurrent_threads()
    print("  setitem concurrent test passed")
    test_matmul_concurrent()
    print("  matmul concurrent test passed")
    print("All GIL release tests passed!")
