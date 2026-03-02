#!/usr/bin/env python3
"""Test thread safety of tensor operations with GIL release."""
import sys
import os
import unittest
import threading
import time

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

try:
    import tenzor.tenzor_core as tz
except ImportError:
    import tenzor_core as tz


class TestThreadSafety(unittest.TestCase):
    """Test concurrent tensor operations from multiple Python threads."""

    @classmethod
    def setUpClass(cls):
        tz.initialize()

    def test_concurrent_tensor_creation(self):
        """Multiple threads creating tensors simultaneously."""
        results = [None] * 4
        errors = []

        def create_tensors(idx):
            try:
                for _ in range(50):
                    t = tz.zeros([10, 10], tz.dtype.float32, tz.Device.cpu())
                    _ = t.shape
                results[idx] = True
            except Exception as e:
                errors.append(str(e))
                results[idx] = False

        threads = [threading.Thread(target=create_tensors, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")
        self.assertTrue(all(results))

    def test_concurrent_math_ops(self):
        """Multiple threads performing math operations."""
        errors = []

        def math_ops(idx):
            try:
                a = tz.ones([50, 50], tz.dtype.float32, tz.Device.cpu())
                b = tz.ones([50, 50], tz.dtype.float32, tz.Device.cpu())
                for _ in range(20):
                    c = tz.add(a, b)
                    d = tz.mul(a, b)
                    _ = c.shape
                    _ = d.shape
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = [threading.Thread(target=math_ops, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_concurrent_device_transfer(self):
        """Multiple threads transferring tensors to CPU (GIL release path)."""
        errors = []

        def transfer_ops(idx):
            try:
                t = tz.ones([20, 20], tz.dtype.float32, tz.Device.cpu())
                for _ in range(30):
                    cpu_t = t.cpu()
                    _ = cpu_t.shape
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = [threading.Thread(target=transfer_ops, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_concurrent_dtype_cast(self):
        """Multiple threads casting dtypes concurrently."""
        errors = []

        def cast_ops(idx):
            try:
                t = tz.ones([10, 10], tz.dtype.float32, tz.Device.cpu())
                for _ in range(20):
                    f64 = t.to(tz.dtype.float64)
                    f32 = f64.to(tz.dtype.float32)
                    _ = f32.shape
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = [threading.Thread(target=cast_ops, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_concurrent_backward_passes(self):
        """Multiple threads running independent backward passes."""
        errors = []

        def backward_ops(idx):
            try:
                for _ in range(10):
                    a = tz.Variable(tz.ones([5, 5], tz.dtype.float32, tz.Device.cpu()), True)
                    b = tz.Variable(tz.ones([5, 5], tz.dtype.float32, tz.Device.cpu()), True)
                    c = a + b
                    c.backward(tz.ones([5, 5], tz.dtype.float32, tz.Device.cpu()))
                    assert a.grad is not None, "Expected gradient on a"
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = [threading.Thread(target=backward_ops, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")


def run_tests():
    tz.initialize()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestThreadSafety)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
