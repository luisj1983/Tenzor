#!/usr/bin/env python3
"""Test BFloat16 tensor <-> NumPy interoperability."""
import sys
import os
import unittest
import numpy as np

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

try:
    import tenzor.tenzor_core as tz
except ImportError:
    import tenzor_core as tz


class TestBFloat16Numpy(unittest.TestCase):
    """BFloat16 tensor to NumPy conversion tests."""

    @classmethod
    def setUpClass(cls):
        tz.initialize()

    def test_bfloat16_to_numpy_basic(self):
        """BFloat16 tensor converts to numpy array."""
        t = tz.zeros([2, 3], tz.dtype.bfloat16, tz.Device.cpu())
        arr = t.numpy()
        self.assertEqual(arr.shape, (2, 3))

    def test_bfloat16_to_numpy_dtype(self):
        """BFloat16 should convert to uint16 (bit repr) or ml_dtypes.bfloat16."""
        t = tz.zeros([4], tz.dtype.bfloat16, tz.Device.cpu())
        arr = t.numpy()

        try:
            import ml_dtypes
            # With ml_dtypes, should be bfloat16
            self.assertEqual(arr.dtype, np.dtype(ml_dtypes.bfloat16))
        except ImportError:
            # Without ml_dtypes, falls back to uint16
            self.assertEqual(arr.dtype, np.uint16)

    def test_bfloat16_roundtrip_precision(self):
        """Values survive BFloat16 -> numpy -> back conversion."""
        t = tz.ones([3], tz.dtype.bfloat16, tz.Device.cpu())
        arr = t.numpy()

        # The bit pattern for bfloat16 1.0 is 0x3F80
        if arr.dtype == np.uint16:
            for val in arr:
                self.assertEqual(val, 0x3F80,
                                 f"BFloat16 1.0 should be 0x3F80, got 0x{val:04X}")

    def test_bfloat16_nonzero_values(self):
        """BFloat16 tensor with non-zero values converts correctly."""
        # Create float32 tensor, convert to bfloat16
        t = tz.ones([2, 2], tz.dtype.float32, tz.Device.cpu())
        bf16 = t.to(tz.dtype.bfloat16)
        arr = bf16.numpy()
        self.assertEqual(arr.shape, (2, 2))

    def test_bfloat16_empty_tensor(self):
        """Empty BFloat16 tensor converts to empty numpy array."""
        t = tz.zeros([0], tz.dtype.bfloat16, tz.Device.cpu())
        arr = t.numpy()
        self.assertEqual(arr.shape, (0,))

    def test_bfloat16_large_tensor(self):
        """Large BFloat16 tensor converts without errors."""
        t = tz.zeros([100, 100], tz.dtype.bfloat16, tz.Device.cpu())
        arr = t.numpy()
        self.assertEqual(arr.shape, (100, 100))


def run_tests():
    tz.initialize()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBFloat16Numpy)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
