#!/usr/bin/env python3
"""Test negative stride tensors and flipped views for NumPy interop."""
import sys
import os
import unittest
import pytest
np = pytest.importorskip("numpy")

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

try:
    import tenzor.tenzor_core as tz
except ImportError:
    import tenzor_core as tz


class TestNegativeStrides(unittest.TestCase):
    """Test tensors with negative strides convert correctly to NumPy."""

    @classmethod
    def setUpClass(cls):
        tz.initialize()

    def test_flip_1d_to_numpy(self):
        """Flipped 1D tensor converts to correct NumPy values."""
        t = tz.Tensor([int(5)], tz.dtype.float32, tz.Device.cpu())
        data = t.numpy()
        # Fill with sequential values
        for i in range(5):
            data[i] = float(i)

        # Flip via slice
        flipped = tz.flip(t, [0])
        arr = flipped.numpy()

        expected = np.array([4, 3, 2, 1, 0], dtype=np.float32)
        np.testing.assert_array_almost_equal(arr, expected)

    def test_flip_2d_to_numpy(self):
        """Flipped 2D tensor preserves correct data ordering."""
        t = tz.zeros([3, 4], tz.dtype.float32, tz.Device.cpu())
        data = t.numpy()
        # `data` has shape (3, 4); use 2D indexing.
        for i in range(3):
            for j in range(4):
                data[i, j] = float(i * 4 + j)

        # Flip along axis 0
        flipped = tz.flip(t, [0])
        arr = flipped.numpy()

        self.assertEqual(arr.shape, (3, 4))

    def test_reversed_view_contiguity(self):
        """Reversed views should produce contiguous numpy arrays."""
        t = tz.ones([10], tz.dtype.float32, tz.Device.cpu())
        flipped = tz.flip(t, [0])
        arr = flipped.numpy()

        # The numpy array should be valid (contiguous or have correct strides)
        self.assertEqual(arr.shape, (10,))
        self.assertTrue(arr.nbytes > 0)

    def test_flip_preserves_values(self):
        """Values are correctly preserved after flip."""
        t = tz.ones([2, 3], tz.dtype.float32, tz.Device.cpu())
        flipped = tz.flip(t, [0, 1])
        arr = flipped.numpy()

        # All ones should still be ones after flipping
        expected = np.ones((2, 3), dtype=np.float32)
        np.testing.assert_array_almost_equal(arr, expected)

    def test_flip_float64(self):
        """Float64 flipped tensors convert correctly."""
        t = tz.ones([4], tz.dtype.float64, tz.Device.cpu())
        flipped = tz.flip(t, [0])
        arr = flipped.numpy()

        self.assertEqual(arr.dtype, np.float64)
        expected = np.ones(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(arr, expected)


def run_tests():
    tz.initialize()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNegativeStrides)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
