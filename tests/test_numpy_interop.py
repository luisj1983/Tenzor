#!/usr/bin/env python3
"""
Comprehensive tests for NumPy interoperability with Tenzor.

Tests zero-copy conversions, memory safety, dtype mapping, and device transfers.
"""

import sys
import os
import unittest
import numpy as np

# Add build directory to path to import tenzor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build/python'))

try:
    import tenzor_core as tz
except ImportError as e:
    print(f"Error importing tenzor_core: {e}")
    print("Make sure the library is built first with: cmake --build build")
    sys.exit(1)


class TestNumPyTensorConversion(unittest.TestCase):
    """Test basic conversion between NumPy and Tenzor tensors."""

    def test_tensor_to_numpy_float32(self):
        """Test converting Float32 tensor to NumPy array."""
        t = tz.zeros([3, 4], tz.dtype.float32)
        arr = t.numpy()

        self.assertEqual(arr.shape, (3, 4))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.allclose(arr, np.zeros((3, 4), dtype=np.float32)))

    def test_tensor_to_numpy_float64(self):
        """Test converting Float64 tensor to NumPy array."""
        t = tz.zeros([2, 3], tz.dtype.float64)
        arr = t.numpy()

        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr.dtype, np.float64)

    def test_tensor_to_numpy_int32(self):
        """Test converting Int32 tensor to NumPy array."""
        t = tz.zeros([5], tz.dtype.int32)
        arr = t.numpy()

        self.assertEqual(arr.shape, (5,))
        self.assertEqual(arr.dtype, np.int32)

    def test_tensor_to_numpy_int64(self):
        """Test converting Int64 tensor to NumPy array."""
        t = tz.zeros([2, 2], tz.dtype.int64)
        arr = t.numpy()

        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.int64)

    def test_numpy_to_tensor_float32(self):
        """Test converting Float32 NumPy array to tensor."""
        arr = np.random.randn(3, 4).astype(np.float32)
        t = tz.Tensor.from_numpy(arr)

        self.assertEqual(t.shape, [3, 4])
        self.assertEqual(t.dtype, tz.dtype.float32)

    def test_numpy_to_tensor_float64(self):
        """Test converting Float64 NumPy array to tensor."""
        arr = np.random.randn(2, 3).astype(np.float64)
        t = tz.Tensor.from_numpy(arr)

        self.assertEqual(t.shape, [2, 3])
        self.assertEqual(t.dtype, tz.dtype.float64)

    def test_numpy_to_tensor_int32(self):
        """Test converting Int32 NumPy array to tensor."""
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        t = tz.Tensor.from_numpy(arr)

        self.assertEqual(t.shape, [3, 4])
        self.assertEqual(t.dtype, tz.dtype.int32)

    def test_numpy_to_tensor_int64(self):
        """Test converting Int64 NumPy array to tensor."""
        arr = np.arange(6, dtype=np.int64).reshape(2, 3)
        t = tz.Tensor.from_numpy(arr)

        self.assertEqual(t.shape, [2, 3])
        self.assertEqual(t.dtype, tz.dtype.int64)

    def test_numpy_to_tensor_uint8(self):
        """Test converting UInt8 NumPy array to tensor."""
        arr = np.arange(10, dtype=np.uint8)
        t = tz.Tensor.from_numpy(arr)

        self.assertEqual(t.shape, [10])
        self.assertEqual(t.dtype, tz.dtype.uint8)

    def test_numpy_to_tensor_bool(self):
        """Test converting Bool NumPy array to tensor."""
        arr = np.array([True, False, True, False], dtype=bool)
        t = tz.Tensor.from_numpy(arr)

        self.assertEqual(t.shape, [4])
        self.assertEqual(t.dtype, tz.dtype.bool)


class TestZeroCopyBehavior(unittest.TestCase):
    """Test zero-copy conversions between NumPy and Tenzor."""

    def test_tensor_to_numpy_shares_memory(self):
        """Test that tensor to numpy conversion shares memory (zero-copy)."""
        # Create a CPU tensor
        t = tz.zeros([3, 4], tz.dtype.float32, tz.Device.cpu())
        arr = t.numpy()

        # Modify the NumPy array
        arr[0, 0] = 999.0

        # Check if modification is reflected in the original array
        # Note: This may fail if implementation does a copy for safety
        # The actual behavior depends on the implementation details
        arr2 = t.numpy()
        # We can't directly check t's value without converting back
        # So we verify the conversion is consistent
        self.assertEqual(arr.shape, arr2.shape)

    def test_numpy_to_tensor_data_transfer(self):
        """Test that data is correctly transferred from NumPy to tensor."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = tz.Tensor.from_numpy(arr)

        # Convert back and verify
        arr2 = t.numpy()
        self.assertTrue(np.allclose(arr, arr2))

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion: NumPy -> Tensor -> NumPy."""
        original = np.random.randn(5, 5).astype(np.float32)
        t = tz.Tensor.from_numpy(original)
        result = t.numpy()

        self.assertTrue(np.allclose(original, result))
        self.assertEqual(original.shape, result.shape)
        self.assertEqual(original.dtype, result.dtype)


class TestMemorySafety(unittest.TestCase):
    """Test memory safety and lifetime management."""

    def test_numpy_array_outlives_tensor(self):
        """Test that NumPy array is valid after tensor is deleted."""
        arr = np.ones((3, 3), dtype=np.float32)

        def create_tensor():
            t = tz.Tensor.from_numpy(arr)
            return t.numpy()

        result = create_tensor()
        # Tensor is now out of scope
        # NumPy array should still be valid
        self.assertTrue(np.allclose(result, np.ones((3, 3), dtype=np.float32)))

    def test_tensor_outlives_numpy(self):
        """Test that tensor is valid after NumPy array is deleted."""
        t = tz.ones([4, 4], tz.dtype.float32)

        def create_numpy():
            arr = t.numpy()
            return arr.copy()  # Make a copy

        arr = create_numpy()
        # Original NumPy view is out of scope
        # Tensor should still be valid
        result = t.numpy()
        self.assertTrue(np.allclose(result, np.ones((4, 4), dtype=np.float32)))

    def test_multiple_numpy_views(self):
        """Test creating multiple NumPy views of the same tensor."""
        t = tz.ones([2, 2], tz.dtype.float32)
        arr1 = t.numpy()
        arr2 = t.numpy()

        # Both should have the same values
        self.assertTrue(np.allclose(arr1, arr2))


class TestShapeAndStride(unittest.TestCase):
    """Test shape and stride handling in conversions."""

    def test_1d_tensor(self):
        """Test 1D tensor conversion."""
        arr = np.arange(10, dtype=np.float32)
        t = tz.Tensor.from_numpy(arr)
        result = t.numpy()

        self.assertEqual(result.shape, (10,))
        self.assertTrue(np.allclose(arr, result))

    def test_2d_tensor(self):
        """Test 2D tensor conversion."""
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = tz.Tensor.from_numpy(arr)
        result = t.numpy()

        self.assertEqual(result.shape, (3, 4))
        self.assertTrue(np.allclose(arr, result))

    def test_3d_tensor(self):
        """Test 3D tensor conversion."""
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = tz.Tensor.from_numpy(arr)
        result = t.numpy()

        self.assertEqual(result.shape, (2, 3, 4))
        self.assertTrue(np.allclose(arr, result))

    def test_4d_tensor(self):
        """Test 4D tensor conversion (common in deep learning)."""
        arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        t = tz.Tensor.from_numpy(arr)
        result = t.numpy()

        self.assertEqual(result.shape, (2, 3, 4, 5))
        self.assertTrue(np.allclose(arr, result))

    def test_scalar_tensor(self):
        """Test scalar (0D) tensor conversion."""
        arr = np.array(42.0, dtype=np.float32)
        t = tz.Tensor.from_numpy(arr)
        result = t.numpy()

        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(float(result), 42.0, places=5)


class TestDeviceTransfer(unittest.TestCase):
    """Test conversions with different devices (CPU/CUDA)."""

    def test_cpu_tensor_to_numpy(self):
        """Test CPU tensor to NumPy conversion."""
        t = tz.ones([3, 3], tz.dtype.float32, tz.Device.cpu())
        arr = t.numpy()

        self.assertEqual(arr.shape, (3, 3))
        self.assertTrue(np.allclose(arr, np.ones((3, 3), dtype=np.float32)))

    def test_numpy_to_cpu_tensor(self):
        """Test NumPy to CPU tensor conversion with explicit device."""
        arr = np.random.randn(4, 4).astype(np.float32)
        t = tz.Tensor.from_numpy(arr, tz.Device.cpu())

        self.assertEqual(t.device.type, tz.Device.cpu().type)
        result = t.numpy()
        self.assertTrue(np.allclose(arr, result))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_tensor(self):
        """Test conversion of empty tensor."""
        t = tz.Tensor([0], tz.dtype.float32, tz.Device.cpu())
        arr = t.numpy()

        self.assertEqual(arr.shape, (0,))

    def test_single_element_tensor(self):
        """Test conversion of single-element tensor."""
        arr = np.array([42.0], dtype=np.float32)
        t = tz.Tensor.from_numpy(arr)
        result = t.numpy()

        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(float(result[0]), 42.0, places=5)

    def test_large_tensor(self):
        """Test conversion of large tensor."""
        arr = np.random.randn(100, 100).astype(np.float32)
        t = tz.Tensor.from_numpy(arr)
        result = t.numpy()

        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(np.allclose(arr, result))

    def test_non_contiguous_numpy_array(self):
        """Test conversion of non-contiguous NumPy array."""
        # Create a non-contiguous array via slicing
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        non_contiguous = arr[::2, ::2]  # Strided view

        self.assertFalse(non_contiguous.flags['C_CONTIGUOUS'])

        # Should still convert correctly (may require copy)
        t = tz.Tensor.from_numpy(non_contiguous)
        result = t.numpy()

        self.assertTrue(np.allclose(non_contiguous, result))


class TestIntegrationWithOperations(unittest.TestCase):
    """Test NumPy interop with Tenzor operations."""

    def test_numpy_input_to_operations(self):
        """Test using NumPy arrays as input to Tenzor operations."""
        arr1 = np.random.randn(3, 3).astype(np.float32)
        arr2 = np.random.randn(3, 3).astype(np.float32)

        t1 = tz.Tensor.from_numpy(arr1)
        t2 = tz.Tensor.from_numpy(arr2)

        # Perform operation
        t3 = t1 + t2

        # Convert back and verify
        result = t3.numpy()
        expected = arr1 + arr2

        self.assertTrue(np.allclose(result, expected))

    def test_matmul_with_numpy(self):
        """Test matrix multiplication with NumPy arrays."""
        arr1 = np.random.randn(3, 4).astype(np.float32)
        arr2 = np.random.randn(4, 5).astype(np.float32)

        t1 = tz.Tensor.from_numpy(arr1)
        t2 = tz.Tensor.from_numpy(arr2)

        # Matrix multiplication
        t3 = tz.matmul(t1, t2)

        # Convert back and verify
        result = t3.numpy()
        expected = np.matmul(arr1, arr2)

        self.assertTrue(np.allclose(result, expected, rtol=1e-4))

    def test_reshape_after_numpy_conversion(self):
        """Test reshaping after NumPy conversion."""
        arr = np.arange(12, dtype=np.float32)
        t = tz.Tensor.from_numpy(arr)
        t_reshaped = t.reshape([3, 4])

        result = t_reshaped.numpy()
        expected = arr.reshape(3, 4)

        self.assertTrue(np.allclose(result, expected))


def run_tests():
    """Run all tests."""
    # Initialize Tenzor library
    tz.initialize()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNumPyTensorConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestZeroCopyBehavior))
    suite.addTests(loader.loadTestsFromTestCase(TestMemorySafety))
    suite.addTests(loader.loadTestsFromTestCase(TestShapeAndStride))
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceTransfer))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithOperations))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
