#!/usr/bin/env python3
"""
Simple test to verify NumPy interoperability works.
"""

import sys
import os
import numpy as np

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))

try:
    import test_numpy_core as tz
except ImportError as e:
    print(f"Error importing test_numpy_core: {e}")
    sys.exit(1)

def test_basic_conversion():
    """Test basic NumPy <-> Tensor conversion."""
    print("Test 1: Basic NumPy to Tensor conversion")

    # Initialize library
    tz.initialize()

    # Create NumPy array
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    print(f"  NumPy array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  NumPy array:\n{arr}")

    # Convert to tensor
    t = tz.Tensor.from_numpy(arr)
    print(f"  Tensor shape: {t.shape}, dtype: {t.dtype}")

    # Convert back to NumPy
    arr2 = t.numpy()
    print(f"  Converted back shape: {arr2.shape}, dtype: {arr2.dtype}")
    print(f"  Converted back:\n{arr2}")

    # Verify
    if np.allclose(arr, arr2):
        print("  ✓ PASSED: Arrays match")
    else:
        print("  ✗ FAILED: Arrays don't match")
        return False

    return True

def test_zeros():
    """Test creating zeros tensor and converting to NumPy."""
    print("\nTest 2: Zeros tensor to NumPy")

    # Create zeros tensor
    t = tz.zeros([3, 4], tz.dtype.float32)
    print(f"  Tensor shape: {t.shape}, dtype: {t.dtype}")

    # Convert to NumPy
    arr = t.numpy()
    print(f"  NumPy array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  NumPy array:\n{arr}")

    # Verify
    if np.allclose(arr, np.zeros((3, 4), dtype=np.float32)):
        print("  ✓ PASSED: Zeros tensor converted correctly")
    else:
        print("  ✗ FAILED: Zeros tensor incorrect")
        return False

    return True

def test_ones():
    """Test creating ones tensor and converting to NumPy."""
    print("\nTest 3: Ones tensor to NumPy")

    # Create ones tensor
    t = tz.ones([2, 3], tz.dtype.float32)
    print(f"  Tensor shape: {t.shape}, dtype: {t.dtype}")

    # Convert to NumPy
    arr = t.numpy()
    print(f"  NumPy array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  NumPy array:\n{arr}")

    # Verify
    if np.allclose(arr, np.ones((2, 3), dtype=np.float32)):
        print("  ✓ PASSED: Ones tensor converted correctly")
    else:
        print("  ✗ FAILED: Ones tensor incorrect")
        return False

    return True

def test_dtypes():
    """Test different data types."""
    print("\nTest 4: Different dtypes")

    dtypes = [
        (np.float32, tz.dtype.float32, "float32"),
        (np.float64, tz.dtype.float64, "float64"),
        (np.int32, tz.dtype.int32, "int32"),
        (np.int64, tz.dtype.int64, "int64"),
        (np.uint8, tz.dtype.uint8, "uint8"),
        (bool, tz.dtype.bool, "bool"),
    ]

    for np_dtype, tz_dtype, name in dtypes:
        print(f"  Testing {name}...")
        arr = np.array([1, 2, 3, 4], dtype=np_dtype)
        t = tz.Tensor.from_numpy(arr)
        arr2 = t.numpy()

        if np.allclose(arr, arr2):
            print(f"    ✓ {name} PASSED")
        else:
            print(f"    ✗ {name} FAILED")
            return False

    print("  ✓ PASSED: All dtypes converted correctly")
    return True

def test_multidimensional():
    """Test multi-dimensional arrays."""
    print("\nTest 5: Multi-dimensional arrays")

    shapes = [
        (10,),
        (3, 4),
        (2, 3, 4),
        (2, 3, 4, 5),
    ]

    for shape in shapes:
        print(f"  Testing shape {shape}...")
        arr = np.random.randn(*shape).astype(np.float32)
        t = tz.Tensor.from_numpy(arr)
        arr2 = t.numpy()

        if arr2.shape == shape and np.allclose(arr, arr2):
            print(f"    ✓ Shape {shape} PASSED")
        else:
            print(f"    ✗ Shape {shape} FAILED")
            return False

    print("  ✓ PASSED: All shapes converted correctly")
    return True

def main():
    print("=" * 60)
    print("NumPy Interoperability Tests")
    print("=" * 60)

    tests = [
        test_basic_conversion,
        test_zeros,
        test_ones,
        test_dtypes,
        test_multidimensional,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
