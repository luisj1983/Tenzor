#!/usr/bin/env python3
"""
Test script to verify the tensor assignment fixes:
1. Non-contiguous tensor assignment
2. Broadcasting assignment
"""

import sys
import os

# Add the build directory to the path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'python')
sys.path.insert(0, build_dir)

import tenzor
import numpy as np


def test_non_contiguous_assignment():
    """Test assignment to non-contiguous tensors."""
    print("Testing non-contiguous tensor assignment...")

    # Create a tensor and transpose it (makes it non-contiguous)
    t1 = tenzor.Tensor([2, 3], tenzor.DType.Float32)
    t1.fill_(1.0)

    # Transpose creates a non-contiguous view
    t2 = t1.transpose(0, 1)

    # Create source tensor
    src = tenzor.Tensor([3, 2], tenzor.DType.Float32)
    src.fill_(5.0)

    print(f"  Source contiguous: {src.is_contiguous()}")
    print(f"  Target contiguous: {t2.is_contiguous()}")

    # This should now work (previously threw "Non-contiguous tensor assignment not yet implemented")
    try:
        t2[:] = src
        print("  ✓ Non-contiguous assignment succeeded!")

        # Verify the values were copied correctly
        np_data = t2.numpy()
        if np.allclose(np_data, 5.0):
            print("  ✓ Values verified correct!")
        else:
            print(f"  ✗ Values incorrect: {np_data}")
            return False

    except RuntimeError as e:
        print(f"  ✗ Failed with error: {e}")
        return False

    return True


def test_broadcasting_assignment():
    """Test assignment with broadcasting."""
    print("\nTesting broadcasting assignment...")

    # Create destination tensor [3, 4]
    dst = tenzor.Tensor([3, 4], tenzor.DType.Float32)
    dst.fill_(0.0)

    # Create source tensor [1, 4] that should broadcast to [3, 4]
    src = tenzor.Tensor([1, 4], tenzor.DType.Float32)
    src.fill_(7.0)

    print(f"  Destination shape: {dst.shape()}")
    print(f"  Source shape: {src.shape()}")

    # This should now work (previously threw "Broadcasting assignment not yet fully implemented")
    try:
        dst[:] = src
        print("  ✓ Broadcasting assignment succeeded!")

        # Verify the values were broadcast correctly
        np_data = dst.numpy()
        expected = np.full((3, 4), 7.0)
        if np.allclose(np_data, expected):
            print("  ✓ Values verified correct!")
            print(f"  Result:\n{np_data}")
        else:
            print(f"  ✗ Values incorrect:\n{np_data}")
            print(f"  Expected:\n{expected}")
            return False

    except RuntimeError as e:
        print(f"  ✗ Failed with error: {e}")
        return False

    return True


def test_broadcasting_different_dims():
    """Test broadcasting with different number of dimensions."""
    print("\nTesting broadcasting with different dimensions...")

    # Create destination tensor [2, 3, 4]
    dst = tenzor.Tensor([2, 3, 4], tenzor.DType.Float32)
    dst.fill_(0.0)

    # Create source tensor [3, 1] that should broadcast to [2, 3, 4]
    src = tenzor.Tensor([3, 1], tenzor.DType.Float32)
    for i in range(3):
        idx = tenzor.Tensor([i], tenzor.DType.Int64)
        # Fill each row with different value
        src[i] = float(i + 1)

    print(f"  Destination shape: {dst.shape()}")
    print(f"  Source shape: {src.shape()}")

    try:
        dst[:] = src
        print("  ✓ Broadcasting assignment with different dims succeeded!")

        # Verify the values
        np_data = dst.numpy()
        print(f"  Result shape: {np_data.shape}")

        # Each row should have the broadcast pattern
        success = True
        for i in range(2):
            for j in range(3):
                expected_val = float(j + 1)
                if not np.allclose(np_data[i, j, :], expected_val):
                    print(f"  ✗ Values incorrect at [{i},{j}]: {np_data[i, j, :]}")
                    success = False
                    break

        if success:
            print("  ✓ Values verified correct!")
        return success

    except RuntimeError as e:
        print(f"  ✗ Failed with error: {e}")
        return False


def test_scalar_broadcasting():
    """Test broadcasting from scalar."""
    print("\nTesting scalar broadcasting...")

    # Create destination tensor [2, 3]
    dst = tenzor.Tensor([2, 3], tenzor.DType.Float32)
    dst.fill_(0.0)

    # Create scalar tensor
    src = tenzor.Tensor([], tenzor.DType.Float32)
    src.fill_(9.0)

    print(f"  Destination shape: {dst.shape()}")
    print(f"  Source shape (scalar): {src.shape()}")

    try:
        dst[:] = src
        print("  ✓ Scalar broadcasting assignment succeeded!")

        # Verify all values are 9.0
        np_data = dst.numpy()
        if np.allclose(np_data, 9.0):
            print("  ✓ Values verified correct!")
        else:
            print(f"  ✗ Values incorrect: {np_data}")
            return False

    except RuntimeError as e:
        print(f"  ✗ Failed with error: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("Testing Tensor Assignment Fixes")
    print("=" * 60)

    all_passed = True

    # Run all tests
    all_passed &= test_non_contiguous_assignment()
    all_passed &= test_broadcasting_assignment()
    all_passed &= test_broadcasting_different_dims()
    all_passed &= test_scalar_broadcasting()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
